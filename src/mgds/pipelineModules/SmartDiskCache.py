import concurrent
import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import threading
import time
from collections.abc import Callable
from typing import Any

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.SingleVariationRandomAccessPipelineModule import SingleVariationRandomAccessPipelineModule

import torch

import xxhash
from tqdm import tqdm

CACHE_VERSION = 3

# Bumped whenever ``_augment_cache_with_missing_names`` changes behaviorally so
# caches stamped by an older version get re-augmented on the new path.
SCHEMA_METHOD = "shape_v1"

# Sentinel resolution key used for sourceless / text-style entries that have
# no resolution dimension. Lives inside ``entry['variants']`` like any other
# variant key, but the cache filename collapses to bare ``hash12``.
NO_RESOLUTION_KEY = "_"

# Aggregate names whose value at load time can be reconstructed from cache
# metadata (variant key + entry filepath) without reading the .pt. When every
# configured aggregate is in this set, _load_aggregate_cache skips torch.load
# entirely — the .pt is dominated by split tensors that aggregate-load doesn't
# need, so deriving these instead saves multi-GB of I/O per epoch on large
# datasets. Falls back to torch.load per item when the variant key won't parse
# (NO_RESOLUTION_KEY, malformed) or frame-dim-enabled video where 'HxW' can't
# encode the frame dim.
_DERIVABLE_AGGREGATES = frozenset({"crop_resolution", "image_path"})


class CachingStoppedException(Exception):
    pass


class SmartDiskCache(
    PipelineModule,
    SingleVariationRandomAccessPipelineModule,
):
    def __init__(
        self,
        cache_dir: str,
        split_names: list[str] | None = None,
        aggregate_names: list[str] | None = None,
        variations_in_name: str | None = None,
        balancing_in_name: str | None = None,
        balancing_strategy_in_name: str | None = None,
        variations_group_in_name: str | list[str] | None = None,
        group_enabled_in_name: str | None = None,
        before_cache_fun: Callable[[], None] | None = None,
        stop_check_fun: Callable[[], bool] | None = None,
        modeltype: str = "",
        source_path_in_name: str | None = None,
        sourceless: bool = False,
        tolerate_missing_source: bool = False,
        resolution_from_upstream: bool = False,
        bucket_method_provider: Callable[[], str] | None = None,
        rebucket_provider: Callable[[float], list[str]] | None = None,
        aspect_bucketing=None,
        extra_watched_paths_in_names: list[str] | None = None,
        trust_cache: bool = False,
        content_key_in_name: str | None = None,
        build_max_workers: int | None = None,
    ):
        super().__init__()
        self.cache_dir = cache_dir
        self._real_cache_dir = os.path.realpath(cache_dir)

        self.split_names = [] if split_names is None else split_names
        self.aggregate_names = [] if aggregate_names is None else aggregate_names

        self.variations_in_name = variations_in_name
        self.balancing_in_name = balancing_in_name
        self.balancing_strategy_in_name = balancing_strategy_in_name
        self.variations_group_in_names = (
            [variations_group_in_name] if isinstance(variations_group_in_name, str) else variations_group_in_name
        )

        self.group_enabled_in_name = group_enabled_in_name

        self.before_cache_fun = (lambda: None) if before_cache_fun is None else before_cache_fun
        self.stop_check_fun = stop_check_fun or (lambda: False)

        self.modeltype = modeltype
        self.source_path_in_name = source_path_in_name
        self.sourceless = sourceless
        # Per-instance trust mode: skip per-file mtime/hash/.pt-existence
        # validation for entries already in the index (same semantics as the
        # OT_SKIP_CACHE_VALIDATION=1 env var, but config-driven so it follows
        # each run's TrainConfig instead of lingering process state).
        self.trust_cache = trust_cache
        # When True, a missing source file is treated as "trust the cache
        # entry as-is" instead of forcing a rebuild. Used by sidecar caches
        # whose source path may legitimately not exist on disk (e.g. mask
        # caches where the per-image mask file is optional and synthesised
        # by an upstream pipeline module when absent). Once cached, those
        # entries stay valid even though no source file is there to stat.
        self.tolerate_missing_source = tolerate_missing_source
        # Sidecar caches whose source path differs from the image path
        # (e.g. mask cache keyed on -masklabel.png) must serve the variant
        # matching the *image's* current bucket, not whatever variant key
        # happens to come first in this entry's dict. With this flag,
        # per-entry validation pulls crop_resolution from upstream and
        # uses it to set the active variant key, and drift recovery is
        # skipped (its variant-derived aspect can disagree with upstream
        # when the sidecar's own aspect differs from the image's).
        self.resolution_from_upstream = resolution_from_upstream

        # Optional providers for multi-resolution variant caching. Both default
        # to None for text caches and any data loader that doesn't construct
        # an AspectBucketing module — drift detection auto-disables in that
        # case and the cache behaves as a single-variant store keyed on the
        # cached resolution (or NO_RESOLUTION_KEY when there's no resolution
        # at all).
        self.bucket_method_provider = bucket_method_provider
        self.rebucket_provider = rebucket_provider
        # Handle to the upstream AspectBucketing module. When set, the cache
        # builder uses ``aspect_bucketing._target_override`` to drive get_item
        # to a specific target_resolution per build pass, enabling lazy
        # multi-variant builds without re-decoding images. None for text
        # caches and any pipeline without aspect bucketing.
        self.aspect_bucketing = aspect_bucketing

        # Content-addressed reuse: name of an upstream item (e.g. the final
        # post-augmentation ``prompt`` string) whose value uniquely determines
        # this cache's split payload for a given (variation, index). When set,
        # ``_build_cache_entry`` consults a persistent content_index.json
        # (content hash -> .pt filename) before walking upstream: a hit is
        # served by copying the existing .pt (refreshing per-entry __concept
        # metadata) instead of re-running the encoder. This makes caption
        # edits cost one encode per *changed line* instead of one per
        # (file, variation): editing 1 of 6 lines reuses the other 5, a bulk
        # "append trigger word line to every file" encodes the new line once
        # for the whole dataset, and re-ordering lines costs zero encodes.
        # Only meaningful for caches whose entries have no resolution
        # dimension (text caches) — reuse is gated on ``resolution is None``.
        self.content_key_in_name = content_key_in_name

        # content hash -> .pt filename, persisted as content_index.json in
        # the cache dir. Loaded lazily on the first build pass that could use
        # it; entries whose .pt vanished (GC, manual deletion) are pruned at
        # load time. Guarded by _index_lock.
        self._content_index: dict[str, str] = {}
        self._content_index_loaded = False
        self._content_index_dirty = False

        # before_cache_fun is deferred to the first variation that actually
        # needs an upstream encode (see _ensure_before_cache_called). A build
        # pass served entirely by dedup or content-addressed copies never
        # moves the encoder onto the GPU at all.
        self._before_cache_called = False
        self._before_cache_lock = threading.Lock()

        # When set (> 1), the build pass fans out over a dedicated
        # ThreadPoolExecutor with this many workers instead of the shared
        # pipeline executor (sized by dataloader_threads, default 2). Used by
        # text caches so enough encode requests are in flight to fill the
        # encoder's batch collector. The pipeline walker is thread-safe
        # (thread-local per-module item caches), and the encoder modules
        # serialize their forwards internally, so extra walkers are safe.
        self.build_max_workers = build_max_workers

        # Additional per-sample sidecar in-name fields whose resolved paths
        # should be watched alongside ``source_path_in_name``. Touching one of
        # these files (e.g. a -masklabel.png sidecar that controls the
        # ``latent_mask`` baked into the .pt) invalidates the entry. Empty by
        # default: every existing caller is a no-op.
        self.extra_watched_paths_in_names = list(extra_watched_paths_in_names or [])

        # filepath -> {in_name: normpath} populated during __refresh_cache and
        # read by _validate_entry / _build_cache_entry / _try_dedup so build
        # threads don't need to walk the upstream pipeline themselves.
        self._extra_paths_by_filepath: dict[str, dict[str, str]] = {}

        self.group_variations = {}
        self.group_indices = {}
        self.group_output_samples = {}
        self.group_full_indices = {}
        self.group_balancing_strategy = {}
        self.group_balancing = {}
        self.variations_initialized = False
        self._sourceless_filepaths = []

        self.cache_index = None
        self._index_lock = threading.Lock()
        self._last_flush_time = 0.0
        self._source_path_cache = {}
        self._aggregate_cache = {}
        # Source filepaths whose cache entries have already been validated in
        # this process. Once a filepath is in this set we skip re-validating
        # it for the rest of the run — the dataset is static within a run
        # (users use repeats, not changing samples_per_epoch) so the
        # per-epoch revalidation was pure overhead.
        self._session_validated_filepaths: set[str] = set()

        # Filled at the start of __refresh_cache by a single os.scandir of the
        # cache dir, then consulted for .pt existence checks instead of one
        # os.path.isfile syscall per (entry, variation). Updated by builders.
        self._existing_pt_files: set[str] = set()
        # Filled at the start of __refresh_cache by parallel os.scandir of the
        # source files' parent dirs. Lookup replaces per-file os.path.getmtime
        # in _validate_entry.
        self._source_mtimes: dict[str, float] = {}
        # Resolution variant key in use this session, per source filepath.
        # Set during validation/build; consulted by get_item and
        # _load_aggregate_cache to load the right variant's .pt. Empty for
        # text caches (no resolution dimension) — those callers fall back to
        # _any_variant_cache_file. Reset at the top of __refresh_cache.
        self._active_key_by_filepath: dict[str, str] = {}

    def length(self) -> int:
        if not self.variations_initialized:
            name = self.split_names[0] if len(self.split_names) > 0 else self.aggregate_names[0]
            return self._get_previous_length(name)
        else:
            return sum(x for x in self.group_output_samples.values())

    def get_inputs(self) -> list[str]:
        inputs = self.split_names + self.aggregate_names
        if self.source_path_in_name:
            inputs = inputs + [self.source_path_in_name]
        if self.content_key_in_name:
            inputs = inputs + [self.content_key_in_name]
        if self.variations_in_name:
            inputs = (
                inputs
                + [self.variations_in_name]
                + [self.balancing_in_name]
                + [self.balancing_strategy_in_name]
                + self.variations_group_in_names
                + [self.group_enabled_in_name]
            )
        return inputs

    def get_outputs(self) -> list[str]:
        outputs = self.split_names + self.aggregate_names
        if self.sourceless:
            outputs.append("concept")
        return outputs

    def __string_key(self, data: list[Any]) -> str:
        json_data = json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(",", ":"), indent=None)
        return hashlib.sha256(json_data.encode("utf-8")).hexdigest()

    def _hash_file(self, filepath: str) -> str:
        h = xxhash.xxh64()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _hash_to_filename(self, full_hash: str) -> str:
        return full_hash[:12]

    def _get_cache_json_path(self) -> str:
        return os.path.join(self.cache_dir, "cache.json")

    def _load_cache_index(self) -> dict:
        cache_path = self._get_cache_json_path()
        tmp_path = cache_path + ".tmp"
        bak_path = cache_path + ".bak"

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                for p in (tmp_path, bak_path):
                    with contextlib.suppress(OSError):
                        os.remove(p)
                self._migrate_legacy_index_in_place(data)
                return data
            except (json.JSONDecodeError, OSError):
                pass

        if os.path.exists(tmp_path):
            try:
                with open(tmp_path, "r") as f:
                    data = json.load(f)
                os.replace(tmp_path, cache_path)
                self._migrate_legacy_index_in_place(data)
                return data
            except (json.JSONDecodeError, OSError):
                pass

        if os.path.exists(bak_path):
            try:
                with open(bak_path, "r") as f:
                    data = json.load(f)
                os.replace(bak_path, cache_path)
                self._migrate_legacy_index_in_place(data)
                return data
            except (json.JSONDecodeError, OSError):
                pass

        return {"version": CACHE_VERSION, "entries": {}, "hash_index": {}}

    @staticmethod
    def _migrate_legacy_index_in_place(idx: dict) -> None:
        """Lift v2 cache-index entries into v3 single-variant shape.

        v2 entries had top-level ``cache_file`` and ``resolution`` fields. v3
        moves those into ``variants[<resolution_key>]['cache_file']`` so a
        single source file can hold multiple resolution variants side-by-side.
        Existing on-disk ``.pt`` files are reused verbatim — no rebuild — and
        stay referenced under whatever resolution they were originally built
        for. New variants get added as the user trains at new resolutions.

        Idempotent. Cheap (dict munging, no I/O). Called every time the index
        is loaded.
        """
        if idx.get("version", 0) >= CACHE_VERSION:
            return
        for entry in idx.get("entries", {}).values():
            if "variants" in entry:
                continue
            legacy_cache_file = entry.pop("cache_file", None)
            legacy_resolution = entry.pop("resolution", None)
            if legacy_cache_file:
                key = legacy_resolution if legacy_resolution else NO_RESOLUTION_KEY
                entry["variants"] = {key: {"cache_file": legacy_cache_file}}
            else:
                entry["variants"] = {}
            entry["cache_version"] = CACHE_VERSION
        idx["version"] = CACHE_VERSION

    def _save_cache_index(self, compact: bool = False):
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = self._get_cache_json_path()
        tmp_path = cache_path + ".tmp"
        bak_path = cache_path + ".bak"

        with self._index_lock:
            with open(tmp_path, "w") as f:
                json.dump(self.cache_index, f, indent=None if compact else 2)

            if os.path.exists(cache_path):
                with contextlib.suppress(OSError):
                    shutil.copy2(cache_path, bak_path)

            os.replace(tmp_path, cache_path)

    def _flush_cache_index(self):
        now = time.monotonic()
        if now - self._last_flush_time >= 30.0:
            self._save_cache_index(compact=True)
            self._save_content_index()
            self._last_flush_time = now

    # ------------------------------------------------------------------
    # Content-addressed reuse (content_key_in_name)
    # ------------------------------------------------------------------

    def _get_content_index_path(self) -> str:
        return os.path.join(self.cache_dir, "content_index.json")

    def _ensure_content_index_loaded(self):
        """Load content_index.json once per refresh, pruning entries whose
        .pt no longer exists on disk. Must run after _scan_existing_pt_files
        so the existence check is a set lookup, not a syscall per entry.
        Only called when a build pass has items to build, so refreshes with
        a fully-valid cache never pay the JSON load.
        """
        if self._content_index_loaded or not self.content_key_in_name:
            return
        self._content_index_loaded = True
        self._content_index = {}
        self._content_index_dirty = False
        try:
            with open(self._get_content_index_path(), "r") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        entries = raw.get("entries", {}) if isinstance(raw, dict) else {}
        pruned = {h: pt for h, pt in entries.items() if isinstance(pt, str) and pt in self._existing_pt_files}
        self._content_index = pruned
        if len(pruned) != len(entries):
            self._content_index_dirty = True

    def _save_content_index(self):
        if not self.content_key_in_name or not self._content_index_dirty:
            return
        path = self._get_content_index_path()
        tmp_path = path + ".tmp"
        with self._index_lock:
            payload = {"version": 1, "entries": dict(self._content_index)}
            self._content_index_dirty = False
        try:
            with open(tmp_path, "w") as f:
                json.dump(payload, f, indent=None, separators=(",", ":"))
            os.replace(tmp_path, path)
        except OSError:
            with contextlib.suppress(OSError):
                os.remove(tmp_path)

    def _content_hash(self, text: str) -> str:
        """Hash of everything that determines a cached variation's payload:
        the exact upstream text plus the cache schema and modeltype. Two
        different models (or schema configurations) sharing a cache dir can
        therefore never alias each other's entries.
        """
        required = sorted(set(self.split_names) | set(self.aggregate_names))
        payload = json.dumps(
            [CACHE_VERSION, self.modeltype, required, text],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        return xxhash.xxh64(payload.encode("utf-8")).hexdigest()

    def _get_content_key_text(self, in_variation: int, in_index: int) -> str | None:
        """Resolve the content key (post-augmentation prompt) for one
        variation. Walks only the cheap string-processing chain — the
        encoder modules consume this name, they don't produce it, so the
        walk never triggers a forward pass.
        """
        if not self.content_key_in_name:
            return None
        try:
            value = self._get_previous_item(in_variation, self.content_key_in_name, in_index)
        except Exception:
            return None
        return value if isinstance(value, str) else None

    def _register_content_pt(self, content_hash: str | None, pt_name: str):
        if content_hash is None:
            return
        with self._index_lock:
            existing = self._content_index.get(content_hash)
            if existing is None or existing not in self._existing_pt_files:
                self._content_index[content_hash] = pt_name
                self._content_index_dirty = True

    def _try_content_reuse(
        self,
        content_hash: str | None,
        pt_name: str,
        in_variation: int,
        in_index: int,
    ) -> bool:
        """Serve one variation by copying a content-identical .pt.

        The donor was produced by the same upstream pipeline for the exact
        same text (hash covers modeltype + schema + text), so its split
        tensors are byte-identical to what a fresh encode would store. Only
        the per-entry ``__concept_*`` metadata can differ — the donor may
        belong to another concept — so those fields are refreshed from the
        current upstream walk before saving under this entry's name.
        Returns False on any inconsistency; the caller falls through to a
        normal encode.
        """
        if content_hash is None:
            return False
        with self._index_lock:
            donor_name = self._content_index.get(content_hash)
        if not donor_name or donor_name == pt_name or donor_name not in self._existing_pt_files:
            return False
        donor_path = os.path.join(self._real_cache_dir, donor_name)
        try:
            cache_data = torch.load(donor_path, weights_only=False, map_location="cpu")
        except Exception:
            return False
        if not isinstance(cache_data, dict):
            return False
        required = set(self.split_names) | set(self.aggregate_names)
        present = {k for k in cache_data if not k.startswith("__")}
        if not required.issubset(present):
            return False

        cache_data["__cache_version"] = CACHE_VERSION
        cache_data["__modeltype"] = self.modeltype
        if self.source_path_in_name:
            for key in [k for k in cache_data if k.startswith("__concept_")]:
                del cache_data[key]
            try:
                concept = self._get_previous_item(in_variation, "concept", in_index)
                if concept is not None and isinstance(concept, dict):
                    cache_data["__concept_loss_weight"] = concept.get("loss_weight", 1.0)
                    cache_data["__concept_type"] = concept.get("type", "STANDARD")
                    cache_data["__concept_name"] = concept.get("name", "")
                    cache_data["__concept_path"] = concept.get("path", "")
                    cache_data["__concept_seed"] = concept.get("seed", 0)
            except Exception:
                pass

        real_pt_path = os.path.join(self._real_cache_dir, pt_name)
        tmp_path = real_pt_path + f".{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            torch.save(cache_data, tmp_path)
            os.replace(tmp_path, real_pt_path)
        except OSError:
            with contextlib.suppress(OSError):
                os.remove(tmp_path)
            return False
        return True

    def _ensure_before_cache_called(self):
        """Run before_cache_fun exactly once per refresh, on the first
        variation that actually needs an upstream encode. Build passes
        served entirely from dedup or content-addressed copies skip the
        model device shuffle entirely. Thread-safe: concurrent builders
        block until the first caller's prepare finishes.
        """
        if self._before_cache_called:
            return
        with self._before_cache_lock:
            if self._before_cache_called:
                return
            if self.before_cache_fun is not None:
                self.before_cache_fun()
            self._before_cache_called = True

    def _get_resolution_string(self, in_variation: int, in_index: int) -> str | None:
        if "crop_resolution" not in self.aggregate_names and not self.resolution_from_upstream:
            return None
        # Walk upstream at this module's current_variation, not the in-space
        # variation we were called with. crop_resolution is variation-
        # invariant by design, but the walker treats SingleVariation modules
        # as strict — when an upstream cache (image cache, in the mask-cache
        # case) has current_variation = N and we ask with variation 0, it
        # raises and we silently lose resolution for every item. Aligning
        # with current_variation keeps the path open across resumed runs.
        variation = self.current_variation if self.current_variation >= 0 else in_variation
        try:
            res = self._get_previous_item(variation, "crop_resolution", in_index)
        except Exception:
            return None
        if res is not None:
            return f"{res[0]}x{res[1]}"
        return None

    def _make_cache_file(self, full_hash: str, resolution: str | None) -> str:
        hash12 = self._hash_to_filename(full_hash)
        if resolution:
            return f"{hash12}_{resolution}"
        return hash12

    @staticmethod
    def _resolution_key(resolution: str | None) -> str:
        return resolution if resolution else NO_RESOLUTION_KEY

    def _target_int_for_resolution_key(self, resolution: str | None) -> int | None:
        """Recover the AspectBucketing target_resolution int from a bucket key.

        Used by the cache build pass to drive ``aspect_bucketing._target_override``
        for a queued (filepath, resolution) item. Looks up the exact (h, w) in
        ``bucket_resolutions`` (an authoritative reverse map populated by
        ``AspectBucketing.start``). Returns None for text caches, sourceless
        flows, or any key shape we can't parse — callers treat None as "no
        override needed".
        """
        if self.aspect_bucketing is None or not resolution:
            return None
        if resolution == NO_RESOLUTION_KEY:
            return None
        parts = resolution.split("x")
        if len(parts) != 2:
            return None
        try:
            h = int(parts[0])
            w = int(parts[1])
        except ValueError:
            return None
        bucket_resolutions = getattr(self.aspect_bucketing, "bucket_resolutions", None) or {}
        for target, buckets in bucket_resolutions.items():
            if (h, w) in buckets:
                return target
        return None

    def _compute_bucket_method(self) -> str | None:
        """Hash of the upstream bucket config, or None when no provider is wired.

        None means "drift detection disabled" — the cache behaves as a single-
        variant store and never re-derives keys. Used by text caches and any
        data loader without an AspectBucketing module.
        """
        if self.bucket_method_provider is None:
            return None
        try:
            return self.bucket_method_provider()
        except Exception:
            return None

    def _populate_active_keys(self, filepaths) -> None:
        """Set this session's active variant key per filepath to the first
        registered variant. Used by the session-skip and fast-validate paths
        which don't run per-entry validation. Reads the dict ordering set up
        by drift recovery (when applicable) so the first key is the one for
        the current bucket config.

        Skipped under ``resolution_from_upstream`` — that mode resolves the
        active key per-item at ``get_item`` time from upstream
        ``crop_resolution`` so a stale dict ordering can't pin the wrong
        variant for the session.
        """
        if self.resolution_from_upstream:
            return
        entries = self.cache_index.get("entries", {})
        for fp in filepaths:
            entry = entries.get(fp)
            if entry is None:
                continue
            self._active_key_by_filepath[fp] = next(
                iter(entry.get("variants", {}).keys()),
                NO_RESOLUTION_KEY,
            )

    def _drift_recovery_pass(self) -> None:
        """Re-derive variant keys from cached aspects when bucket config drifted.

        For each entry:
          1. Parse aspect from any existing variant key (e.g. ``"896x640"``
             → 1.4).
          2. Call the rebucket provider to get the resolution keys the
             *current* bucket config would assign for that aspect.
          3. Reuse any pre-existing .pt file matching a derived key (cache
             hit via dedup, no rebuild). Other keys are left for the
             per-index loop to rebuild.
          4. Reorder ``entry['variants']`` so newly-derived keys come first.
             This way the "first variant" heuristic in validation/get_item
             picks the current-config key.
          5. Pre-set ``_active_key_by_filepath`` to the first derived key
             for each known filepath so the per-index loop validates
             against the new key (and rebuilds it if no .pt exists).

        Pure arithmetic — no image decode. O(N) cheap work.
        """
        if self.rebucket_provider is None or self.resolution_from_upstream:
            return
        entries = self.cache_index.get("entries")
        if not entries:
            return
        # Build a reverse map filename → filepath so we can pre-set the
        # active key per filepath. Entries are keyed by filepath already, so
        # this is just a list comprehension.
        derived = 0
        reused = 0
        with self._index_lock:
            for entry in entries.values():
                if entry.get("modeltype") != self.modeltype:
                    continue
                variants = entry.get("variants") or {}
                aspect = self._aspect_from_variant_keys(variants)
                if aspect is None:
                    continue
                try:
                    new_keys = self.rebucket_provider(aspect)
                except Exception:
                    continue
                if not new_keys:
                    continue
                file_hash = entry.get("hash")
                if not file_hash:
                    continue
                for nk in new_keys:
                    if nk in variants:
                        continue
                    cache_file = self._make_cache_file(
                        file_hash,
                        None if nk == NO_RESOLUTION_KEY else nk,
                    )
                    if f"{cache_file}_1.pt" in self._existing_pt_files:
                        variants[nk] = {"cache_file": cache_file}
                        reused += 1
                    derived += 1
                # Move newly-derived keys to the front so the "first
                # variant" picks them, preserving any reused .pt links.
                new_keys_set = set(new_keys)
                reordered = {nk: variants[nk] for nk in new_keys if nk in variants}
                reordered.update((k, v) for k, v in variants.items() if k not in new_keys_set)
                entry["variants"] = reordered
                # Active key is now resolved per-epoch from upstream in
                # ``__refresh_cache`` and ``get_item``. Don't pin to the
                # first derived key — that biased every image to the
                # smallest target on the next run. The reordering above
                # still helps text-cache callers via the
                # ``_any_variant_cache_file`` first-key fallback.
        if derived:
            print(
                f"SmartDiskCache: bucket method drift — re-derived {derived} variant keys "
                f"({reused} reused from disk, no image decode)."
            )

    @staticmethod
    def _aspect_from_variant_keys(variants: dict) -> float | None:
        """Recover aspect ratio from any HxW-shaped variant key. None if
        none of the keys parse (e.g. only ``NO_RESOLUTION_KEY`` present).
        """
        for k in variants:
            if k == NO_RESOLUTION_KEY:
                continue
            parts = k.split("x")
            if len(parts) != 2:
                continue
            try:
                h = int(parts[0])
                w = int(parts[1])
            except ValueError:
                continue
            if w <= 0:
                continue
            return h / w
        return None

    def _pt_path(self, cache_file: str, variation: int) -> str:
        return os.path.join(self.cache_dir, f"{cache_file}_{variation + 1}.pt")

    def _real_pt_path(self, cache_file: str, variation: int) -> str:
        return os.path.join(self._real_cache_dir, f"{cache_file}_{variation + 1}.pt")

    def _scan_existing_pt_files(self) -> set[str]:
        """Single os.scandir of the real cache dir; returns set of .pt filenames.

        On Windows, os.scandir uses FindFirstFile/FindNextFile which returns
        attributes in the same syscall as the directory enumeration, so
        DirEntry.is_file() needs no extra stat. This collapses the
        per-(entry, variation) os.path.isfile loop to one syscall per dir.
        """
        found: set[str] = set()
        try:
            with os.scandir(self._real_cache_dir) as it:
                for e in it:
                    if e.name.endswith(".pt") and e.is_file(follow_symlinks=False):
                        found.add(e.name)
        except OSError:
            pass
        return found

    def _bulk_stat_source_files(self, filepaths) -> dict[str, float]:
        """Parallel os.scandir per parent dir; returns {normpath: mtime}.

        Replaces N×os.path.getmtime syscalls with K×os.scandir (K = number of
        distinct parent dirs). Files that don't exist are simply absent from
        the returned dict (so callers must treat a missing key as "rebuild").
        """
        by_parent: dict[str, set[str]] = {}
        for fp in filepaths:
            by_parent.setdefault(os.path.dirname(fp), set()).add(os.path.basename(fp))

        if not by_parent:
            return {}

        mtimes: dict[str, float] = {}
        lock = threading.Lock()

        def scan_one(parent: str, names: set[str]):
            local: dict[str, float] = {}
            try:
                with os.scandir(parent) as it:
                    for e in it:
                        if e.name in names:
                            with contextlib.suppress(OSError):
                                local[os.path.normpath(os.path.join(parent, e.name))] = e.stat().st_mtime
            except OSError:
                pass
            if local:
                with lock:
                    mtimes.update(local)

        futures = [self._state.executor.submit(scan_one, p, n) for p, n in by_parent.items()]
        for f in futures:
            f.result()
        return mtimes

    def _compute_watched_fingerprints(self, entries: dict) -> dict[str, tuple[int, float]] | None:
        """Per-parent-dir fingerprint of (count, mtime_sum) restricted to the
        basenames present in `entries`, plus any sidecar paths each entry
        recorded under ``sidecar_mtimes``. Unrelated files in the same dir
        don't perturb the fingerprint. Returns None if any directory is
        unreachable.
        """
        by_parent: dict[str, set[str]] = {}
        for fp, entry in entries.items():
            by_parent.setdefault(os.path.dirname(fp), set()).add(os.path.basename(fp))
            sidecar_paths = entry.get("sidecar_mtimes") if isinstance(entry, dict) else None
            if sidecar_paths:
                for sidecar_path in sidecar_paths:
                    by_parent.setdefault(os.path.dirname(sidecar_path), set()).add(os.path.basename(sidecar_path))

        fp_map: dict[str, tuple[int, float]] = {}
        for parent, names in by_parent.items():
            count = 0
            mtime_sum = 0.0
            try:
                with os.scandir(parent) as it:
                    for e in it:
                        if e.name not in names:
                            continue
                        try:
                            mtime = e.stat().st_mtime
                        except OSError:
                            continue
                        count += 1
                        mtime_sum += mtime
            except OSError:
                return None
            fp_map[parent] = (count, mtime_sum)
        return fp_map

    def _validate_entry(
        self, filepath: str, entry: dict, requested_key: str, variations: int, current_mtime: float | None
    ) -> str:
        """Validate ``entry``'s variant for ``requested_key`` against disk.

        Returns one of:
          - ``'valid'``: variant present, all .pt files exist, and each .pt
            holds every name in ``split_names + aggregate_names``.
          - ``'missing_variant'``: entry is fine but doesn't have this
            resolution key — caller queues a rebuild for *just this variant*
            and keeps the entry plus its other variants intact.
          - ``'missing_pt'``: variant key is registered but a .pt is missing
            from disk. Caller drops the broken variant and rebuilds it.
          - ``'incomplete_schema'``: .pt exists but is missing one or more
            required schema names (e.g. masked_training was enabled after
            this .pt was originally cached). Caller rebuilds the variant.
          - ``'content_changed'``: source file changed; full entry rebuild.
          - ``'rebuild'``: source file unreadable.
        """
        if entry["modeltype"] != self.modeltype:
            raise RuntimeError(
                f"Cache modeltype mismatch for '{filepath}': "
                f"cached as '{entry['modeltype']}', current model is '{self.modeltype}'. "
                f"Delete the cache directory or use a separate cache_dir for this model type."
            )

        if current_mtime is None:
            if not self.tolerate_missing_source:
                return "rebuild"
            # Source file is gone but the cache has an entry; trust it.
            # Reuse the equality-branch checks below by pinning current_mtime
            # to the stored value so we don't fall through to the hash path
            # (which would also fail without a readable file).
            current_mtime = entry["mtime"]

        variants = entry.get("variants", {})
        variant = variants.get(requested_key)

        if current_mtime == entry["mtime"]:
            if not self._check_sidecars(filepath, entry):
                return "content_changed"
            if variant is None:
                return "missing_variant"
            cache_file = variant["cache_file"]
            for v in range(variations):
                if f"{cache_file}_{v + 1}.pt" not in self._existing_pt_files:
                    return "missing_pt"
            if not self._variant_schema_is_complete(variant, cache_file, variations):
                return "incomplete_schema"
            return "valid"

        file_hash = self._hash_file(filepath)
        if file_hash != entry["hash"]:
            return "content_changed"

        entry["mtime"] = current_mtime

        if not self._check_sidecars(filepath, entry):
            return "content_changed"

        if variant is None:
            return "missing_variant"
        cache_file = variant["cache_file"]
        for v in range(variations):
            if f"{cache_file}_{v + 1}.pt" not in self._existing_pt_files:
                return "missing_pt"

        if not self._variant_schema_is_complete(variant, cache_file, variations):
            return "incomplete_schema"

        return "valid"

    def _variant_schema_is_complete(self, variant: dict, cache_file: str, variations: int) -> bool:
        """True if every required schema name is present in this variant's .pt.

        Uses an index-stored ``schema_keys`` list as a fast path so we avoid
        re-reading every .pt on every validation pass; legacy variants
        without that field get a one-time peek that populates it. Future
        validations are then cheap dict comparisons.
        """
        required = set(self.split_names) | set(self.aggregate_names)
        if not required:
            return True
        cached_keys = variant.get("schema_keys")
        if cached_keys is not None:
            return required.issubset(cached_keys)

        # Legacy variant: peek the .pt once and stamp schema_keys.
        real_pt = self._real_pt_path(cache_file, 0)
        try:
            data = torch.load(real_pt, weights_only=False, map_location="cpu")
        except Exception:
            return False
        present = sorted(k for k in data if not k.startswith("__"))
        with self._index_lock:
            variant["schema_keys"] = present
        return required.issubset(present)

    def _detect_cache_schema_drift(self) -> set[str]:
        """Return the currently-required names that aren't in on-disk cache files.

        Schema drift happens when a setting like ``masked_training`` is toggled
        between runs: ``split_names``/``aggregate_names`` now include keys that
        weren't written when the existing ``.pt`` files were built. All entries
        are produced by the same code path, so spot-checking one valid cache
        file is enough to determine the on-disk schema.
        """
        if not self.cache_index or not self.cache_index.get("entries"):
            return set()
        required = set(self.split_names) | set(self.aggregate_names)
        if not required:
            return set()

        for entry in self.cache_index["entries"].values():
            if entry.get("modeltype") != self.modeltype:
                continue
            cache_file = self._any_variant_cache_file(entry)
            if not cache_file:
                continue
            pt = self._real_pt_path(cache_file, 0)
            if not os.path.isfile(pt):
                continue
            try:
                cached = torch.load(pt, weights_only=False, map_location="cpu")
            except Exception:
                continue
            cached_keys = {k for k in cached if not k.startswith("__")}
            return required - cached_keys

        return set()

    @staticmethod
    def _any_variant_cache_file(entry: dict) -> str | None:
        """Return *any* variant's cache_file, or None if the entry has none.

        Used by code paths that don't care which resolution they're inspecting
        (sentinel templating, schema-drift spot check, sourceless aggregate
        load) — they just need *some* representative .pt.
        """
        variants = entry.get("variants") or {}
        for variant in variants.values():
            cache_file = variant.get("cache_file")
            if cache_file:
                return cache_file
        return None

    def _active_cache_file(self, filepath: str, entry: dict) -> str | None:
        """Cache file for the variant chosen for ``filepath`` this session.

        Falls back to any-variant when no active key has been recorded (e.g.
        sourceless mode, or a freshly-loaded entry not yet validated).
        """
        key = self._active_key_by_filepath.get(filepath)
        if key is not None:
            variant = entry.get("variants", {}).get(key)
            if variant is not None:
                cache_file = variant.get("cache_file")
                if cache_file:
                    return cache_file
        return self._any_variant_cache_file(entry)

    @staticmethod
    def _resize_to_ref_shape(value, ref_shape):
        """Force ``value``'s spatial dims to match ``ref_shape`` via interpolation.

        Why: when augmenting a cache built under different settings
        (e.g. ``masked_training`` toggled, which adds
        ``mask_augmentation`` modules to the upstream chain), the
        upstream pipeline can produce a different ``crop_resolution``
        than what's stored alongside ``latent_image``. Saving an
        augmented value with a divergent spatial shape later breaks
        ``torch.stack`` inside the dataloader's collate. Resizing
        keeps downstream batching working without rebuilding the
        cache from scratch -- the mask is approximate when the cache
        crosses pipelines, but a full re-encode of every entry is
        unacceptable for large datasets.
        """
        if ref_shape is None or not torch.is_tensor(value):
            return value
        if value.dim() < 2 or tuple(value.shape[-2:]) == ref_shape:
            return value

        needs_batch_dim = value.dim() == 3
        x = value.unsqueeze(0) if needs_batch_dim else value
        orig_dtype = x.dtype
        x = torch.nn.functional.interpolate(
            x.to(torch.float32),
            size=tuple(ref_shape),
            mode="bilinear",
            align_corners=False,
        ).to(orig_dtype)
        return x.squeeze(0) if needs_batch_dim else x

    def _augment_cache_with_missing_names(self, target_names: set[str]):
        """Ensure each cached entry has every name in ``target_names``.

        Walks every entry in the on-disk index, resolves it back to an
        ``in_index`` via the upstream pipeline, and (per variation):

        1. Loads the cached ``.pt``.
        2. Skips names that are already present and whose spatial shape
           matches ``latent_image``.
        3. For names that are missing **or** shape-mismatched, computes
           a fresh value via ``_get_previous_item`` and resizes it to the
           cached ``latent_image`` shape if the upstream-produced value
           has different spatial dims.
        4. Writes the merged dict back atomically.

        Existing correctly-shaped keys are preserved untouched, so this
        is a targeted backfill, not a full rebuild.
        """
        if not target_names:
            return

        sorted_targets = sorted(target_names)
        print(
            f"SmartDiskCache: ensuring cache contains {sorted_targets} "
            f"with shapes consistent against cached latent_image."
        )

        # Map every cached filepath to an upstream in_index and its variation count.
        augment_items: list[tuple[str, int, int]] = []
        seen_paths: set[str] = set()
        for group_key, variations in self.group_variations.items():
            for in_index in self.group_full_indices.get(group_key, []):
                filepath = self._get_source_path(0, in_index)
                if filepath is None:
                    continue
                filepath = os.path.normpath(filepath)
                if filepath in seen_paths:
                    continue
                entry = self.cache_index["entries"].get(filepath)
                if entry is None:
                    continue
                if entry.get("modeltype") != self.modeltype:
                    continue
                seen_paths.add(filepath)
                augment_items.append((filepath, in_index, variations))

        if not augment_items:
            return

        if self.before_cache_fun is not None:
            self.before_cache_fun()

        current_device = torch.cuda.current_device() if torch.cuda.is_available() else None

        def fn(filepath, in_index, variations, current_device):
            if torch.cuda.is_available() and current_device is not None:
                torch.cuda.set_device(current_device)

            entry = self.cache_index["entries"].get(filepath)
            if entry is None:
                return filepath, "no_entry"
            variants = entry.get("variants") or {}
            if not variants:
                return filepath, "no_cache_file"

            wrote_anything = False
            for variant in variants.values():
                cache_file = variant.get("cache_file")
                if not cache_file:
                    continue
                for v in range(variations):
                    real_pt = self._real_pt_path(cache_file, v)
                    if not os.path.isfile(real_pt):
                        continue
                    try:
                        cache_data = torch.load(real_pt, weights_only=False, map_location="cpu")
                    except Exception as e:
                        return filepath, f"load_failed: {e}"

                    # Reference shape from cached latent_image; everything spatial
                    # in the same .pt must match this so collate_fn can stack the
                    # batch without crashing.
                    ref_shape = None
                    latent_image = cache_data.get("latent_image")
                    if torch.is_tensor(latent_image) and latent_image.dim() >= 2:
                        ref_shape = tuple(latent_image.shape[-2:])

                    needs_work: list[str] = []
                    for name in sorted_targets:
                        cached_val = cache_data.get(name)
                        if cached_val is None:
                            needs_work.append(name)
                            continue
                        if (
                            ref_shape is not None
                            and torch.is_tensor(cached_val)
                            and cached_val.dim() >= 2
                            and tuple(cached_val.shape[-2:]) != ref_shape
                        ):
                            needs_work.append(name)

                    if not needs_work:
                        continue

                    try:
                        with torch.no_grad():
                            for name in needs_work:
                                value = self._get_previous_item(v, name, in_index)
                                value = self._resize_to_ref_shape(value, ref_shape)
                                cache_data[name] = self.__clone_for_cache(value)
                    except Exception as e:
                        return filepath, f"compute_failed: {e}"

                    tmp_path = real_pt + f".{os.getpid()}.{threading.get_ident()}.aug.tmp"
                    try:
                        torch.save(cache_data, tmp_path)
                        os.replace(tmp_path, real_pt)
                        wrote_anything = True
                    except Exception as e:
                        with contextlib.suppress(OSError):
                            os.remove(tmp_path)
                        return filepath, f"save_failed: {e}"

            return filepath, "augmented" if wrote_anything else "already_consistent"

        failed: list[tuple[str, str]] = []
        skipped = 0
        with tqdm(total=len(augment_items), smoothing=0.1, desc="augmenting cache") as bar:
            fs = [self._state.executor.submit(fn, fp, idx, var, current_device) for fp, idx, var in augment_items]
            for count, f in enumerate(concurrent.futures.as_completed(fs), 1):
                try:
                    fp, status = f.result()
                except Exception:
                    self._state.executor.shutdown(wait=True, cancel_futures=True)
                    raise
                if status == "already_consistent":
                    skipped += 1
                elif status not in ("augmented", "no_entry", "no_cache_file"):
                    failed.append((fp, status))
                if status in ("augmented", "already_consistent"):
                    # Stamp every variant of this entry with the current
                    # schema so future _validate_entry calls don't re-trigger
                    # augmentation. Only entries whose augment actually
                    # succeeded get stamped — failures stay flagged.
                    entry = self.cache_index["entries"].get(fp)
                    if entry is not None:
                        new_keys = sorted(set(self.split_names) | set(self.aggregate_names))
                        with self._index_lock:
                            for variant in (entry.get("variants") or {}).values():
                                variant["schema_keys"] = new_keys
                if count % 250 == 0:
                    self._torch_gc()
                bar.update(1)
                if self.stop_check_fun():
                    self._state.executor.shutdown(wait=True, cancel_futures=True)
                    raise CachingStoppedException

        succeeded = len(augment_items) - len(failed) - skipped
        print(
            f"SmartDiskCache: augmentation complete "
            f"({succeeded} updated, {skipped} already consistent, "
            f"{len(failed)} failed)."
        )
        if failed:
            for fp, reason in failed[:10]:
                print(f"  augment failed: {fp}: {reason}")
            if len(failed) > 10:
                print(f"  ... and {len(failed) - 10} more")

    def _fast_validate(self) -> bool:
        """Fast validation: per-watched-file directory fingerprint + random spot check.

        Skips the expensive per-file validation loop when nothing has changed.
        Returns True if cache appears valid.

        The fingerprint is restricted to source basenames in the cache index
        plus any explicitly-watched sidecars (e.g. -masklabel.png paths recorded
        on each entry's ``sidecar_mtimes``). Unrelated files in the same
        directory don't perturb the fingerprint. Renames within a directory
        that don't shift count or mtime sum are caught by the spot check
        downstream.
        """
        last_validated = self.cache_index.get("last_validated")
        if last_validated is None:
            return False

        entries = self.cache_index.get("entries", {})
        if not entries:
            return False

        stored_fp_raw = self.cache_index.get("watched_fingerprints")
        if stored_fp_raw is None:
            # Legacy cache without fingerprint — force a full pass so the
            # fingerprint gets written for next time.
            return False

        # Force migration: sidecar watching is configured but some entries
        # pre-date the feature and have no recorded sidecar state. Fall
        # through to slow validation so _check_sidecars can populate it.
        if self.extra_watched_paths_in_names and any(("sidecar_mtimes" not in e) for e in entries.values()):
            return False
        current_fp = self._compute_watched_fingerprints(entries)
        if current_fp is None:
            return False
        # JSON round-trip turns tuples into lists; normalise before compare.
        stored_fp = {k: tuple(v) for k, v in stored_fp_raw.items()}
        if stored_fp != current_fp:
            return False

        # Spot-check: mtime compare + .pt existence (no hashing)
        # Small datasets: check everything (still instant). Large: random sample.
        all_keys = list(entries.keys())
        if len(all_keys) <= 100:
            sample_keys = all_keys
            sample_size = len(all_keys)
        else:
            sample_size = min(50, max(10, len(all_keys) // 20))
            sample_keys = random.sample(all_keys, sample_size)

        for filepath in sample_keys:
            entry = entries[filepath]
            if entry.get("modeltype") != self.modeltype:
                return False
            try:
                current_mtime = os.path.getmtime(filepath)
            except OSError:
                if not self.tolerate_missing_source:
                    return False
                # Missing source under tolerate mode: cache entry is
                # authoritative. Skip the mtime equality check; only the
                # .pt existence check below still applies.
                current_mtime = entry.get("mtime")
            if current_mtime != entry.get("mtime"):
                return False
            cache_file = self._any_variant_cache_file(entry)
            if cache_file and f"{cache_file}_1.pt" not in self._existing_pt_files:
                return False

        self._fast_validate_sample_size = sample_size
        return True

    def _add_to_hash_index(self, file_hash: str, filepath: str):
        if file_hash not in self.cache_index["hash_index"]:
            self.cache_index["hash_index"][file_hash] = []
        if filepath not in self.cache_index["hash_index"][file_hash]:
            self.cache_index["hash_index"][file_hash].append(filepath)

    def _remove_from_hash_index(self, file_hash: str, filepath: str):
        if file_hash in self.cache_index["hash_index"]:
            paths = self.cache_index["hash_index"][file_hash]
            if filepath in paths:
                paths.remove(filepath)
            if not paths:
                del self.cache_index["hash_index"][file_hash]

    def _try_dedup(self, filepath: str, file_hash: str, resolution: str | None, mtime: float) -> bool:
        """Reuse an existing entry's variant when content+resolution match.

        Hash-index lookup finds entries built from the same source bytes; we
        only dedup when one of those entries already has a variant at the
        requested resolution key. Different resolution requests of the same
        content are still allowed to share a single entry — we just register
        an additional variant under a different key.
        """
        key = self._resolution_key(resolution)
        with self._index_lock:
            if file_hash not in self.cache_index["hash_index"]:
                return False

            for existing_path in self.cache_index["hash_index"][file_hash]:
                existing_entry = self.cache_index["entries"].get(existing_path)
                if existing_entry is None:
                    continue
                if existing_entry["modeltype"] != self.modeltype:
                    continue
                existing_variants = existing_entry.get("variants", {})
                variant = existing_variants.get(key)
                if variant is None:
                    continue
                cache_file = variant["cache_file"]

                target_entry = self.cache_index["entries"].get(filepath)
                if target_entry is None:
                    sidecar_mtimes, sidecar_hashes = self._compute_sidecar_state(
                        self._extra_paths_by_filepath.get(filepath, {})
                    )
                    target_entry = {
                        "filename": os.path.basename(filepath),
                        "hash": file_hash,
                        "mtime": mtime,
                        "modeltype": self.modeltype,
                        "variants": {},
                        "cache_version": CACHE_VERSION,
                        "sidecar_mtimes": sidecar_mtimes,
                        "sidecar_hashes": sidecar_hashes,
                    }
                    self.cache_index["entries"][filepath] = target_entry
                dedup_variant = {
                    "cache_file": cache_file,
                    "schema_keys": variant.get("schema_keys"),
                }
                # Propagate the stamped crop_resolution so the dedup'd
                # entry's agg fast path matches what split-fetch loads
                # from the shared .pt (same file → same value).
                existing_cr = variant.get("crop_resolution")
                if existing_cr is not None:
                    dedup_variant["crop_resolution"] = existing_cr
                target_entry["variants"][key] = dedup_variant
                self._add_to_hash_index(file_hash, filepath)
                return True

            return False

    def __clone_for_cache(self, x: Any):
        if isinstance(x, torch.Tensor):
            return x.clone()
        return x

    def _build_cache_entry(
        self,
        filepath: str,
        file_hash: str,
        resolution: str | None,
        mtime: float,
        group_key: str,
        in_index: int,
        variations: int,
        current_device,
    ):
        cache_file = self._make_cache_file(file_hash, resolution)
        required_schema = set(self.split_names) | set(self.aggregate_names)

        # Capture the actual stored crop_resolution (from either the
        # newly-written cache_data or the reused-existing .pt) so we can
        # stamp it on the variant index entry below. Lets
        # _try_synthesize_aggregate serve the .pt's true value without
        # torch.load AND without parsing the variant key string —
        # parsed-key synthesis breaks whenever the .pt's stored
        # crop_resolution diverges from the key (bucket-overlap
        # ambiguity in _target_int_for_resolution_key, drift recovery
        # linking an out-of-grid key to an old .pt, or AspectBucketing
        # config changes between build and read).
        stored_crop_resolution = None

        # Content-addressed reuse only applies to caches without a resolution
        # dimension (text caches): a donor .pt built for one resolution must
        # never be copied into a variant slot for another.
        use_content_key = self.content_key_in_name is not None and resolution is None

        for v in range(variations):
            pt_name = f"{cache_file}_{v + 1}.pt"
            content_hash = None
            if use_content_key:
                text = self._get_content_key_text(v, in_index)
                if text is not None:
                    content_hash = self._content_hash(text)
            if pt_name in self._existing_pt_files:
                # Reuse the existing .pt only if it actually contains every
                # required schema name. An incomplete .pt (e.g. cached
                # before masked_training was enabled) would otherwise be
                # silently re-registered and quietly sentinel-padded at
                # training time, dropping that sample from the loss.
                if not required_schema:
                    continue
                real_pt = self._real_pt_path(cache_file, v)
                try:
                    existing = torch.load(real_pt, weights_only=False, map_location="cpu")
                except Exception:
                    existing = None
                if isinstance(existing, dict):
                    existing_keys = {k for k in existing if not k.startswith("__")}
                    if required_schema.issubset(existing_keys):
                        if stored_crop_resolution is None:
                            stored_crop_resolution = existing.get("crop_resolution")
                        # Register only the hash stamped at write time — the
                        # freshly computed hash may not describe this .pt if
                        # the seed or pipeline layout changed since it was
                        # built (different SelectRandomText draws). A wrong
                        # registration would poison reuse for *other* files.
                        if use_content_key:
                            stamped = existing.get("__content_hash")
                            if isinstance(stamped, str):
                                self._register_content_pt(stamped, pt_name)
                        continue
                # else: fall through to rewrite below.

            if content_hash is not None and self._try_content_reuse(content_hash, pt_name, v, in_index):
                with self._index_lock:
                    self._existing_pt_files.add(pt_name)
                self._register_content_pt(content_hash, pt_name)
                continue

            self._ensure_before_cache_called()
            cache_data = {}
            with torch.no_grad():
                for name in self.split_names:
                    cache_data[name] = self.__clone_for_cache(self._get_previous_item(v, name, in_index))
                for name in self.aggregate_names:
                    cache_data[name] = self.__clone_for_cache(self._get_previous_item(v, name, in_index))
            cache_data["__cache_version"] = CACHE_VERSION
            cache_data["__modeltype"] = self.modeltype
            if content_hash is not None:
                # Write-time stamp: the only trustworthy source for content
                # index registration on later runs (see the reuse branch).
                cache_data["__content_hash"] = content_hash
            if self.source_path_in_name:
                try:
                    concept = self._get_previous_item(v, "concept", in_index)
                    if concept is not None and isinstance(concept, dict):
                        cache_data["__concept_loss_weight"] = concept.get("loss_weight", 1.0)
                        cache_data["__concept_type"] = concept.get("type", "STANDARD")
                        cache_data["__concept_name"] = concept.get("name", "")
                        cache_data["__concept_path"] = concept.get("path", "")
                        cache_data["__concept_seed"] = concept.get("seed", 0)
                except Exception:
                    pass

            real_pt_path = self._real_pt_path(cache_file, v)
            real_tmp_path = real_pt_path + f".{os.getpid()}.{threading.get_ident()}.tmp"
            torch.save(cache_data, real_tmp_path)
            os.replace(real_tmp_path, real_pt_path)

            if stored_crop_resolution is None:
                stored_crop_resolution = cache_data.get("crop_resolution")

            with self._index_lock:
                self._existing_pt_files.add(pt_name)
            self._register_content_pt(content_hash, pt_name)

        sidecar_mtimes, sidecar_hashes = self._compute_sidecar_state(self._extra_paths_by_filepath.get(filepath, {}))
        # Stamp the source image's exact resolution onto the entry.
        # Used by _fast_resolution_string to compute the per-epoch
        # variant key with the *real* aspect, not the quantized aspect
        # recovered from a bucket key — buckets at different targets use
        # different aspect grids, so the recovered approximation can
        # land on a different bucket than the actual image would,
        # mismatching validation's predicted variant against get_item's
        # actual variant lookup and crashing AspectBatchSorting later.
        # Pulled from the walker cache (CalcAspect already ran during
        # the split-fetch loop above), so this is free.
        original_resolution = None
        try:
            res = self._get_previous_item(0, "original_resolution", in_index)
            if res is not None and len(res) >= 2:
                original_resolution = [int(res[-2]), int(res[-1])]
        except Exception:
            original_resolution = None
        key = self._resolution_key(resolution)
        with self._index_lock:
            entry = self.cache_index["entries"].get(filepath)
            if entry is None:
                entry = {
                    "filename": os.path.basename(filepath),
                    "hash": file_hash,
                    "mtime": mtime,
                    "modeltype": self.modeltype,
                    "variants": {},
                    "cache_version": CACHE_VERSION,
                    "sidecar_mtimes": sidecar_mtimes,
                    "sidecar_hashes": sidecar_hashes,
                }
                if original_resolution is not None:
                    entry["original_resolution"] = original_resolution
                self.cache_index["entries"][filepath] = entry
            else:
                # Refresh metadata; the source content may have changed and
                # been re-hashed since this entry was first created.
                entry["hash"] = file_hash
                entry["mtime"] = mtime
                entry["modeltype"] = self.modeltype
                entry["cache_version"] = CACHE_VERSION
                entry["sidecar_mtimes"] = sidecar_mtimes
                entry["sidecar_hashes"] = sidecar_hashes
                if original_resolution is not None:
                    entry["original_resolution"] = original_resolution
                if "variants" not in entry:
                    entry["variants"] = {}
            variant_record = {
                "cache_file": cache_file,
                "schema_keys": sorted(set(self.split_names) | set(self.aggregate_names)),
            }
            # Stamp the .pt's actual stored crop_resolution so the agg
            # fast path can serve it without torch.load and — critically —
            # without round-tripping the variant key string through bucket
            # math that may not agree with what's on disk.
            if stored_crop_resolution is not None:
                try:
                    cr_list = [int(x) for x in stored_crop_resolution]
                    if len(cr_list) >= 2:
                        variant_record["crop_resolution"] = cr_list
                except (TypeError, ValueError):
                    pass
            entry["variants"][key] = variant_record
            self._add_to_hash_index(file_hash, filepath)

    def _get_source_path(self, in_variation: int, in_index: int) -> str | None:
        if self.source_path_in_name:
            return self._get_previous_item(0, self.source_path_in_name, in_index)
        return None

    def _get_extra_paths(self, in_index: int) -> dict[str, str]:
        """Resolve each watched sidecar in-name to its normpath'd value.

        Names that resolve to None / empty string are omitted, so callers that
        iterate the result see only sidecars the upstream pipeline actually
        produced for this index. Existence on disk is not checked here — that's
        a per-validation concern in _check_sidecars.
        """
        if not self.extra_watched_paths_in_names:
            return {}
        out: dict[str, str] = {}
        for name in self.extra_watched_paths_in_names:
            try:
                value = self._get_previous_item(0, name, 0 if in_index is None else in_index)
            except Exception:
                continue
            if value is None or value == "":
                continue
            out[name] = os.path.normpath(value)
        return out

    def _compute_sidecar_state(self, extra_paths: dict[str, str]) -> tuple[dict[str, float], dict[str, str]]:
        """Snapshot per-sidecar (mtime, hash) for entry-build time.

        Missing sidecars are omitted from both dicts — their absence is itself
        the recorded state. _check_sidecars treats a missing entry as "stored
        missing"; if the file appears later, the (stored missing, currently
        present) case triggers a rebuild.
        """
        mtimes: dict[str, float] = {}
        hashes: dict[str, str] = {}
        for path in extra_paths.values():
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            try:
                file_hash = self._hash_file(path)
            except OSError:
                continue
            mtimes[path] = mtime
            hashes[path] = file_hash
        return mtimes, hashes

    def _check_sidecars(self, filepath: str, entry: dict) -> bool:
        """Validate this entry's stored sidecar state against on-disk truth.

        Returns True if every watched sidecar is consistent with what was on
        disk when the entry was built (or freshly populated on first read for
        a legacy entry that lacks the metadata). Returns False to signal a
        rebuild — sidecar added, removed, or content changed.

        Mirrors the primary-source mtime→hash escalation in _validate_entry,
        so a touch-only mtime change doesn't trigger a VAE re-encode when the
        bytes haven't actually changed.
        """
        if not self.extra_watched_paths_in_names:
            return True

        expected_paths = self._extra_paths_by_filepath.get(filepath, {})
        if not expected_paths:
            # No sidecars resolved for this sample. Treat as "all watched
            # sidecars currently absent" and compare against the stored state
            # the same way (entry may still claim sidecars exist).
            pass

        stored_mtimes = entry.get("sidecar_mtimes")
        stored_hashes = entry.get("sidecar_hashes")

        # Backward-compat: legacy entries pre-date this feature. Populate the
        # state from current disk and treat as valid this run.
        if stored_mtimes is None or stored_hashes is None:
            mtimes, hashes = self._compute_sidecar_state(expected_paths)
            with self._index_lock:
                entry["sidecar_mtimes"] = mtimes
                entry["sidecar_hashes"] = hashes
            return True

        watched_paths = set(expected_paths.values())
        for path in watched_paths | set(stored_mtimes.keys()):
            current_mtime = self._source_mtimes.get(path)
            if current_mtime is None:
                try:
                    current_mtime = os.path.getmtime(path)
                except OSError:
                    current_mtime = None

            stored_mtime = stored_mtimes.get(path)
            stored_hash = stored_hashes.get(path)

            # Sidecar add/remove cases.
            if stored_mtime is None and current_mtime is None:
                continue
            if stored_mtime is None and current_mtime is not None:
                # File appeared since last cache build.
                return False
            if stored_mtime is not None and current_mtime is None:
                # File deleted since last cache build.
                return False

            # Both present.
            if current_mtime == stored_mtime:
                continue

            # mtime drift — fall back to hashing.
            try:
                current_hash = self._hash_file(path)
            except OSError:
                return False
            if current_hash != stored_hash:
                return False

            # Same content, refresh stored mtime so the next fast-path
            # fingerprint pass matches.
            with self._index_lock:
                stored_mtimes[path] = current_mtime

        return True

    def __init_variations(self):
        if self.variations_in_name is not None:
            group_variations = {}
            group_indices = {}
            group_balancing = {}
            group_balancing_strategy = {}

            for in_index in range(self._get_previous_length(self.variations_in_name)):
                if self.group_enabled_in_name and not self._get_previous_item(0, self.group_enabled_in_name, in_index):
                    continue

                variations = self._get_previous_item(0, self.variations_in_name, in_index)
                balancing = self._get_previous_item(0, self.balancing_in_name, in_index)
                balancing_strategy = self._get_previous_item(0, self.balancing_strategy_in_name, in_index)
                group_key = self.__string_key(
                    [self._get_previous_item(0, name, in_index) for name in self.variations_group_in_names]
                )

                if group_key not in group_variations:
                    group_variations[group_key] = variations

                if group_key not in group_indices:
                    group_indices[group_key] = []
                group_indices[group_key].append(in_index)

                if group_key not in group_balancing:
                    group_balancing[group_key] = balancing

                if group_key not in group_balancing_strategy:
                    group_balancing_strategy[group_key] = balancing_strategy

            group_output_samples = {}
            for group_key, balancing in group_balancing.items():
                balancing_strategy = group_balancing_strategy[group_key]
                if balancing_strategy == "REPEATS":
                    group_output_samples[group_key] = int(math.floor(len(group_indices[group_key]) * balancing))
                if balancing_strategy == "SAMPLES":
                    group_output_samples[group_key] = int(balancing)
        else:
            first_previous_name = self.split_names[0] if len(self.split_names) > 0 else self.aggregate_names[0]

            group_variations = {"": 1}
            group_indices = {"": list(range(self._get_previous_length(first_previous_name)))}
            group_output_samples = {"": len(group_indices[""])}
            group_balancing_strategy = {}
            group_balancing = {}

        self.group_variations = group_variations
        self.group_full_indices = {k: list(v) for k, v in group_indices.items()}
        self.group_indices = group_indices
        self.group_output_samples = group_output_samples
        self.group_balancing_strategy = group_balancing_strategy
        self.group_balancing = group_balancing

        self.variations_initialized = True

    def __get_input_index(self, out_variation: int, out_index: int) -> tuple[str, int, int, int]:
        offset = 0
        for group_key, group_output_samples in self.group_output_samples.items():
            if out_index >= group_output_samples + offset:
                offset += group_output_samples
                continue

            variations = self.group_variations[group_key]
            local_index = (out_index - offset) + (out_variation * self.group_output_samples[group_key])
            in_variation = (local_index // len(self.group_indices[group_key])) % variations
            group_index = local_index % len(self.group_indices[group_key])
            in_index = self.group_indices[group_key][group_index]

            return group_key, in_variation, group_index, in_index
        return None

    def __reshuffle_samples(self, out_variation: int):
        for group_key, strategy in self.group_balancing_strategy.items():
            if strategy == "SAMPLES":
                rand = self._get_rand(out_variation)
                shuffled = list(self.group_full_indices[group_key])
                rand.shuffle(shuffled)
                sample_count = int(self.group_balancing[group_key])
                self.group_indices[group_key] = shuffled[:sample_count]

    def __init_sourceless(self):
        self.cache_index = self._load_cache_index()
        if not self.cache_index["entries"]:
            raise RuntimeError(
                "Sourceless training enabled but cache is empty. "
                "Build the cache first with sourceless_training disabled."
            )

        for filepath, entry in self.cache_index["entries"].items():
            if entry.get("cache_version", 0) < CACHE_VERSION:
                raise RuntimeError(
                    f"Cache for '{os.path.basename(filepath)}' was built with an older format. "
                    f"Rebuild your cache with the latest version for sourceless training."
                )
            if entry.get("modeltype") != self.modeltype:
                raise RuntimeError(
                    f"Cache modeltype mismatch: cached as '{entry.get('modeltype')}', "
                    f"current model is '{self.modeltype}'. "
                    f"Change your cache directory or rebuild the cache."
                )

        self._sourceless_filepaths = sorted(self.cache_index["entries"].keys())

        for fp in self._sourceless_filepaths:
            entry = self.cache_index["entries"][fp]
            cache_file = self._any_variant_cache_file(entry)
            if cache_file is None:
                raise RuntimeError(f"Sourceless training: cache entry for '{fp}' has no variants. Rebuild your cache.")
            pt_path = self._real_pt_path(cache_file, 0)
            if not os.path.isfile(pt_path):
                raise RuntimeError(f"Sourceless training: cache file '{pt_path}' is missing. Rebuild your cache.")

        n = len(self._sourceless_filepaths)
        self.group_variations = {"": 1}
        self.group_indices = {"": list(range(n))}
        self.group_full_indices = {"": list(range(n))}
        self.group_output_samples = {"": n}
        self.group_balancing_strategy = {}
        self.group_balancing = {}
        self.variations_initialized = True

        self._aggregate_cache = {}
        if self.aggregate_names:
            with tqdm(total=len(self._sourceless_filepaths), smoothing=0.1, desc="loading aggregate cache") as bar:
                for _group_index, fp in enumerate(self._sourceless_filepaths):
                    entry = self.cache_index["entries"].get(fp)
                    if entry is None:
                        bar.update(1)
                        continue
                    cache_file = self._any_variant_cache_file(entry)
                    if cache_file is None:
                        bar.update(1)
                        continue
                    real_path = self._real_pt_path(cache_file, 0)
                    try:
                        cached = torch.load(real_path, weights_only=False, map_location="cpu")
                        agg_data = {}
                        for name in self.aggregate_names:
                            if name in cached:
                                agg_data[name] = cached[name]
                        if agg_data:
                            self._aggregate_cache[(fp, 0)] = agg_data
                    except Exception:
                        pass
                    bar.update(1)

    def __refresh_cache_sourceless(self, out_variation: int):
        if not self.variations_initialized:
            self.__init_sourceless()

    def __refresh_cache(self, out_variation: int):
        if not self.variations_initialized:
            self.__init_variations()
        self.__reshuffle_samples(out_variation)

        self.cache_index = self._load_cache_index()
        os.makedirs(self.cache_dir, exist_ok=True)
        self._source_path_cache = {}
        self._aggregate_cache = {}
        self._active_key_by_filepath = {}
        self._extra_paths_by_filepath = {}
        self._content_index_loaded = False
        self._before_cache_called = False

        # Bucket-method drift: the AspectBucketing config may have changed
        # since the cache was last validated (e.g. user edited
        # target_resolutions or quantization). Compare the stamped
        # bucket_method against what the upstream provider says now. On
        # drift, run the aspect-math recovery pass to register variants for
        # the new keys without re-decoding any source images.
        #
        # When stored is None (caches migrated from v2, or built before
        # bucket_method stamping landed), we also run drift recovery — we
        # can't assume the cache was built under the *current* config, and
        # the recovery pass is cheap (O(N) aspect math, no image decode) on
        # the happy path where keys already match.
        current_bucket_method = self._compute_bucket_method()
        stored_bucket_method = self.cache_index.get("bucket_method")
        bucket_drift = current_bucket_method is not None and stored_bucket_method != current_bucket_method

        # Schema drift: if split_names/aggregate_names changed since the cache
        # was built (e.g. masked_training was just enabled), the on-disk .pt
        # files won't contain the new keys. Also re-augment when the stored
        # schema_method doesn't match -- caches stamped by an older augment
        # version may have shape-inconsistent values that need fixing in
        # place. Run before any fast-path return so downstream readers
        # always find what they need.
        required_schema = sorted(set(self.split_names) | set(self.aggregate_names))
        stored_schema = self.cache_index.get("schema")
        stored_method = self.cache_index.get("schema_method")
        if self.cache_index.get("entries") and (stored_schema != required_schema or stored_method != SCHEMA_METHOD):
            targets: set[str] = set()
            if stored_schema is not None and stored_method != SCHEMA_METHOD:
                # Cache was stamped by an older augment that may have written
                # shape-inconsistent values. Re-augment every required name so
                # the new shape-correction logic can fix them in place.
                targets = set(required_schema)
            else:
                missing = self._detect_cache_schema_drift()
                if missing:
                    targets = missing

            if targets:
                self._augment_cache_with_missing_names(targets)

            self.cache_index["schema"] = required_schema
            self.cache_index["schema_method"] = SCHEMA_METHOD
            self._save_cache_index()

        # Resolve the source filepaths this call will deliver.
        required_filepaths: set[str] = set()
        index_to_filepath: dict[int, str] = {}
        required_sidecar_paths: set[str] = set()
        for group_key in self.group_variations:
            for in_index in self.group_indices[group_key]:
                filepath = self._get_source_path(0, in_index)
                if filepath is None:
                    continue
                filepath = os.path.normpath(filepath)
                index_to_filepath[in_index] = filepath
                required_filepaths.add(filepath)
                if self.extra_watched_paths_in_names:
                    extras = self._get_extra_paths(in_index)
                    if extras:
                        self._extra_paths_by_filepath[filepath] = extras
                        required_sidecar_paths.update(extras.values())

        # --- Session skip path ---
        # If every required filepath was already validated earlier in this
        # process and is still present in the on-disk index, skip validation
        # entirely. This avoids per-epoch revalidation on static datasets.
        # Bypassed on bucket drift: drift recovery needs to run once before we
        # can trust session-skip again. Also bypassed when multi-resolution
        # variants are configured — each epoch picks a different target per
        # item, so we have to validate (and lazy-build) for the new keys.
        multi_target = (
            self.aspect_bucketing is not None and len(getattr(self.aspect_bucketing, "bucket_resolutions", {})) > 1
        )
        if (
            not bucket_drift
            and not multi_target
            and required_filepaths
            and self.cache_index.get("entries")
            and required_filepaths.issubset(self._session_validated_filepaths)
            and all(fp in self.cache_index["entries"] for fp in required_filepaths)
        ):
            self._source_path_cache = dict(index_to_filepath)
            self._populate_active_keys(required_filepaths)
            print(
                f"SmartDiskCache: Skipped re-validation ({len(required_filepaths)} entries already validated this run)"
            )
            self._load_aggregate_cache(out_variation)
            return

        # --- Trust mode (trust_cache ctor flag or OT_SKIP_CACHE_VALIDATION=1) ---
        # Skip per-file mtime/hash/.pt-existence validation. Any filepath
        # already in the on-disk index is trusted; only missing filepaths are
        # cached. Modeltype is still verified up-front to fail loud on
        # accidentally reusing another model's cache.
        skip_validation = self.trust_cache or os.environ.get("OT_SKIP_CACHE_VALIDATION") == "1"
        if skip_validation:
            missing = sum(1 for fp in required_filepaths if fp not in self.cache_index.get("entries", {}))
            print(
                f"SmartDiskCache: trust mode active ({self.cache_dir}) — "
                f"{len(required_filepaths) - missing} entries trusted without validation, "
                f"{missing} missing from index (will be cached)"
            )
        if skip_validation and self.cache_index.get("entries"):
            entries = self.cache_index["entries"]
            for fp in required_filepaths:
                entry = entries.get(fp)
                if entry is not None and entry.get("modeltype") != self.modeltype:
                    raise RuntimeError(
                        f"Cache modeltype mismatch for '{fp}': "
                        f"cached as '{entry.get('modeltype')}', current model is '{self.modeltype}'. "
                        f"Delete the cache directory or use a separate cache_dir for this model type."
                    )

        # Single os.scandir of the cache dir — used by fast and full paths.
        self._existing_pt_files = self._scan_existing_pt_files()

        # Bucket-method drift recovery: aspect-math derives new variant keys
        # from cached resolutions and links any pre-existing .pt files. Pure
        # arithmetic, no image decode. Runs before validation so the per-index
        # loop sees the new variants registered.
        if bucket_drift:
            self._drift_recovery_pass()

        # --- Fast validation path ---
        # Bypassed on bucket drift: the recovery pass may have queued new
        # variants that need rebuild, so we have to fall through. Also
        # bypassed under multi-target aspect bucketing: the per-epoch
        # required variant rotates, and fast-validate only spot-checks
        # whichever variant is currently registered — it would skip the
        # lazy-build pass for this epoch's missing key.
        if (
            not skip_validation
            and not bucket_drift
            and not multi_target
            and self.cache_index.get("entries")
            and self._fast_validate()
        ):
            all_in_index = all(fp in self.cache_index["entries"] for fp in required_filepaths)
            if all_in_index:
                self._source_path_cache = dict(index_to_filepath)
                self._populate_active_keys(required_filepaths)
                n = len(self.cache_index["entries"])
                checked = getattr(self, "_fast_validate_sample_size", "?")
                print(f"SmartDiskCache: Fast validation passed ({n} entries, {checked} spot-checked)")
                self._session_validated_filepaths.update(required_filepaths)
                self._load_aggregate_cache(out_variation)
                return

            # Index mismatch — fall through to full validation
            self._source_path_cache = {}

        # Bulk-stat all source files once via parallel os.scandir per parent dir.
        # Subsequent _validate_entry calls read mtimes from this dict instead of
        # firing one os.path.getmtime syscall per file. Skipped under trust mode
        # since we won't be calling _validate_entry at all.
        if not skip_validation:
            self._source_mtimes = self._bulk_stat_source_files(required_filepaths | required_sidecar_paths)
        else:
            self._source_mtimes = {}

        # Clear fast-validation token during full validation. Preserved under
        # trust mode: a trust run neither verifies nor invalidates source
        # state, so whatever guarantee the token carried still holds — popping
        # it here would permanently knock later non-trust runs off the fast
        # path (the re-stamp at the end is also gated on full validation).
        if not skip_validation:
            self.cache_index.pop("last_validated", None)

        files_built = 0
        files_skipped = 0
        files_failed = []

        for group_key in self.group_variations:
            variations = self.group_variations[group_key]

            # _validate_entry is invariant in `in_variation` (source path uses
            # variation 0; resolution is variation-independent in the bucketing
            # pipeline; .pt existence iterates `range(variations)` internally).
            # Validate each in_index exactly once and dedupe across needed
            # variations afterwards.
            items_to_build_by_index: dict[int, tuple] = {}

            with tqdm(total=len(self.group_indices[group_key]), smoothing=0.1, desc="validating cache") as bar:
                for group_index, in_index in enumerate(self.group_indices[group_key]):
                    # Trust-mode early skip: avoid upstream pipeline calls
                    # (_get_source_path triggers crop_resolution upstream,
                    # which can do per-image I/O on slow cloud storage)
                    if skip_validation:
                        cached_fp = index_to_filepath.get(in_index)
                        if cached_fp is not None and cached_fp in self.cache_index["entries"]:
                            self._source_path_cache[in_index] = cached_fp
                            files_skipped += 1
                            bar.update(1)
                            continue

                    filepath = self._get_source_path(0, in_index)
                    if filepath is None:
                        bar.update(1)
                        continue

                    filepath = os.path.normpath(filepath)
                    self._source_path_cache[in_index] = filepath

                    entry = self.cache_index["entries"].get(filepath)
                    if entry is not None:
                        if skip_validation:
                            files_skipped += 1
                            bar.update(1)
                            continue
                        # Active key for this epoch: resolve from upstream
                        # when this cache has resolution variants (aspect
                        # bucketing or explicit upstream-resolution mode).
                        # Fast path: when the entry already has variants
                        # we recover aspect from any variant key and ask
                        # AspectBucketing for the would-be bucket without
                        # decoding the image (saved ~250ms/file on disk).
                        # Slow path (``_get_resolution_string``) only runs
                        # when there's no aspect to recover, i.e. fresh
                        # entries the very first time we see them.
                        if self.resolution_from_upstream or self.aspect_bucketing is not None:
                            resolution = self._fast_resolution_string(entry, out_variation, in_index)
                            if resolution is None:
                                resolution = self._get_resolution_string(out_variation, in_index)
                            active_key = self._resolution_key(resolution)
                            self._active_key_by_filepath[filepath] = active_key
                        else:
                            active_key = self._active_key_by_filepath.get(filepath)
                            if active_key is None:
                                active_key = next(
                                    iter(entry.get("variants", {}).keys()),
                                    NO_RESOLUTION_KEY,
                                )
                                self._active_key_by_filepath[filepath] = active_key
                        mtime = self._source_mtimes.get(filepath)
                        status = self._validate_entry(filepath, entry, active_key, variations, mtime)
                        if status == "valid":
                            files_skipped += 1
                            bar.update(1)
                            continue
                        if status == "missing_variant":
                            # Entry is fine but doesn't have a variant for
                            # this epoch's expected key. Rebuild that
                            # variant only — leave the entry and other
                            # variants in place. Reuse the fast key we
                            # computed above; falls back to the upstream
                            # walk if the fast path returned None.
                            resolution = self._fast_resolution_string(entry, out_variation, in_index)
                            if resolution is None:
                                resolution = self._get_resolution_string(out_variation, in_index)
                            self._active_key_by_filepath[filepath] = self._resolution_key(resolution)
                            items_to_build_by_index[in_index] = (
                                filepath,
                                group_key,
                                out_variation,
                                in_index,
                                group_index,
                                variations,
                                resolution,
                            )
                            bar.update(1)
                            continue
                        # Otherwise: rebuild. Drop the stale entry.
                        with self._index_lock:
                            old_hash = entry.get("hash")
                            if old_hash:
                                self._remove_from_hash_index(old_hash, filepath)
                            if filepath in self.cache_index["entries"]:
                                del self.cache_index["entries"][filepath]
                            self._active_key_by_filepath.pop(filepath, None)

                    # Rebuild path: now we DO need the current resolution.
                    # This opens the image (via CalcAspect) but only for files
                    # that are missing or stale, not for the whole dataset.
                    resolution = self._get_resolution_string(out_variation, in_index)
                    self._active_key_by_filepath[filepath] = self._resolution_key(resolution)
                    items_to_build_by_index[in_index] = (
                        filepath,
                        group_key,
                        out_variation,
                        in_index,
                        group_index,
                        variations,
                        resolution,
                    )
                    bar.update(1)

            items_to_build = list(items_to_build_by_index.values())

            if not items_to_build:
                continue

            # before_cache_fun is NOT called here anymore — it's deferred to
            # the first variation that needs an actual upstream encode (see
            # _ensure_before_cache_called). The content index, however, must
            # be ready before any builder can consult it.
            self._ensure_content_index_loaded()

            seen_paths = set()
            unique_items = []
            for item in items_to_build:
                fp = item[0]
                if fp not in seen_paths:
                    seen_paths.add(fp)
                    unique_items.append(item)

            self._last_flush_time = time.monotonic()

            # Dedicated build pool when configured: the shared pipeline
            # executor is sized for training-time loading (dataloader_threads,
            # default 2), far too narrow to keep an encoder batch collector
            # fed during a cache build.
            if self.build_max_workers is not None and self.build_max_workers > 1:
                build_pool_ctx = concurrent.futures.ThreadPoolExecutor(max_workers=self.build_max_workers)
            else:
                build_pool_ctx = contextlib.nullcontext(self._state.executor)

            with (
                build_pool_ctx as build_executor,
                tqdm(total=len(unique_items), smoothing=0.1, desc="caching") as bar,
            ):

                def fn(
                    filepath, group_key, in_variation, in_index, group_index, variations, resolution, current_device
                ):
                    if torch.cuda.is_available() and current_device is not None:
                        torch.cuda.set_device(current_device)

                    try:
                        mtime = os.path.getmtime(filepath)
                    except OSError:
                        if not self.tolerate_missing_source:
                            return filepath, "missing"
                        # Synthetic source: no file to stat, no bytes to
                        # hash. Use mtime=0 and a per-filepath synthetic
                        # hash so distinct synthetic entries don't dedup
                        # together (same content would, but the fallback
                        # hash is intentionally derived from the path).
                        mtime = 0.0
                        file_hash = xxhash.xxh64(filepath.encode("utf-8")).hexdigest()
                    else:
                        try:
                            file_hash = self._hash_file(filepath)
                        except OSError:
                            return filepath, "hash_failed"

                    if self._try_dedup(filepath, file_hash, resolution, mtime):
                        entry = self.cache_index["entries"][filepath]
                        key = self._resolution_key(resolution)
                        variant = entry.get("variants", {}).get(key)
                        cf = variant["cache_file"] if variant else None
                        all_present = cf is not None and all(
                            f"{cf}_{v + 1}.pt" in self._existing_pt_files for v in range(variations)
                        )
                        if all_present:
                            return filepath, "dedup"

                    try:
                        self._build_cache_entry(
                            filepath,
                            file_hash,
                            resolution,
                            mtime,
                            group_key,
                            in_index,
                            variations,
                            current_device,
                        )
                    except Exception as e:
                        return filepath, f"build_failed: {e}"

                    return filepath, "built"

                current_device = torch.cuda.current_device() if torch.cuda.is_available() else None

                # Group queued items by AspectBucketing target_resolution.
                # Each upstream walk reads ``aspect_bucketing._target_override``
                # to pin the bucket choice — that attribute is shared state,
                # so concurrent workers MUST share the same target. We run
                # one parallel batch per target with the override held for
                # that batch, then clear and move on. For text caches and
                # configs without aspect bucketing, every item lands in a
                # single ``None`` group and the behaviour is identical to
                # before.
                target_groups: dict[int | None, list[tuple]] = {}
                for item in unique_items:
                    resolution_str = item[6]
                    target = self._target_int_for_resolution_key(resolution_str)
                    target_groups.setdefault(target, []).append(item)

                for target, group_items in target_groups.items():
                    prev_override = None
                    ab = self.aspect_bucketing
                    if ab is not None and target is not None:
                        prev_override = getattr(ab, "_target_override", None)
                        ab._target_override = target
                    try:
                        fs = [
                            build_executor.submit(
                                fn,
                                filepath,
                                group_key,
                                in_variation,
                                in_index,
                                group_index,
                                variations,
                                resolution,
                                current_device,
                            )
                            for (
                                filepath,
                                group_key,
                                in_variation,
                                in_index,
                                group_index,
                                variations,
                                resolution,
                            ) in group_items
                        ]
                        for build_count, f in enumerate(concurrent.futures.as_completed(fs), 1):
                            try:
                                filepath, status = f.result()
                            except Exception:
                                build_executor.shutdown(wait=True, cancel_futures=True)
                                raise
                            if status == "built" or status == "dedup":
                                files_built += 1
                            elif status.startswith("build_failed") or status == "missing" or status == "hash_failed":
                                files_failed.append((filepath, status))
                                print(f"Warning: failed to cache '{filepath}': {status}")
                            if build_count % 250 == 0:
                                self._torch_gc()
                            self._flush_cache_index()
                            bar.update(1)
                            if self.stop_check_fun():
                                build_executor.shutdown(wait=True, cancel_futures=True)
                                self._save_cache_index()
                                self._save_content_index()
                                print(
                                    f"SmartDiskCache: Stopped early. Cached {files_built} files this session, {files_skipped} reused from cache."
                                )
                                raise CachingStoppedException
                    finally:
                        if ab is not None and target is not None:
                            ab._target_override = prev_override

        if not skip_validation:
            self.cache_index["last_validated"] = time.time()
            # Per-watched-file fingerprint: ignored if any parent dir is
            # unreachable (returns None); a missing fingerprint just means the
            # next run pays for one extra full validation pass.
            self.cache_index["watched_fingerprints"] = (
                self._compute_watched_fingerprints(self.cache_index["entries"]) or {}
            )
        self.cache_index["schema"] = required_schema
        self.cache_index["schema_method"] = SCHEMA_METHOD
        if current_bucket_method is not None:
            self.cache_index["bucket_method"] = current_bucket_method
        self._save_cache_index()
        self._save_content_index()

        # Mark every required filepath that ended up with a valid entry as
        # validated for this process so subsequent epochs can skip outright.
        entries = self.cache_index.get("entries", {})
        self._session_validated_filepaths.update(fp for fp in required_filepaths if fp in entries)

        total = files_built + files_skipped + len(files_failed)
        if total > 0:
            print(
                f"SmartDiskCache: Cached {files_built}/{total} files. {files_skipped} reused from cache. {len(files_failed)} failed."
            )
        if files_failed:
            for fp, reason in files_failed[:10]:
                print(f"  {fp}: {reason}")
            if len(files_failed) > 10:
                print(f"  ... and {len(files_failed) - 10} more")

        self._load_aggregate_cache(out_variation)

    def _fast_resolution_string(
        self,
        entry: dict,
        in_variation: int,
        in_index: int,
    ) -> str | None:
        """Compute the per-epoch variant key for an entry without
        decoding its source image.

        Recovers aspect from any cached variant key in ``entry`` (the
        same trick ``_drift_recovery_pass`` uses), then asks
        ``AspectBucketing.variant_key_from_aspect`` to reproduce the
        rand.choice + bucket math that ``get_item`` would run. Skips
        the ``CalcAspect → LoadImage`` chain that ``_get_resolution_string``
        otherwise triggers per cache hit — which was costing ~250 ms
        per file across the whole validation pass on every epoch under
        multi-target bucketing.

        Returns None when no aspect can be recovered (fresh entry,
        text-cache shape) or AspectBucketing can't produce a key
        (fixed WxH config, missing target). Caller falls back to the
        slow upstream walk in those cases.
        """
        if self.aspect_bucketing is None:
            return None
        # Prefer the stamped original_resolution (exact aspect) over the
        # variant-key approximation. Bucket aspect grids differ across
        # targets; recovering aspect from a 768-target variant key and
        # asking for the 512-target bucket can land on a different
        # bucket than the image's actual aspect would — that mismatch
        # is exactly what causes AspectBatchSorting to bucket items at
        # sort time differently from what get_item loads at fetch time.
        orig = entry.get("original_resolution")
        aspect: float | None = None
        if isinstance(orig, (list, tuple)) and len(orig) >= 2:
            try:
                h = float(orig[-2])
                w = float(orig[-1])
                if w > 0:
                    aspect = h / w
            except (TypeError, ValueError):
                aspect = None
        if aspect is None:
            # Legacy entries (pre-stamping) — walk for original_resolution
            # upstream and lazy-stamp on the entry so subsequent epochs
            # hit the fast path. This costs one image decode per legacy
            # file the FIRST time it's seen; after that the stamp keeps
            # validation cheap forever. We avoid the variant-key
            # approximation entirely here because its rounding error
            # near bucket boundaries was the original source of the
            # cross-entry shape mismatches.
            walk_variation = self.current_variation if self.current_variation >= 0 else in_variation
            try:
                upstream = self._get_previous_item(walk_variation, "original_resolution", in_index)
            except Exception:
                upstream = None
            if isinstance(upstream, (list, tuple)) and len(upstream) >= 2:
                try:
                    h = float(upstream[-2])
                    w = float(upstream[-1])
                    if w > 0:
                        aspect = h / w
                        entry["original_resolution"] = [int(h), int(w)]
                except (TypeError, ValueError):
                    aspect = None
        if aspect is None:
            # Last resort: approximate aspect from any variant key. Used
            # only when both the stamp is missing AND upstream is
            # unavailable (e.g. text caches that have no resolution at all).
            variants = entry.get("variants") or {}
            aspect = self._aspect_from_variant_keys(variants)
        if aspect is None:
            return None
        try:
            return self.aspect_bucketing.variant_key_from_aspect(in_variation, in_index, aspect)
        except Exception:
            return None

    def _try_synthesize_aggregate(
        self,
        filepath: str,
        entry: dict,
        in_variation: int,
        in_index: int,
    ) -> dict | None:
        """Reconstruct agg_data from cache metadata, no .pt I/O.

        Resolves the per-epoch variant key via ``_fast_resolution_string``
        (aspect from cached variant keys, rand.choice over targets — no
        image decode). Falls back to ``_get_resolution_string`` only
        when the entry has no aspect to recover yet; that path *does*
        decode but is rare after the build pass has run.

        Returns the synthesized dict, or None when synthesis fails
        (variant key won't parse, or the resolved key is
        ``NO_RESOLUTION_KEY``). The caller is expected to have already
        gated entry: every configured aggregate name must be in
        ``_DERIVABLE_AGGREGATES`` and ``frame_dim_enabled`` must be
        False (else the 2D ``HxW`` key can't represent the cached value).
        """
        resolution = self._fast_resolution_string(entry, in_variation, in_index)
        if resolution is None:
            resolution = self._get_resolution_string(in_variation, in_index)
        if resolution is None or resolution == NO_RESOLUTION_KEY:
            return None

        # Synthesis is only safe when the computed variant key actually
        # exists on the entry. Otherwise split-fetch's get_item falls back
        # to _active_cache_file at a different key, and AspectBatchSorting
        # would bucket by a crop_resolution that no on-disk .pt for this
        # (filepath, variation, in_index) produces — torch.stack then
        # blows up in collate on the mismatched latent shapes. Falling
        # through to None routes the item to the slow path, where the
        # same _active_cache_file fallback reads the real crop_resolution
        # off the .pt that split-fetch will load.
        variants = entry.get("variants") or {}
        variant = variants.get(self._resolution_key(resolution))
        if variant is None:
            return None

        # Prefer the stamped crop_resolution from the variant entry —
        # that's the value torch.load would return for the .pt this
        # variant points to. Parsing (h, w) out of the variant key
        # string is *almost* the same answer, but the two diverge when
        # drift recovery linked an out-of-grid key to an old .pt or
        # when _target_int_for_resolution_key returned an ambiguous
        # target during the build pass — in either case the .pt's
        # stored crop_resolution is the truth, and AspectBatchSorting
        # has to bucket by exactly that or the per-item .pt that
        # split-fetch later loads won't stack with its batchmates.
        # Legacy variant records (pre-stamping) fall through to None
        # → slow path → torch.load → lazy-stamp the field below.
        stored_cr = variant.get("crop_resolution")
        if not isinstance(stored_cr, (list, tuple)) or len(stored_cr) < 2:
            return None
        try:
            h = int(stored_cr[-2])
            w = int(stored_cr[-1])
        except (TypeError, ValueError):
            return None

        agg_data: dict = {}
        for name in self.aggregate_names:
            if name == "crop_resolution":
                agg_data[name] = (h, w)
            elif name == "image_path":
                agg_data[name] = filepath
            else:
                # _DERIVABLE_AGGREGATES gate should have prevented this,
                # but a defensive None forces the caller to .pt-load this
                # item if we ever extend the set without updating here.
                return None
        return agg_data

    def _load_aggregate_cache(self, out_variation: int):
        if not self.aggregate_names:
            return

        # Aggregates that vary per resolution variant (most importantly
        # ``crop_resolution`` — consumed by AspectBatchSorting to form
        # batches) must be loaded from the SAME variant that ``get_item``
        # will return latents from. With multi-target bucketing the
        # variant rotates per (variation, index), so we need the
        # per-item key to pick the right .pt — not the session-pinned
        # active key.
        per_item_key = self.resolution_from_upstream or self.aspect_bucketing is not None

        # Fast path: when every aggregate name is derivable from cache
        # metadata (variant key parses to crop_resolution; entry key is
        # image_path), skip torch.load entirely. The .pt is dominated by
        # split latent tensors that aggregate-load doesn't need — reading
        # the whole file just to extract a 2-tuple + string costs multi-GB
        # of disk I/O per epoch on large datasets.
        #
        # Correctness relies on synth producing the SAME crop_resolution
        # that ``get_item``'s split fetch will load for the same
        # (variation, in_index). That now holds: synth's
        # ``_fast_resolution_string`` uses the entry's stamped
        # ``original_resolution`` (exact aspect) and the same
        # ``AspectBucketing.variant_key_from_aspect`` that get_item walks
        # to, and the per-in_index agg cache key keeps writes for the
        # same filepath at different in_index distinct.
        #
        # Video pipelines (frame_dim_enabled=True) stay on the slow path:
        # the 2D HxW variant key can't represent the 3D
        # (frames, h, w) crop_resolution they cache.
        ab = self.aspect_bucketing
        frame_dim = bool(getattr(ab, "frame_dim_enabled", False)) if ab is not None else False
        fast_path = (
            not frame_dim and bool(self.aggregate_names) and set(self.aggregate_names).issubset(_DERIVABLE_AGGREGATES)
        )

        load_items = []
        for group_key in self.group_variations:
            start_index = self.group_output_samples[group_key] * out_variation
            end_index = self.group_output_samples[group_key] * (out_variation + 1) - 1

            start_variation = start_index // len(self.group_indices[group_key])
            end_variation = end_index // len(self.group_indices[group_key])

            variations = self.group_variations[group_key]
            needed_variations = [(x % variations) for x in range(start_variation, end_variation + 1)]

            for in_variation in needed_variations:
                for _group_index, in_index in enumerate(self.group_indices[group_key]):
                    filepath = self._source_path_cache.get(in_index)
                    if filepath is None:
                        continue
                    cache_entry = self.cache_index["entries"].get(filepath)
                    if cache_entry is None:
                        continue
                    variation = in_variation % variations
                    load_items.append((filepath, cache_entry, variation, in_variation, in_index))

        if fast_path:
            slow_items = []
            # Progress bar even on the fast path: this loop is O(N) dict
            # ops plus per-item RNG/bucket math via AspectBucketing, but
            # on a 22k-sample dataset it's still a few seconds of work
            # and worth showing so it doesn't look like a freeze after
            # the 'Cached X/Y files' line.
            with tqdm(total=len(load_items), smoothing=0.1, desc="loading aggregate cache") as bar:
                for filepath, cache_entry, variation, in_variation, in_index in load_items:
                    agg_data = self._try_synthesize_aggregate(filepath, cache_entry, in_variation, in_index)
                    if agg_data is None:
                        slow_items.append((filepath, cache_entry, variation, in_variation, in_index))
                        bar.update(1)
                        continue
                    # Key by (filepath, variation, in_index) — see the
                    # slow-path write below for the rationale.
                    self._aggregate_cache[(filepath, variation, in_index)] = agg_data
                    bar.update(1)
            if not slow_items:
                return
            load_items = slow_items

        stamped_any = False
        with tqdm(total=len(load_items), smoothing=0.1, desc="loading aggregate cache") as bar:
            for filepath, cache_entry, variation, in_variation, in_index in load_items:
                cache_file = None
                variant_for_stamp = None
                if per_item_key:
                    # Fast resolve via the stamped original_resolution.
                    # Falls back to the upstream image-decode walk only
                    # when no aspect can be recovered.
                    resolution = self._fast_resolution_string(cache_entry, in_variation, in_index)
                    if resolution is None:
                        resolution = self._get_resolution_string(in_variation, in_index)
                    if resolution is not None:
                        variants = cache_entry.get("variants", {})
                        variant = variants.get(self._resolution_key(resolution))
                        if variant is not None:
                            cache_file = variant.get("cache_file")
                            variant_for_stamp = variant
                if cache_file is None:
                    cache_file = self._active_cache_file(filepath, cache_entry)
                    if variant_for_stamp is None and cache_file is not None:
                        # Locate the active-fallback variant so we can
                        # stamp its crop_resolution too. Iterating the
                        # variants dict is O(num_variants_per_file),
                        # typically ≤ a handful.
                        for v in (cache_entry.get("variants") or {}).values():
                            if v.get("cache_file") == cache_file:
                                variant_for_stamp = v
                                break
                if cache_file is None:
                    bar.update(1)
                    continue
                real_path = self._real_pt_path(cache_file, variation)
                try:
                    cached = torch.load(real_path, weights_only=False, map_location="cpu")
                    agg_data = {}
                    for name in self.aggregate_names:
                        if name in cached:
                            agg_data[name] = cached[name]
                    if agg_data:
                        # Key by (filepath, variation, in_index): the same
                        # filepath can appear at multiple in_index values
                        # (across concepts or repeats) and each in_index
                        # has its own rand.choice'd variant — see the
                        # matching get_item lookup for the rationale.
                        self._aggregate_cache[(filepath, variation, in_index)] = agg_data
                        # Lazy-stamp the variant entry's crop_resolution
                        # from what we just read off disk. Subsequent
                        # epochs (and other agg loads on the same entry)
                        # then hit the fast path in
                        # _try_synthesize_aggregate without paying for
                        # another torch.load. Old caches built before
                        # this field was stamped migrate themselves on
                        # first read.
                        if variant_for_stamp is not None and variant_for_stamp.get("crop_resolution") is None:
                            stored = agg_data.get("crop_resolution")
                            if isinstance(stored, (list, tuple)) and len(stored) >= 2:
                                try:
                                    cr_list = [int(x) for x in stored]
                                    with self._index_lock:
                                        variant_for_stamp["crop_resolution"] = cr_list
                                    stamped_any = True
                                except (TypeError, ValueError):
                                    pass
                except Exception:
                    pass
                bar.update(1)

        if stamped_any:
            # Persist the lazy-stamped crop_resolution fields so the next
            # process / next epoch hits the fast path without re-running
            # torch.load on every entry.
            self._save_cache_index()

    def start(self, out_variation: int):
        if self.sourceless:
            self.__refresh_cache_sourceless(out_variation)
        else:
            self.__refresh_cache(out_variation)
        self._ensure_blank_sentinel()

    def _ensure_blank_sentinel(self):
        """Persist a zero-tensor sentinel for cache-miss fallback.

        Why: files that failed to cache (build_failed / missing / hash_failed)
        leave gaps in the index. At training time the text encoder has been
        offloaded to CPU, so re-encoding those gaps risks both a device
        mismatch and an OOM. Returning zeros for those few samples is
        preferable to crashing training.
        """
        if not self.cache_index:
            return

        sentinel_name = self.cache_index.get("blank_sentinel")
        required = set(self.split_names) | set(self.aggregate_names)
        if sentinel_name:
            existing_path = os.path.join(self._real_cache_dir, sentinel_name)
            if os.path.isfile(existing_path):
                try:
                    existing = torch.load(existing_path, weights_only=False, map_location="cpu")
                    existing_keys = {k for k in existing if not k.startswith("__")}
                    if existing.get("__modeltype") == self.modeltype and required.issubset(existing_keys):
                        return
                except Exception:
                    pass

        template = None
        for entry in self.cache_index.get("entries", {}).values():
            if entry.get("modeltype") != self.modeltype:
                continue
            cache_file = self._any_variant_cache_file(entry)
            if not cache_file:
                continue
            pt = self._real_pt_path(cache_file, 0)
            if os.path.isfile(pt):
                try:
                    template = torch.load(pt, weights_only=False, map_location="cpu")
                    break
                except Exception:
                    template = None
        if template is None:
            return

        sentinel = {"__cache_version": CACHE_VERSION, "__modeltype": self.modeltype}
        for name in self.split_names + self.aggregate_names:
            v = template.get(name)
            if torch.is_tensor(v):
                sentinel[name] = torch.zeros_like(v)
            elif v is not None:
                sentinel[name] = v

        out_name = "blank_sentinel.pt"
        out_path = os.path.join(self._real_cache_dir, out_name)
        tmp_path = out_path + f".{os.getpid()}.tmp"
        try:
            torch.save(sentinel, tmp_path)
            os.replace(tmp_path, out_path)
        except OSError:
            return

        with self._index_lock:
            self.cache_index["blank_sentinel"] = out_name
        self._save_cache_index()

    def _load_blank_sentinel(self) -> dict | None:
        if not self.cache_index:
            return None
        sentinel_name = self.cache_index.get("blank_sentinel")
        if not sentinel_name:
            return None
        sentinel_path = os.path.join(self._real_cache_dir, sentinel_name)
        if not os.path.isfile(sentinel_path):
            return None
        try:
            return torch.load(sentinel_path, weights_only=False, map_location=self.pipeline.device)
        except Exception:
            return None

    def get_item(self, index: int, requested_name: str = None) -> dict:
        result = self.__get_input_index(self.current_variation, index)
        if result is None:
            return {requested_name: self._get_previous_item(self.current_variation, requested_name, index)}

        group_key, in_variation, group_index, in_index = result

        filepath = self._sourceless_filepaths[group_index] if self.sourceless else self._source_path_cache.get(in_index)

        if filepath is not None:
            cache_entry = self.cache_index["entries"].get(filepath)

            if cache_entry is not None:
                variation = in_variation % self.group_variations[group_key]

                if requested_name in self.aggregate_names:
                    # Key the lookup on (filepath, variation, in_index) — not
                    # just (filepath, variation). When the same source file
                    # appears across multiple concepts (or under repeats), it
                    # is queried at distinct in_index values, and each one
                    # picks a different rand.choice target (the seed is per
                    # (variation, index)). The variant's stored crop_resolution
                    # therefore differs per in_index. Keying only by filepath
                    # meant the last agg-load write clobbered earlier ones,
                    # leaving the sort-time agg out of sync with what split-
                    # fetch's _get_resolution_string would resolve for this
                    # specific in_index — AspectBatchSorting bucketed by the
                    # stale crop_resolution and torch.stack later blew up on
                    # the mismatched latent shapes.
                    agg_data = self._aggregate_cache.get((filepath, variation, in_index))
                    if agg_data is not None:
                        # Return a shallow copy. The walker stashes the
                        # returned dict in its per-module item_cache and
                        # then update()s it when a split name is queried at
                        # the same (variation, index). Without the copy that
                        # update would mutate our entry in _aggregate_cache.
                        return dict(agg_data)

                # Per-epoch variant resolve: ask upstream what bucket this
                # (variation, index) wants right now. AspectBucketing's
                # rand.choice is seeded on (variation, index), so each epoch
                # rolls a fresh per-item target — matching the old DiskCache
                # behaviour the new persistent cache otherwise froze. Skipped
                # for caches without resolution variants (text caches), which
                # fall through to ``_active_cache_file``'s any-variant fallback.
                #
                # Fallback path: use ``_active_cache_file`` for missing
                # variants. ``_load_aggregate_cache`` uses the same fallback,
                # so split-fetch's ``.pt`` matches whatever ``crop_resolution``
                # ended up in the aggregate cache for this (filepath, var).
                # Previously this branch went straight to
                # ``_any_variant_cache_file`` — that returns the first
                # variant in dict order, which differs from the aggregate
                # load's active-key fallback whenever the entry has
                # multiple variants. Symptom: AspectBatchSorting buckets
                # by aggregate's crop_resolution; collate gets split-fetch's
                # different-variant latent; torch.stack blows up.
                cache_file: str | None
                if self.resolution_from_upstream or self.aspect_bucketing is not None:
                    # Fast resolve via the stamped original_resolution
                    # before falling back to the upstream image-decode walk.
                    # Per-batch this saves the ~250 ms decode that
                    # ``_get_resolution_string`` would trigger for every
                    # item via CalcAspect → LoadImage.
                    resolution = self._fast_resolution_string(cache_entry, in_variation, in_index)
                    if resolution is None:
                        resolution = self._get_resolution_string(in_variation, in_index)
                    if resolution is not None:
                        key = self._resolution_key(resolution)
                        variants = cache_entry.get("variants", {})
                        variant = variants.get(key)
                        if variant is not None:
                            cache_file = variant["cache_file"]
                        else:
                            cache_file = self._active_cache_file(filepath, cache_entry)
                    else:
                        cache_file = self._active_cache_file(filepath, cache_entry)
                else:
                    cache_file = self._active_cache_file(filepath, cache_entry)
                if cache_file is None:
                    # Entry has no variants — fall through to sentinel.
                    sentinel = self._load_blank_sentinel()
                    if sentinel is not None:
                        item = {}
                        for name in self.split_names + self.aggregate_names:
                            if name in sentinel:
                                item[name] = sentinel[name]
                        return item
                    return {}
                real_cache_path = self._real_pt_path(cache_file, variation)

                cached = torch.load(real_cache_path, weights_only=False, map_location=self.pipeline.device)

                item = {}
                missing_for_file: list[str] = []
                for name in self.split_names + self.aggregate_names:
                    if name in cached:
                        item[name] = cached[name]
                    else:
                        missing_for_file.append(name)
                if missing_for_file:
                    # The schema-drift pass should have populated these, but a
                    # per-file augmentation failure (or a stale .pt that
                    # survived a cache rebuild because its filename matched
                    # an existing on-disk file) can still leave gaps. Borrow
                    # the sentinel's zero-tensors for the missing keys.
                    #
                    # Critical: the sentinel is templated off some arbitrary
                    # entry's spatial shape, which may not match this item's
                    # orientation. Copying its tensors verbatim glues a
                    # landscape mask onto a portrait latent (or vice versa)
                    # and crashes torch.stack downstream when AspectBatchSorting
                    # groups two items by their (matching) crop_resolution but
                    # one of them was sentinel-filled at the wrong shape.
                    # Resize tensor fields to match the item's actual spatial
                    # dims, taking the ref from any already-loaded item tensor.
                    ref_shape = None
                    for v in item.values():
                        if torch.is_tensor(v) and v.dim() >= 2:
                            ref_shape = tuple(v.shape[-2:])
                            break
                    sentinel = self._load_blank_sentinel()
                    if sentinel is not None:
                        for name in missing_for_file:
                            if name not in sentinel:
                                continue
                            sval = sentinel[name]
                            if (
                                torch.is_tensor(sval)
                                and sval.dim() >= 2
                                and ref_shape is not None
                                and tuple(sval.shape[-2:]) != ref_shape
                            ):
                                new_shape = tuple(sval.shape[:-2]) + ref_shape
                                item[name] = torch.zeros(
                                    new_shape,
                                    dtype=sval.dtype,
                                    device=sval.device,
                                )
                            else:
                                item[name] = sval
                if self.sourceless and "__concept_loss_weight" in cached:
                    item["concept"] = {
                        "loss_weight": cached["__concept_loss_weight"],
                        "type": cached.get("__concept_type", "STANDARD"),
                        "name": cached.get("__concept_name", ""),
                        "path": cached.get("__concept_path", ""),
                        "seed": cached.get("__concept_seed", 0),
                    }
                return item

        sentinel = self._load_blank_sentinel()
        if sentinel is not None:
            item = {}
            for name in self.split_names + self.aggregate_names:
                if name in sentinel:
                    item[name] = sentinel[name]
            return item

        if self.before_cache_fun is not None:
            self.before_cache_fun()

        item = {}
        with torch.no_grad():
            if requested_name in self.split_names:
                for name in self.split_names:
                    item[name] = self._get_previous_item(in_variation, name, in_index)
            elif requested_name in self.aggregate_names:
                for name in self.aggregate_names:
                    item[name] = self._get_previous_item(in_variation, name, in_index)
            else:
                for name in self.split_names + self.aggregate_names:
                    item[name] = self._get_previous_item(in_variation, name, in_index)
        return item

    @staticmethod
    def gc_preview(cache_dir: str) -> dict:
        cache_path = os.path.join(cache_dir, "cache.json")
        if not os.path.isfile(cache_path):
            return {"orphan_count": 0, "orphan_bytes": 0}

        with open(cache_path, "r") as f:
            index = json.load(f)

        SmartDiskCache._migrate_legacy_index_in_place(index)
        entries = index.get("entries", {})

        dead_filepaths = [fp for fp in entries if not os.path.isfile(fp)]

        referenced_cache_files = set()
        for fp, entry in entries.items():
            if fp in dead_filepaths:
                continue
            for cf in SmartDiskCache._iter_variant_cache_files(entry):
                for v in range(1, 100):
                    pt = os.path.join(cache_dir, f"{cf}_{v}.pt")
                    if os.path.isfile(pt):
                        referenced_cache_files.add(os.path.normpath(pt))
                    else:
                        break

        # Top-level non-entry .pt files (currently just blank_sentinel.pt)
        # are intentional artifacts, not orphans.
        sentinel_name = index.get("blank_sentinel")
        if sentinel_name:
            referenced_cache_files.add(os.path.normpath(os.path.join(cache_dir, sentinel_name)))

        orphan_count = 0
        orphan_bytes = 0

        for entry in os.scandir(cache_dir):
            if entry.name.endswith(".pt") and entry.is_file():
                if os.path.normpath(entry.path) not in referenced_cache_files:
                    orphan_count += 1
                    orphan_bytes += entry.stat().st_size

        return {"orphan_count": orphan_count, "orphan_bytes": orphan_bytes}

    @staticmethod
    def _iter_variant_cache_files(entry: dict):
        """Yield every cache_file string registered under ``entry['variants']``.

        Used by the static gc helpers which walk all variants for both
        orphan-detection and cleanup.
        """
        for variant in (entry.get("variants") or {}).values():
            cf = variant.get("cache_file")
            if cf:
                yield cf

    @staticmethod
    def gc_clean(cache_dir: str):
        cache_path = os.path.join(cache_dir, "cache.json")
        if not os.path.isfile(cache_path):
            return

        with open(cache_path, "r") as f:
            index = json.load(f)

        SmartDiskCache._migrate_legacy_index_in_place(index)
        entries = index.get("entries", {})
        hash_index = index.get("hash_index", {})

        dead_filepaths = [fp for fp in entries if not os.path.isfile(fp)]

        for fp in dead_filepaths:
            entry = entries.pop(fp)
            file_hash = entry.get("hash", "")
            if file_hash in hash_index:
                paths = hash_index[file_hash]
                if fp in paths:
                    paths.remove(fp)
                if not paths:
                    del hash_index[file_hash]
                    for cf in SmartDiskCache._iter_variant_cache_files(entry):
                        for v in range(1, 100):
                            pt = os.path.join(cache_dir, f"{cf}_{v}.pt")
                            if os.path.isfile(pt):
                                os.remove(pt)
                            else:
                                break

        referenced_cache_files = set()
        for entry in entries.values():
            for cf in SmartDiskCache._iter_variant_cache_files(entry):
                for v in range(1, 100):
                    pt = os.path.join(cache_dir, f"{cf}_{v}.pt")
                    if os.path.isfile(pt):
                        referenced_cache_files.add(os.path.normpath(pt))
                    else:
                        break

        sentinel_name = index.get("blank_sentinel")
        if sentinel_name:
            referenced_cache_files.add(os.path.normpath(os.path.join(cache_dir, sentinel_name)))

        for entry in os.scandir(cache_dir):
            if entry.name.endswith(".pt") and entry.is_file():
                if os.path.normpath(entry.path) not in referenced_cache_files:
                    os.remove(entry.path)

        tmp_path = cache_path + ".tmp"
        bak_path = cache_path + ".bak"
        with open(tmp_path, "w") as f:
            json.dump(index, f, indent=2)
        if os.path.exists(cache_path):
            shutil.copy2(cache_path, bak_path)
        os.replace(tmp_path, cache_path)
