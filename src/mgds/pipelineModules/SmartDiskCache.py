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
SCHEMA_METHOD = 'shape_v1'

# Sentinel resolution key used for sourceless / text-style entries that have
# no resolution dimension. Lives inside ``entry['variants']`` like any other
# variant key, but the cache filename collapses to bare ``hash12``.
NO_RESOLUTION_KEY = '_'


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
            bucket_method_provider: Callable[[], str] | None = None,
            rebucket_provider: Callable[[float], list[str]] | None = None,
    ):
        super().__init__()
        self.cache_dir = cache_dir
        self._real_cache_dir = os.path.realpath(cache_dir)

        self.split_names = [] if split_names is None else split_names
        self.aggregate_names = [] if aggregate_names is None else aggregate_names

        self.variations_in_name = variations_in_name
        self.balancing_in_name = balancing_in_name
        self.balancing_strategy_in_name = balancing_strategy_in_name
        self.variations_group_in_names = \
            [variations_group_in_name] if isinstance(variations_group_in_name, str) else variations_group_in_name

        self.group_enabled_in_name = group_enabled_in_name

        self.before_cache_fun = (lambda: None) if before_cache_fun is None else before_cache_fun
        self.stop_check_fun = stop_check_fun or (lambda: False)

        self.modeltype = modeltype
        self.source_path_in_name = source_path_in_name
        self.sourceless = sourceless

        # Optional providers for multi-resolution variant caching. Both default
        # to None for text caches and any data loader that doesn't construct
        # an AspectBucketing module — drift detection auto-disables in that
        # case and the cache behaves as a single-variant store keyed on the
        # cached resolution (or NO_RESOLUTION_KEY when there's no resolution
        # at all).
        self.bucket_method_provider = bucket_method_provider
        self.rebucket_provider = rebucket_provider

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
        if self.variations_in_name:
            inputs = inputs \
                + [self.variations_in_name] \
                + [self.balancing_in_name] \
                + [self.balancing_strategy_in_name] \
                + self.variations_group_in_names \
                + [self.group_enabled_in_name]
        return inputs

    def get_outputs(self) -> list[str]:
        outputs = self.split_names + self.aggregate_names
        if self.sourceless:
            outputs.append('concept')
        return outputs

    def __string_key(self, data: list[Any]) -> str:
        json_data = json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(',', ':'), indent=None)
        return hashlib.sha256(json_data.encode('utf-8')).hexdigest()

    def _hash_file(self, filepath: str) -> str:
        h = xxhash.xxh64()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                h.update(chunk)
        return h.hexdigest()

    def _hash_to_filename(self, full_hash: str) -> str:
        return full_hash[:12]

    def _get_cache_json_path(self) -> str:
        return os.path.join(self.cache_dir, 'cache.json')

    def _load_cache_index(self) -> dict:
        cache_path = self._get_cache_json_path()
        tmp_path = cache_path + '.tmp'
        bak_path = cache_path + '.bak'

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
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
                with open(tmp_path, 'r') as f:
                    data = json.load(f)
                os.replace(tmp_path, cache_path)
                self._migrate_legacy_index_in_place(data)
                return data
            except (json.JSONDecodeError, OSError):
                pass

        if os.path.exists(bak_path):
            try:
                with open(bak_path, 'r') as f:
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
        if idx.get('version', 0) >= CACHE_VERSION:
            return
        for entry in idx.get('entries', {}).values():
            if 'variants' in entry:
                continue
            legacy_cache_file = entry.pop('cache_file', None)
            legacy_resolution = entry.pop('resolution', None)
            if legacy_cache_file:
                key = legacy_resolution if legacy_resolution else NO_RESOLUTION_KEY
                entry['variants'] = {key: {'cache_file': legacy_cache_file}}
            else:
                entry['variants'] = {}
            entry['cache_version'] = CACHE_VERSION
        idx['version'] = CACHE_VERSION

    def _save_cache_index(self, compact: bool = False):
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = self._get_cache_json_path()
        tmp_path = cache_path + '.tmp'
        bak_path = cache_path + '.bak'

        with self._index_lock:
            with open(tmp_path, 'w') as f:
                json.dump(self.cache_index, f, indent=None if compact else 2)

            if os.path.exists(cache_path):
                with contextlib.suppress(OSError):
                    shutil.copy2(cache_path, bak_path)

            os.replace(tmp_path, cache_path)

    def _flush_cache_index(self):
        now = time.monotonic()
        if now - self._last_flush_time >= 30.0:
            self._save_cache_index(compact=True)
            self._last_flush_time = now

    def _get_resolution_string(self, in_variation: int, in_index: int) -> str | None:
        if 'crop_resolution' in self.aggregate_names:
            res = self._get_previous_item(in_variation, 'crop_resolution', in_index)
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
        """
        entries = self.cache_index.get('entries', {})
        for fp in filepaths:
            entry = entries.get(fp)
            if entry is None:
                continue
            self._active_key_by_filepath[fp] = next(
                iter(entry.get('variants', {}).keys()),
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
        if self.rebucket_provider is None:
            return
        entries = self.cache_index.get('entries')
        if not entries:
            return
        # Build a reverse map filename → filepath so we can pre-set the
        # active key per filepath. Entries are keyed by filepath already, so
        # this is just a list comprehension.
        derived = 0
        reused = 0
        with self._index_lock:
            for filepath, entry in entries.items():
                if entry.get('modeltype') != self.modeltype:
                    continue
                variants = entry.get('variants') or {}
                aspect = self._aspect_from_variant_keys(variants)
                if aspect is None:
                    continue
                try:
                    new_keys = self.rebucket_provider(aspect)
                except Exception:
                    continue
                if not new_keys:
                    continue
                file_hash = entry.get('hash')
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
                        variants[nk] = {'cache_file': cache_file}
                        reused += 1
                    derived += 1
                # Move newly-derived keys to the front so the "first
                # variant" picks them, preserving any reused .pt links.
                new_keys_set = set(new_keys)
                reordered = {nk: variants[nk] for nk in new_keys if nk in variants}
                reordered.update(
                    (k, v) for k, v in variants.items() if k not in new_keys_set
                )
                entry['variants'] = reordered
                # Pin this run's active key to the first derived key —
                # even if its .pt doesn't exist yet (per-index loop will
                # rebuild on 'missing_variant' status).
                self._active_key_by_filepath[filepath] = new_keys[0]
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
            parts = k.split('x')
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
                    if e.name.endswith('.pt') and e.is_file(follow_symlinks=False):
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
        basenames present in `entries`. Unrelated files in the same dir don't
        perturb the fingerprint, so sidecar caption/mask touches won't
        invalidate the fast path. Returns None if any directory is unreachable.
        """
        by_parent: dict[str, set[str]] = {}
        for fp in entries:
            by_parent.setdefault(os.path.dirname(fp), set()).add(os.path.basename(fp))

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

    def _validate_entry(self, filepath: str, entry: dict, requested_key: str, variations: int, current_mtime: float | None) -> str:
        """Validate ``entry``'s variant for ``requested_key`` against disk.

        Returns one of:
          - ``'valid'``: variant present and all .pt files exist.
          - ``'missing_variant'``: entry is fine but doesn't have this
            resolution key — caller queues a rebuild for *just this variant*
            and keeps the entry plus its other variants intact.
          - ``'missing_pt'``: variant key is registered but a .pt is missing.
            Caller drops the broken variant and rebuilds it.
          - ``'content_changed'``: source file changed; full entry rebuild.
          - ``'rebuild'``: source file unreadable.
        """
        if entry['modeltype'] != self.modeltype:
            raise RuntimeError(
                f"Cache modeltype mismatch for '{filepath}': "
                f"cached as '{entry['modeltype']}', current model is '{self.modeltype}'. "
                f"Delete the cache directory or use a separate cache_dir for this model type."
            )

        if current_mtime is None:
            return 'rebuild'

        variants = entry.get('variants', {})
        variant = variants.get(requested_key)

        if current_mtime == entry['mtime']:
            if variant is None:
                return 'missing_variant'
            cache_file = variant['cache_file']
            for v in range(variations):
                if f"{cache_file}_{v + 1}.pt" not in self._existing_pt_files:
                    return 'missing_pt'
            return 'valid'

        file_hash = self._hash_file(filepath)
        if file_hash != entry['hash']:
            return 'content_changed'

        entry['mtime'] = current_mtime

        if variant is None:
            return 'missing_variant'
        cache_file = variant['cache_file']
        for v in range(variations):
            if f"{cache_file}_{v + 1}.pt" not in self._existing_pt_files:
                return 'missing_pt'

        return 'valid'

    def _detect_cache_schema_drift(self) -> set[str]:
        """Return the currently-required names that aren't in on-disk cache files.

        Schema drift happens when a setting like ``masked_training`` is toggled
        between runs: ``split_names``/``aggregate_names`` now include keys that
        weren't written when the existing ``.pt`` files were built. All entries
        are produced by the same code path, so spot-checking one valid cache
        file is enough to determine the on-disk schema.
        """
        if not self.cache_index or not self.cache_index.get('entries'):
            return set()
        required = set(self.split_names) | set(self.aggregate_names)
        if not required:
            return set()

        for entry in self.cache_index['entries'].values():
            if entry.get('modeltype') != self.modeltype:
                continue
            cache_file = self._any_variant_cache_file(entry)
            if not cache_file:
                continue
            pt = self._real_pt_path(cache_file, 0)
            if not os.path.isfile(pt):
                continue
            try:
                cached = torch.load(pt, weights_only=False, map_location='cpu')
            except Exception:
                continue
            cached_keys = {k for k in cached if not k.startswith('__')}
            return required - cached_keys

        return set()

    @staticmethod
    def _any_variant_cache_file(entry: dict) -> str | None:
        """Return *any* variant's cache_file, or None if the entry has none.

        Used by code paths that don't care which resolution they're inspecting
        (sentinel templating, schema-drift spot check, sourceless aggregate
        load) — they just need *some* representative .pt.
        """
        variants = entry.get('variants') or {}
        for variant in variants.values():
            cache_file = variant.get('cache_file')
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
            variant = entry.get('variants', {}).get(key)
            if variant is not None:
                cache_file = variant.get('cache_file')
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
            mode='bilinear',
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
                entry = self.cache_index['entries'].get(filepath)
                if entry is None:
                    continue
                if entry.get('modeltype') != self.modeltype:
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

            entry = self.cache_index['entries'].get(filepath)
            if entry is None:
                return filepath, 'no_entry'
            variants = entry.get('variants') or {}
            if not variants:
                return filepath, 'no_cache_file'

            wrote_anything = False
            for variant in variants.values():
                cache_file = variant.get('cache_file')
                if not cache_file:
                    continue
                for v in range(variations):
                    real_pt = self._real_pt_path(cache_file, v)
                    if not os.path.isfile(real_pt):
                        continue
                    try:
                        cache_data = torch.load(real_pt, weights_only=False, map_location='cpu')
                    except Exception as e:
                        return filepath, f'load_failed: {e}'

                    # Reference shape from cached latent_image; everything spatial
                    # in the same .pt must match this so collate_fn can stack the
                    # batch without crashing.
                    ref_shape = None
                    latent_image = cache_data.get('latent_image')
                    if torch.is_tensor(latent_image) and latent_image.dim() >= 2:
                        ref_shape = tuple(latent_image.shape[-2:])

                    needs_work: list[str] = []
                    for name in sorted_targets:
                        cached_val = cache_data.get(name)
                        if cached_val is None:
                            needs_work.append(name)
                            continue
                        if (ref_shape is not None and torch.is_tensor(cached_val)
                                and cached_val.dim() >= 2
                                and tuple(cached_val.shape[-2:]) != ref_shape):
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
                        return filepath, f'compute_failed: {e}'

                    tmp_path = real_pt + f'.{os.getpid()}.{threading.get_ident()}.aug.tmp'
                    try:
                        torch.save(cache_data, tmp_path)
                        os.replace(tmp_path, real_pt)
                        wrote_anything = True
                    except Exception as e:
                        with contextlib.suppress(OSError):
                            os.remove(tmp_path)
                        return filepath, f'save_failed: {e}'

            return filepath, 'augmented' if wrote_anything else 'already_consistent'

        failed: list[tuple[str, str]] = []
        skipped = 0
        with tqdm(total=len(augment_items), smoothing=0.1, desc='augmenting cache') as bar:
            fs = [self._state.executor.submit(fn, fp, idx, var, current_device)
                  for fp, idx, var in augment_items]
            for count, f in enumerate(concurrent.futures.as_completed(fs), 1):
                try:
                    fp, status = f.result()
                except Exception:
                    self._state.executor.shutdown(wait=True, cancel_futures=True)
                    raise
                if status == 'already_consistent':
                    skipped += 1
                elif status not in ('augmented', 'no_entry', 'no_cache_file'):
                    failed.append((fp, status))
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

        The fingerprint is restricted to source basenames in the cache index, so
        unrelated sidecar files (.txt captions, masks, .npz) added or touched in
        the same parent directory don't invalidate the fast path. Renames within
        a directory that don't shift count or mtime sum are caught by the spot
        check downstream.
        """
        last_validated = self.cache_index.get('last_validated')
        if last_validated is None:
            return False

        entries = self.cache_index.get('entries', {})
        if not entries:
            return False

        stored_fp_raw = self.cache_index.get('watched_fingerprints')
        if stored_fp_raw is None:
            # Legacy cache without fingerprint — force a full pass so the
            # fingerprint gets written for next time.
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
            if entry.get('modeltype') != self.modeltype:
                return False
            try:
                current_mtime = os.path.getmtime(filepath)
            except OSError:
                return False
            if current_mtime != entry.get('mtime'):
                return False
            cache_file = self._any_variant_cache_file(entry)
            if cache_file and f"{cache_file}_1.pt" not in self._existing_pt_files:
                return False

        self._fast_validate_sample_size = sample_size
        return True

    def _add_to_hash_index(self, file_hash: str, filepath: str):
        if file_hash not in self.cache_index['hash_index']:
            self.cache_index['hash_index'][file_hash] = []
        if filepath not in self.cache_index['hash_index'][file_hash]:
            self.cache_index['hash_index'][file_hash].append(filepath)

    def _remove_from_hash_index(self, file_hash: str, filepath: str):
        if file_hash in self.cache_index['hash_index']:
            paths = self.cache_index['hash_index'][file_hash]
            if filepath in paths:
                paths.remove(filepath)
            if not paths:
                del self.cache_index['hash_index'][file_hash]

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
            if file_hash not in self.cache_index['hash_index']:
                return False

            for existing_path in self.cache_index['hash_index'][file_hash]:
                existing_entry = self.cache_index['entries'].get(existing_path)
                if existing_entry is None:
                    continue
                if existing_entry['modeltype'] != self.modeltype:
                    continue
                existing_variants = existing_entry.get('variants', {})
                variant = existing_variants.get(key)
                if variant is None:
                    continue
                cache_file = variant['cache_file']

                target_entry = self.cache_index['entries'].get(filepath)
                if target_entry is None:
                    target_entry = {
                        'filename': os.path.basename(filepath),
                        'hash': file_hash,
                        'mtime': mtime,
                        'modeltype': self.modeltype,
                        'variants': {},
                        'cache_version': CACHE_VERSION,
                    }
                    self.cache_index['entries'][filepath] = target_entry
                target_entry['variants'][key] = {'cache_file': cache_file}
                self._add_to_hash_index(file_hash, filepath)
                return True

            return False

    def __clone_for_cache(self, x: Any):
        if isinstance(x, torch.Tensor):
            return x.clone()
        return x

    def _build_cache_entry(
            self, filepath: str, file_hash: str, resolution: str | None, mtime: float,
            group_key: str, in_index: int, variations: int,
            current_device,
    ):
        cache_file = self._make_cache_file(file_hash, resolution)

        for v in range(variations):
            pt_name = f"{cache_file}_{v + 1}.pt"
            if pt_name in self._existing_pt_files:
                continue

            cache_data = {}
            with torch.no_grad():
                for name in self.split_names:
                    cache_data[name] = self.__clone_for_cache(self._get_previous_item(v, name, in_index))
                for name in self.aggregate_names:
                    cache_data[name] = self.__clone_for_cache(self._get_previous_item(v, name, in_index))
            cache_data['__cache_version'] = CACHE_VERSION
            cache_data['__modeltype'] = self.modeltype
            if self.source_path_in_name:
                try:
                    concept = self._get_previous_item(v, 'concept', in_index)
                    if concept is not None and isinstance(concept, dict):
                        cache_data['__concept_loss_weight'] = concept.get('loss_weight', 1.0)
                        cache_data['__concept_type'] = concept.get('type', 'STANDARD')
                        cache_data['__concept_name'] = concept.get('name', '')
                        cache_data['__concept_path'] = concept.get('path', '')
                        cache_data['__concept_seed'] = concept.get('seed', 0)
                except Exception:
                    pass

            real_pt_path = self._real_pt_path(cache_file, v)
            real_tmp_path = real_pt_path + f'.{os.getpid()}.{threading.get_ident()}.tmp'
            torch.save(cache_data, real_tmp_path)
            os.replace(real_tmp_path, real_pt_path)

            with self._index_lock:
                self._existing_pt_files.add(pt_name)

        key = self._resolution_key(resolution)
        with self._index_lock:
            entry = self.cache_index['entries'].get(filepath)
            if entry is None:
                entry = {
                    'filename': os.path.basename(filepath),
                    'hash': file_hash,
                    'mtime': mtime,
                    'modeltype': self.modeltype,
                    'variants': {},
                    'cache_version': CACHE_VERSION,
                }
                self.cache_index['entries'][filepath] = entry
            else:
                # Refresh metadata; the source content may have changed and
                # been re-hashed since this entry was first created.
                entry['hash'] = file_hash
                entry['mtime'] = mtime
                entry['modeltype'] = self.modeltype
                entry['cache_version'] = CACHE_VERSION
                if 'variants' not in entry:
                    entry['variants'] = {}
            entry['variants'][key] = {'cache_file': cache_file}
            self._add_to_hash_index(file_hash, filepath)

    def _get_source_path(self, in_variation: int, in_index: int) -> str | None:
        if self.source_path_in_name:
            return self._get_previous_item(0, self.source_path_in_name, in_index)
        return None

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
                if balancing_strategy == 'REPEATS':
                    group_output_samples[group_key] = int(math.floor(len(group_indices[group_key]) * balancing))
                if balancing_strategy == 'SAMPLES':
                    group_output_samples[group_key] = int(balancing)
        else:
            first_previous_name = self.split_names[0] if len(self.split_names) > 0 else self.aggregate_names[0]

            group_variations = {'': 1}
            group_indices = {'': list(range(self._get_previous_length(first_previous_name)))}
            group_output_samples = {'': len(group_indices[''])}
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
            if strategy == 'SAMPLES':
                rand = self._get_rand(out_variation)
                shuffled = list(self.group_full_indices[group_key])
                rand.shuffle(shuffled)
                sample_count = int(self.group_balancing[group_key])
                self.group_indices[group_key] = shuffled[:sample_count]

    def __init_sourceless(self):
        self.cache_index = self._load_cache_index()
        if not self.cache_index['entries']:
            raise RuntimeError(
                "Sourceless training enabled but cache is empty. "
                "Build the cache first with sourceless_training disabled."
            )

        for filepath, entry in self.cache_index['entries'].items():
            if entry.get('cache_version', 0) < CACHE_VERSION:
                raise RuntimeError(
                    f"Cache for '{os.path.basename(filepath)}' was built with an older format. "
                    f"Rebuild your cache with the latest version for sourceless training."
                )
            if entry.get('modeltype') != self.modeltype:
                raise RuntimeError(
                    f"Cache modeltype mismatch: cached as '{entry.get('modeltype')}', "
                    f"current model is '{self.modeltype}'. "
                    f"Change your cache directory or rebuild the cache."
                )

        self._sourceless_filepaths = sorted(self.cache_index['entries'].keys())

        for fp in self._sourceless_filepaths:
            entry = self.cache_index['entries'][fp]
            cache_file = self._any_variant_cache_file(entry)
            if cache_file is None:
                raise RuntimeError(
                    f"Sourceless training: cache entry for '{fp}' has no variants. "
                    f"Rebuild your cache."
                )
            pt_path = self._real_pt_path(cache_file, 0)
            if not os.path.isfile(pt_path):
                raise RuntimeError(
                    f"Sourceless training: cache file '{pt_path}' is missing. "
                    f"Rebuild your cache."
                )

        n = len(self._sourceless_filepaths)
        self.group_variations = {'': 1}
        self.group_indices = {'': list(range(n))}
        self.group_full_indices = {'': list(range(n))}
        self.group_output_samples = {'': n}
        self.group_balancing_strategy = {}
        self.group_balancing = {}
        self.variations_initialized = True

        self._aggregate_cache = {}
        if self.aggregate_names:
            with tqdm(total=len(self._sourceless_filepaths), smoothing=0.1, desc='loading aggregate cache') as bar:
                for _group_index, fp in enumerate(self._sourceless_filepaths):
                    entry = self.cache_index['entries'].get(fp)
                    if entry is None:
                        bar.update(1)
                        continue
                    cache_file = self._any_variant_cache_file(entry)
                    if cache_file is None:
                        bar.update(1)
                        continue
                    real_path = self._real_pt_path(cache_file, 0)
                    try:
                        cached = torch.load(real_path, weights_only=False, map_location='cpu')
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
        stored_bucket_method = self.cache_index.get('bucket_method')
        bucket_drift = (
            current_bucket_method is not None
            and stored_bucket_method != current_bucket_method
        )

        # Schema drift: if split_names/aggregate_names changed since the cache
        # was built (e.g. masked_training was just enabled), the on-disk .pt
        # files won't contain the new keys. Also re-augment when the stored
        # schema_method doesn't match -- caches stamped by an older augment
        # version may have shape-inconsistent values that need fixing in
        # place. Run before any fast-path return so downstream readers
        # always find what they need.
        required_schema = sorted(set(self.split_names) | set(self.aggregate_names))
        stored_schema = self.cache_index.get('schema')
        stored_method = self.cache_index.get('schema_method')
        if (self.cache_index.get('entries')
                and (stored_schema != required_schema or stored_method != SCHEMA_METHOD)):
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

            self.cache_index['schema'] = required_schema
            self.cache_index['schema_method'] = SCHEMA_METHOD
            self._save_cache_index()

        # Resolve the source filepaths this call will deliver.
        required_filepaths: set[str] = set()
        index_to_filepath: dict[int, str] = {}
        for group_key in self.group_variations:
            for in_index in self.group_indices[group_key]:
                filepath = self._get_source_path(0, in_index)
                if filepath is None:
                    continue
                filepath = os.path.normpath(filepath)
                index_to_filepath[in_index] = filepath
                required_filepaths.add(filepath)

        # --- Session skip path ---
        # If every required filepath was already validated earlier in this
        # process and is still present in the on-disk index, skip validation
        # entirely. This avoids per-epoch revalidation on static datasets.
        # Bypassed on bucket drift: drift recovery needs to run once before we
        # can trust session-skip again.
        if (not bucket_drift
                and required_filepaths
                and self.cache_index.get('entries')
                and required_filepaths.issubset(self._session_validated_filepaths)
                and all(fp in self.cache_index['entries'] for fp in required_filepaths)):
            self._source_path_cache = dict(index_to_filepath)
            self._populate_active_keys(required_filepaths)
            print(f"SmartDiskCache: Skipped re-validation ({len(required_filepaths)} entries already validated this run)")
            self._load_aggregate_cache(out_variation)
            return

        # --- Trust mode (OT_SKIP_CACHE_VALIDATION=1) ---
        # Skip per-file mtime/hash/.pt-existence validation. Any filepath
        # already in the on-disk index is trusted; only missing filepaths are
        # cached. Modeltype is still verified up-front to fail loud on
        # accidentally reusing another model's cache.
        skip_validation = os.environ.get("OT_SKIP_CACHE_VALIDATION") == "1"
        if skip_validation and self.cache_index.get('entries'):
            entries = self.cache_index['entries']
            for fp in required_filepaths:
                entry = entries.get(fp)
                if entry is not None and entry.get('modeltype') != self.modeltype:
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
        # variants that need rebuild, so we have to fall through.
        if (not skip_validation
                and not bucket_drift
                and self.cache_index.get('entries')
                and self._fast_validate()):
            all_in_index = all(
                fp in self.cache_index['entries'] for fp in required_filepaths
            )
            if all_in_index:
                self._source_path_cache = dict(index_to_filepath)
                self._populate_active_keys(required_filepaths)
                n = len(self.cache_index['entries'])
                checked = getattr(self, '_fast_validate_sample_size', '?')
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
            self._source_mtimes = self._bulk_stat_source_files(required_filepaths)
        else:
            self._source_mtimes = {}

        # Clear fast-validation token during full validation
        self.cache_index.pop('last_validated', None)

        before_cache_fun_called = False
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

            with tqdm(total=len(self.group_indices[group_key]), smoothing=0.1, desc='validating cache') as bar:
                for group_index, in_index in enumerate(self.group_indices[group_key]):
                    # Trust-mode early skip: avoid upstream pipeline calls
                    # (_get_source_path triggers crop_resolution upstream,
                    # which can do per-image I/O on slow cloud storage)
                    if skip_validation:
                        cached_fp = index_to_filepath.get(in_index)
                        if cached_fp is not None and cached_fp in self.cache_index['entries']:
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

                    entry = self.cache_index['entries'].get(filepath)
                    if entry is not None:
                        if skip_validation:
                            files_skipped += 1
                            bar.update(1)
                            continue
                        # Active key for this session: drift recovery may
                        # have pre-set it to a re-derived key. Otherwise
                        # pick the first registered variant — for
                        # bucket_method-matching runs that's the previous
                        # run's key, which is still valid. Trusting the
                        # cached key lets us skip the AspectBucketing
                        # ->LoadImage chain that _get_resolution_string
                        # would otherwise trigger per cache hit.
                        active_key = self._active_key_by_filepath.get(filepath)
                        if active_key is None:
                            active_key = next(
                                iter(entry.get('variants', {}).keys()),
                                NO_RESOLUTION_KEY,
                            )
                            self._active_key_by_filepath[filepath] = active_key
                        mtime = self._source_mtimes.get(filepath)
                        status = self._validate_entry(filepath, entry, active_key, variations, mtime)
                        if status == 'valid':
                            files_skipped += 1
                            bar.update(1)
                            continue
                        if status == 'missing_variant':
                            # Entry is fine but doesn't have a variant for
                            # this session's expected key. Rebuild that
                            # variant only — leave the entry and other
                            # variants in place.
                            resolution = self._get_resolution_string(0, in_index)
                            self._active_key_by_filepath[filepath] = self._resolution_key(resolution)
                            items_to_build_by_index[in_index] = (filepath, group_key, 0, in_index, group_index, variations, resolution)
                            bar.update(1)
                            continue
                        # Otherwise: rebuild. Drop the stale entry.
                        with self._index_lock:
                            old_hash = entry.get('hash')
                            if old_hash:
                                self._remove_from_hash_index(old_hash, filepath)
                            if filepath in self.cache_index['entries']:
                                del self.cache_index['entries'][filepath]
                            self._active_key_by_filepath.pop(filepath, None)

                    # Rebuild path: now we DO need the current resolution.
                    # This opens the image (via CalcAspect) but only for files
                    # that are missing or stale, not for the whole dataset.
                    resolution = self._get_resolution_string(0, in_index)
                    self._active_key_by_filepath[filepath] = self._resolution_key(resolution)
                    items_to_build_by_index[in_index] = (filepath, group_key, 0, in_index, group_index, variations, resolution)
                    bar.update(1)

            items_to_build = list(items_to_build_by_index.values())

            if not items_to_build:
                continue

            if not before_cache_fun_called and self.before_cache_fun is not None:
                before_cache_fun_called = True
                self.before_cache_fun()

            seen_paths = set()
            unique_items = []
            for item in items_to_build:
                fp = item[0]
                if fp not in seen_paths:
                    seen_paths.add(fp)
                    unique_items.append(item)

            self._last_flush_time = time.monotonic()
            with tqdm(total=len(unique_items), smoothing=0.1, desc='caching') as bar:
                def fn(filepath, group_key, in_variation, in_index, group_index, variations, resolution, current_device):
                    if torch.cuda.is_available() and current_device is not None:
                        torch.cuda.set_device(current_device)

                    try:
                        mtime = os.path.getmtime(filepath)
                    except OSError:
                        return filepath, 'missing'

                    try:
                        file_hash = self._hash_file(filepath)
                    except OSError:
                        return filepath, 'hash_failed'

                    if self._try_dedup(filepath, file_hash, resolution, mtime):
                        entry = self.cache_index['entries'][filepath]
                        key = self._resolution_key(resolution)
                        variant = entry.get('variants', {}).get(key)
                        cf = variant['cache_file'] if variant else None
                        all_present = (
                            cf is not None
                            and all(
                                f"{cf}_{v + 1}.pt" in self._existing_pt_files
                                for v in range(variations)
                            )
                        )
                        if all_present:
                            return filepath, 'dedup'

                    try:
                        self._build_cache_entry(
                            filepath, file_hash, resolution, mtime,
                            group_key, in_index, variations, current_device,
                        )
                    except Exception as e:
                        return filepath, f'build_failed: {e}'

                    return filepath, 'built'

                current_device = torch.cuda.current_device() if torch.cuda.is_available() else None

                fs = [self._state.executor.submit(
                    fn, filepath, group_key, in_variation, in_index, group_index, variations, resolution, current_device)
                      for (filepath, group_key, in_variation, in_index, group_index, variations, resolution)
                      in unique_items]
                for build_count, f in enumerate(concurrent.futures.as_completed(fs), 1):
                    try:
                        filepath, status = f.result()
                    except Exception:
                        self._state.executor.shutdown(wait=True, cancel_futures=True)
                        raise
                    if status == 'built' or status == 'dedup':
                        files_built += 1
                    elif status.startswith('build_failed') or status == 'missing' or status == 'hash_failed':
                        files_failed.append((filepath, status))
                        print(f"Warning: failed to cache '{filepath}': {status}")
                    if build_count % 250 == 0:
                        self._torch_gc()
                    self._flush_cache_index()
                    bar.update(1)
                    if self.stop_check_fun():
                        self._state.executor.shutdown(wait=True, cancel_futures=True)
                        self._save_cache_index()
                        print(f"SmartDiskCache: Stopped early. Cached {files_built} files this session, {files_skipped} reused from cache.")
                        raise CachingStoppedException

        if not skip_validation:
            self.cache_index['last_validated'] = time.time()
            # Per-watched-file fingerprint: ignored if any parent dir is
            # unreachable (returns None); a missing fingerprint just means the
            # next run pays for one extra full validation pass.
            self.cache_index['watched_fingerprints'] = (
                self._compute_watched_fingerprints(self.cache_index['entries']) or {}
            )
        self.cache_index['schema'] = required_schema
        self.cache_index['schema_method'] = SCHEMA_METHOD
        if current_bucket_method is not None:
            self.cache_index['bucket_method'] = current_bucket_method
        self._save_cache_index()

        # Mark every required filepath that ended up with a valid entry as
        # validated for this process so subsequent epochs can skip outright.
        entries = self.cache_index.get('entries', {})
        self._session_validated_filepaths.update(
            fp for fp in required_filepaths if fp in entries
        )

        total = files_built + files_skipped + len(files_failed)
        if total > 0:
            print(f"SmartDiskCache: Cached {files_built}/{total} files. {files_skipped} reused from cache. {len(files_failed)} failed.")
        if files_failed:
            for fp, reason in files_failed[:10]:
                print(f"  {fp}: {reason}")
            if len(files_failed) > 10:
                print(f"  ... and {len(files_failed) - 10} more")

        self._load_aggregate_cache(out_variation)

    def _load_aggregate_cache(self, out_variation: int):
        if not self.aggregate_names:
            return

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
                    cache_entry = self.cache_index['entries'].get(filepath)
                    if cache_entry is None:
                        continue
                    variation = in_variation % variations
                    load_items.append((filepath, cache_entry, variation))

        with tqdm(total=len(load_items), smoothing=0.1, desc='loading aggregate cache') as bar:
            for filepath, cache_entry, variation in load_items:
                cache_file = self._active_cache_file(filepath, cache_entry)
                if cache_file is None:
                    bar.update(1)
                    continue
                real_path = self._real_pt_path(cache_file, variation)
                try:
                    cached = torch.load(real_path, weights_only=False, map_location='cpu')
                    agg_data = {}
                    for name in self.aggregate_names:
                        if name in cached:
                            agg_data[name] = cached[name]
                    if agg_data:
                        self._aggregate_cache[(filepath, variation)] = agg_data
                except Exception:
                    pass
                bar.update(1)

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

        sentinel_name = self.cache_index.get('blank_sentinel')
        required = set(self.split_names) | set(self.aggregate_names)
        if sentinel_name:
            existing_path = os.path.join(self._real_cache_dir, sentinel_name)
            if os.path.isfile(existing_path):
                try:
                    existing = torch.load(existing_path, weights_only=False, map_location='cpu')
                    existing_keys = {k for k in existing if not k.startswith('__')}
                    if (existing.get('__modeltype') == self.modeltype
                            and required.issubset(existing_keys)):
                        return
                except Exception:
                    pass

        template = None
        for entry in self.cache_index.get('entries', {}).values():
            if entry.get('modeltype') != self.modeltype:
                continue
            cache_file = self._any_variant_cache_file(entry)
            if not cache_file:
                continue
            pt = self._real_pt_path(cache_file, 0)
            if os.path.isfile(pt):
                try:
                    template = torch.load(pt, weights_only=False, map_location='cpu')
                    break
                except Exception:
                    template = None
        if template is None:
            return

        sentinel = {'__cache_version': CACHE_VERSION, '__modeltype': self.modeltype}
        for name in self.split_names + self.aggregate_names:
            v = template.get(name)
            if torch.is_tensor(v):
                sentinel[name] = torch.zeros_like(v)
            elif v is not None:
                sentinel[name] = v

        out_name = 'blank_sentinel.pt'
        out_path = os.path.join(self._real_cache_dir, out_name)
        tmp_path = out_path + f'.{os.getpid()}.tmp'
        try:
            torch.save(sentinel, tmp_path)
            os.replace(tmp_path, out_path)
        except OSError:
            return

        with self._index_lock:
            self.cache_index['blank_sentinel'] = out_name
        self._save_cache_index()

    def _load_blank_sentinel(self) -> dict | None:
        if not self.cache_index:
            return None
        sentinel_name = self.cache_index.get('blank_sentinel')
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
            cache_entry = self.cache_index['entries'].get(filepath)

            if cache_entry is not None:
                variation = in_variation % self.group_variations[group_key]

                if requested_name in self.aggregate_names:
                    agg_data = self._aggregate_cache.get((filepath, variation))
                    if agg_data is not None:
                        return agg_data

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
                    # per-file augmentation failure can still leave gaps. Borrow
                    # the sentinel's zero-tensors for the missing keys so
                    # downstream readers don't crash on a single bad file.
                    sentinel = self._load_blank_sentinel()
                    if sentinel is not None:
                        for name in missing_for_file:
                            if name in sentinel:
                                item[name] = sentinel[name]
                if self.sourceless and '__concept_loss_weight' in cached:
                    item['concept'] = {
                        'loss_weight': cached['__concept_loss_weight'],
                        'type': cached.get('__concept_type', 'STANDARD'),
                        'name': cached.get('__concept_name', ''),
                        'path': cached.get('__concept_path', ''),
                        'seed': cached.get('__concept_seed', 0),
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
        cache_path = os.path.join(cache_dir, 'cache.json')
        if not os.path.isfile(cache_path):
            return {'orphan_count': 0, 'orphan_bytes': 0}

        with open(cache_path, 'r') as f:
            index = json.load(f)

        SmartDiskCache._migrate_legacy_index_in_place(index)
        entries = index.get('entries', {})

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
        sentinel_name = index.get('blank_sentinel')
        if sentinel_name:
            referenced_cache_files.add(
                os.path.normpath(os.path.join(cache_dir, sentinel_name))
            )

        orphan_count = 0
        orphan_bytes = 0

        for entry in os.scandir(cache_dir):
            if entry.name.endswith('.pt') and entry.is_file():
                if os.path.normpath(entry.path) not in referenced_cache_files:
                    orphan_count += 1
                    orphan_bytes += entry.stat().st_size

        return {'orphan_count': orphan_count, 'orphan_bytes': orphan_bytes}

    @staticmethod
    def _iter_variant_cache_files(entry: dict):
        """Yield every cache_file string registered under ``entry['variants']``.

        Used by the static gc helpers which walk all variants for both
        orphan-detection and cleanup.
        """
        for variant in (entry.get('variants') or {}).values():
            cf = variant.get('cache_file')
            if cf:
                yield cf

    @staticmethod
    def gc_clean(cache_dir: str):
        cache_path = os.path.join(cache_dir, 'cache.json')
        if not os.path.isfile(cache_path):
            return

        with open(cache_path, 'r') as f:
            index = json.load(f)

        SmartDiskCache._migrate_legacy_index_in_place(index)
        entries = index.get('entries', {})
        hash_index = index.get('hash_index', {})

        dead_filepaths = [fp for fp in entries if not os.path.isfile(fp)]

        for fp in dead_filepaths:
            entry = entries.pop(fp)
            file_hash = entry.get('hash', '')
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

        sentinel_name = index.get('blank_sentinel')
        if sentinel_name:
            referenced_cache_files.add(
                os.path.normpath(os.path.join(cache_dir, sentinel_name))
            )

        for entry in os.scandir(cache_dir):
            if entry.name.endswith('.pt') and entry.is_file():
                if os.path.normpath(entry.path) not in referenced_cache_files:
                    os.remove(entry.path)

        tmp_path = cache_path + '.tmp'
        bak_path = cache_path + '.bak'
        with open(tmp_path, 'w') as f:
            json.dump(index, f, indent=2)
        if os.path.exists(cache_path):
            shutil.copy2(cache_path, bak_path)
        os.replace(tmp_path, cache_path)
