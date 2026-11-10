import concurrent
import contextlib
import copy
import errno
import gc
import hashlib
import json
import math
import os
import random
import shutil
import threading
import time
import unicodedata
from collections.abc import Callable
from typing import Any

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModules.ResolutionPolicy import (
    ADAPTIVE_MIN_RESOLUTION,
    all_resolutions_cover_scale,
    nominal_resolution_for_crop,
    parse_adaptive_resolution_ladder,
    scale_resolution_for_crop,
    target_resolutions_for_spec,
    upscale_tolerance_fraction,
)
from mgds.pipelineModuleTypes.SingleVariationRandomAccessPipelineModule import SingleVariationRandomAccessPipelineModule

import torch

import xxhash
from PIL import Image
from tqdm import tqdm

CACHE_VERSION = 3

# Bumped whenever ``_augment_cache_with_missing_names`` changes behaviorally so
# caches stamped by an older version get re-augmented on the new path.
SCHEMA_METHOD = "shape_v1"

# Several SmartDiskCache instances with different configurations can share one
# cache dir (the SFT and DPO loaders of an SFT + DPO run both use cache/image).
# Each configuration ("role") keeps its own bucket/schema stamps and validation
# tokens under ``cache_index['roles'][<signature>]`` so the instances stop
# invalidating each other; the legacy top-level stamps are still mirrored for
# older readers. Both tables are capped, evicting the least recently changed.
MAX_ROLE_STAMPS = 8
MAX_ROLE_VALIDATION_TOKENS = 8

# Sentinel resolution key used for sourceless / text-style entries that have
# no resolution dimension. Lives inside ``entry['variants']`` like any other
# variant key, but the cache filename collapses to bare ``hash12``.
NO_RESOLUTION_KEY = "_"

# Placeholder stored in a queued rebuild item's resolution slot during the
# validation loop. Resolving the real value walks the upstream pipeline and
# decodes the source image — far too slow to do serially per item — so the
# loop defers it and a parallel pass fills these in before the build phase
# groups items by target resolution.
RESOLUTION_PENDING = object()

# Formats whose decoded size read_header_resolution can predict from the
# header. LoadImage decodes these with torchvision, which applies EXIF
# orientation to JPEG and PNG only; anything torchvision rejects falls back to
# Pillow's exif_transpose, which applies it to every format.
HEADER_RESOLUTION_FORMATS = {"JPEG", "MPO", "PNG", "WEBP"}
EXIF_ORIENTED_FORMATS = {"JPEG", "MPO", "PNG"}
EXIF_ORIENTATION_TAG = 0x0112


def read_header_resolution(path: str) -> tuple[int, int] | None:
    """(height, width) LoadImage will decode ``path`` to, from its header alone.

    Returns None whenever the header can't pin that down exactly (unsupported
    or animated formats, a WebP carrying an EXIF rotation that only the
    Pillow fallback would apply), so the caller decodes instead.
    """
    try:
        with Image.open(path) as image:
            image_format = image.format
            if image_format not in HEADER_RESOLUTION_FORMATS:
                return None
            # MPO's extra frames are previews; torchvision decodes the first.
            if image_format != "MPO" and getattr(image, "is_animated", False):
                return None
            width, height = image.size
            # info["exif"] is what the header parse already found; getexif()
            # would decode a PNG to look for a trailing eXIf chunk.
            exif_data = image.info.get("exif")
        orientation = None
        if exif_data:
            exif = Image.Exif()
            exif.load(exif_data)
            orientation = exif.get(EXIF_ORIENTATION_TAG)
    except Exception:
        return None
    rotated = orientation in (5, 6, 7, 8)
    if rotated and image_format not in EXIF_ORIENTED_FORMATS:
        return None
    return (width, height) if rotated else (height, width)


# Aggregate names whose value at load time can be reconstructed from cache
# metadata (variant key + entry filepath) without reading the .pt. When every
# configured aggregate is in this set, _load_aggregate_cache skips torch.load
# entirely — the .pt is dominated by split tensors that aggregate-load doesn't
# need, so deriving these instead saves multi-GB of I/O per epoch on large
# datasets. Falls back to torch.load per item when the variant key won't parse
# (NO_RESOLUTION_KEY, malformed) or frame-dim-enabled video where 'HxW' can't
# encode the frame dim.
_DERIVABLE_AGGREGATES = frozenset({"crop_resolution", "image_path"})


def _replace_retrying(src: str, dst: str, attempts: int = 8, delay: float = 0.05) -> None:
    """os.replace that rides out a transient PermissionError.

    On Windows, replacing a file another handle has open (a reader, or a second
    build worker renaming its own copy of the same content-addressed .pt at the
    same moment) fails with WinError 5. Content-addressed names make that race
    routine when two samples share a caption, so retry briefly with backoff.
    """
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:  # noqa: PERF203 - the retry is the point
            if attempt == attempts - 1:
                raise
            time.sleep(delay * (2**attempt))


# Windows' ERROR_TOO_MANY_LINKS: the file already has as many hard links as
# the filesystem allows (1,023 on NTFS).
_WINERROR_TOO_MANY_LINKS = 1142

# Content reuse stops trying hard links for the rest of a refresh after this
# many consecutive failures that are not "donor full" (a filesystem without
# hard links fails every time).
MAX_CONTENT_LINK_FAILURES = 8


# Names Windows resolves to a device whatever the directory holds (with any
# extension on older versions).
_DOS_DEVICE_NAMES = frozenset(
    ["con", "prn", "aux", "nul", "conin$", "conout$", *(f"com{i}" for i in range(1, 10))]
    + [f"lpt{i}" for i in range(1, 10)]
)


def _listing_key(name: str) -> str:
    """A directory entry's name, folded so that an ASCII name any supported
    filesystem would resolve to this entry has the same key.

    ASCII names fold by case. Other names are upper-cased, compatibility-
    decomposed and case-folded, which is at least as coarse as what NTFS, APFS
    and HFS+ do when they match a name (dotless i and long s reach ASCII
    through their upper case, the Kelvin sign through its decomposition).
    Folding too much is harmless: a key that matches only sends the caller to
    the filesystem.
    """
    if name.isascii():
        return name.lower()
    return unicodedata.normalize("NFKD", name.upper()).casefold()


def _list_dir_keys(parent: str) -> set[str] | None:
    """``_listing_key`` of every entry of ``parent``; None if it can't be listed."""
    try:
        with os.scandir(parent) as it:
            return {_listing_key(e.name) for e in it}
    except OSError:
        return None


def _listing_rules_out(name: str, keys: set[str] | None) -> bool:
    """True when a listing of a directory proves it holds no file ``name``.

    Only a plain name is ruled out this way: ASCII, not ending in a dot or a
    space (Windows strips those), without ``~`` (8.3 short names) or ``:``
    (streams), and not a DOS device name. Anything else, and any name whose
    key the listing holds, is left to the filesystem, so the only answer that
    can differ from asking it is for a file created after the listing.
    """
    if keys is None or not name or not name.isascii():
        return False
    if name[-1] in ". " or "~" in name or ":" in name:
        return False
    if name.split(".", 1)[0].rstrip(" ").lower() in _DOS_DEVICE_NAMES:
        return False
    return name.lower() not in keys


@contextlib.contextmanager
def _gc_heap_frozen(collect_first: bool = False):
    """Keep everything alive right now out of a build loop's forced collections.

    The build loops call ``_torch_gc()`` every 250 items, and each of those
    full collections re-walks every GC-tracked container in the process. With
    a loaded index that is millions of dicts and lists which are not garbage
    and are not about to become garbage. ``gc.freeze()`` parks all currently
    tracked objects in the permanent generation, so the collections only visit
    what the loop itself allocates. Frozen objects are still freed by
    reference counting; only reference cycles among them wait for a thaw.

    The heap is thawed again on exit, so a standalone mgds leaves the
    collector as it found it. The exception is a host that had already frozen
    objects when the loop started: it manages the permanent generation itself
    (OneTrainer freezes at every epoch start and thaws at the next one and at
    the end of the run), and thawing here would undo its freeze. The objects
    added here then stay frozen until the host's next ``gc.unfreeze()``.

    ``collect_first`` runs one full collection before freezing, so that
    garbage which is already dead (left by whatever ran before the loop, a
    model load for instance) is released instead of being parked along with
    the live objects for as long as the loop runs.
    """
    host_frozen = gc.get_freeze_count() > 0
    if collect_first:
        gc.collect()
    gc.freeze()
    try:
        yield
    finally:
        if not host_frozen:
            gc.unfreeze()


# _gc_paused bookkeeping: how many blocks currently hold the collector paused,
# and whether it was enabled when the first of them started.
_gc_pause_lock = threading.Lock()
_gc_pause_depth = 0
_gc_pause_was_enabled = False


@contextlib.contextmanager
def _gc_paused():
    """Pause the cyclic garbage collector while plain data is parsed or dumped.

    Parsing cache.json allocates millions of containers and serializing it
    walks them; neither can create garbage cycles, yet the allocation counts
    keep triggering full collections that re-walk the half-built index. The
    collector's previous enabled state is restored on exit. Blocks may nest or
    overlap across threads (a periodic flush next to build workers): the
    collector stays paused until the last one leaves.
    """
    global _gc_pause_depth, _gc_pause_was_enabled
    with _gc_pause_lock:
        if _gc_pause_depth == 0:
            _gc_pause_was_enabled = gc.isenabled()
            gc.disable()
        _gc_pause_depth += 1
    try:
        yield
    finally:
        with _gc_pause_lock:
            _gc_pause_depth -= 1
            if _gc_pause_depth == 0 and _gc_pause_was_enabled:
                gc.enable()


# Caption strings a sample_prompt_path cache stamps as sourceless runtime
# values. Blanked (keys kept) when store_sourceless_prompts is False.
SOURCELESS_PROMPT_NAMES = ("prompt", "prompt_1", "prompt_2")


class CachingStoppedException(Exception):
    pass


class SmartDiskCache(
    PipelineModule,
    SingleVariationRandomAccessPipelineModule,
):
    # Class-level defaults for the caption-privacy knobs (see __init__), so
    # instances built without __init__ (tests, offline tools) keep the
    # historical behavior.
    store_sourceless_prompts = True
    scrub_split_names: tuple[str, ...] | list[str] = ()
    _disk_index_may_hold_prompts = False

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
        adaptive_resolution: bool = False,
        additional_resolution_in_names: list[str] | None = None,
        adaptive_resolution_max: int | None = None,
        adaptive_resolution_ladder: str | list[int] | None = None,
        store_sourceless_prompts: bool = True,
        scrub_split_names: list[str] | None = None,
        header_resolution_path_in_names: dict[str, str] | None = None,
        header_resolution_blocked_in_names: list[str] | None = None,
        upscale_tolerance: float = 0.0,
    ):
        super().__init__()
        self.cache_dir = cache_dir
        self._real_cache_dir = os.path.realpath(cache_dir)

        self.split_names = [] if split_names is None else split_names
        self.aggregate_names = [] if aggregate_names is None else aggregate_names

        # Caption privacy (both opt-in, neither part of any cache fingerprint,
        # so toggling them never invalidates existing entries):
        # - store_sourceless_prompts=False stamps "" for every prompt /
        #   prompt_1 / prompt_2 runtime value (index rows and the legacy .pt
        #   __sourceless_values block). The keys stay, so sourceless
        #   validation still sees a resolved row; sourceless runs then serve
        #   "" for those names. Prompts already stored in a loaded index are
        #   blanked on load and persisted by the next save.
        # - scrub_split_names lists split tensors (e.g. token ids) that are
        #   written as zeros_like on every .pt write: same shape and dtype,
        #   no content. Only safe for names nothing downstream reads the
        #   values of (e.g. tokens next to a cached hidden state + mask).
        self.store_sourceless_prompts = store_sourceless_prompts
        self.scrub_split_names = list(scrub_split_names or [])
        # True while cache.json on disk may still hold prompts that this
        # instance blanked on load; the next save then skips the .bak copy
        # (which would preserve them) and removes any stale one.
        self._disk_index_may_hold_prompts = False

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
        self.adaptive_resolution = adaptive_resolution
        self.additional_resolution_in_names = additional_resolution_in_names or []
        self.adaptive_resolution_max = adaptive_resolution_max
        self.adaptive_resolution_ladder = adaptive_resolution_ladder
        # Mirrors AspectBucketing.upscale_tolerance for the paths that re-derive
        # tier eligibility without the (unstarted) bucketing module: sourceless
        # variant selection and the crop-resolution fallback. 0 = no upscale.
        self.upscale_tolerance = upscale_tolerance_fraction(upscale_tolerance)
        # Sourceless entries whose tolerance-preferred tier variant is not
        # cached and that train at a lower cached tier instead (see
        # _reconstructed_adaptive_variants); reported once per init.
        self._tier_fallback_entries: set[int] = set()

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

        # Verified donor payloads (content hash -> loaded cache dict) for the
        # current build pass. The headline reuse scenario copies one donor to
        # thousands of recipients; without this memo each copy re-deserializes
        # the same bytes from disk. Cleared at the start and end of every
        # refresh so donor tensors don't outlive the build pass.
        self._content_reuse_memo: dict[str, dict] = {}
        # What each memoized donor's file held when it was read: (donor .pt
        # name, key order, repr of every non-tensor value). A recipient whose
        # payload is exactly that is hard-linked to the donor instead of
        # written again (see _try_content_reuse). Same lifetime as the memo.
        self._content_donor_meta: dict[str, tuple[str, tuple, dict[str, str]]] = {}
        # Consecutive failed link attempts this refresh (see
        # MAX_CONTENT_LINK_FAILURES).
        self._content_link_failures = 0

        # Memoized blank sentinel dict; _load_blank_sentinel fires on every
        # cache-miss get_item, which is per sample per epoch in the worst
        # case. Invalidated whenever the sentinel is rewritten or refreshed.
        self._blank_sentinel_memo: dict | None = None

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

        # Resolution name -> path name (e.g. original_resolution ->
        # image_path) for resolving a new item's variant key from its source
        # files' headers instead of decoding them through the upstream chain.
        # Only valid while nothing between the image load and its CalcAspect
        # changes the size; ``header_resolution_blocked_in_names`` are
        # per-item flags (e.g. a size-changing augmentation's enable setting)
        # that send the item down the full walk when truthy.
        self.header_resolution_path_in_names = header_resolution_path_in_names or {}
        self.header_resolution_blocked_in_names = header_resolution_blocked_in_names or []

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
        # Serializes index-file writes (open/copy/replace) without holding
        # _index_lock across disk I/O — builder threads only contend on the
        # in-memory dict, not on the periodic flush's multi-MB write.
        self._index_io_lock = threading.Lock()
        # (size, mtime_ns) of cache.json as of our last load/save. The module
        # is the only writer within a run, so a matching stat means the
        # in-memory index is current and the per-epoch reload can be skipped.
        self._index_disk_stat = None
        self._last_flush_time = 0.0
        # Seconds between periodic index checkpoints. Adaptive: see
        # _flush_cache_index.
        self._flush_interval = 120.0
        self._source_path_cache = {}
        self._aggregate_cache = {}
        # Source filepaths whose cache entries have already been validated in
        # this process. Once a filepath is in this set we skip re-validating
        # it for the rest of the run — the dataset is static within a run
        # (users use repeats, not changing samples_per_epoch) so the
        # per-epoch revalidation was pure overhead.
        self._session_validated_filepaths: set[str] = set()
        # Variant-granular counterpart to _session_validated_filepaths, for the
        # multi-resolution session skip. Under multi-target bucketing a filepath
        # maps to several resolution variants and the required one rotates per
        # epoch, so filepath-granularity can't express "this epoch's variant was
        # already validated". Holds (filepath, resolution_key) pairs. Like the
        # filepath set, it is session-scoped (never reset per epoch).
        self._session_validated_variants: set[tuple[str, str]] = set()

        # Filled at the start of __refresh_cache by a single os.scandir of the
        # cache dir, then consulted for .pt existence checks instead of one
        # os.path.isfile syscall per (entry, variation). Updated by builders.
        self._existing_pt_files: set[str] = set()
        # Filled at the start of __refresh_cache by parallel os.scandir of the
        # source files' parent dirs. Lookup replaces per-file os.path.getmtime
        # in _validate_entry.
        self._source_mtimes: dict[str, float] = {}
        # parent dir -> listing keys of everything that same scan saw there
        # (only kept when sidecars are watched). Lets _check_sidecars skip the
        # failing stat of an optional sidecar that is not on disk. Dropped
        # once a build starts and at the end of the refresh.
        self._scanned_dir_keys: dict[str, set[str]] | None = None
        # Resolution variant key in use this session, per source filepath.
        # Set during validation/build; consulted by get_item and
        # _load_aggregate_cache to load the right variant's .pt. Empty for
        # text caches (no resolution dimension) — those callers fall back to
        # _any_variant_cache_file. Reset at the top of __refresh_cache.
        self._active_key_by_filepath: dict[str, str] = {}
        # True once the current refresh changed the in-memory index in a way
        # that has to reach disk. Gates the end-of-refresh save, so a refresh
        # that changed nothing neither rewrites a multi-hundred-MB cache.json
        # nor forces every other instance sharing the dir into a full re-parse.
        # Sticky for the whole refresh (a mid-build flush doesn't clear it);
        # reset at the top of __refresh_cache. Set it whenever in doubt.
        self._index_dirty = False
        # (out_variation, {in_index: resolution}) proven by the adaptive
        # fast-path guard and reused by _load_aggregate_cache in the same
        # refresh instead of re-resolving every row. Reset each refresh.
        self._epoch_keys: tuple[int, dict[int, str]] | None = None

    @staticmethod
    def _normpath(path: Any) -> str:
        return os.path.normpath(str(path))

    def _status_label(self) -> str:
        name = os.path.basename(os.path.normpath(self.cache_dir))
        return name or self.cache_dir

    def _status(self, message: str) -> None:
        # Cache phase reporting is opt-in to keep stdout clean during normal
        # runs. Set OT_SMARTCACHE_VERBOSE=1 to surface the per-phase lines.
        if os.environ.get("OT_SMARTCACHE_VERBOSE") == "1":
            print(f"SmartDiskCache[{self._status_label()}]: {message}", flush=True)

    def _safe_previous_item(self, variation: int, name: str | None, index: int) -> Any:
        if not name:
            return None
        try:
            return self._get_previous_item(variation, name, index)
        except Exception:
            return None

    def _sourceless_metadata(self, in_index: int, variations: int) -> dict:
        linked_paths = {}
        for name in ("image_path", "sample_prompt_path", "mask_path", "cond_path"):
            value = self._safe_previous_item(0, name, in_index)
            if value:
                linked_paths[name] = self._normpath(value)

        group_values = []
        if self.variations_group_in_names:
            group_values = [self._safe_previous_item(0, name, in_index) for name in self.variations_group_in_names]

        # Store the derived key, not the values. Every consumer of the old
        # ``group_values`` field immediately reduced it to
        # ``__string_key(group_values)`` and compared that for equality —
        # nothing ever read the values back. Keeping the raw list meant
        # embedding a full copy of every variations-group input (for text
        # caches that includes the concept's ``text`` config block, which
        # carries the tag-dropout whitelist) into the entry-level
        # ``sourceless`` block AND again into every ``sourceless_rows``
        # row. At ~14 KB per copy across ~96k entries that was ~2.4 GB of
        # the index — and the whole index is re-serialized under
        # ``_index_lock`` on every periodic flush, freezing all build
        # workers. The sha256 is 64 bytes and preserves the key exactly.
        group_key = self.__string_key(group_values)

        return {
            "source_path_in_name": self.source_path_in_name,
            "source_index": int(in_index),
            "variations": int(variations or 1),
            "balancing": self._safe_previous_item(0, self.balancing_in_name, in_index),
            "balancing_strategy": self._safe_previous_item(0, self.balancing_strategy_in_name, in_index),
            "group_key": group_key,
            "group_enabled": self._safe_previous_item(0, self.group_enabled_in_name, in_index)
            if self.group_enabled_in_name
            else True,
            "linked_paths": linked_paths,
        }

    def _entry_variation_count(self, entry: dict) -> int:
        max_variations = 1
        for variant in (entry.get("variants") or {}).values():
            cache_file = variant.get("cache_file")
            if not cache_file:
                continue
            count = 0
            while os.path.isfile(self._real_pt_path(cache_file, count)):
                count += 1
            max_variations = max(max_variations, count or 1)
        return max_variations

    def _sourceless_counterpart_path(self, anchor_path: str, anchor_entry: dict | None) -> str:
        if self.source_path_in_name in (None, "image_path"):
            return self._normpath(anchor_path)

        linked_paths = ((anchor_entry or {}).get("sourceless") or {}).get("linked_paths") or {}
        linked = linked_paths.get(self.source_path_in_name)
        if linked:
            return self._normpath(linked)

        if self.source_path_in_name == "sample_prompt_path":
            return self._normpath(os.path.splitext(anchor_path)[0] + ".txt")

        raise RuntimeError(
            f"Sourceless training cannot map image cache entry '{anchor_path}' "
            f"to required source '{self.source_path_in_name}'. Rebuild the cache with the latest mgds."
        )

    def _is_sourceless_anchor_cache(self) -> bool:
        return self.source_path_in_name == "image_path" or "image_path" in self.aggregate_names

    def _entry_sourceless_row_records(self, filepath: str, entry: dict) -> list[dict]:
        records = []
        rows = entry.get("sourceless_rows") or {}
        if isinstance(rows, dict):
            for row_key, row in rows.items():
                if not isinstance(row, dict):
                    continue
                metadata = row.get("metadata") or {}
                if not isinstance(metadata, dict):
                    metadata = {}
                source_index = metadata.get("source_index")
                if source_index is None:
                    try:
                        source_index = int(row_key)
                    except (TypeError, ValueError):
                        source_index = 1 << 60
                records.append(
                    {
                        "filepath": filepath,
                        "source_index": int(source_index),
                        "metadata": metadata,
                    }
                )

        if records:
            return records

        metadata = entry.get("sourceless") or {}
        source_index = metadata.get("source_index", 1 << 60) if isinstance(metadata, dict) else 1 << 60
        return [
            {
                "filepath": filepath,
                "source_index": int(source_index),
                "metadata": metadata if isinstance(metadata, dict) else {},
            }
        ]

    def _init_sourceless_groups_from_records(self, records: list[dict]) -> None:
        group_variations = {}
        group_indices = {}
        group_balancing = {}
        group_balancing_strategy = {}
        has_metadata = False

        for in_index, record in enumerate(records):
            meta = record.get("metadata") or {}
            if meta:
                has_metadata = True

            enabled = meta.get("group_enabled", True)
            if not enabled:
                continue

            group_key = self._metadata_group_key(meta)
            variations = int(
                meta.get("variations")
                or self._entry_variation_count(self.cache_index["entries"][record["filepath"]])
                or 1
            )
            balancing = meta.get("balancing")
            if balancing is None:
                balancing = 1.0
            balancing_strategy = meta.get("balancing_strategy") or "REPEATS"

            group_variations.setdefault(group_key, variations)
            group_indices.setdefault(group_key, []).append(in_index)
            group_balancing.setdefault(group_key, balancing)
            group_balancing_strategy.setdefault(group_key, balancing_strategy)

        if not has_metadata:
            n = len(records)
            variations = max(
                (self._entry_variation_count(self.cache_index["entries"][record["filepath"]]) for record in records),
                default=1,
            )
            self.group_variations = {"": variations}
            self.group_indices = {"": list(range(n))}
            self.group_full_indices = {"": list(range(n))}
            self.group_output_samples = {"": n}
            self.group_balancing_strategy = {}
            self.group_balancing = {}
            return

        group_output_samples = {}
        for group_key, indices in group_indices.items():
            balancing_strategy = group_balancing_strategy[group_key]
            balancing = group_balancing[group_key]
            if balancing_strategy == "SAMPLES":
                group_output_samples[group_key] = int(balancing)
            else:
                group_output_samples[group_key] = int(math.floor(len(indices) * balancing))

        self.group_variations = group_variations
        self.group_indices = group_indices
        self.group_full_indices = {k: list(v) for k, v in group_indices.items()}
        self.group_output_samples = group_output_samples
        self.group_balancing_strategy = group_balancing_strategy
        self.group_balancing = group_balancing

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
            if self.source_path_in_name == "sample_prompt_path":
                outputs.extend(["prompt", "prompt_1", "prompt_2"])
        return outputs

    def __string_key(self, data: list[Any]) -> str:
        json_data = json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(",", ":"), indent=None)
        return hashlib.sha256(json_data.encode("utf-8")).hexdigest()

    def _metadata_group_key(self, meta: Any) -> str:
        """Variations-group key of a stamped sourceless metadata block.

        Current stamps carry the derived key directly. Caches written before
        that change carry the raw ``group_values`` list instead, so the key is
        recomputed from it — including the historical "missing key means
        ``[""]``" default, so a legacy stamp keeps hashing to the same key it
        always did. An empty list is NOT the same as a missing field and is
        hashed as-is, matching the old read path.
        """
        if not isinstance(meta, dict):
            meta = {}
        group_key = meta.get("group_key")
        if isinstance(group_key, str):
            return group_key
        group_values = meta.get("group_values")
        if group_values is None:
            group_values = [""]
        return self.__string_key(group_values)

    def _hash_file(self, filepath: str) -> str:
        h = xxhash.xxh64()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    # Parallel pre-hash of files whose mtime changed (_prehash_changed_mtimes).
    # The reads are large so that both the read and the xxhash update spend
    # their time with the GIL released.
    PREHASH_WORKERS = 4
    PREHASH_CHUNK_BYTES = 1 << 20

    def _validation_hash(self, path: str) -> str:
        """``_hash_file(path)`` for the validation loop, served from the
        current group's pre-hash when that covered ``path``."""
        digest = (getattr(self, "_prehashed", None) or {}).get(path)
        if digest is not None:
            return digest
        return self._hash_file(path)

    @staticmethod
    def _stamp_hashed_size(entry: dict, file_hash: str, size: int | None) -> None:
        """Record on ``entry`` the length of the content ``file_hash`` was
        taken from (or drop the record when that length is not known).

        ``size_hash`` names the hash the size belongs to. A writer that knows
        nothing of these fields (an older mgds) replaces ``hash`` and leaves
        them behind; the mismatch then voids the size instead of letting it
        speak for content it was not measured on.
        """
        if size is None:
            entry.pop("size", None)
            entry.pop("size_hash", None)
        else:
            entry["size"] = int(size)
            entry["size_hash"] = file_hash

    @staticmethod
    def _recorded_size(entry: dict) -> int | None:
        """Length of the content ``entry['hash']`` was taken from, if recorded."""
        size = entry.get("size")
        if type(size) is not int or size < 0:
            return None
        file_hash = entry.get("hash")
        if not file_hash or entry.get("size_hash") != file_hash:
            return None
        return size

    @staticmethod
    def _unchanged_size(filepath: str, before: os.stat_result) -> int | None:
        """The file's size if it still has the size and mtime of ``before``
        (a stat taken before it was hashed), else None: only then is that
        size the length of what was hashed."""
        try:
            after = os.stat(filepath)
        except OSError:
            return None
        if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
            return None
        return before.st_size

    def _size_rules_out_recorded_content(self, filepath: str, entry: dict) -> bool:
        """True when ``filepath`` is not as long as the content ``entry`` was
        hashed from, which settles "content changed" without reading it.

        Entries without a recorded size (every cache built before sizes were
        recorded) are never ruled out and hash as before. The bulk scan's
        size is only a hint, since it comes from the directory entry, which
        NTFS can leave behind the file itself (it is updated per hard link):
        a stat of the file decides.
        """
        size = self._recorded_size(entry)
        if size is None:
            return False
        scanned = (getattr(self, "_source_sizes", None) or {}).get(filepath)
        if scanned is None or scanned == size:
            return False
        try:
            return os.stat(filepath).st_size != size
        except OSError:
            return False

    def _prehash_changed_mtimes(self, filepaths) -> int:
        """Hash in a thread pool what validating ``filepaths`` would hash one
        file at a time on the main thread.

        That is every source whose scanned mtime differs from the stored one
        and, for entries whose source is unchanged, every recorded sidecar
        whose mtime differs: a copy or sync that kept the bytes but not the
        mtimes makes that the whole dataset. The digests equal ``_hash_file``'s
        (xxh64 does not depend on the read size) and are handed to the loop by
        ``_validation_hash``, so its decisions are the same. A file the pool
        could not read is left to the loop, which fails on it as before.
        Returns the number of digests taken.
        """
        self._prehashed: dict[str, str] = {}
        # path -> bytes read for its digest (stamped as the entry's size).
        self._prehashed_sizes: dict[str, int] = {}
        entries = self.cache_index.get("entries", {})
        mtimes = self._source_mtimes
        sizes = getattr(self, "_source_sizes", None) or {}
        watch_sidecars = bool(self.extra_watched_paths_in_names)

        def changed_sidecars(entry: dict) -> list[str]:
            stored = entry.get("sidecar_mtimes")
            if not watch_sidecars or not isinstance(stored, dict) or entry.get("sidecar_hashes") is None:
                return []
            return [
                path
                for path, stored_mtime in stored.items()
                if stored_mtime is not None and mtimes.get(path) not in (None, stored_mtime)
            ]

        changed_sources: list[str] = []
        sidecars: set[str] = set()
        for filepath in dict.fromkeys(filepaths):
            entry = entries.get(filepath)
            if not isinstance(entry, dict) or entry.get("modeltype") != self.modeltype:
                continue
            if entry.get("shared_file_conflict"):
                continue
            current = mtimes.get(filepath)
            if current is None:
                # No source to hash; its sidecars are only checked when a
                # missing source is tolerated.
                if self.tolerate_missing_source:
                    sidecars.update(changed_sidecars(entry))
            elif current == entry.get("mtime"):
                sidecars.update(changed_sidecars(entry))
            else:
                recorded = self._recorded_size(entry)
                if recorded is not None and sizes.get(filepath, recorded) != recorded:
                    # Its length differs from the recorded one: the loop
                    # settles that without a hash (see
                    # _size_rules_out_recorded_content).
                    continue
                changed_sources.append(filepath)

        def hash_one(path: str) -> tuple[str, str | None, int]:
            h = xxhash.xxh64()
            read = 0
            try:
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(self.PREHASH_CHUNK_BYTES), b""):
                        h.update(chunk)
                        read += len(chunk)
            except OSError:
                return path, None, read
            return path, h.hexdigest(), read

        if len(changed_sources) + len(sidecars) < 2:
            return 0

        workers = max(1, min(self.PREHASH_WORKERS, os.cpu_count() or 1))
        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool,
            tqdm(total=len(changed_sources), smoothing=0.1, desc="hashing changed files") as bar,
        ):
            # Sources first: the loop only looks at the sidecars of an entry
            # whose source still has its recorded content.
            for path, digest, read in pool.map(hash_one, changed_sources):
                if digest is not None:
                    self._prehashed[path] = digest
                    self._prehashed_sizes[path] = read
                    if digest == entries[path].get("hash"):
                        sidecars.update(changed_sidecars(entries[path]))
                bar.update(1)
            todo = sorted(sidecars.difference(self._prehashed))
            bar.total += len(todo)
            bar.refresh()
            for path, digest, _read in pool.map(hash_one, todo):
                if digest is not None:
                    self._prehashed[path] = digest
                bar.update(1)
        return len(self._prehashed)

    def _hash_to_filename(self, full_hash: str) -> str:
        return full_hash[:12]

    def _get_cache_json_path(self) -> str:
        return os.path.join(self.cache_dir, "cache.json")

    @staticmethod
    def _read_index_file(path: str) -> dict:
        # The collector is paused for the parse (see _gc_paused).
        with open(path, "r", encoding="utf-8") as f, _gc_paused():
            return json.load(f)

    def _load_cache_index(self) -> dict:
        cache_path = self._get_cache_json_path()
        tmp_path = cache_path + ".tmp"
        bak_path = cache_path + ".bak"

        if os.path.exists(cache_path):
            try:
                data = self._read_index_file(cache_path)
                for p in (tmp_path, bak_path):
                    with contextlib.suppress(OSError):
                        os.remove(p)
                self._migrate_loaded_index(data)
                return data
            except (json.JSONDecodeError, OSError):
                pass

        if os.path.exists(tmp_path):
            try:
                data = self._read_index_file(tmp_path)
                os.replace(tmp_path, cache_path)
                self._migrate_loaded_index(data, recovered=True)
                return data
            except (json.JSONDecodeError, OSError):
                pass

        if os.path.exists(bak_path):
            try:
                data = self._read_index_file(bak_path)
                os.replace(bak_path, cache_path)
                self._migrate_loaded_index(data, recovered=True)
                return data
            except (json.JSONDecodeError, OSError):
                pass

        return {"version": CACHE_VERSION, "entries": {}, "hash_index": {}}

    def _migrate_loaded_index(self, data: dict, recovered: bool = False) -> None:
        """Run the on-load migrations; anything they changed must be saved.

        ``recovered`` marks an index restored from .tmp/.bak, which is also
        persisted by the next save rather than trusted to stay in place.
        """
        legacy = data.get("version", 0) < CACHE_VERSION
        self._migrate_legacy_index_in_place(data)
        stamped, flagged = self._migrate_dedup_keys(data)
        if flagged:
            print(
                f"SmartDiskCache ({self._status_label()}): {flagged} entries share a cache file with a sample of a "
                "different pair/mask identity (same chosen image, different rejected image or mask); all but the "
                "pair each file was built from are rebuilt into their own files on the next normal cache pass."
            )
        interned = self._migrate_intern_sourceless_concepts(data)
        blanked = 0
        if not self.store_sourceless_prompts:
            blanked = self._blank_stored_prompts(data)
            if blanked:
                self._disk_index_may_hold_prompts = True
        if legacy or interned or recovered or blanked or stamped or flagged:
            self._index_dirty = True

    @staticmethod
    def _blank_prompt_block(block: Any) -> int:
        """Blank prompt values of one per-variation runtime block; returns count."""
        if not isinstance(block, dict):
            return 0
        blanked = 0
        for values in block.values():
            if not isinstance(values, dict):
                continue
            for name in SOURCELESS_PROMPT_NAMES:
                if name in values and values[name] != "":
                    values[name] = ""
                    blanked += 1
        return blanked

    @classmethod
    def _blank_stored_prompts(cls, cache_index: dict) -> int:
        """Blank every prompt string stamped into an index (rows + legacy
        entry-level block), keeping the keys. Returns the number blanked."""
        blanked = 0
        for entry in (cache_index.get("entries") or {}).values():
            if not isinstance(entry, dict):
                continue
            for row in (entry.get("sourceless_rows") or {}).values():
                if isinstance(row, dict):
                    blanked += cls._blank_prompt_block(row.get("runtime_values"))
            blanked += cls._blank_prompt_block(entry.get("sourceless_runtime_values"))
        return blanked

    def _scrub_cache_payload(self, cache_data: dict) -> None:
        """Apply the caption-privacy knobs to a .pt payload about to be saved."""
        for name in self.scrub_split_names:
            value = cache_data.get(name)
            if torch.is_tensor(value):
                cache_data[name] = torch.zeros_like(value)
        if not self.store_sourceless_prompts:
            values = cache_data.get("__sourceless_values")
            if isinstance(values, dict):
                for name in SOURCELESS_PROMPT_NAMES:
                    if name in values:
                        values[name] = ""

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

    def _snapshot_index_stat(self) -> tuple[int, int] | None:
        try:
            st = os.stat(self._get_cache_json_path())
        except OSError:
            return None
        return (st.st_size, st.st_mtime_ns)

    def _cache_index_is_stale(self) -> bool:
        """True when cache.json on disk differs from what this process last
        loaded or saved (external writer, e.g. gc_clean from another run)."""
        return self._snapshot_index_stat() != self._index_disk_stat

    def _save_cache_index(self, compact: bool = True, backup: bool = True):
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = self._get_cache_json_path()
        tmp_path = cache_path + ".tmp"
        bak_path = cache_path + ".bak"

        # Serialize under the index lock, but write outside it: holding
        # _index_lock across the file write + backup copy stalled every
        # builder thread that needed to register an entry while the periodic
        # flush was writing a multi-MB index to disk. The collector is paused
        # for the dump (see _gc_paused).
        with self._index_lock, _gc_paused():
            if compact:
                # cache.json is machine-only (every reader uses json.load).
                # ``indent`` also forces the pure-Python encoder instead of the
                # C one, which profiled ~6x slower on a large index.
                payload = json.dumps(self.cache_index, separators=(",", ":"))
            else:
                payload = json.dumps(self.cache_index, indent=2)

        with self._index_io_lock:
            # Write bytes directly: a text-mode write on Windows funnels the
            # multi-MB payload through the locale codec (cp1252 charmap), which
            # profiled slower than the json serialization itself. The payload
            # is pure ASCII (ensure_ascii=True), so utf-8 bytes are readable by
            # any past or future reader regardless of its text encoding.
            with open(tmp_path, "wb") as f:
                f.write(payload.encode("utf-8"))

            # The .bak copy is crash insurance for the final save; copying a
            # multi-MB file on every periodic flush was pure overhead. Skipped
            # (and any stale .bak dropped) while the on-disk index may still
            # hold prompts this instance blanked: backing it up would keep
            # the very captions store_sourceless_prompts=False removes.
            purge_prompt_backup = self._disk_index_may_hold_prompts
            if backup and not purge_prompt_backup and os.path.exists(cache_path):
                with contextlib.suppress(OSError):
                    shutil.copy2(cache_path, bak_path)

            os.replace(tmp_path, cache_path)
            if purge_prompt_backup:
                with contextlib.suppress(OSError):
                    os.remove(bak_path)
                self._disk_index_may_hold_prompts = False
            self._index_disk_stat = self._snapshot_index_stat()

    def _flush_cache_index(self):
        """Periodic crash-recovery checkpoint of the index.

        Spacing adapts to what a checkpoint actually costs. ``json.dumps``
        runs under ``_index_lock``, so for its whole duration every build
        worker blocks on ``_try_dedup``/``_build_cache_entry`` *and* the
        drain loop that owns the progress bar is stuck here — the build
        goes completely silent. On a large index that dumps takes tens of
        seconds, so a fixed 120s cadence spent a double-digit percentage of
        the run frozen. Targeting 5% keeps the checkpoint cheap on any index
        size; the end-of-build ``_save_cache_index`` is what guarantees
        durability, this only bounds how much progress a crash can lose.
        """
        now = time.monotonic()
        if now - self._last_flush_time < self._flush_interval:
            return

        started = time.monotonic()
        self._save_cache_index(compact=True, backup=False)
        self._save_content_index()
        elapsed = time.monotonic() - started

        self._flush_interval = max(120.0, 20.0 * elapsed)
        # Measured from completion, not entry: a checkpoint that itself took
        # longer than the interval would otherwise re-fire immediately.
        self._last_flush_time = time.monotonic()

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
            with open(self._get_content_index_path(), "r", encoding="utf-8") as f:
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
            with open(tmp_path, "wb") as f:
                f.write(json.dumps(payload, indent=None, separators=(",", ":")).encode("utf-8"))
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

        When the refresh changes nothing (same concept and stamped values,
        nothing to scrub), the copy would hold exactly what the donor's file
        holds, so this entry's name is hard-linked to that file instead of
        writing the bytes a second time. The link is safe because nothing
        edits a cache file in place: every writer saves to a temporary name
        and renames it over the target, which re-points that one name and
        leaves the file the other names share untouched. Any link failure
        (no hard-link support, the link limit, a vanished donor) falls back
        to the copy; ``OT_SMARTCACHE_NO_HARDLINKS=1`` turns linking off.
        """
        if content_hash is None:
            return False
        with self._index_lock:
            donor_name = self._content_index.get(content_hash)
        if not donor_name or donor_name == pt_name or donor_name not in self._existing_pt_files:
            return False
        donor_path = os.path.join(self._real_cache_dir, donor_name)
        cache_data = self._content_reuse_memo.get(content_hash)
        if cache_data is None:
            try:
                loaded = torch.load(donor_path, weights_only=False, map_location="cpu")
            except Exception:
                return False
            if not isinstance(loaded, dict):
                return False
            # The content index can go stale while the donor file still
            # exists (the .pt rewritten in place under a different draw, with
            # the old hash mapping never removed). The write-time stamp is
            # the only proof the donor still holds the payload this hash
            # describes — without this check a stale mapping silently copies
            # another caption's embeddings.
            if loaded.get("__content_hash") != content_hash:
                return False
            required = set(self.split_names) | set(self.aggregate_names)
            present = {k for k in loaded if not k.startswith("__")}
            if not required.issubset(present):
                return False
            self._content_reuse_memo[content_hash] = loaded
            self.__dict__.setdefault("_content_donor_meta", {})[content_hash] = (
                donor_name,
                tuple(loaded),
                {key: repr(value) for key, value in loaded.items() if not torch.is_tensor(value)},
            )
            cache_data = loaded
        donor_data = cache_data
        # Shallow copy per recipient: metadata refresh below must not leak
        # between recipients sharing the memoized donor payload.
        cache_data = dict(cache_data)

        cache_data["__cache_version"] = CACHE_VERSION
        cache_data["__modeltype"] = self.modeltype
        self._stamp_sourceless_runtime_values(cache_data, in_variation, in_index)
        # The donor may predate the privacy knobs; the copy must not.
        self._scrub_cache_payload(cache_data)

        real_pt_path = os.path.join(self._real_cache_dir, pt_name)
        tmp_path = real_pt_path + f".{os.getpid()}.{threading.get_ident()}.tmp"

        donor_full = False
        if self._content_links_enabled() and self._is_donor_file_payload(
            content_hash, donor_name, donor_data, cache_data
        ):
            try:
                os.link(donor_path, tmp_path)
                _replace_retrying(tmp_path, real_pt_path)
            except OSError as error:
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)
                donor_full = error.errno == errno.EMLINK or getattr(error, "winerror", None) == _WINERROR_TOO_MANY_LINKS
                if not donor_full and error.errno not in (errno.ENOENT, errno.EEXIST):
                    self._content_link_failures = getattr(self, "_content_link_failures", 0) + 1
            else:
                # rename() does nothing when both names already are one file
                # (POSIX) and then leaves the temporary name behind.
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)
                self._content_link_failures = 0
                return True

        try:
            torch.save(cache_data, tmp_path)
            _replace_retrying(tmp_path, real_pt_path)
        except OSError:
            with contextlib.suppress(OSError):
                os.remove(tmp_path)
            return False
        if donor_full:
            # The donor has as many links as the filesystem allows. This copy
            # holds the same payload and takes over as the donor, so the next
            # recipients link to it instead of each being written in full.
            with self._index_lock:
                self._existing_pt_files.add(pt_name)
                self._content_index[content_hash] = pt_name
                self._content_index_dirty = True
                meta = self._content_donor_meta.get(content_hash)
                if meta is not None:
                    self._content_donor_meta[content_hash] = (pt_name, *meta[1:])
        return True

    def _content_links_enabled(self) -> bool:
        if os.environ.get("OT_SMARTCACHE_NO_HARDLINKS") == "1":
            return False
        return getattr(self, "_content_link_failures", 0) < MAX_CONTENT_LINK_FAILURES

    def _is_donor_file_payload(self, content_hash: str, donor_name: str, donor: dict, candidate: dict) -> bool:
        """True when saving ``candidate`` would store exactly what the donor's
        file holds, so that a hard link to the file is as good as the copy.

        Judged against the snapshot taken when the donor was read (key order
        and the repr of every non-tensor value, i.e. the version, model type,
        content hash and every stamped concept / runtime value), not against
        the memoized dict, whose nested values a refresh may have touched.
        Tensors must be the donor's own objects; a scrubbed tensor is new and
        only passes when the donor's was all zeros already. Anything unsure
        answers False and the recipient is written as a copy.
        """
        meta = (getattr(self, "_content_donor_meta", None) or {}).get(content_hash)
        if meta is None or meta[0] != donor_name:
            return False
        _, keys, reprs = meta
        if tuple(candidate) != keys:
            return False
        for key, value in candidate.items():
            if key in reprs:
                if torch.is_tensor(value) or repr(value) != reprs[key]:
                    return False
                continue
            other = donor.get(key)
            if value is other:
                continue
            if not (
                torch.is_tensor(value)
                and torch.is_tensor(other)
                and value.dtype == other.dtype
                and value.shape == other.shape
                and not bool(other.any())
                and not bool(value.any())
            ):
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

    def _needs_resolution(self) -> bool:
        """True when this cache keys variants by resolution at all.

        Mirrors the early-out gate in ``_get_resolution_string`` so callers
        can tell "resolution is None because there isn't one" (text caches)
        apart from "resolution is expensive to compute" (image caches, where
        the upstream walk decodes the source file)."""
        return "crop_resolution" in self.aggregate_names or self.resolution_from_upstream

    def _get_resolution_string(self, in_variation: int, in_index: int) -> str | None:
        if not self._needs_resolution():
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

    def _header_resolution_string(self, in_variation: int, in_index: int) -> tuple[str, dict] | None:
        """A new item's variant key from its source files' headers.

        Feeds the header sizes to ``_fast_resolution_string`` as if they were
        the build's stamped source resolutions, so the key is the one
        validation will expect for that item. Returns ``(key,
        source_resolutions)`` or None to fall back to ``_get_resolution_string``
        (no header config, a blocking flag set, a header that can't be read).
        """
        if not self.header_resolution_path_in_names or self.aspect_bucketing is None:
            return None
        if getattr(self.aspect_bucketing, "frame_dim_enabled", False):
            return None
        needed = ["original_resolution", *self.additional_resolution_in_names]
        if any(name not in self.header_resolution_path_in_names for name in needed):
            return None

        variation = self.current_variation if self.current_variation >= 0 else in_variation
        source_resolutions = {}
        try:
            for name in self.header_resolution_blocked_in_names:
                if self._get_previous_item(variation, name, in_index):
                    return None
            for name in needed:
                path = self._get_previous_item(variation, self.header_resolution_path_in_names[name], in_index)
                resolution = read_header_resolution(path) if path else None
                if resolution is None:
                    return None
                source_resolutions[name] = list(resolution)
        except Exception:
            return None

        entry = {
            "original_resolution": source_resolutions["original_resolution"],
            "source_resolutions": source_resolutions,
        }
        key = self._fast_resolution_string(entry, in_variation, in_index)
        if key is None:
            return None
        return key, source_resolutions

    def _resolve_new_item(self, in_variation: int, in_index: int) -> tuple[str | None, dict | None]:
        header = self._header_resolution_string(in_variation, in_index)
        if header is not None:
            return header
        return self._get_resolution_string(in_variation, in_index), None

    def _make_cache_file(self, full_hash: str, resolution: str | None) -> str:
        hash12 = self._hash_to_filename(full_hash)
        if resolution:
            return f"{hash12}_{resolution}"
        return hash12

    @staticmethod
    def _dedup_key(file_hash: str, sidecar_hashes: dict | None) -> str:
        """Identity of what a cache file holds: the source bytes plus every
        sidecar that feeds the cached tensors (a DPO pair's rejected image,
        the chosen and rejected masks, ...).

        Two samples may only share a .pt (dedup) and a cache_file name when
        this key matches. Keying on the source hash alone let every pair with
        the same chosen image collapse onto one file, so pairs with a
        different rejected image or mask trained on another pair's tensors.
        Without sidecars the key is the plain content hash, so single-image
        caches keep exactly their old keys and file names.
        """
        if not sidecar_hashes:
            return file_hash
        h = xxhash.xxh64()
        h.update(file_hash.encode("utf-8"))
        for sidecar_hash in sorted(sidecar_hashes.values()):
            h.update(b"|")
            h.update(str(sidecar_hash).encode("utf-8"))
        return h.hexdigest()

    @classmethod
    def _entry_dedup_key(cls, entry: dict) -> str:
        key = entry.get("dedup_key")
        if key:
            return key
        return cls._dedup_key(entry.get("hash", ""), entry.get("sidecar_hashes"))

    @classmethod
    def _migrate_dedup_keys(cls, idx: dict) -> int:
        """Stamp ``dedup_key`` on every entry, rebuild ``hash_index`` on it and
        flag entries whose cache file is shared with a sample of a different
        key (``shared_file_conflict``).

        Caches built before sidecars entered the key may hold one file for
        several pairs that share a chosen image but not a rejected image or
        mask; only one of them is the pair the file was actually built from.
        The flag makes _validate_entry rebuild every other member of such a
        group into its own file on the next normal cache pass (the holder is
        recognised from the .pt's image_path and kept; no file is deleted)
        and makes sourceless / trust mode refuse to serve them. Idempotent;
        returns ``(entries stamped with a key, entries flagged)``.
        """
        entries = idx.get("entries")
        if not isinstance(entries, dict):
            return 0, 0
        hash_index: dict[str, list[str]] = {}
        by_cache_file: dict[str, list[str]] = {}
        stamped = 0
        for filepath, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            key = entry.get("dedup_key")
            if not key:
                key = entry["dedup_key"] = cls._dedup_key(entry.get("hash", ""), entry.get("sidecar_hashes"))
                stamped += 1
            hash_index.setdefault(key, []).append(filepath)
            for cache_file in cls._iter_variant_cache_files(entry):
                by_cache_file.setdefault(cache_file, []).append(filepath)
        idx["hash_index"] = hash_index
        flagged: set[str] = set()
        for paths in by_cache_file.values():
            if len(paths) < 2:
                continue
            # The bug: same source bytes, different sidecars, one file. Group
            # the file's entries by source hash and flag every group whose
            # members disagree on the dedup key (that is, on the sidecars).
            by_source: dict[str, list[str]] = {}
            for fp in paths:
                by_source.setdefault(entries[fp].get("hash", ""), []).append(fp)
            for same_source in by_source.values():
                if len(same_source) > 1 and len({entries[fp]["dedup_key"] for fp in same_source}) > 1:
                    flagged.update(same_source)
        for filepath, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            if filepath in flagged:
                entry["shared_file_conflict"] = True
            else:
                entry.pop("shared_file_conflict", None)
        return stamped, len(flagged)

    def _peek_pt_holder(self, cache_file: str) -> str | None:
        """The source path a variant's first .pt was built from (its stamped
        ``image_path``), read without touching the tensors; None when the file
        is unreadable or carries no path. Memoised per cache file."""
        memo = self.__dict__.setdefault("_pt_holder_memo", {})
        if cache_file in memo:
            return memo[cache_file]
        holder = None
        try:
            data = torch.load(self._real_pt_path(cache_file, 0), weights_only=False, map_location="cpu", mmap=True)
            value = data.get("image_path") if isinstance(data, dict) else None
            if isinstance(value, str) and value:
                holder = value
        except Exception:  # noqa: BLE001 - an unreadable .pt simply has no known holder
            holder = None
        memo[cache_file] = holder
        return holder

    @staticmethod
    def _same_source_path(a: str | None, b: str | None) -> bool:
        if not a or not b:
            return False
        return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))

    def _entry_holds_its_files(self, filepath: str, entry: dict) -> bool:
        """True when every variant file of a conflict-flagged entry was built
        from this entry's own source (the .pt's image_path is this path)."""
        cache_files = list(self._iter_variant_cache_files(entry))
        if not cache_files:
            return False
        return all(self._same_source_path(self._peek_pt_holder(cf), filepath) for cf in cache_files)

    def _shared_file_victims(self, filepaths) -> list[str]:
        """Among ``filepaths``, the conflict-flagged entries whose cache file
        holds another sample's tensors. Holders are unflagged on the way."""
        entries = self.cache_index.get("entries", {})
        victims = []
        for filepath in filepaths:
            entry = entries.get(filepath)
            if not isinstance(entry, dict) or not entry.get("shared_file_conflict"):
                continue
            if self._entry_holds_its_files(filepath, entry):
                entry.pop("shared_file_conflict", None)
                self._index_dirty = True
            else:
                victims.append(filepath)
        return victims

    def _rename_legacy_sidecar_files(self) -> int:
        """Give every legacy, source-hash-named file whose holder has sidecars
        the name its holder's dedup key now dictates.

        Before sidecars entered the key, a pair's file was ``<source12>_<res>``.
        A later twin WITHOUT sidecars has that very name as its own (its key is
        the source hash), so rebuilding it would only re-link to, or overwrite,
        the holder's file and the conflict could never resolve. Renaming the
        holder's file (metadata-only, no tensor I/O, index updated in step)
        frees the name; the twin then builds into it on this same pass. A file
        referenced by several entries moves only when the entry it was built
        from (its stamped image_path) is the one with sidecars; the others
        keep a dangling link and are rebuilt by validation. Nothing is ever
        overwritten: a name already taken is left alone. Files named by a
        no-sidecar entry's source hash are their own name already, so
        single-image caches are untouched. Returns the number of files moved.
        """
        entries = self.cache_index.get("entries") or {}
        referencing: dict[str, list[str]] = {}
        for filepath, entry in entries.items():
            for cache_file in self._iter_variant_cache_files(entry):
                referencing.setdefault(cache_file, []).append(filepath)
        moved = 0
        for filepath, entry in entries.items():
            if entry.get("modeltype") != self.modeltype:
                continue
            dedup_key = self._entry_dedup_key(entry)
            source_hash = entry.get("hash", "")
            if not source_hash or dedup_key == source_hash:
                continue
            for key, variant in list((entry.get("variants") or {}).items()):
                cache_file = variant.get("cache_file")
                resolution = None if key == NO_RESOLUTION_KEY else key
                if not cache_file or cache_file != self._make_cache_file(source_hash, resolution):
                    continue
                new_cache_file = self._make_cache_file(dedup_key, resolution)
                if f"{cache_file}_1.pt" not in self._existing_pt_files:
                    continue
                others = [fp for fp in referencing.get(cache_file, []) if fp != filepath]
                if others and not self._same_source_path(self._peek_pt_holder(cache_file), filepath):
                    continue  # a twin built this file; this entry is a victim and rebuilds
                names = []
                v = 0
                while f"{cache_file}_{v + 1}.pt" in self._existing_pt_files:
                    names.append((f"{cache_file}_{v + 1}.pt", f"{new_cache_file}_{v + 1}.pt"))
                    v += 1
                if any(new in self._existing_pt_files for _, new in names):
                    continue  # never overwrite a file another entry may reference
                for old, new in names:
                    _replace_retrying(os.path.join(self._real_cache_dir, old), os.path.join(self._real_cache_dir, new))
                    self._existing_pt_files.discard(old)
                    self._existing_pt_files.add(new)
                with self._index_lock:
                    variant["cache_file"] = new_cache_file
                    if not others:
                        entry.pop("shared_file_conflict", None)
                self.__dict__.setdefault("_pt_holder_memo", {}).pop(cache_file, None)
                self._index_dirty = True
                moved += 1
        if moved:
            self._status(f"moved {moved} legacy cache files to their sidecar-aware names")
            self._save_cache_index()
        return moved

    def _refuse_shared_file_victims(self, filepaths, mode: str) -> None:
        victims = self._shared_file_victims(filepaths)
        if not victims:
            return
        raise RuntimeError(
            f"{mode}: {len(victims)} pair entries in {self.cache_dir} share a cache file with a different pair "
            "(same chosen image, different rejected image or mask) and would train on the other pair's "
            "tensors. Run a normal cache pass with the source files present (skip cache validation OFF) to "
            f"rebuild them into their own files. First: {os.path.basename(victims[0])}"
        )

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

    def _role_signature(self) -> str:
        """Identity of this instance's cache configuration.

        Instances sharing one cache dir with different schemas, resolution
        inputs or watched sidecars (the SFT and DPO loaders of one run) used to
        overwrite each other's single top-level stamps, so every refresh saw
        bucket and schema drift from the other loader. Stamps are kept per
        signature instead. Variations-group inputs are left out: they never
        change .pt content, and validation tokens are additionally keyed by
        the dataset digest.
        """
        payload = {
            "schema": sorted(set(self.split_names) | set(self.aggregate_names)),
            "additional_resolutions": list(getattr(self, "additional_resolution_in_names", None) or []),
            "watched": sorted(getattr(self, "extra_watched_paths_in_names", None) or []),
            "source": self.source_path_in_name,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    @staticmethod
    def _dataset_digest(filepaths) -> str:
        """Order-independent digest of the source paths a refresh delivers.

        Validation tokens are keyed by it so two loaders with one role but
        different datasets (e.g. the training and validation loaders) each
        keep a token whose fingerprint covers exactly their own rows.
        """
        h = hashlib.sha256()
        for filepath in sorted(filepaths):
            h.update(filepath.encode("utf-8", "surrogatepass"))
            h.update(b"\0")
        return h.hexdigest()[:16]

    def _role_block(self, role_sig: str) -> dict | None:
        roles = self.cache_index.get("roles")
        if not isinstance(roles, dict):
            return None
        role = roles.get(role_sig)
        return role if isinstance(role, dict) else None

    def _role_stamps(self, role_sig: str) -> dict:
        """bucket_method / schema / schema_method as last stamped for this role.

        Migration: an index without a ``roles`` table (written before per-role
        stamps, or by an older mgds) and a role missing from the table both
        seed from the legacy top-level stamps. That is exactly the pre-role
        behaviour, so at worst one drift/validation pass stamps the role.
        """
        role = self._role_block(role_sig)
        source = role if role is not None else self.cache_index
        return {name: source.get(name) for name in ("bucket_method", "schema", "schema_method")}

    def _role_validation_token(self, role_sig: str, digest: str) -> dict | None:
        """The fast-validation token this role + dataset may rely on, or None."""
        role = self._role_block(role_sig)
        if role is not None:
            token = (role.get("validations") or {}).get(digest)
            if isinstance(token, dict):
                return dict(token, scope="required")
            return None
        if isinstance(self.cache_index.get("roles"), dict):
            # The table exists but this role never validated: a token another
            # configuration wrote can't vouch for this one's rows.
            return None
        # Pre-role index: the legacy global token, checked over every entry
        # exactly as before.
        return {
            "last_validated": self.cache_index.get("last_validated"),
            "watched_fingerprints": self.cache_index.get("watched_fingerprints"),
            "scope": "all",
        }

    def _ensure_role_block(self, role_sig: str) -> dict:
        roles = self.cache_index.get("roles")
        if not isinstance(roles, dict):
            roles = {}
            self.cache_index["roles"] = roles
        role = roles.get(role_sig)
        if not isinstance(role, dict):
            role = {}
            roles[role_sig] = role
        return role

    def _prune_roles(self, keep_sig: str) -> None:
        roles = self.cache_index.get("roles")
        if not isinstance(roles, dict) or len(roles) <= MAX_ROLE_STAMPS:
            return

        def updated(sig: str) -> float:
            role = roles.get(sig)
            return (role.get("updated") or 0.0) if isinstance(role, dict) else 0.0

        stale = sorted((sig for sig in roles if sig != keep_sig), key=updated)
        for sig in stale[: len(roles) - MAX_ROLE_STAMPS]:
            del roles[sig]

    def _stamp_role(self, role_sig: str, **stamps) -> bool:
        """Record this role's stamps; True when anything persisted changed.

        The same values are mirrored to the legacy top-level keys so an older
        mgds reading this index still finds (the most recent role's) stamps.
        """
        with self._index_lock:
            changed = self._role_block(role_sig) is None
            role = self._ensure_role_block(role_sig)
            for name, value in stamps.items():
                if role.get(name) != value:
                    role[name] = value
                    changed = True
                self.cache_index[name] = value
            if changed:
                role["updated"] = time.time()
                self._prune_roles(role_sig)
        return changed

    def _pop_role_validation(self, role_sig: str, digest: str) -> dict | None:
        role = self._role_block(role_sig)
        validations = role.get("validations") if role is not None else None
        if not isinstance(validations, dict):
            return None
        token = validations.pop(digest, None)
        return token if isinstance(token, dict) else None

    def _stamp_role_validation(self, role_sig: str, digest: str, token: dict) -> None:
        with self._index_lock:
            role = self._ensure_role_block(role_sig)
            validations = role.get("validations")
            if not isinstance(validations, dict):
                validations = {}
                role["validations"] = validations
            validations[digest] = token
            if len(validations) > MAX_ROLE_VALIDATION_TOKENS:

                def validated_at(key: str) -> float:
                    value = validations.get(key)
                    return (value.get("last_validated") or 0.0) if isinstance(value, dict) else 0.0

                stale = sorted((key for key in validations if key != digest), key=validated_at)
                for key in stale[: len(validations) - MAX_ROLE_VALIDATION_TOKENS]:
                    del validations[key]
            # Legacy mirror for older readers. Their global fingerprint won't
            # match this scoped one unless the scopes coincide, which only
            # costs them a full validation.
            self.cache_index["last_validated"] = token.get("last_validated")
            self.cache_index["watched_fingerprints"] = token.get("watched_fingerprints")

    @staticmethod
    def _same_fingerprints(stored, current) -> bool:
        if not isinstance(stored, dict) or not isinstance(current, dict):
            return False
        try:
            return {k: tuple(v) for k, v in stored.items()} == {k: tuple(v) for k, v in current.items()}
        except TypeError:
            return False

    def _missing_schema_names(self, filepaths) -> set[str]:
        """Required names absent from any recorded variant schema of ``filepaths``.

        Another role sharing the cache dir can rebuild a shared file's variant
        with its narrower schema. Checked from the index ``schema_keys`` only
        (no .pt reads); variants without a recorded schema are left to
        ``_validate_entry``'s one-time peek.
        """
        required = set(self.split_names) | set(self.aggregate_names)
        if not required:
            return set()
        entries = self.cache_index.get("entries", {})
        missing: set[str] = set()
        for filepath in filepaths:
            entry = entries.get(filepath)
            if not isinstance(entry, dict) or entry.get("modeltype") != self.modeltype:
                continue
            for variant in (entry.get("variants") or {}).values():
                schema_keys = variant.get("schema_keys")
                if schema_keys is None or not variant.get("cache_file"):
                    continue
                if not required.issubset(schema_keys):
                    missing.update(required.difference(schema_keys))
        return missing

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

    def _epoch_required_variants(self, index_to_filepath, out_variation):
        """The (filepath, resolution_key) pairs this epoch will request.

        Derives each key exactly as the validation loop, get_item and
        _load_aggregate_cache do: ``_fast_resolution_string(entry, out_variation,
        in_index)`` recovers aspect from the stamped ``original_resolution`` and
        replays the per-(out_variation, in_index) RNG bucket choice — no image
        decode. One key per in_index (``_validate_entry`` is invariant in
        in_variation), seeded on the epoch variation so it matches precisely what
        fetch will look up this epoch.

        Returns ``(required, resolvable)``. ``resolvable`` is ``False`` when any
        required entry is missing or its key can't be recovered without a decode
        — the session-skip caller must then fall through to full validation.
        """
        required: set[tuple[str, str]] = set()
        resolvable = True
        entries = self.cache_index.get("entries", {})
        for group_key in self.group_variations:
            for in_index in self.group_indices[group_key]:
                filepath = index_to_filepath.get(in_index)
                if filepath is None:
                    continue
                entry = entries.get(filepath)
                if entry is None:
                    resolvable = False
                    continue
                resolution = self._fast_resolution_string(entry, out_variation, in_index)
                if resolution is None:
                    resolvable = False
                    continue
                required.add((filepath, self._resolution_key(resolution)))
        return required, resolvable

    @staticmethod
    def _resolution_stamp_token(entry: dict) -> tuple:
        """The entry fields ``_fast_resolution_string`` may stamp.

        It replaces (never mutates in place) ``original_resolution`` and
        ``source_resolutions``, so holding the previous objects and comparing
        by value detects every stamp it wrote.
        """
        return (
            entry.get("adaptive_resolution_applied"),
            entry.get("adaptive_max_resolution"),
            entry.get("original_resolution"),
            entry.get("source_resolutions"),
        )

    def _epoch_key(self, out_variation: int, in_index: int) -> str | None:
        epoch_keys = getattr(self, "_epoch_keys", None)
        if epoch_keys is None or epoch_keys[0] != out_variation:
            return None
        return epoch_keys[1].get(in_index)

    def _adaptive_fast_path_guard(self, index_to_filepath, out_variation):
        """Prove this epoch's adaptive variants exist before skipping validation.

        ``_fast_validate`` only spot-checks that *some* variant of a sampled
        entry has a ``_1.pt``. Under adaptive resolution an entry can hold
        several tiers, and when the tier a row needs is missing, get_item
        silently falls back to a different tier or the blank sentinel. The
        adaptive fast path therefore also requires, for every row:

        * the source resolutions its key depends on are already stamped (never
          reads upstream: an unstamped row needs full validation);
        * the resolved variant has a cache_file, every variation's .pt is in
          ``_existing_pt_files``, and its ``schema_keys`` cover this cache;
        * no watched sidecar the row now resolves to exists unrecorded.

        Returns ``((keys, pairs, stamps_changed), None)`` on success, where
        ``keys`` maps in_index to its resolution string and ``pairs`` holds the
        (filepath, key) set; ``(None, reason)`` on the first failure.
        """
        entries = self.cache_index.get("entries", {})
        required_names = set(self.split_names) | set(self.aggregate_names)
        additional_names = list(getattr(self, "additional_resolution_in_names", None) or [])
        keys: dict[int, str] = {}
        pairs: set[tuple[str, str]] = set()
        stamps_changed = False
        # parent dir -> listing keys, taken on the first sidecar looked up there.
        sidecar_dirs: dict[str, set[str] | None] = {}
        for group_key in self.group_variations:
            variations = self.group_variations[group_key]
            for in_index in self.group_indices[group_key]:
                filepath = index_to_filepath.get(in_index)
                if filepath is None:
                    continue
                entry = entries.get(filepath)
                if entry is None:
                    return None, f"{filepath} is not indexed"
                if entry.get("modeltype") != self.modeltype:
                    return None, f"{filepath} was cached for another model type"
                # Mirror _fast_resolution_string's lookups exactly, so a row it
                # would resolve by walking upstream is rejected here instead.
                source_map = entry.get("source_resolutions") or {}
                resolution_values = [source_map.get("original_resolution", entry.get("original_resolution"))]
                resolution_values.extend(source_map.get(name) for name in additional_names)
                for value in resolution_values:
                    try:
                        stamped = isinstance(value, (list, tuple)) and len(value) >= 2 and float(value[-1]) > 0
                    except (TypeError, ValueError):
                        stamped = False
                    if not stamped:
                        return None, f"{filepath} has unstamped source resolutions"
                stored_sidecars = entry.get("sidecar_mtimes") or {}
                for sidecar in (self._extra_paths_by_filepath.get(filepath) or {}).values():
                    if sidecar not in stored_sidecars and self._sidecar_exists(sidecar, sidecar_dirs):
                        return None, f"{filepath} has an unrecorded sidecar"
                before = self._resolution_stamp_token(entry)
                resolution = self._fast_resolution_string(entry, out_variation, in_index)
                if self._resolution_stamp_token(entry) != before:
                    # Persisted by whichever path runs next: the fast path
                    # saves on success, full validation's end save on decline.
                    stamps_changed = True
                    self._index_dirty = True
                if resolution is None:
                    return None, f"{filepath} has no resolvable variant key"
                key = self._resolution_key(resolution)
                variant = (entry.get("variants") or {}).get(key)
                cache_file = variant.get("cache_file") if isinstance(variant, dict) else None
                if not cache_file:
                    return None, f"{filepath} is missing variant {key}"
                for v in range(variations):
                    if f"{cache_file}_{v + 1}.pt" not in self._existing_pt_files:
                        return None, f"{filepath} variant {key} is missing {cache_file}_{v + 1}.pt"
                schema_keys = variant.get("schema_keys")
                if required_names and (schema_keys is None or not required_names.issubset(schema_keys)):
                    return None, f"{filepath} variant {key} has an incomplete recorded schema"
                keys[in_index] = resolution
                pairs.add((filepath, key))
        return (keys, pairs, stamps_changed), None

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
                # Files are named by the dedup key (source + sidecars); entries
                # built before sidecars entered the key still carry files named
                # by the source hash alone, so try both.
                name_keys = dict.fromkeys((self._entry_dedup_key(entry), file_hash))
                for nk in new_keys:
                    if nk in variants:
                        continue
                    for name_key in name_keys:
                        cache_file = self._make_cache_file(
                            name_key,
                            None if nk == NO_RESOLUTION_KEY else nk,
                        )
                        if f"{cache_file}_1.pt" in self._existing_pt_files:
                            variants[nk] = {"cache_file": cache_file}
                            reused += 1
                            break
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

    @staticmethod
    def _sidecar_exists(path: str, listings: dict[str, set[str] | None]) -> bool:
        """``os.path.exists(path)`` for an optional sidecar, without the failing
        stat when its directory's listing proves it absent.

        ``listings`` caches one listing per parent directory, taken on first
        use. Whatever the listing can't rule out (see ``_listing_rules_out``)
        is asked of the filesystem as before.
        """
        parent, name = os.path.split(path)
        if parent not in listings:
            listings[parent] = _list_dir_keys(parent)
        if _listing_rules_out(name, listings[parent]):
            return False
        return os.path.exists(path)

    def _scan_rules_out(self, path: str) -> bool:
        """True when this refresh's bulk scan listed ``path``'s directory and
        that listing proves the file absent (see ``_listing_rules_out``)."""
        listings = getattr(self, "_scanned_dir_keys", None)
        if not listings:
            return False
        parent, name = os.path.split(path)
        return _listing_rules_out(name, listings.get(parent))

    def _bulk_stat_source_files(self, filepaths) -> dict[str, float]:
        """Parallel os.scandir per parent dir; returns {normpath: mtime}.

        Replaces N×os.path.getmtime syscalls with K×os.scandir (K = number of
        distinct parent dirs). Files that don't exist are simply absent from
        the returned dict (so callers must treat a missing key as "rebuild").

        With watched sidecars, every name each fully listed directory holds is
        also kept (``_scanned_dir_keys``), so ``_check_sidecars`` can tell an
        optional sidecar is absent without a failing stat per file.
        """
        by_parent: dict[str, set[str]] = {}
        for fp in filepaths:
            by_parent.setdefault(os.path.dirname(fp), set()).add(os.path.basename(fp))

        keep_listings = bool(getattr(self, "extra_watched_paths_in_names", None))
        listings: dict[str, set[str]] = {}
        self._scanned_dir_keys = listings
        # {normpath: size in bytes}, from the same stat as the mtime. Only a
        # hint (see _size_rules_out_recorded_content).
        sizes: dict[str, int] = {}
        self._source_sizes = sizes

        if not by_parent:
            return {}

        mtimes: dict[str, float] = {}
        lock = threading.Lock()

        def scan_one(parent: str, names: set[str]):
            local: dict[str, float] = {}
            local_sizes: dict[str, int] = {}
            seen: set[str] | None = set() if keep_listings else None
            try:
                with os.scandir(parent) as it:
                    for e in it:
                        if seen is not None:
                            seen.add(_listing_key(e.name))
                        if e.name in names:
                            with contextlib.suppress(OSError):
                                stat = e.stat()
                                path = os.path.normpath(os.path.join(parent, e.name))
                                local[path] = stat.st_mtime
                                local_sizes[path] = stat.st_size
            except OSError:
                # A listing that broke off proves nothing about what it lacks.
                seen = None
            if local or seen is not None:
                with lock:
                    mtimes.update(local)
                    sizes.update(local_sizes)
                    if seen is not None:
                        listings[parent] = seen

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

        if entry.get("shared_file_conflict"):
            # Flagged on load: this entry's file is shared with a sample of a
            # different pair/mask identity. The pair the file was built from
            # keeps it; every other member rebuilds into its own file (the
            # shared file is left in place for its holder).
            if self._entry_holds_its_files(filepath, entry):
                with self._index_lock:
                    entry.pop("shared_file_conflict", None)
                self._index_dirty = True
            else:
                return "content_changed"

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

        # A file whose length differs from the content the entry was hashed
        # from cannot hold the same bytes: no need to read it to find out.
        if self._size_rules_out_recorded_content(filepath, entry):
            return "content_changed"

        file_hash = self._validation_hash(filepath)
        if file_hash != entry["hash"]:
            return "content_changed"

        entry["mtime"] = current_mtime
        hashed_size = (getattr(self, "_prehashed_sizes", None) or {}).get(filepath)
        if hashed_size is not None:
            self._stamp_hashed_size(entry, file_hash, hashed_size)
        self._index_dirty = True

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
        self._index_dirty = True
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

    @staticmethod
    def _atomic_write_json(path: str, payload: dict) -> None:
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)

    @classmethod
    def _adaptive_prune_plan(
        cls,
        cache_dir: str,
        additional_resolution_names: list[str] | None = None,
        additional_resolution_provider: Callable[[str, dict], dict[str, list[int]] | None] | None = None,
        adaptive_resolution_ladder: str | list[int] | None = None,
        adaptive_resolution_max: int | None = None,
        upscale_tolerance: float = 0.0,
    ) -> tuple[dict, dict, dict]:
        """Build a read-only plan for removing cache variants that upscale.

        Only index metadata and file headers supplied by the optional caller
        are consulted; tensor payloads are never loaded. ``upscale_tolerance``
        keeps variants a loader with that tolerance may still train.
        """
        upscale_tolerance = upscale_tolerance_fraction(upscale_tolerance)
        index_path = os.path.join(cache_dir, "cache.json")
        with open(index_path, "r", encoding="utf-8") as handle:
            original_index = json.load(handle)
        updated_index = copy.deepcopy(original_index)
        additional_resolution_names = additional_resolution_names or []

        affected_paths = []
        removed_variants = []
        skipped_paths = []
        entries_without_usable_variant = []

        for filepath, entry in (updated_index.get("entries") or {}).items():
            if entry.get("adaptive_resolution_applied") is False:
                continue
            source_map = dict(entry.get("source_resolutions") or {})
            if "original_resolution" not in source_map and entry.get("original_resolution") is not None:
                source_map["original_resolution"] = entry["original_resolution"]
            missing_names = [
                name
                for name in ["original_resolution", *additional_resolution_names]
                if not isinstance(source_map.get(name), (list, tuple)) or len(source_map[name]) < 2
            ]
            if missing_names and additional_resolution_provider is not None:
                resolved = additional_resolution_provider(filepath, entry) or {}
                source_map.update(resolved)
                missing_names = [
                    name
                    for name in ["original_resolution", *additional_resolution_names]
                    if not isinstance(source_map.get(name), (list, tuple)) or len(source_map[name]) < 2
                ]
            if missing_names:
                skipped_paths.append({"filepath": filepath, "missing": missing_names})
                continue

            primary = source_map["original_resolution"]
            all_sources = [
                source_map["original_resolution"],
                *(source_map[name] for name in additional_resolution_names),
            ]
            variants = entry.get("variants") or {}
            remove_keys = []
            missing_variant_metadata = []
            entry_maximum = entry.get("adaptive_max_resolution", adaptive_resolution_max)
            try:
                if entry_maximum is None:
                    adaptive_floor = min(parse_adaptive_resolution_ladder(adaptive_resolution_ladder))
                else:
                    adaptive_floor = min(
                        target_resolutions_for_spec(
                            str(entry_maximum),
                            True,
                            adaptive_resolution_ladder,
                        )
                    )
            except (TypeError, ValueError):
                adaptive_floor = ADAPTIVE_MIN_RESOLUTION
            for key, variant in variants.items():
                crop = variant.get("crop_resolution")
                if not isinstance(crop, (list, tuple)) or len(crop) < 2:
                    parts = key.split("x")
                    crop = parts if len(parts) == 2 else None
                if crop is None:
                    missing_variant_metadata.append(f"crop_resolution:{key}")
                    continue
                try:
                    crop_h, crop_w = int(crop[-2]), int(crop[-1])
                except (TypeError, ValueError):
                    missing_variant_metadata.append(f"crop_resolution:{key}")
                    continue
                if crop_h <= 0 or crop_w <= 0:
                    missing_variant_metadata.append(f"crop_resolution:{key}")
                    continue
                # The smallest configured tier is an intentional training
                # floor: retain it even when the source must be enlarged once.
                if (
                    variant.get("target_resolution") == adaptive_floor
                    or nominal_resolution_for_crop((crop_h, crop_w)) <= adaptive_floor + 1
                ):
                    continue
                scale_resolution = scale_resolution_for_crop(primary, (crop_h, crop_w))
                if not all_resolutions_cover_scale(all_sources, scale_resolution, upscale_tolerance):
                    remove_keys.append(key)

            if missing_variant_metadata:
                skipped_paths.append({"filepath": filepath, "missing": missing_variant_metadata})

            if not remove_keys:
                continue
            affected_paths.append(filepath)
            for key in remove_keys:
                record = variants.pop(key)
                removed_variants.append(
                    {
                        "filepath": filepath,
                        "variant": key,
                        "cache_file": record.get("cache_file"),
                    }
                )
            if not any(value.get("cache_file") for value in variants.values()):
                entries_without_usable_variant.append(filepath)

        retained_cache_files = {
            variant.get("cache_file")
            for entry in (updated_index.get("entries") or {}).values()
            for variant in (entry.get("variants") or {}).values()
            if variant.get("cache_file")
        }
        removed_cache_files = {
            item["cache_file"]
            for item in removed_variants
            if item.get("cache_file") and item["cache_file"] not in retained_cache_files
        }
        files = []
        if os.path.isdir(cache_dir):
            for item in os.scandir(cache_dir):
                if not item.is_file() or not item.name.endswith(".pt"):
                    continue
                cache_file = item.name.rsplit("_", 1)[0]
                if cache_file in removed_cache_files:
                    files.append(
                        {
                            "path": item.path,
                            "name": item.name,
                            "bytes": item.stat().st_size,
                        }
                    )

        result = {
            "affected_entries": len(affected_paths),
            "affected_entry_paths": affected_paths,
            "removed_variants": len(removed_variants),
            "variant_records": removed_variants,
            "files": len(files),
            "file_records": files,
            "bytes": sum(item["bytes"] for item in files),
            "skipped_missing_metadata": len(skipped_paths),
            "skipped_entries": skipped_paths,
            "entries_without_usable_variant": len(entries_without_usable_variant),
            "entries_without_usable_variant_paths": entries_without_usable_variant,
        }
        return original_index, updated_index, result

    @classmethod
    def preview_adaptive_prune(
        cls,
        cache_dir: str,
        additional_resolution_names: list[str] | None = None,
        additional_resolution_provider: Callable[[str, dict], dict[str, list[int]] | None] | None = None,
        adaptive_resolution_ladder: str | list[int] | None = None,
        adaptive_resolution_max: int | None = None,
        upscale_tolerance: float = 0.0,
    ) -> dict:
        _, _, result = cls._adaptive_prune_plan(
            cache_dir,
            additional_resolution_names,
            additional_resolution_provider,
            adaptive_resolution_ladder,
            adaptive_resolution_max,
            upscale_tolerance,
        )
        return result

    @classmethod
    def clean_adaptive_prune(
        cls,
        cache_dir: str,
        additional_resolution_names: list[str] | None = None,
        additional_resolution_provider: Callable[[str, dict], dict[str, list[int]] | None] | None = None,
        adaptive_resolution_ladder: str | list[int] | None = None,
        adaptive_resolution_max: int | None = None,
        upscale_tolerance: float = 0.0,
    ) -> dict:
        _, updated_index, result = cls._adaptive_prune_plan(
            cache_dir,
            additional_resolution_names,
            additional_resolution_provider,
            adaptive_resolution_ladder,
            adaptive_resolution_max,
            upscale_tolerance,
        )
        index_path = os.path.join(cache_dir, "cache.json")

        # Remove records first. If deletion is interrupted, the remaining .pt
        # files are harmless orphans and can be reclaimed on the next run.
        cls._atomic_write_json(index_path, updated_index)

        deleted_files = 0
        delete_errors = []
        deleted_cache_files = set()
        for file_record in result["file_records"]:
            try:
                os.remove(file_record["path"])
                deleted_files += 1
                name = file_record["name"]
                deleted_cache_files.add(name.rsplit("_", 1)[0])
            except OSError as exc:  # noqa: PERF203 - continue deleting independent files
                delete_errors.append({"path": file_record["path"], "error": str(exc)})

        content_index_path = os.path.join(cache_dir, "content_index.json")
        if deleted_cache_files and os.path.isfile(content_index_path):
            try:
                with open(content_index_path, "r", encoding="utf-8") as handle:
                    content_index = json.load(handle)
                entries = content_index.get("entries") or {}

                def references_deleted_file(value) -> bool:
                    return isinstance(value, str) and any(
                        value == cache_file or value.startswith(f"{cache_file}_") for cache_file in deleted_cache_files
                    )

                content_index["entries"] = {
                    key: value for key, value in entries.items() if not references_deleted_file(value)
                }
                cls._atomic_write_json(content_index_path, content_index)
            except (OSError, ValueError):
                # The image index mutation is already safe and authoritative;
                # a malformed auxiliary content index must not roll it back.
                pass

        return {
            **result,
            "deleted_files": deleted_files,
            "delete_errors": delete_errors,
        }

    def _active_cache_file(self, filepath: str, entry: dict) -> str | None:
        """Cache file for the variant chosen for ``filepath`` this session.

        Falls back to another source-size-eligible variant when no active key
        has been recorded (or the requested variant failed to build).  This
        prevents a stale high-resolution cache file from bypassing
        AspectBucketing's no-upscale selection at load time.
        """
        eligible = dict(self._training_eligible_variants(entry))
        key = self._active_key_by_filepath.get(filepath)
        if key is not None:
            variant = eligible.get(key)
            if variant is not None:
                cache_file = variant.get("cache_file")
                if cache_file:
                    return cache_file
        for variant in eligible.values():
            cache_file = variant.get("cache_file")
            if cache_file:
                return cache_file
        return None

    @staticmethod
    def _variant_for_cache_file(entry: dict, cache_file: str | None) -> dict | None:
        if cache_file is None:
            return None
        for variant in (entry.get("variants") or {}).values():
            if variant.get("cache_file") == cache_file:
                return variant
        return None

    def _capped_adaptive_tiers(self, maximum: int | None) -> list[int]:
        """Adaptive tiers an entry stamped with ``maximum`` may train at, ascending.

        Equals the tier list AspectBucketing derives from the item's own target
        spec: that spec's maximum plus every ladder value at or below it.
        """
        if maximum is None:
            return []
        try:
            return target_resolutions_for_spec(
                str(int(maximum)),
                True,
                getattr(self, "adaptive_resolution_ladder", None),
            )
        except (TypeError, ValueError):
            return []

    def _reconstructed_tier_buckets(self, tiers: list[int]) -> dict[int, set[tuple[int, int]]]:
        """Bucket grid per tier from an unstarted bucketing module; {} if unavailable."""
        reconstruct = getattr(getattr(self, "aspect_bucketing", None), "reconstruct_bucket_resolutions", None)
        if reconstruct is None:
            return {}
        try:
            return {tier: {(int(h), int(w)) for h, w in reconstruct(tier)} for tier in tiers}
        except Exception:
            return {}

    def _reconstructed_adaptive_variants(
        self,
        entry: dict,
        variants: list[tuple[str, dict]],
        source_resolution: tuple[int, int],
        all_source_resolutions: list,
    ) -> list[tuple[str, dict]]:
        """Sourceless twin of the started-AspectBucketing branch.

        Rebuilds the bucket the item's own tiers (capped by its stamped
        maximum) assign to its aspect, applies the same no-upscale eligibility
        with the smallest-tier fallback, and keeps the cached variant with that
        exact key. Returns [] when the bucketing module can't reconstruct its
        grid or that variant isn't cached; the caller then falls back to the
        crop-resolution rules.
        """
        reconstruct = getattr(getattr(self, "aspect_bucketing", None), "reconstruct_bucket_for_aspect", None)
        if reconstruct is None:
            return []
        tiers = self._capped_adaptive_tiers(
            entry.get("adaptive_max_resolution", getattr(self, "adaptive_resolution_max", None))
        )
        if not tiers:
            return []
        tolerance = getattr(self, "upscale_tolerance", 0.0)
        try:
            aspect = source_resolution[0] / source_resolution[1]
            eligible = [
                tier
                for tier in tiers
                if all_resolutions_cover_scale(
                    all_source_resolutions,
                    scale_resolution_for_crop(source_resolution, reconstruct(aspect, tier)),
                    tolerance,
                )
            ]
            bucket_h, bucket_w = reconstruct(aspect, max(eligible) if eligible else min(tiers))
        except Exception:
            return []
        key = f"{bucket_h}x{bucket_w}"
        selected = [(variant_key, value) for variant_key, value in variants if variant_key == key]
        if not selected and tolerance > 0.0:
            # The tier the tolerance allows was never cached (the cache was
            # built under a stricter rule); the caller's crop-resolution path
            # serves the best cached tier instead. Counted for the init report.
            fallbacks = self.__dict__.setdefault("_tier_fallback_entries", set())
            fallbacks.add(id(entry))
        return selected

    def _training_eligible_variants(self, entry: dict) -> list[tuple[str, dict]]:
        """Cached variants this source may train at without an upscale.

        Sourced runs use the live AspectBucketing target map, which also lets
        us reject every variant if the required smallest-tier fallback is
        absent.  Sourceless runs have no started bucketing module: in adaptive
        mode they rebuild the bucket grid from the unstarted module and pick
        the same variant a sourced run would; otherwise (or when that variant
        isn't cached) they recover the rule from the stamped original/crop
        resolutions and fall back to the smallest cached tier when the source
        fits none.
        """
        variants = [(key, value) for key, value in (entry.get("variants") or {}).items() if value.get("cache_file")]
        if not variants:
            return []
        adaptive_enabled = getattr(self, "adaptive_resolution", False) and entry.get(
            "adaptive_resolution_applied", True
        )

        source_resolution_map = dict(entry.get("source_resolutions") or {})
        original = source_resolution_map.get("original_resolution", entry.get("original_resolution"))
        if not isinstance(original, (list, tuple)) or len(original) < 2:
            return [] if adaptive_enabled else variants
        try:
            source_h = int(original[-2])
            source_w = int(original[-1])
        except (TypeError, ValueError):
            return [] if adaptive_enabled else variants
        if source_h <= 0 or source_w <= 0:
            return [] if adaptive_enabled else variants

        additional_resolutions = []
        for name in getattr(self, "additional_resolution_in_names", []):
            additional = source_resolution_map.get(name)
            if not isinstance(additional, (list, tuple)) or len(additional) < 2:
                # In adaptive mode an unstamped required source means there is
                # no proof that an existing high tier is safe. Validation will
                # resolve and stamp it before building the required variant.
                if adaptive_enabled:
                    return []
                continue
            additional_resolutions.append(additional)
        all_source_resolutions = [original, *additional_resolutions]

        bucket_resolutions = getattr(self.aspect_bucketing, "bucket_resolutions", None) or {}
        target_filter = getattr(self.aspect_bucketing, "eligible_target_resolutions", None)
        if bucket_resolutions and target_filter is not None:
            try:
                targets = target_filter((source_h, source_w), list(bucket_resolutions), additional_resolutions)
                if adaptive_enabled and targets:
                    targets = [max(targets)]
                aspect = source_h / source_w
                allowed_keys = {
                    f"{h}x{w}"
                    for target in targets
                    for h, w in [self.aspect_bucketing.bucket_for_aspect(aspect, target)]
                }
                return [(key, value) for key, value in variants if key in allowed_keys]
            except Exception:
                # Preserve compatibility with third-party bucketing modules;
                # the crop-resolution path below provides the same guard.
                pass
        elif adaptive_enabled:
            selected = self._reconstructed_adaptive_variants(
                entry, variants, (source_h, source_w), all_source_resolutions
            )
            if selected:
                return selected

        parsed: list[tuple[str, dict, int, int]] = []
        for key, value in variants:
            crop = value.get("crop_resolution")
            if not isinstance(crop, (list, tuple)) or len(crop) < 2:
                parts = key.split("x")
                crop = parts if len(parts) == 2 else None
            if crop is None:
                continue
            try:
                crop_h = int(crop[-2])
                crop_w = int(crop[-1])
            except (TypeError, ValueError):
                continue
            if crop_h > 0 and crop_w > 0:
                parsed.append((key, value, crop_h, crop_w))

        if not parsed:
            return variants
        fitting = []
        tolerance = getattr(self, "upscale_tolerance", 0.0)
        for key, value, crop_h, crop_w in parsed:
            scale_resolution = scale_resolution_for_crop((source_h, source_w), (crop_h, crop_w))
            if all_resolutions_cover_scale(all_source_resolutions, scale_resolution, tolerance):
                fitting.append((key, value, crop_h, crop_w))
        maximum = entry.get("adaptive_max_resolution", getattr(self, "adaptive_resolution_max", None))
        if maximum is not None:
            # Quantized buckets of a tier can exceed its nominal size (the 768
            # tier's 640x960 is 784 nominal), so tier membership also counts.
            capped_buckets = set().union(
                *self._reconstructed_tier_buckets(self._capped_adaptive_tiers(maximum)).values()
            )

            def within_maximum(value: dict, crop_h: int, crop_w: int) -> bool:
                if nominal_resolution_for_crop((crop_h, crop_w)) <= maximum + 1:
                    return True
                if (crop_h, crop_w) in capped_buckets:
                    return True
                stamped_target = value.get("target_resolution")
                return isinstance(stamped_target, int) and stamped_target <= maximum

            fitting = [item for item in fitting if within_maximum(item[1], item[2], item[3])]
        if fitting:
            if adaptive_enabled:
                largest = max(nominal_resolution_for_crop((h, w)) for _, _, h, w in fitting)
                return [
                    (key, value)
                    for key, value, h, w in fitting
                    if abs(nominal_resolution_for_crop((h, w)) - largest) < 1
                ]
            return [(key, value) for key, value, _, _ in fitting]

        adaptive_floor = ADAPTIVE_MIN_RESOLUTION
        if adaptive_enabled:
            maximum = entry.get("adaptive_max_resolution", getattr(self, "adaptive_resolution_max", None))
            try:
                if maximum is None:
                    adaptive_floor = min(
                        parse_adaptive_resolution_ladder(getattr(self, "adaptive_resolution_ladder", None))
                    )
                else:
                    adaptive_floor = min(
                        target_resolutions_for_spec(
                            str(maximum),
                            True,
                            getattr(self, "adaptive_resolution_ladder", None),
                        )
                    )
            except (TypeError, ValueError):
                adaptive_floor = ADAPTIVE_MIN_RESOLUTION
            floor_buckets = self._reconstructed_tier_buckets([adaptive_floor]).get(adaptive_floor, set())
            floor_variants = [
                (key, value)
                for key, value, crop_h, crop_w in parsed
                if value.get("target_resolution") == adaptive_floor or (crop_h, crop_w) in floor_buckets
            ]
            if floor_variants:
                return floor_variants
        smallest_nominal = min(nominal_resolution_for_crop((h, w)) for _, _, h, w in parsed)
        if adaptive_enabled and smallest_nominal > adaptive_floor + 1:
            return []
        return [
            (key, value)
            for key, value, h, w in parsed
            if abs(nominal_resolution_for_crop((h, w)) - smallest_nominal) < 1
        ]

    def _sourceless_variant(self, entry: dict, in_index: int) -> dict | None:
        """Pick the resolution-bucket variant for a sourceless row.

        AspectBucketing is placeholdered out of the sourceless pipeline (its
        variant_key_from_aspect can't walk upstream), so instead of freezing
        every image to one bucket we rotate deterministically among the entry's
        CACHED variants by (epoch, in_index): each epoch serves a different
        cached resolution, giving mixed-resolution training. Crucially both
        _try_synthesize_aggregate and get_item call this with the same
        current_variation + in_index, so the crop_resolution stamped into the
        aggregate cache always matches the variant whose latent get_item loads
        — AspectBatchSorting and collate stay shape-consistent.
        """
        usable = sorted(self._training_eligible_variants(entry))
        if not usable:
            return None
        if len(usable) == 1:
            return usable[0][1]
        if getattr(self, "adaptive_resolution", False) and entry.get("adaptive_resolution_applied", True):
            return max(
                (value for _, value in usable),
                key=lambda value: nominal_resolution_for_crop(value.get("crop_resolution", (0, 0))),
            )
        epoch = self.current_variation if self.current_variation >= 0 else 0
        pick = (epoch * 0x9E3779B1 + in_index * 0x85EBCA77) % len(usable)
        return usable[pick][1]

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
                    self._scrub_cache_payload(cache_data)

                    tmp_path = real_pt + f".{os.getpid()}.{threading.get_ident()}.aug.tmp"
                    try:
                        torch.save(cache_data, tmp_path)
                        _replace_retrying(tmp_path, real_pt)
                        wrote_anything = True
                    except Exception as e:
                        with contextlib.suppress(OSError):
                            os.remove(tmp_path)
                        return filepath, f"save_failed: {e}"

            return filepath, "augmented" if wrote_anything else "already_consistent"

        failed: list[tuple[str, str]] = []
        skipped = 0
        # Frozen for the same reason as the build loop: the index is loaded
        # and the periodic _torch_gc below would re-walk it every 250 items.
        with (
            self._frozen_loop_heap(),
            tqdm(total=len(augment_items), smoothing=0.1, desc="augmenting cache") as bar,
        ):
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

    def _frozen_loop_heap(self):
        """``_gc_heap_frozen`` for one of this refresh's build or augment loops.

        The first loop of a refresh collects before it freezes; the loops
        after it (one per group) only freeze, since each full collection
        walks the whole index again.
        """
        collect_first = not getattr(self, "_refresh_heap_collected", False)
        self._refresh_heap_collected = True
        return _gc_heap_frozen(collect_first)

    def _fast_validate(self, required_filepaths=None, token: dict | None = None) -> bool:
        """Fast validation: per-watched-file directory fingerprint + random spot check.

        Skips the expensive per-file validation loop when nothing has changed.
        Returns True if cache appears valid.

        The fingerprint is restricted to source basenames in the cache index
        plus any explicitly-watched sidecars (e.g. -masklabel.png paths recorded
        on each entry's ``sidecar_mtimes``). Unrelated files in the same
        directory don't perturb the fingerprint. Renames within a directory
        that don't shift count or mtime sum are caught by the spot check
        downstream.

        ``token`` carries ``last_validated`` / ``watched_fingerprints`` /
        ``scope`` for the caller's role and dataset (``_role_validation_token``);
        None reads the legacy top-level stamps. With ``scope == "required"`` the
        sidecar-migration check, fingerprint and spot-check cover only
        ``required_filepaths``: rows another loader sharing this cache dir owns
        can neither invalidate nor vouch for this loader's rows.
        """
        if token is None:
            token = {
                "last_validated": self.cache_index.get("last_validated"),
                "watched_fingerprints": self.cache_index.get("watched_fingerprints"),
                "scope": "all",
            }
        last_validated = token.get("last_validated")
        if last_validated is None:
            return False

        entries = self.cache_index.get("entries", {})
        if not entries:
            return False
        if required_filepaths is not None and token.get("scope") == "required":
            entries = {fp: entries[fp] for fp in required_filepaths if fp in entries}
            if not entries:
                return False

        stored_fp_raw = token.get("watched_fingerprints")
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

    def _try_dedup(
        self,
        filepath: str,
        file_hash: str,
        resolution: str | None,
        mtime: float,
        in_index: int,
        variations: int,
        dedup_key: str | None = None,
        sidecar_state: tuple[dict[str, float], dict[str, str]] | None = None,
        source_size: int | None = None,
    ) -> bool:
        """Reuse an existing entry's variant when content+resolution match.

        Hash-index lookup finds entries built from the same source bytes AND
        the same sidecars (``dedup_key``; the plain content hash when the
        cache has no sidecars); we only dedup when one of those entries
        already has a variant at the requested resolution key. Different
        resolution requests of the same content are still allowed to share a
        single entry — we just register an additional variant under a
        different key. An entry still flagged as sharing its file with a
        different pair is never a donor: its file may hold the other pair.
        """
        key = self._resolution_key(resolution)
        dedup_key = dedup_key or file_hash
        runtime_values_by_variation = (
            self._sourceless_runtime_values_by_variation(variations, in_index) if self.source_path_in_name else {}
        )
        with self._index_lock:
            if dedup_key not in self.cache_index["hash_index"]:
                return False

            for existing_path in self.cache_index["hash_index"][dedup_key]:
                existing_entry = self.cache_index["entries"].get(existing_path)
                if existing_entry is None:
                    continue
                if existing_entry["modeltype"] != self.modeltype:
                    continue
                if existing_entry.get("shared_file_conflict"):
                    continue
                existing_variants = existing_entry.get("variants", {})
                variant = existing_variants.get(key)
                if variant is None:
                    continue
                cache_file = variant["cache_file"]

                target_entry = self.cache_index["entries"].get(filepath)
                if target_entry is None:
                    sidecar_mtimes, sidecar_hashes = sidecar_state or self._compute_sidecar_state(
                        self._extra_paths_by_filepath.get(filepath, {})
                    )
                    target_entry = {
                        "filename": os.path.basename(filepath),
                        "hash": file_hash,
                        "dedup_key": dedup_key,
                        "mtime": mtime,
                        "modeltype": self.modeltype,
                        "variants": {},
                        "cache_version": CACHE_VERSION,
                        "sidecar_mtimes": sidecar_mtimes,
                        "sidecar_hashes": sidecar_hashes,
                        "sourceless": self._sourceless_metadata(in_index, variations),
                    }
                    self._stamp_hashed_size(target_entry, file_hash, source_size)
                    row = target_entry.setdefault("sourceless_rows", {}).setdefault(str(in_index), {})
                    row["metadata"] = target_entry["sourceless"]
                    # Always set runtime_values (even {}); key presence is the
                    # "row resolved" marker the presence predicate latches on.
                    self._store_row_runtime_values(target_entry, in_index, runtime_values_by_variation)
                    self.cache_index["entries"][filepath] = target_entry
                else:
                    target_entry["dedup_key"] = dedup_key
                    target_entry.pop("shared_file_conflict", None)
                    target_entry["sourceless"] = self._sourceless_metadata(in_index, variations)
                    row = target_entry.setdefault("sourceless_rows", {}).setdefault(str(in_index), {})
                    row["metadata"] = target_entry["sourceless"]
                    self._store_row_runtime_values(target_entry, in_index, runtime_values_by_variation)
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
                self._add_to_hash_index(dedup_key, filepath)
                return True

            return False

    def __clone_for_cache(self, x: Any):
        if isinstance(x, torch.Tensor):
            return x.clone()
        return x

    def _sourceless_runtime_values(self, in_variation: int, in_index: int) -> dict:
        values = {}

        concept = self._safe_previous_item(in_variation, "concept", in_index)
        if isinstance(concept, dict):
            values["concept"] = copy.deepcopy(concept)

        if self.source_path_in_name == "sample_prompt_path":
            for name in SOURCELESS_PROMPT_NAMES:
                value = self._safe_previous_item(in_variation, name, in_index)
                if value is not None:
                    # Key kept even when blanked: sourceless serving and the
                    # row-presence checks only look for the key.
                    values[name] = copy.deepcopy(value) if self.store_sourceless_prompts else ""

        return values

    def _sourceless_runtime_values_by_variation(self, variations: int, in_index: int) -> dict:
        values_by_variation = {}
        for v in range(int(variations or 1)):
            values = self._sourceless_runtime_values(v, in_index)
            if values:
                values_by_variation[str(v)] = values
        return values_by_variation

    # ------------------------------------------------------------------
    # Concept interning
    #
    # The per-variation runtime values embed the *concept* — an ~80-field
    # config dict that is identical for every sample of a concept and every
    # variation of a sample. Embedding a deep copy per (entry, row, variation)
    # exploded cache.json to multiple GB (the concept block dwarfs the few
    # hundred bytes of prompt it travels with) and made the per-epoch
    # json.dumps of the index the dominant cost — tens of GB of transient RAM
    # plus minutes of wall time at every epoch boundary. Instead we keep one
    # copy per distinct concept in cache_index['concepts'] and reference it by
    # a content hash; the per-variation block then carries only the prompt(s).
    # ------------------------------------------------------------------

    @staticmethod
    def _concept_identity(concept: dict) -> str:
        payload = json.dumps(concept, sort_keys=True, default=str)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def _intern_concept(self, concept: dict) -> str | None:
        if not isinstance(concept, dict):
            return None
        concepts = self.cache_index.setdefault("concepts", {})
        cid = self._concept_identity(concept)
        if cid not in concepts:
            concepts[cid] = concept
        return cid

    def _split_runtime_values_by_variation(self, runtime_values_by_variation: dict) -> tuple[str | None, dict]:
        """Intern the (variation-invariant) concept out of a runtime block.

        Returns ``(concept_id, prompts_by_variation)`` where the per-variation
        mapping is the same minus its bulky ``concept`` entry — the concept now
        lives once in ``cache_index['concepts']`` keyed by ``concept_id``. The
        concept is determined by the row (index), not the variation, so a single
        id describes every variation of the row.
        """
        concept_id = None
        prompts_by_variation = {}
        for var, values in (runtime_values_by_variation or {}).items():
            if not isinstance(values, dict):
                prompts_by_variation[var] = values
                continue
            concept = values.get("concept")
            if concept_id is None and isinstance(concept, dict):
                concept_id = self._intern_concept(concept)
            prompts_by_variation[var] = {k: v for k, v in values.items() if k != "concept"}
        return concept_id, prompts_by_variation

    def _store_row_runtime_values(self, entry: dict, in_index: int, runtime_values_by_variation: dict) -> None:
        """Write a row's runtime values in interned form.

        Replaces the old pattern of stamping the full block into BOTH
        ``entry['sourceless_runtime_values']`` and the per-row slot (the index
        then serialized every concept twice). Only the per-row slot is written;
        the read path treats the entry-level slot as a legacy fallback.
        """
        concept_id, prompts_by_variation = self._split_runtime_values_by_variation(runtime_values_by_variation)
        row = entry.setdefault("sourceless_rows", {}).setdefault(str(in_index), {})
        row["runtime_values"] = prompts_by_variation
        if concept_id is not None:
            row["concept_id"] = concept_id
        else:
            row.pop("concept_id", None)
        # Drop any stale entry-level duplicate left by an older build/migration.
        entry.pop("sourceless_runtime_values", None)

    def _attach_interned_concept(self, runtime_values: dict, holder: dict) -> dict:
        """Re-attach the interned concept to a per-variation runtime block.

        ``holder`` is the row (or, for legacy row-less entries, the entry) that
        carries the ``concept_id``. Returns a shallow copy with a deep-copied
        concept so downstream mutation can't poison the shared interned object —
        matching the old per-consumer deepcopy semantics. Legacy blocks that
        still embed their own concept are returned unchanged.
        """
        if "concept" in runtime_values:
            return runtime_values
        cid = holder.get("concept_id")
        if cid is None:
            return runtime_values
        concept = (self.cache_index.get("concepts") or {}).get(cid)
        if not isinstance(concept, dict):
            return runtime_values
        merged = dict(runtime_values)
        merged["concept"] = copy.deepcopy(concept)
        return merged

    @staticmethod
    def _migrate_intern_sourceless_concepts(cache_index: dict) -> int:
        """Collapse embedded per-variation concept dicts into a shared table.

        Idempotent: a cache already in interned form (rows carry ``concept_id``,
        no embedded ``concept``) yields 0 and is left untouched. Returns the
        number of embedded concept copies removed, for reporting. Run on load
        and by the offline migration script so existing multi-GB caches shrink
        on their next save without a rebuild.
        """
        entries = cache_index.get("entries")
        if not isinstance(entries, dict):
            return 0
        concepts = cache_index.setdefault("concepts", {})

        def intern(concept: dict) -> str:
            payload = json.dumps(concept, sort_keys=True, default=str)
            cid = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
            if cid not in concepts:
                concepts[cid] = concept
            return cid

        def strip(block: dict, holder: dict) -> int:
            removed = 0
            cid = holder.get("concept_id")
            for values in block.values():
                if isinstance(values, dict) and "concept" in values:
                    concept = values.pop("concept")
                    removed += 1
                    if cid is None and isinstance(concept, dict):
                        cid = intern(concept)
            if cid is not None:
                holder["concept_id"] = cid
            return removed

        removed = 0
        for entry in entries.values():
            if not isinstance(entry, dict):
                continue
            rows = entry.get("sourceless_rows")
            has_rows = isinstance(rows, dict) and bool(rows)
            if has_rows:
                for row in rows.values():
                    rv = row.get("runtime_values") if isinstance(row, dict) else None
                    if isinstance(rv, dict):
                        removed += strip(rv, row)
                # The entry-level block is a pure duplicate of the rows here.
                entry_block = entry.get("sourceless_runtime_values")
                if isinstance(entry_block, dict):
                    removed += sum(
                        1 for values in entry_block.values() if isinstance(values, dict) and "concept" in values
                    )
                entry.pop("sourceless_runtime_values", None)
            else:
                entry_block = entry.get("sourceless_runtime_values")
                if isinstance(entry_block, dict):
                    removed += strip(entry_block, entry)
        return removed

    def _set_entry_sourceless_runtime_values(self, entry: dict, variations: int, in_index: int) -> None:
        # Plain setter. Resolves runtime values (concept / prompts) via the
        # upstream pipeline, so callers MUST presence-gate before invoking this
        # (see _upgrade_sourceless_runtime_values) — resolving is the expensive
        # part. No value-equality check: comparing a JSON-round-tripped stored
        # value against a freshly-resolved live object is unreliable (type and
        # key-order coercion) and was the source of the never-converging
        # re-save loop.
        if not self.source_path_in_name:
            return

        # Always write the runtime_values slot, even when empty: its key
        # presence is the "this row has been resolved" marker the presence
        # predicate latches on. Popping it (the old behavior) made no-concept
        # rows look perpetually unstamped, so they were re-resolved every epoch.
        values_by_variation = self._sourceless_runtime_values_by_variation(variations, in_index)
        self._store_row_runtime_values(entry, in_index, values_by_variation)

    def _set_entry_sourceless_metadata(self, entry: dict, variations: int, in_index: int) -> None:
        # Plain setter; see _set_entry_sourceless_runtime_values for why there
        # is no repr-based change detection here.
        if not self.source_path_in_name:
            return

        metadata = self._sourceless_metadata(in_index, variations)
        entry["sourceless"] = metadata
        row = entry.setdefault("sourceless_rows", {}).setdefault(str(in_index), {})
        row["metadata"] = metadata

    def _sourceless_metadata_present(self, entry: dict, in_index: int) -> bool:
        # Per-(entry, in_index) presence predicate used to skip already-stamped
        # rows before the expensive resolve. A row is "present" when its
        # metadata is stamped AND its runtime_values slot has been written
        # (the key exists — its value may be an empty dict for a no-concept row;
        # we check key PRESENCE, not truthiness, so legitimately-empty runtime
        # values still latch and the pass converges).
        #
        # The entry-level "sourceless" block is only a fallback for truly legacy
        # single-row caches with no per-row structure: a multi-row entry (e.g.
        # the same source path duplicated across rows) must be judged per row,
        # otherwise row 0's entry-level block would mask an unstamped row 1.
        rows = entry.get("sourceless_rows")
        if isinstance(rows, dict) and rows:
            row = rows.get(str(in_index))
            return isinstance(row, dict) and bool(row.get("metadata")) and "runtime_values" in row
        return bool(entry.get("sourceless")) and "sourceless_runtime_values" in entry

    def _entry_has_any_sourceless_metadata(self, entry: dict) -> bool:
        # Index-agnostic presence check used by the sourceless-init guard: the
        # entry carries usable metadata via either the legacy entry-level block
        # or any stamped per-row metadata.
        if entry.get("sourceless"):
            return True
        rows = entry.get("sourceless_rows") or {}
        return any(isinstance(row, dict) and row.get("metadata") for row in rows.values())

    def _sourceless_index_metadata_ready(self, index_to_filepath: dict[int, str]) -> bool:
        if not self.source_path_in_name:
            return True

        entries = self.cache_index.get("entries", {})
        for in_index, filepath in index_to_filepath.items():
            entry = entries.get(filepath)
            if entry is None:
                continue
            if not self._sourceless_metadata_present(entry, in_index):
                return False
        return True

    def _pt_built_from(self, cached: dict, filepath: str | None) -> bool:
        """Whether a loaded .pt was built from ``filepath``. A .pt without a
        stamped image_path (or no path to compare) is taken at its word, as
        before; one stamped with another sample's path (a deduped twin) must
        not lend that sample its concept."""
        holder = cached.get("image_path") if isinstance(cached, dict) else None
        if not isinstance(holder, str) or not holder or not filepath:
            return True
        return self._same_source_path(holder, filepath)

    def _sourceless_runtime_values_for_row(
        self, entry: dict, in_index: int, variation: int, cached: dict, filepath: str | None = None
    ) -> dict | None:
        source_indices = getattr(self, "_sourceless_source_indices", None)
        source_index = source_indices[in_index] if source_indices and in_index < len(source_indices) else in_index
        row = (entry.get("sourceless_rows") or {}).get(str(source_index))
        if isinstance(row, dict):
            runtime_values = (row.get("runtime_values") or {}).get(str(variation))
            if isinstance(runtime_values, dict):
                # Interned form carries the concept by id on the row; legacy
                # form embeds it in the block — _attach handles both.
                return self._attach_interned_concept(runtime_values, row)

        runtime_values = (entry.get("sourceless_runtime_values") or {}).get(str(variation))
        if isinstance(runtime_values, dict):
            return self._attach_interned_concept(runtime_values, entry)

        # Last resort, the values baked into the .pt: only when the .pt is
        # this entry's own. A deduped twin's would hand this sample the
        # other pair's concept (its beta, loss weight, type), silently.
        if not self._pt_built_from(cached, filepath):
            raise RuntimeError(
                f"Sourceless training: cache entry for '{filepath}' carries no runtime values of its own and its "
                f"cache file was built from '{cached.get('image_path')}', so its concept cannot be served. "
                "Run a normal cache pass (skip cache validation OFF) to stamp the entry's own values."
            )
        runtime_values = cached.get("__sourceless_values")
        return runtime_values if isinstance(runtime_values, dict) else None

    def _stamp_sourceless_runtime_values(self, cache_data: dict, in_variation: int, in_index: int) -> bool:
        if not self.source_path_in_name:
            return False

        previous_values = repr(cache_data.get("__sourceless_values"))
        previous_legacy = tuple(
            repr(cache_data.get(key))
            for key in (
                "__concept_loss_weight",
                "__concept_type",
                "__concept_name",
                "__concept_path",
                "__concept_seed",
            )
        )

        for key in [k for k in cache_data if k.startswith("__concept_")]:
            del cache_data[key]

        runtime_values = self._sourceless_runtime_values(in_variation, in_index)
        if runtime_values:
            cache_data["__sourceless_values"] = runtime_values
        else:
            cache_data.pop("__sourceless_values", None)

        concept = runtime_values.get("concept")
        if isinstance(concept, dict):
            cache_data["__concept_loss_weight"] = concept.get("loss_weight", 1.0)
            cache_data["__concept_type"] = concept.get("type", "STANDARD")
            cache_data["__concept_name"] = concept.get("name", "")
            cache_data["__concept_path"] = concept.get("path", "")
            cache_data["__concept_seed"] = concept.get("seed", 0)

        current_legacy = tuple(
            repr(cache_data.get(key))
            for key in (
                "__concept_loss_weight",
                "__concept_type",
                "__concept_name",
                "__concept_path",
                "__concept_seed",
            )
        )
        return previous_values != repr(cache_data.get("__sourceless_values")) or previous_legacy != current_legacy

    def _save_pt_atomic(self, cache_data: dict, real_pt_path: str) -> None:
        real_tmp_path = real_pt_path + f".{os.getpid()}.{threading.get_ident()}.tmp"
        torch.save(cache_data, real_tmp_path)
        try:
            _replace_retrying(real_tmp_path, real_pt_path)
        except OSError:
            with contextlib.suppress(OSError):
                os.remove(real_tmp_path)
            raise

    def _sourceless_row_group_key(self, metadata: Any) -> str | None:
        """Group key recomputed from a stamped row's metadata.

        Mirrors _init_sourceless_groups_from_records' defaulting (missing
        group_values -> [""]) so a key computed here compares equal to the
        live group key of an unchanged concept.
        """
        if not isinstance(metadata, dict) or not metadata:
            return None
        try:
            return self._metadata_group_key(metadata)
        except TypeError:
            return None

    def _find_remappable_sourceless_row(self, entry: dict, group_key: str, variations: int) -> dict | None:
        """Find a stamped row whose content is still valid at a shifted index.

        Everything stamped on a row is a function of the (unchanged) source
        file and its concept, not of the row's global dataset position — the
        position only seeds RNG draws, and those are frozen at bake time
        anyway. So a row stamped under an old index is reusable for a new
        index as long as it belongs to the same group and was baked for the
        same variation count. Rows without group metadata (legacy stamps)
        never match and fall through to the slow resolve.
        """
        rows = entry.get("sourceless_rows")
        if not isinstance(rows, dict):
            return None
        # Without variations_in_name there is a single "" group whose key is
        # not derived from group values, so any stamped row of the file fits.
        match_any_group = self.variations_in_name is None
        for row in rows.values():
            if not isinstance(row, dict) or "runtime_values" not in row:
                continue
            metadata = row.get("metadata")
            if not isinstance(metadata, dict) or not metadata:
                continue
            if int(metadata.get("variations") or 1) != int(variations or 1):
                continue
            if not match_any_group and self._sourceless_row_group_key(metadata) != group_key:
                continue
            return row
        return None

    def _prune_orphaned_sourceless_rows(self, index_to_filepath: dict[int, str]) -> int:
        """Drop row stamps keyed by indices a file no longer occupies.

        Only entries referenced by the current dataset are touched: an entry
        for a file that left the dataset keeps its rows, since another config
        sharing this cache dir may still resolve through them (and a stale
        row costs only index bytes — it can also seed a future remap).
        """
        valid_keys_by_filepath: dict[str, set[str]] = {}
        for in_index, filepath in index_to_filepath.items():
            valid_keys_by_filepath.setdefault(filepath, set()).add(str(in_index))

        entries = self.cache_index.get("entries", {})
        pruned = 0
        for filepath, valid_keys in valid_keys_by_filepath.items():
            entry = entries.get(filepath)
            rows = entry.get("sourceless_rows") if isinstance(entry, dict) else None
            if not isinstance(rows, dict):
                continue
            for key in [k for k in rows if k not in valid_keys]:
                del rows[key]
                pruned += 1
        return pruned

    def _upgrade_sourceless_runtime_values(self, index_to_filepath: dict[int, str]) -> int:
        if not self.source_path_in_name:
            return 0

        upgraded = 0
        remapped = 0
        resolved_since_save = 0
        last_save = time.monotonic()
        entries = self.cache_index.get("entries", {})
        total = sum(len(indices) for indices in self.group_indices.values())
        with tqdm(total=total, smoothing=0.1, desc="updating sourceless metadata") as bar:
            for group_key, indices in self.group_indices.items():
                variations = int(self.group_variations.get(group_key, 1) or 1)
                for in_index in indices:
                    bar.update(1)
                    filepath = index_to_filepath.get(in_index)
                    if filepath is None:
                        continue
                    entry = entries.get(filepath)
                    if entry is None:
                        continue
                    # Presence-gate BEFORE resolving: _set_entry_* walk the
                    # upstream pipeline, and the per-module memo only caches a
                    # single index, so resolving every index re-runs the whole
                    # chain. Skipping already-stamped rows here keeps this
                    # pass cheap for anything already baked.
                    if self._sourceless_metadata_present(entry, in_index):
                        continue

                    # Rekey fast path: rows are stamped under their global
                    # dataset position, so adding/removing files anywhere
                    # shifts every later in_index and makes the whole dataset
                    # look unstamped. Reuse the stamp filed under the old
                    # index instead of re-walking the pipeline; only genuinely
                    # new files take the slow resolve below.
                    candidate = self._find_remappable_sourceless_row(entry, group_key, variations)
                    if candidate is not None:
                        metadata = dict(candidate["metadata"])
                        metadata["source_index"] = int(in_index)
                        entry["sourceless"] = metadata
                        row = entry.setdefault("sourceless_rows", {}).setdefault(str(in_index), {})
                        row["metadata"] = metadata
                        row["runtime_values"] = copy.deepcopy(candidate.get("runtime_values"))
                        if not self.store_sourceless_prompts:
                            self._blank_prompt_block(row["runtime_values"])
                        if "concept_id" in candidate:
                            row["concept_id"] = candidate["concept_id"]
                        else:
                            row.pop("concept_id", None)
                        upgraded += 1
                        remapped += 1
                        continue

                    self._set_entry_sourceless_metadata(entry, variations, in_index)
                    self._set_entry_sourceless_runtime_values(entry, variations, in_index)
                    upgraded += 1
                    resolved_since_save += 1
                    # Periodic flush so an interrupt doesn't discard a long
                    # slow-path run. Gated on count AND elapsed time because
                    # a save serializes the whole multi-MB index.
                    if resolved_since_save >= 1000 and time.monotonic() - last_save >= 120:
                        self._save_cache_index()
                        resolved_since_save = 0
                        last_save = time.monotonic()

        # Prune AFTER the loop: the rekey fast path reads old-index rows.
        pruned = self._prune_orphaned_sourceless_rows(index_to_filepath)
        if remapped or pruned:
            self._status(
                f"sourceless metadata: {remapped} rows rekeyed after index shift, "
                f"{upgraded - remapped} resolved, {pruned} stale rows pruned"
            )
        if upgraded or pruned:
            self._save_cache_index()
        return upgraded

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
        dedup_key: str | None = None,
        sidecar_state: tuple[dict[str, float], dict[str, str]] | None = None,
        source_size: int | None = None,
    ):
        # The file is named after what it holds: source bytes plus sidecars
        # (see _dedup_key), so pairs that share a chosen image but not a
        # rejected image or mask never write into one another's file.
        dedup_key = dedup_key or file_hash
        cache_file = self._make_cache_file(dedup_key, resolution)
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
                if isinstance(existing, dict) and not self._pt_built_from(existing, filepath):
                    # The file on disk under this entry's name was built from
                    # another sample. It is only ours to reuse when that
                    # sample is a twin of the same identity (same dedup key);
                    # a legacy file of a sidecar twin would otherwise hand us
                    # its rejected latent or mask again, and must never be
                    # overwritten while that twin still references it.
                    holder = self.cache_index["entries"].get(os.path.normpath(existing.get("image_path", "")))
                    if holder is None or self._entry_dedup_key(holder) != dedup_key:
                        raise RuntimeError(
                            f"Cache file {pt_name} belongs to '{existing.get('image_path')}' (a different "
                            f"pair/mask identity) and cannot be reused or overwritten for '{filepath}'. "
                            "Run a normal cache pass with skip cache validation OFF, which moves such legacy "
                            "files to their sidecar-aware names first."
                        )
                if isinstance(existing, dict):
                    existing_keys = {k for k in existing if not k.startswith("__")}
                    if required_schema.issubset(existing_keys):
                        if stored_crop_resolution is None:
                            stored_crop_resolution = existing.get("crop_resolution")
                        # Do NOT rewrite the .pt to (re)stamp sourceless runtime
                        # values: metadata lives in cache.json (the read path is
                        # index-first), so rewriting tensors here only churned
                        # the 200GB+ archive every epoch. Sourceless runtime
                        # values are stamped into the index in the entry
                        # create/refresh branch below.
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
            self._stamp_sourceless_runtime_values(cache_data, v, in_index)
            self._scrub_cache_payload(cache_data)

            real_pt_path = self._real_pt_path(cache_file, v)
            self._save_pt_atomic(cache_data, real_pt_path)

            if stored_crop_resolution is None:
                stored_crop_resolution = cache_data.get("crop_resolution")

            with self._index_lock:
                self._existing_pt_files.add(pt_name)
            self._register_content_pt(content_hash, pt_name)

        sidecar_mtimes, sidecar_hashes = sidecar_state or self._compute_sidecar_state(
            self._extra_paths_by_filepath.get(filepath, {})
        )
        runtime_values_by_variation = (
            self._sourceless_runtime_values_by_variation(variations, in_index) if self.source_path_in_name else {}
        )
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
        source_resolutions = {}
        for resolution_name in ["original_resolution", *self.additional_resolution_in_names]:
            try:
                res = self._get_previous_item(0, resolution_name, in_index)
                if res is not None and len(res) >= 2:
                    source_resolutions[resolution_name] = [int(res[-2]), int(res[-1])]
            except Exception:  # noqa: PERF203 - each optional resolution is independent
                continue
        original_resolution = source_resolutions.get("original_resolution")
        adaptive_max_resolution = None
        if self.adaptive_resolution and self.aspect_bucketing is not None:
            try:
                target_spec = self.aspect_bucketing._target_resolutions_for_item(0, in_index)
                adaptive_max_resolution = max(
                    target_resolutions_for_spec(
                        str(target_spec),
                        True,
                        getattr(self, "adaptive_resolution_ladder", None),
                    ),
                    default=None,
                )
            except Exception:
                adaptive_max_resolution = None
        key = self._resolution_key(resolution)
        with self._index_lock:
            entry = self.cache_index["entries"].get(filepath)
            if entry is None:
                entry = {
                    "filename": os.path.basename(filepath),
                    "hash": file_hash,
                    "dedup_key": dedup_key,
                    "mtime": mtime,
                    "modeltype": self.modeltype,
                    "variants": {},
                    "cache_version": CACHE_VERSION,
                    "sidecar_mtimes": sidecar_mtimes,
                    "sidecar_hashes": sidecar_hashes,
                }
                self._stamp_hashed_size(entry, file_hash, source_size)
                if original_resolution is not None:
                    entry["original_resolution"] = original_resolution
                if source_resolutions:
                    entry["source_resolutions"] = source_resolutions
                if adaptive_max_resolution is not None:
                    entry["adaptive_max_resolution"] = adaptive_max_resolution
                if self.adaptive_resolution:
                    entry["adaptive_resolution_applied"] = adaptive_max_resolution is not None
                entry["sourceless"] = self._sourceless_metadata(in_index, variations)
                row = entry.setdefault("sourceless_rows", {}).setdefault(str(in_index), {})
                row["metadata"] = entry["sourceless"]
                # Always set runtime_values (even {}): its key presence is the
                # "row resolved" marker the presence predicate latches on.
                self._store_row_runtime_values(entry, in_index, runtime_values_by_variation)
                self.cache_index["entries"][filepath] = entry
            else:
                # Refresh metadata; the source content may have changed and
                # been re-hashed since this entry was first created.
                entry["hash"] = file_hash
                self._stamp_hashed_size(entry, file_hash, source_size)
                entry["dedup_key"] = dedup_key
                entry.pop("shared_file_conflict", None)
                entry["mtime"] = mtime
                entry["modeltype"] = self.modeltype
                entry["cache_version"] = CACHE_VERSION
                entry["sidecar_mtimes"] = sidecar_mtimes
                entry["sidecar_hashes"] = sidecar_hashes
                if original_resolution is not None:
                    entry["original_resolution"] = original_resolution
                if source_resolutions:
                    entry["source_resolutions"] = source_resolutions
                if adaptive_max_resolution is not None:
                    entry["adaptive_max_resolution"] = adaptive_max_resolution
                if self.adaptive_resolution:
                    entry["adaptive_resolution_applied"] = adaptive_max_resolution is not None
                entry["sourceless"] = self._sourceless_metadata(in_index, variations)
                row = entry.setdefault("sourceless_rows", {}).setdefault(str(in_index), {})
                row["metadata"] = entry["sourceless"]
                # Always set runtime_values (even {}); see the create branch.
                self._store_row_runtime_values(entry, in_index, runtime_values_by_variation)
                if "variants" not in entry:
                    entry["variants"] = {}
            variant_record = {
                "cache_file": cache_file,
                "schema_keys": sorted(set(self.split_names) | set(self.aggregate_names)),
            }
            target_resolution = self._target_int_for_resolution_key(resolution)
            if target_resolution is not None:
                variant_record["target_resolution"] = target_resolution
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
            self._add_to_hash_index(dedup_key, filepath)

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
                # The sidecars are part of the dedup identity: re-key the entry
                # now that they are known, so no later sample dedups onto it
                # by the source hash alone.
                old_key = self._entry_dedup_key(entry)
                new_key = self._dedup_key(entry.get("hash", ""), hashes)
                if new_key != old_key:
                    self._remove_from_hash_index(old_key, filepath)
                    self._add_to_hash_index(new_key, filepath)
                entry["dedup_key"] = new_key
            self._index_dirty = True
            return True

        watched_paths = set(expected_paths.values())
        for path in watched_paths | set(stored_mtimes.keys()):
            current_mtime = self._source_mtimes.get(path)
            # A sidecar the bulk scan's listing proves absent needs no stat
            # that would only fail; every other miss is asked as before.
            if current_mtime is None and not self._scan_rules_out(path):
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
                current_hash = self._validation_hash(path)
            except OSError:
                return False
            if current_hash != stored_hash:
                return False

            # Same content, refresh stored mtime so the next fast-path
            # fingerprint pass matches.
            with self._index_lock:
                stored_mtimes[path] = current_mtime
            self._index_dirty = True

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
        # Entries whose file holds a different pair must never be served
        # silently; without the sources there is nothing to rebuild from.
        self._refuse_shared_file_victims(list(self.cache_index["entries"]), "Sourceless training")

        state = self._state.sourceless_cache_state
        entries = self.cache_index["entries"]

        if self._is_sourceless_anchor_cache() or "anchor_paths" not in state:
            records = []
            for filepath, entry in entries.items():
                records.extend(self._entry_sourceless_row_records(filepath, entry))
            if self.adaptive_resolution:
                eligible_records = []
                dropped_records = 0
                self._tier_fallback_entries = set()
                for record in records:
                    entry = entries.get(record["filepath"], {})
                    if self._training_eligible_variants(entry):
                        eligible_records.append(record)
                    else:
                        dropped_records += 1
                records = eligible_records
                if dropped_records:
                    self._status(
                        f"adaptive sourceless resolution: dropped {dropped_records} rows without a safe cached tier"
                    )
                if self._tier_fallback_entries:
                    print(
                        f"SmartDiskCache[{self._status_label()}]: upscale tolerance "
                        f"{self.upscale_tolerance:.1%} would move {len(self._tier_fallback_entries)} entries to a "
                        "tier that is not cached; they train at their best cached tier until the cache is "
                        "rebuilt with the sources present.",
                        flush=True,
                    )
            has_source_indices = any(record.get("source_index") != 1 << 60 for record in records)
            if has_source_indices:
                records = sorted(records, key=lambda record: (record.get("source_index", 1 << 60), record["filepath"]))
            else:
                records = sorted(records, key=lambda record: record["filepath"])
            self._sourceless_filepaths = [record["filepath"] for record in records]
            self._sourceless_source_indices = [
                record.get("source_index", index) for index, record in enumerate(records)
            ]
            self._init_sourceless_groups_from_records(records)
            state["anchor_paths"] = list(self._sourceless_filepaths)
            state["anchor_source_indices"] = list(self._sourceless_source_indices)
            state["anchor_entries"] = {path: entries[path] for path in self._sourceless_filepaths}
            state["group_variations"] = dict(self.group_variations)
            state["group_indices"] = {k: list(v) for k, v in self.group_indices.items()}
            state["group_full_indices"] = {k: list(v) for k, v in self.group_full_indices.items()}
            state["group_output_samples"] = dict(self.group_output_samples)
            state["group_balancing_strategy"] = dict(self.group_balancing_strategy)
            state["group_balancing"] = dict(self.group_balancing)
        else:
            anchor_paths = state["anchor_paths"]
            anchor_source_indices = state.get("anchor_source_indices") or list(range(len(anchor_paths)))
            anchor_entries = state.get("anchor_entries", {})
            self._sourceless_filepaths = [
                self._sourceless_counterpart_path(anchor_path, anchor_entries.get(anchor_path))
                for anchor_path in anchor_paths
            ]
            self._sourceless_source_indices = list(anchor_source_indices)
            missing = [path for path in self._sourceless_filepaths if path not in entries]
            if missing:
                raise RuntimeError(
                    "Sourceless training cache alignment failed: "
                    f"{len(missing)} required {self.source_path_in_name or 'source'} entries are missing. "
                    f"First missing: {missing[0]}"
                )
            self.group_variations = dict(state["group_variations"])
            self.group_indices = {k: list(v) for k, v in state["group_indices"].items()}
            self.group_full_indices = {k: list(v) for k, v in state["group_full_indices"].items()}
            self.group_output_samples = dict(state["group_output_samples"])
            self.group_balancing_strategy = dict(state["group_balancing_strategy"])
            self.group_balancing = dict(state["group_balancing"])

        # One listing of the cache dir answers "is this .pt there?" for every
        # entry. A name the listing lacks (missing, a symlink, a name whose
        # case differs on a case-insensitive filesystem) is still asked of the
        # filesystem, so the outcome is the one the per-entry isfile gave.
        listed_pt_files = self._scan_existing_pt_files()
        for fp in self._sourceless_filepaths:
            entry = self.cache_index["entries"][fp]
            # Hard guard: refuse to train sourceless on an entry with no baked
            # sourceless metadata. _entry_sourceless_row_records silently falls
            # back to source_index=1<<60 + empty metadata for such entries,
            # which would misalign/empty the run. This is the safety net for
            # caches built before the bake, or under trust / skip-validation
            # mode (which never stamps pre-existing entries).
            if not self._entry_has_any_sourceless_metadata(entry):
                raise RuntimeError(
                    f"Sourceless training: cache entry for '{fp}' is missing sourceless metadata. "
                    "The cache was built before sourceless metadata was baked, or under "
                    "trust/skip-validation mode which does not bake it. Run a normal cache pass "
                    "(only_cache, with source files present and 'skip cache validation' OFF) to bake "
                    "the metadata into cache.json, then retry sourceless training."
                )
            cache_file = self._any_variant_cache_file(entry)
            if cache_file is None:
                raise RuntimeError(f"Sourceless training: cache entry for '{fp}' has no variants. Rebuild your cache.")
            pt_path = self._real_pt_path(cache_file, 0)
            if f"{cache_file}_1.pt" not in listed_pt_files and not os.path.isfile(pt_path):
                raise RuntimeError(f"Sourceless training: cache file '{pt_path}' is missing. Rebuild your cache.")

        self.variations_initialized = True
        self._source_path_cache = dict(enumerate(self._sourceless_filepaths))

        self._aggregate_cache = {}
        if not self.aggregate_names:
            return None
        # Reported to __refresh_cache_sourceless: the variation whose aggregate
        # values this load took from index metadata alone, else None.
        variation = self.current_variation if self.current_variation >= 0 else 0
        return variation if self._load_aggregate_cache(variation) else None

    def __refresh_cache_sourceless(self, out_variation: int):
        index_only_variation = None
        if not self.variations_initialized:
            self._status(f"loading sourceless cache index from {self.cache_dir}")
            index_only_variation = self.__init_sourceless()
            self._status(
                f"sourceless cache ready "
                f"({len(self.cache_index.get('entries', {}))} indexed entries, "
                f"{sum(self.group_output_samples.values())} output samples)"
            )
        indices_before = dict(self.group_indices)
        self.__reshuffle_samples(out_variation)
        if index_only_variation == out_variation and self.group_indices == indices_before:
            # The init above has just loaded the aggregate values of this very
            # variation over these very rows, and took all of them from index
            # metadata: it read no cache file and stamped nothing. A second
            # load would recompute the same values from the same inputs, so
            # the first is kept. (Any other case reloads below, as before: a
            # load that had to read cache files stamps the index, which the
            # reload then serves from; a SAMPLES group changes the rows.)
            return
        self._aggregate_cache = {}
        if self.aggregate_names:
            self._status(f"preloading aggregate values for epoch {out_variation}")
            self._load_aggregate_cache(out_variation)

    def __refresh_cache(self, out_variation: int):
        if not self.variations_initialized:
            self._status("initializing cache groups")
            self.__init_variations()
        self.__reshuffle_samples(out_variation)

        self._index_dirty = False
        self._epoch_keys = None
        self._refresh_heap_collected = False

        # Reload the index from disk only when we don't hold one yet or the
        # file changed under us (external writer, e.g. gc from another run).
        # This process is otherwise the only writer, so re-parsing a
        # potentially huge cache.json on every epoch start was pure waste.
        if self.cache_index is None or self._cache_index_is_stale():
            self._status(f"loading cache index from {self.cache_dir}")
            self.cache_index = self._load_cache_index()
            self._index_disk_stat = self._snapshot_index_stat()
            if self._disk_index_may_hold_prompts:
                # Prompts were blanked on load (store_sourceless_prompts off).
                # Persist now: the fast-validation paths below return without
                # saving, which would leave the captions on disk indefinitely.
                self._save_cache_index()
        os.makedirs(self.cache_dir, exist_ok=True)
        self._source_path_cache = {}
        self._aggregate_cache = {}
        self._active_key_by_filepath = {}
        self._extra_paths_by_filepath = {}
        self._content_index_loaded = False
        self._content_reuse_memo = {}
        self._content_donor_meta = {}
        self._content_link_failures = 0
        self._blank_sentinel_memo = None
        self._before_cache_called = False
        indexed_entries = len(self.cache_index.get("entries", {}))
        sample_rows = sum(len(indices) for indices in self.group_indices.values())
        self._status(
            f"preparing epoch {out_variation} "
            f"({sample_rows} rows, {len(self.group_indices)} groups, {indexed_entries} indexed entries)"
        )

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
        #
        # Stamps are compared per role (see _role_signature): instances with
        # different configurations sharing this cache dir each keep their own,
        # so one no longer reads the other's stamp as drift on every refresh.
        role_sig = self._role_signature()
        role_stamps = self._role_stamps(role_sig)
        current_bucket_method = self._compute_bucket_method()
        stored_bucket_method = role_stamps["bucket_method"]
        bucket_drift = current_bucket_method is not None and stored_bucket_method != current_bucket_method

        # Schema drift: if split_names/aggregate_names changed since the cache
        # was built (e.g. masked_training was just enabled), the on-disk .pt
        # files won't contain the new keys. Also re-augment when the stored
        # schema_method doesn't match -- caches stamped by an older augment
        # version may have shape-inconsistent values that need fixing in
        # place. Run before any fast-path return so downstream readers
        # always find what they need.
        required_schema = sorted(set(self.split_names) | set(self.aggregate_names))
        stored_schema = role_stamps["schema"]
        stored_method = role_stamps["schema_method"]
        if self.cache_index.get("entries") and (stored_schema != required_schema or stored_method != SCHEMA_METHOD):
            self._status("checking cache schema drift")
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

            self._stamp_role(role_sig, schema=required_schema, schema_method=SCHEMA_METHOD)
            self.cache_index["last_role"] = role_sig
            self._index_dirty = True
            self._save_cache_index()

        # Resolve the source filepaths this call will deliver.
        self._status("resolving source paths for cache validation")
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
        self._status(f"resolved {len(required_filepaths)} unique source paths ({len(required_sidecar_paths)} sidecars)")

        # Cross-role schema check: when another role saved since this one last
        # looked, it may have rebuilt a shared file's variant with a narrower
        # schema. Index-only (schema_keys), so cheap; augments just the missing
        # names. Runs in every mode, trust included, before any skip path.
        if self.cache_index.get("entries") and self.cache_index.get("last_role") != role_sig:
            missing_names = self._missing_schema_names(required_filepaths)
            if missing_names:
                self._status(f"another cache role left {sorted(missing_names)} missing; augmenting")
                self._augment_cache_with_missing_names(missing_names)
                self._index_dirty = True
                self.cache_index["last_role"] = role_sig
                self._save_cache_index()
            else:
                # In memory only: persisted with the next real save. A reload
                # after another role's save brings back that role's marker.
                self.cache_index["last_role"] = role_sig

        skip_validation = self.trust_cache or os.environ.get("OT_SKIP_CACHE_VALIDATION") == "1"
        if not skip_validation:
            if not self._sourceless_index_metadata_ready(index_to_filepath):
                self._index_dirty = True
                self._status("updating sourceless cache index metadata")
                upgraded = self._upgrade_sourceless_runtime_values(index_to_filepath)
                if upgraded:
                    self._status(f"baked sourceless cache index metadata for {upgraded} entries")

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
            # Trust mode skips the validation that would rebuild them: refuse
            # entries whose file holds a different pair instead of serving it.
            self._refuse_shared_file_victims(required_filepaths, "Trust mode")

        # --- Variant-aware session skip (multi-resolution) ---
        # The filepath-granular skip above is bypassed under multi-resolution
        # bucketing because a filepath then maps to several resolution variants
        # and the required one rotates per epoch, so "filepath validated" can't
        # tell us "this epoch's variant was validated". Track (filepath, key)
        # pairs instead: recompute this epoch's required variants (deterministic,
        # no image decode — same out_variation-seeded roll validation/get_item
        # use) and skip when every one was already validated earlier this run. A
        # bucket a file hasn't been rolled into yet isn't in the set, so it falls
        # through to full validation and lazy-build. Both get_item and
        # _load_aggregate_cache re-derive the variant key per item under
        # multi-target, so the skip needs no _active_key_by_filepath priming —
        # only the guarantee (checked here) that every requested variant exists.
        if (
            not skip_validation
            and multi_target
            and not bucket_drift
            and required_filepaths
            and self.cache_index.get("entries")
            and self._session_validated_variants
        ):
            required_variants, resolvable = self._epoch_required_variants(index_to_filepath, out_variation)
            if resolvable and required_variants and required_variants.issubset(self._session_validated_variants):
                self._source_path_cache = dict(index_to_filepath)
                print(
                    f"SmartDiskCache: Skipped re-validation ({len(required_variants)} variants already validated this run)"
                )
                self._load_aggregate_cache(out_variation)
                return

        # Single os.scandir of the cache dir — used by fast and full paths.
        self._status("scanning existing cache tensors")
        self._existing_pt_files = self._scan_existing_pt_files()

        # Legacy files named by the source hash alone but built from an entry
        # with sidecars move to that entry's dedup-key name, so a twin without
        # sidecars (whose own name IS the source-hash name) can be built
        # into a file of its own instead of re-linking to the holder's.
        if not skip_validation:
            self._rename_legacy_sidecar_files()

        # Bucket-method drift recovery: aspect-math derives new variant keys
        # from cached resolutions and links any pre-existing .pt files. Pure
        # arithmetic, no image decode. Runs before validation so the per-index
        # loop sees the new variants registered.
        if bucket_drift:
            self._index_dirty = True
            self._status("cache bucket method changed; recovering variant links")
            self._drift_recovery_pass()

        # --- Fast validation path ---
        # Bypassed on bucket drift: the recovery pass may have queued new
        # variants that need rebuild, so we have to fall through. Also
        # bypassed under random multi-target aspect bucketing: the per-epoch
        # required variant rotates, and fast-validate only spot-checks
        # whichever variant is currently registered — it would skip the
        # lazy-build pass for this epoch's missing key.
        #
        # Adaptive resolution is multi-target but not random: every row always
        # takes its largest eligible tier. There the fast path runs, gated by
        # _adaptive_fast_path_guard proving every row's required variant is on
        # disk; if the guard declines, full validation runs exactly as before.
        adaptive_single_target = (
            multi_target
            and bool(getattr(self, "adaptive_resolution", False))
            and bool(getattr(self.aspect_bucketing, "adaptive_resolution", False))
        )
        dataset_digest = self._dataset_digest(required_filepaths) if not skip_validation else None
        if (
            not skip_validation
            and not bucket_drift
            and (not multi_target or adaptive_single_target)
            and self.cache_index.get("entries")
            # Entries flagged on load as sharing their file with a different
            # pair must reach _validate_entry, which rebuilds them.
            and not any(
                (self.cache_index["entries"].get(fp) or {}).get("shared_file_conflict") for fp in required_filepaths
            )
            and self._fast_validate(required_filepaths, self._role_validation_token(role_sig, dataset_digest))
        ):
            all_in_index = all(fp in self.cache_index["entries"] for fp in required_filepaths)
            if all_in_index and adaptive_single_target:
                verified, failure = self._adaptive_fast_path_guard(index_to_filepath, out_variation)
                if verified is not None:
                    epoch_keys, verified_variants, stamps_changed = verified
                    self._source_path_cache = dict(index_to_filepath)
                    for in_index, resolution in epoch_keys.items():
                        self._active_key_by_filepath[index_to_filepath[in_index]] = self._resolution_key(resolution)
                    n = len(self.cache_index["entries"])
                    checked = getattr(self, "_fast_validate_sample_size", "?")
                    print(
                        f"SmartDiskCache: Fast validation passed ({n} entries, {checked} spot-checked, "
                        f"{len(verified_variants)} adaptive variants verified)"
                    )
                    self._session_validated_filepaths.update(required_filepaths)
                    self._session_validated_variants.update(verified_variants)
                    if stamps_changed:
                        # The guard's resolve re-stamped adaptive fields (e.g. a
                        # concept's resolution cap changed); persist them.
                        self._index_dirty = True
                        self.cache_index["last_role"] = role_sig
                        self._save_cache_index()
                    self._epoch_keys = (out_variation, epoch_keys)
                    self._load_aggregate_cache(out_variation)
                    return
                print(f"SmartDiskCache: adaptive fast validation declined ({failure}); running full validation")
            elif all_in_index:
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
            self._status(f"statting {len(required_filepaths)} source files and {len(required_sidecar_paths)} sidecars")
            self._source_mtimes = self._bulk_stat_source_files(required_filepaths | required_sidecar_paths)
        else:
            self._source_mtimes = {}
            self._scanned_dir_keys = None

        # Clear fast-validation token during full validation. Preserved under
        # trust mode: a trust run neither verifies nor invalidates source
        # state, so whatever guarantee the token carried still holds — popping
        # it here would permanently knock later non-trust runs off the fast
        # path (the re-stamp at the end is also gated on full validation).
        previous_token = None
        if not skip_validation:
            self.cache_index.pop("last_validated", None)
            previous_token = self._pop_role_validation(role_sig, dataset_digest)

        files_built = 0
        files_skipped = 0
        files_failed = []
        header_mismatches = []
        self._status("validating cache entries")

        for group_key in self.group_variations:
            variations = self.group_variations[group_key]

            # _validate_entry is invariant in `in_variation` (source path uses
            # variation 0; resolution is variation-independent in the bucketing
            # pipeline; .pt existence iterates `range(variations)` internally).
            # Validate each in_index exactly once and dedupe across needed
            # variations afterwards.
            items_to_build_by_index: dict[int, tuple] = {}

            # Files this group's loop would hash one at a time (mtime changed,
            # content probably not) are hashed in parallel first; the loop
            # then finds their digests in _validation_hash.
            if not skip_validation:
                self._prehash_changed_mtimes(
                    index_to_filepath[in_index]
                    for in_index in self.group_indices[group_key]
                    if in_index in index_to_filepath
                )

            with tqdm(total=len(self.group_indices[group_key]), smoothing=0.1, desc="validating cache") as bar:
                for group_index, in_index in enumerate(self.group_indices[group_key]):
                    # Trust-mode early skip: avoid upstream pipeline calls
                    # (_get_source_path triggers crop_resolution upstream,
                    # which can do per-image I/O on slow cloud storage)
                    if skip_validation:
                        cached_fp = index_to_filepath.get(in_index)
                        if cached_fp is not None and cached_fp in self.cache_index["entries"]:
                            if self.adaptive_resolution and self.aspect_bucketing is not None:
                                trusted_entry = self.cache_index["entries"][cached_fp]
                                stamp_before = self._resolution_stamp_token(trusted_entry)
                                resolution = self._fast_resolution_string(trusted_entry, out_variation, in_index)
                                if self._resolution_stamp_token(trusted_entry) != stamp_before:
                                    self._index_dirty = True
                                if resolution is None:
                                    resolution = self._get_resolution_string(out_variation, in_index)
                                active_key = self._resolution_key(resolution)
                                self._active_key_by_filepath[cached_fp] = active_key
                                variant = (trusted_entry.get("variants") or {}).get(active_key)
                                if not variant or not variant.get("cache_file"):
                                    # Trust skips source staleness checks, not
                                    # creation of the tier this policy needs.
                                    items_to_build_by_index[in_index] = (
                                        cached_fp,
                                        group_key,
                                        out_variation,
                                        in_index,
                                        group_index,
                                        variations,
                                        resolution,
                                    )
                                    self._source_path_cache[in_index] = cached_fp
                                    bar.update(1)
                                    continue
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
                            stamp_before = self._resolution_stamp_token(entry)
                            resolution = self._fast_resolution_string(entry, out_variation, in_index)
                            if resolution is None:
                                resolution = self._get_resolution_string(out_variation, in_index)
                            if self._resolution_stamp_token(entry) != stamp_before:
                                self._index_dirty = True
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
                        if status in ("missing_pt", "incomplete_schema"):
                            # Only this variant is broken (_validate_entry's
                            # contract): drop and rebuild just the variant.
                            # The entry, its hash link, and its other
                            # variants are still valid — deleting the whole
                            # entry would orphan their .pt files and hand
                            # them to the next gc sweep.
                            with self._index_lock:
                                (entry.get("variants") or {}).pop(active_key, None)
                            self._index_dirty = True
                            resolution = None if active_key == NO_RESOLUTION_KEY else active_key
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
                        # Otherwise (content_changed / rebuild): the source
                        # itself changed, every variant is stale. Drop the
                        # entry.
                        with self._index_lock:
                            old_hash = self._entry_dedup_key(entry)
                            if old_hash:
                                self._remove_from_hash_index(old_hash, filepath)
                            if filepath in self.cache_index["entries"]:
                                del self.cache_index["entries"][filepath]
                            self._active_key_by_filepath.pop(filepath, None)
                        self._index_dirty = True

                    # Rebuild path: now we DO need the current resolution —
                    # but computing it opens the image through the full
                    # upstream chain (mask augmentation included), which at
                    # ~hundreds of ms per file would serialize hours of decode
                    # into this loop. Queue a placeholder instead; the
                    # parallel resolve pass below fills it in before the
                    # build phase needs it.
                    if self._needs_resolution():
                        resolution = RESOLUTION_PENDING
                    else:
                        resolution = None
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

            # The pre-hashed digests were for this group's loop only.
            self._prehashed = {}
            self._prehashed_sizes = {}

            # Parallel resolve pass: fill in the resolutions the loop above
            # deferred. Where the source headers give the exact decoded size
            # (_header_resolution_string), the key comes from those without a
            # decode; otherwise each resolve repeats exactly the upstream walk
            # the serial code used to run inline — same variation and index,
            # so the per-index seeded RNG lands on the same bucket — fanned
            # out across threads since the walk decodes the source image. No
            # ``_target_override`` is held here, matching the old inline
            # behaviour where each item picks its own epoch-rotated target.
            pending = [ii for ii, item in items_to_build_by_index.items() if item[6] is RESOLUTION_PENDING]
            # in_index -> header-read source resolutions, checked against the
            # decoded sizes the build stamps.
            header_estimates: dict[int, dict] = {}
            if pending:
                resolve_workers = max(4, min(16, os.cpu_count() or 8))
                with (
                    concurrent.futures.ThreadPoolExecutor(max_workers=resolve_workers) as resolve_pool,
                    tqdm(total=len(pending), smoothing=0.1, desc="resolving new items") as resolve_bar,
                ):
                    future_to_index = {
                        resolve_pool.submit(self._resolve_new_item, out_variation, ii): ii for ii in pending
                    }
                    for future in concurrent.futures.as_completed(future_to_index):
                        ii = future_to_index[future]
                        resolution, estimate = future.result()
                        if estimate is not None:
                            header_estimates[ii] = estimate
                        item = items_to_build_by_index[ii]
                        self._active_key_by_filepath[item[0]] = self._resolution_key(resolution)
                        items_to_build_by_index[ii] = item[:6] + (resolution,)
                        resolve_bar.update(1)

            items_to_build = list(items_to_build_by_index.values())

            if not items_to_build:
                continue

            # The scan's directory listings are only trusted while they are
            # fresh: a build can run for hours, so the groups validated after
            # it stat a missing sidecar themselves again.
            self._scanned_dir_keys = None

            # Builds, dedups and failures all change the index. Claim the
            # role marker before the first periodic flush can persist .pt
            # writes, so other roles re-check schemas even after a crash.
            self._index_dirty = True
            self.cache_index["last_role"] = role_sig

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

            # The index loaded above is long-lived and made of millions of
            # GC-tracked containers, and the loop below forces a full
            # collection every 250 items. Frozen for the duration of this
            # group's build, those collections skip it (see _gc_heap_frozen).
            with (
                self._frozen_loop_heap(),
                build_pool_ctx as build_executor,
                tqdm(total=len(unique_items), smoothing=0.1, desc="caching") as bar,
            ):

                def fn(
                    filepath,
                    group_key,
                    in_variation,
                    in_index,
                    group_index,
                    variations,
                    resolution,
                    current_device,
                    header_estimates=header_estimates,
                ):
                    if torch.cuda.is_available() and current_device is not None:
                        torch.cuda.set_device(current_device)

                    # Length of what gets hashed below, recorded on the entry
                    # next to the hash; None when it can't be vouched for.
                    source_size = None
                    try:
                        source_stat = os.stat(filepath)
                        mtime = source_stat.st_mtime
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
                        source_size = self._unchanged_size(filepath, source_stat)

                    # Sidecars (a pair's rejected image, masks) are hashed once
                    # here: they are part of the identity the file is named and
                    # deduped by, and the entry records their state.
                    sidecar_state = self._compute_sidecar_state(self._extra_paths_by_filepath.get(filepath, {}))
                    dedup_key = self._dedup_key(file_hash, sidecar_state[1])

                    if self._try_dedup(
                        filepath,
                        file_hash,
                        resolution,
                        mtime,
                        in_index,
                        variations,
                        dedup_key=dedup_key,
                        sidecar_state=sidecar_state,
                        source_size=source_size,
                    ):
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
                            dedup_key=dedup_key,
                            sidecar_state=sidecar_state,
                            source_size=source_size,
                        )
                    except Exception as e:
                        return filepath, f"build_failed: {e}"

                    estimate = header_estimates.get(in_index)
                    if estimate is not None:
                        stamped = (self.cache_index["entries"].get(filepath) or {}).get("source_resolutions") or {}
                        for name, header_resolution in estimate.items():
                            decoded = stamped.get(name)
                            if decoded is not None and list(decoded) != header_resolution:
                                header_mismatches.append((filepath, name, header_resolution, list(decoded)))

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

                # Submit each group bucket by bucket: the pool runs items in
                # submission order, so concurrent workers then encode images of
                # one shape, which a batching encoder (EncodeVAE's collector)
                # can stack. Nothing depends on build order: workers already
                # complete items in arbitrary order.
                for group_items in target_groups.values():
                    group_items.sort(key=lambda item: (item[6] or "", item[3]))

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

        self._scanned_dir_keys = None

        if header_mismatches:
            # The build keyed these by a header size the decode disagreed with.
            # Their .pt files and stamps hold the decoded truth, so the next
            # validation computes the right key and rebuilds that variant.
            examples = "; ".join(
                f"'{fp}' {name} header {h[0]}x{h[1]} vs decoded {d[-2]}x{d[-1]}"
                for fp, name, h, d in header_mismatches[:3]
            )
            print(
                f"Warning: {len(header_mismatches)} image header size(s) disagreed with the decoded size; "
                f"those entries are rebuilt on the next cache validation. {examples}"
            )

        if not skip_validation:
            # No unconditional re-stamp here: entries (re)built or deduped this
            # pass already stamped their index metadata in-context, and the
            # presence-gated pre-validation pass backfills any stragglers. A
            # second full sweep every epoch was pure churn.
            #
            # Per-watched-file fingerprint over this loader's rows only (the
            # token is keyed by role + dataset digest): ignored if any parent
            # dir is unreachable (returns None); a missing fingerprint just
            # means the next run pays for one extra full validation pass. A
            # clean pass reproducing the previous fingerprint keeps that token
            # instead of bumping its timestamp, so it doesn't force a save.
            entries = self.cache_index["entries"]
            scoped_entries = {fp: entries[fp] for fp in required_filepaths if fp in entries}
            fingerprint = self._compute_watched_fingerprints(scoped_entries) or {}
            if (
                previous_token is not None
                and not self._index_dirty
                and self._same_fingerprints(previous_token.get("watched_fingerprints"), fingerprint)
            ):
                token = previous_token
            else:
                token = {"last_validated": time.time(), "watched_fingerprints": fingerprint}
                self._index_dirty = True
            self._stamp_role_validation(role_sig, dataset_digest, token)
        final_stamps = {"schema": required_schema, "schema_method": SCHEMA_METHOD}
        if current_bucket_method is not None:
            final_stamps["bucket_method"] = current_bucket_method
        if self._stamp_role(role_sig, **final_stamps):
            self._index_dirty = True
        # Only write when this refresh changed something: a no-op rewrite of
        # a huge index costs seconds here and a full re-parse in every other
        # instance sharing the dir (its stale check sees the new stat).
        if self._index_dirty:
            self.cache_index["last_role"] = role_sig
            self._save_cache_index()
        self._save_content_index()

        # Mark every required filepath that ended up with a valid entry as
        # validated for this process so subsequent epochs can skip outright.
        entries = self.cache_index.get("entries", {})
        self._session_validated_filepaths.update(fp for fp in required_filepaths if fp in entries)

        # Variant-granular counterpart for the multi-resolution skip: record the
        # (filepath, key) this epoch built or confirmed (one per in_index, the
        # out_variation-rolled bucket). The `key in variants` guard means a
        # missing/failed variant can't poison a future skip, and `fp not in
        # failed_fps` drops files whose rebuild errored. Only under multi_target
        # — the filepath-level set above already covers single-resolution.
        if multi_target:
            required_variants, _ = self._epoch_required_variants(index_to_filepath, out_variation)
            failed_fps = {fp for fp, _ in files_failed}
            for fp, key in required_variants:
                if fp in failed_fps:
                    continue
                entry = entries.get(fp)
                if entry is not None and key in entry.get("variants", {}):
                    self._session_validated_variants.add((fp, key))

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

        # Free memoized donor payloads — they're only useful within one
        # build pass and would otherwise hold tensors for the whole epoch.
        self._content_reuse_memo = {}
        self._content_donor_meta = {}

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
        if getattr(self, "adaptive_resolution", False):
            try:
                target_spec = self.aspect_bucketing._target_resolutions_for_item(in_variation, in_index)
                targets = target_resolutions_for_spec(
                    str(target_spec),
                    True,
                    getattr(self, "adaptive_resolution_ladder", None),
                )
                entry["adaptive_resolution_applied"] = bool(targets)
                if targets:
                    entry["adaptive_max_resolution"] = max(targets)
            except Exception:
                pass
        # Prefer the stamped original_resolution (exact aspect) over the
        # variant-key approximation. Bucket aspect grids differ across
        # targets; recovering aspect from a 768-target variant key and
        # asking for the 512-target bucket can land on a different
        # bucket than the image's actual aspect would — that mismatch
        # is exactly what causes AspectBatchSorting to bucket items at
        # sort time differently from what get_item loads at fetch time.
        source_resolution_map = dict(entry.get("source_resolutions") or {})
        orig = source_resolution_map.get("original_resolution", entry.get("original_resolution"))
        aspect: float | None = None
        original_resolution: tuple[int, int] | None = None
        if isinstance(orig, (list, tuple)) and len(orig) >= 2:
            try:
                h = float(orig[-2])
                w = float(orig[-1])
                if w > 0:
                    aspect = h / w
                    original_resolution = (int(h), int(w))
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
                        original_resolution = (int(h), int(w))
                        entry["original_resolution"] = list(original_resolution)
                        source_resolution_map["original_resolution"] = list(original_resolution)
                        entry["source_resolutions"] = source_resolution_map
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

        additional_resolutions = []
        for resolution_name in getattr(self, "additional_resolution_in_names", []):
            additional = source_resolution_map.get(resolution_name)
            if not isinstance(additional, (list, tuple)) or len(additional) < 2:
                walk_variation = self.current_variation if self.current_variation >= 0 else in_variation
                try:
                    additional = self._get_previous_item(walk_variation, resolution_name, in_index)
                except Exception:
                    additional = None
                if isinstance(additional, (list, tuple)) and len(additional) >= 2:
                    additional = [int(additional[-2]), int(additional[-1])]
                    source_resolution_map[resolution_name] = additional
                    entry["source_resolutions"] = source_resolution_map
            if not isinstance(additional, (list, tuple)) or len(additional) < 2:
                if getattr(self, "adaptive_resolution", False):
                    return None
                continue
            additional_resolutions.append((int(additional[-2]), int(additional[-1])))
        try:
            if original_resolution is not None:
                resolution_selector = getattr(self.aspect_bucketing, "variant_key_from_resolutions", None)
                if resolution_selector is not None:
                    return resolution_selector(in_variation, in_index, original_resolution, additional_resolutions)
                resolution_selector = getattr(self.aspect_bucketing, "variant_key_from_resolution", None)
                if resolution_selector is not None:
                    return resolution_selector(in_variation, in_index, original_resolution)
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
        if self.sourceless:
            # Rotate among cached buckets (see _sourceless_variant). get_item
            # picks the identical variant, so the synthesized crop_resolution
            # matches the latent it loads — and we skip the torch.load per row
            # that made the sourceless aggregate load ~2000x slower than sourced.
            variant = self._sourceless_variant(entry, in_index)
            if variant is None:
                return None
        elif self.aspect_bucketing is None and not self.resolution_from_upstream:
            # Non-sourceless caches with no bucketing: get_item serves the
            # active (any-)variant; synthesize from that same variant.
            variant = self._variant_for_cache_file(entry, self._active_cache_file(filepath, entry))
            if variant is None:
                return None
        else:
            # Keys the adaptive fast-path guard already proved this refresh.
            resolution = self._epoch_key(in_variation, in_index)
            if resolution is None:
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
            variant = (entry.get("variants") or {}).get(self._resolution_key(resolution))
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

    def _load_aggregate_cache(self, out_variation: int) -> bool:
        """Fill ``_aggregate_cache`` for ``out_variation``.

        Returns True when every row was synthesized from index metadata: no
        cache file was read, so no variant was lazily stamped from one and the
        index was not saved.
        """
        if not self.aggregate_names:
            return False

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
        self._status(f"loading aggregate cache for {len(load_items)} rows")

        if fast_path:
            slow_items = []
            # Progress bar even on the fast path: this loop is O(N) dict
            # ops plus per-item RNG/bucket math via AspectBucketing, but
            # on a 22k-sample dataset it's still a few seconds of work
            # and worth showing so it doesn't look like a freeze after
            # the 'Cached X/Y files' line.
            with tqdm(total=len(load_items), smoothing=0.1, desc="loading aggregate cache") as bar:
                for filepath, cache_entry, variation, in_variation, in_index in load_items:
                    # Rotation resolved with the epoch variation so the
                    # synthesized crop_resolution matches the variant
                    # get_item's split fetch loads this epoch.
                    agg_data = self._try_synthesize_aggregate(filepath, cache_entry, out_variation, in_index)
                    if agg_data is None:
                        slow_items.append((filepath, cache_entry, variation, in_variation, in_index))
                        bar.update(1)
                        continue
                    # Key by (filepath, variation, in_index) — see the
                    # slow-path write below for the rationale.
                    self._aggregate_cache[(filepath, variation, in_index)] = agg_data
                    bar.update(1)
            if not slow_items:
                self._status("aggregate cache ready")
                return True
            load_items = slow_items
            self._status(f"{len(load_items)} aggregate rows need tensor reads")

        stamped_any = False
        with tqdm(total=len(load_items), smoothing=0.1, desc="loading aggregate cache") as bar:
            for filepath, cache_entry, variation, _in_variation, in_index in load_items:
                cache_file = None
                variant_for_stamp = None
                if per_item_key:
                    # Fast resolve via the stamped original_resolution.
                    # Falls back to the upstream image-decode walk only
                    # when no aspect can be recovered. Rotation is seeded
                    # on the epoch variation, matching validation/get_item.
                    resolution = self._epoch_key(out_variation, in_index)
                    if resolution is None:
                        resolution = self._fast_resolution_string(cache_entry, out_variation, in_index)
                    if resolution is None:
                        resolution = self._get_resolution_string(out_variation, in_index)
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
                    source_name = getattr(self, "source_path_in_name", None)
                    if (
                        source_name
                        and source_name in agg_data
                        and not self._same_source_path(agg_data[source_name], filepath)
                    ):
                        # A deduped .pt stamps its builder's path; serve this
                        # entry's own, as _try_synthesize_aggregate does.
                        agg_data[source_name] = filepath
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
        self._status("aggregate cache ready")
        return False

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
        # Prefer a variant whose recorded schema covers every required name.
        # Loaders with different schemas share a cache dir (SFT + DPO); a
        # template taken from the other loader's narrower .pt produced a
        # sentinel missing this loader's names, so it was rewritten on every
        # start. Bounded attempts, then the original any-entry fallback.
        attempts = 0
        for entry in self.cache_index.get("entries", {}).values():
            if template is not None or attempts >= 8:
                break
            if entry.get("modeltype") != self.modeltype:
                continue
            for variant in (entry.get("variants") or {}).values():
                cache_file = variant.get("cache_file")
                schema_keys = variant.get("schema_keys")
                if not cache_file or schema_keys is None or not required.issubset(schema_keys):
                    continue
                pt = self._real_pt_path(cache_file, 0)
                if not os.path.isfile(pt):
                    continue
                attempts += 1
                try:
                    candidate = torch.load(pt, weights_only=False, map_location="cpu")
                except Exception:
                    break
                if required.issubset(k for k in candidate if not k.startswith("__")):
                    template = candidate
                break
        for entry in self.cache_index.get("entries", {}).values():
            if template is not None:
                break
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
            sentinel_changed = self.cache_index.get("blank_sentinel") != out_name
            self.cache_index["blank_sentinel"] = out_name
        self._blank_sentinel_memo = None
        # The .pt is already replaced atomically; the index only needs a write
        # when the recorded name changed.
        if sentinel_changed:
            self._save_cache_index()

    def _load_blank_sentinel(self) -> dict | None:
        # Memoized: this fires per cache-miss get_item, i.e. per sample per
        # epoch for every failed-to-cache file. Invalidated on refresh and
        # whenever the sentinel is rewritten.
        if self._blank_sentinel_memo is not None:
            return self._blank_sentinel_memo
        if not self.cache_index:
            return None
        sentinel_name = self.cache_index.get("blank_sentinel")
        if not sentinel_name:
            return None
        sentinel_path = os.path.join(self._real_cache_dir, sentinel_name)
        if not os.path.isfile(sentinel_path):
            return None
        try:
            sentinel = torch.load(sentinel_path, weights_only=False, map_location=self.pipeline.device)
        except Exception:
            return None
        self._blank_sentinel_memo = sentinel
        return sentinel

    def get_item(self, index: int, requested_name: str = None) -> dict:
        result = self.__get_input_index(self.current_variation, index)
        if result is None:
            return {requested_name: self._get_previous_item(self.current_variation, requested_name, index)}

        group_key, in_variation, group_index, in_index = result

        filepath = self._sourceless_filepaths[in_index] if self.sourceless else self._source_path_cache.get(in_index)

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

                # Runtime-value-only request (concept / prompt, e.g. from
                # VariationSorting's grouping pass at epoch start). These live in
                # the index runtime_values, so serve them WITHOUT torch.load —
                # the sourced pipeline resolves concept/prompt from a lightweight
                # upstream module too, never from the cached .pt. Loading the
                # whole tensor .pt here just to return a concept dict meant
                # reading the entire cache off disk to group samples.
                if (
                    self.sourceless
                    and requested_name is not None
                    and requested_name not in self.split_names
                    and requested_name not in self.aggregate_names
                ):
                    runtime_values = self._sourceless_runtime_values_for_row(cache_entry, in_index, variation, {})
                    if isinstance(runtime_values, dict) and requested_name in runtime_values:
                        return {requested_name: runtime_values[requested_name]}

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
                if self.sourceless:
                    # Rotate among cached resolution buckets per (epoch,
                    # in_index). _try_synthesize_aggregate picks the identical
                    # variant, so the crop_resolution it cached matches the
                    # latent loaded here (no AspectBatchSorting/collate mismatch).
                    sl_variant = self._sourceless_variant(cache_entry, in_index)
                    cache_file = sl_variant.get("cache_file") if sl_variant else None
                elif self.resolution_from_upstream or self.aspect_bucketing is not None:
                    # Fast resolve via the stamped original_resolution
                    # before falling back to the upstream image-decode walk.
                    # Per-batch this saves the ~250 ms decode that
                    # ``_get_resolution_string`` would trigger for every
                    # item via CalcAspect → LoadImage.
                    #
                    # The rotation is seeded on the *epoch* variation — the
                    # same variation ``__refresh_cache`` used to validate and
                    # lazy-build this epoch's variants. Using ``in_variation``
                    # here (always 0 when variations=1) froze serving to the
                    # epoch-0 bucket forever while validation kept building
                    # per-epoch variants that were never loaded.
                    rotation_variation = self.current_variation if self.current_variation >= 0 else in_variation
                    resolution = self._fast_resolution_string(cache_entry, rotation_variation, in_index)
                    if resolution is None:
                        resolution = self._get_resolution_string(rotation_variation, in_index)
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
                source_name = getattr(self, "source_path_in_name", None)
                if source_name and source_name in item and not self._same_source_path(item[source_name], filepath):
                    # A deduped .pt stamps the path of the sample it was built
                    # from; the served path is this entry's own identity (as
                    # the aggregate fast path already serves it).
                    item[source_name] = filepath
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
                if self.sourceless:
                    runtime_values = self._sourceless_runtime_values_for_row(
                        cache_entry, in_index, variation, cached, filepath
                    )
                    if isinstance(runtime_values, dict):
                        for name, value in runtime_values.items():
                            item.setdefault(name, value)
                if self.sourceless and "__concept_loss_weight" in cached and self._pt_built_from(cached, filepath):
                    item.setdefault(
                        "concept",
                        {
                            "loss_weight": cached["__concept_loss_weight"],
                            "type": cached.get("__concept_type", "STANDARD"),
                            "name": cached.get("__concept_name", ""),
                            "path": cached.get("__concept_path", ""),
                            "seed": cached.get("__concept_seed", 0),
                        },
                    )
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

        with open(cache_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        SmartDiskCache._migrate_legacy_index_in_place(index)
        entries = index.get("entries", {})

        dead_filepaths = [fp for fp in entries if not os.path.isfile(fp)]

        # One scandir, then pure set lookups. The walk must not stop at the
        # first missing variation suffix — a gap (e.g. _1.pt lost out-of-band
        # while _2.pt/_3.pt survive) would otherwise turn the still-indexed
        # higher variations into "orphans".
        existing_pt_sizes = {
            e.name: e.stat().st_size for e in os.scandir(cache_dir) if e.name.endswith(".pt") and e.is_file()
        }

        referenced_names = set()
        for fp, entry in entries.items():
            if fp in dead_filepaths:
                continue
            for cf in SmartDiskCache._iter_variant_cache_files(entry):
                for v in range(1, 100):
                    name = f"{cf}_{v}.pt"
                    if name in existing_pt_sizes:
                        referenced_names.add(name)

        # Top-level non-entry .pt files (currently just blank_sentinel.pt)
        # are intentional artifacts, not orphans.
        sentinel_name = index.get("blank_sentinel")
        if sentinel_name:
            referenced_names.add(sentinel_name)

        orphan_count = 0
        orphan_bytes = 0
        # Names that are hard links of one file (content reuse links a
        # duplicate caption to its donor) share their bytes: removing orphans
        # frees them only when every name of the file is an orphan.
        # (device, inode) -> [orphan names seen, link count, size]
        linked: dict[tuple[int, int], list[int]] = {}

        for name, size in existing_pt_sizes.items():
            if name not in referenced_names:
                orphan_count += 1
                try:
                    st = os.stat(os.path.join(cache_dir, name))
                except OSError:
                    st = None
                if st is None or st.st_nlink <= 1 or not st.st_ino:
                    orphan_bytes += size
                else:
                    linked.setdefault((st.st_dev, st.st_ino), [0, st.st_nlink, size])[0] += 1

        orphan_bytes += sum(size for names, links, size in linked.values() if names >= links)

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

        with open(cache_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        SmartDiskCache._migrate_legacy_index_in_place(index)
        SmartDiskCache._migrate_dedup_keys(index)
        entries = index.get("entries", {})
        hash_index = index.get("hash_index", {})

        dead_filepaths = [fp for fp in entries if not os.path.isfile(fp)]

        # One scandir, then pure set lookups. None of the variation walks may
        # stop at the first missing suffix — a gap (e.g. _1.pt lost
        # out-of-band while _2.pt/_3.pt survive) would otherwise leave the
        # higher variations out of the referenced set and delete them.
        existing_pt_names = {e.name for e in os.scandir(cache_dir) if e.name.endswith(".pt") and e.is_file()}

        for fp in dead_filepaths:
            entry = entries.pop(fp)
            file_hash = SmartDiskCache._entry_dedup_key(entry)
            if file_hash in hash_index:
                paths = hash_index[file_hash]
                if fp in paths:
                    paths.remove(fp)
                if not paths:
                    del hash_index[file_hash]
                    for cf in SmartDiskCache._iter_variant_cache_files(entry):
                        for v in range(1, 100):
                            name = f"{cf}_{v}.pt"
                            if name in existing_pt_names:
                                os.remove(os.path.join(cache_dir, name))
                                existing_pt_names.discard(name)

        referenced_names = set()
        for entry in entries.values():
            for cf in SmartDiskCache._iter_variant_cache_files(entry):
                for v in range(1, 100):
                    name = f"{cf}_{v}.pt"
                    if name in existing_pt_names:
                        referenced_names.add(name)

        sentinel_name = index.get("blank_sentinel")
        if sentinel_name:
            referenced_names.add(sentinel_name)

        for name in existing_pt_names:
            if name not in referenced_names:
                os.remove(os.path.join(cache_dir, name))

        tmp_path = cache_path + ".tmp"
        bak_path = cache_path + ".bak"
        with open(tmp_path, "wb") as f:
            f.write(json.dumps(index, separators=(",", ":")).encode("utf-8"))
        if os.path.exists(cache_path):
            shutil.copy2(cache_path, bak_path)
        os.replace(tmp_path, cache_path)
