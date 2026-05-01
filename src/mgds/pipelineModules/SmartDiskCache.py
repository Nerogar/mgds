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

CACHE_VERSION = 2

# Bumped whenever ``_augment_cache_with_missing_names`` changes behaviorally so
# caches stamped by an older version get re-augmented on the new path.
SCHEMA_METHOD = 'shape_v1'


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
                return data
            except (json.JSONDecodeError, OSError):
                pass

        if os.path.exists(tmp_path):
            try:
                with open(tmp_path, 'r') as f:
                    data = json.load(f)
                os.replace(tmp_path, cache_path)
                return data
            except (json.JSONDecodeError, OSError):
                pass

        if os.path.exists(bak_path):
            try:
                with open(bak_path, 'r') as f:
                    data = json.load(f)
                os.replace(bak_path, cache_path)
                return data
            except (json.JSONDecodeError, OSError):
                pass

        return {"version": CACHE_VERSION, "entries": {}, "hash_index": {}}

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

    def _pt_path(self, cache_file: str, variation: int) -> str:
        return os.path.join(self.cache_dir, f"{cache_file}_{variation + 1}.pt")

    def _real_pt_path(self, cache_file: str, variation: int) -> str:
        return os.path.join(self._real_cache_dir, f"{cache_file}_{variation + 1}.pt")

    def _validate_entry(self, filepath: str, entry: dict, resolution: str | None, variations: int) -> str:
        if entry['modeltype'] != self.modeltype:
            raise RuntimeError(
                f"Cache modeltype mismatch for '{filepath}': "
                f"cached as '{entry['modeltype']}', current model is '{self.modeltype}'. "
                f"Delete the cache directory or use a separate cache_dir for this model type."
            )

        if resolution and entry.get('resolution') and resolution != entry['resolution']:
            return 'resolution_changed'

        try:
            current_mtime = os.path.getmtime(filepath)
        except OSError:
            return 'rebuild'

        if current_mtime == entry['mtime']:
            for v in range(variations):
                if not os.path.isfile(self._real_pt_path(entry['cache_file'], v)):
                    return 'missing_pt'
            return 'valid'

        file_hash = self._hash_file(filepath)
        if file_hash != entry['hash']:
            return 'content_changed'

        entry['mtime'] = current_mtime

        for v in range(variations):
            if not os.path.isfile(self._real_pt_path(entry['cache_file'], v)):
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
            cache_file = entry.get('cache_file')
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
            cache_file = entry.get('cache_file')
            if not cache_file:
                return filepath, 'no_cache_file'

            wrote_anything = False
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
        """Fast validation: directory mtime check + random spot check.

        Skips the expensive per-file validation loop when nothing has changed.
        Returns True if cache appears valid.
        """
        last_validated = self.cache_index.get('last_validated')
        if last_validated is None:
            return False

        entries = self.cache_index.get('entries', {})
        if not entries:
            return False

        # Check if any source directory was modified after last validation
        # (catches added/removed/renamed files)
        parent_dirs = set()
        for filepath in entries:
            parent_dirs.add(os.path.dirname(filepath))

        for d in parent_dirs:
            try:
                dir_mtime = os.path.getmtime(d)
            except OSError:
                return False
            if dir_mtime > last_validated:
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
            cache_file = entry.get('cache_file')
            if cache_file and not os.path.isfile(self._real_pt_path(cache_file, 0)):
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
        with self._index_lock:
            if file_hash not in self.cache_index['hash_index']:
                return False

            for existing_path in self.cache_index['hash_index'][file_hash]:
                existing_entry = self.cache_index['entries'].get(existing_path)
                if existing_entry is None:
                    continue
                if existing_entry['modeltype'] != self.modeltype:
                    continue
                if resolution and existing_entry.get('resolution') != resolution:
                    continue
                cache_file = existing_entry['cache_file']

                self.cache_index['entries'][filepath] = {
                    'filename': os.path.basename(filepath),
                    'hash': file_hash,
                    'mtime': mtime,
                    'modeltype': self.modeltype,
                    'resolution': resolution,
                    'cache_file': cache_file,
                    'cache_version': CACHE_VERSION,
                }
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
            real_pt = self._real_pt_path(cache_file, v)
            if os.path.isfile(real_pt):
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
            self.cache_index['entries'][filepath] = {
                'filename': os.path.basename(filepath),
                'hash': file_hash,
                'mtime': mtime,
                'modeltype': self.modeltype,
                'resolution': resolution,
                'cache_file': cache_file,
                'cache_version': CACHE_VERSION,
            }
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
            pt_path = self._real_pt_path(entry['cache_file'], 0)
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
                    real_path = self._real_pt_path(entry['cache_file'], 0)
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
        if (required_filepaths
                and self.cache_index.get('entries')
                and required_filepaths.issubset(self._session_validated_filepaths)
                and all(fp in self.cache_index['entries'] for fp in required_filepaths)):
            self._source_path_cache = dict(index_to_filepath)
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

        # --- Fast validation path ---
        if not skip_validation and self.cache_index.get('entries') and self._fast_validate():
            all_in_index = all(
                fp in self.cache_index['entries'] for fp in required_filepaths
            )
            if all_in_index:
                self._source_path_cache = dict(index_to_filepath)
                n = len(self.cache_index['entries'])
                checked = getattr(self, '_fast_validate_sample_size', '?')
                print(f"SmartDiskCache: Fast validation passed ({n} entries, {checked} spot-checked)")
                self._session_validated_filepaths.update(required_filepaths)
                self._load_aggregate_cache(out_variation)
                return

            # Index mismatch — fall through to full validation
            self._source_path_cache = {}

        # Clear fast-validation token during full validation
        self.cache_index.pop('last_validated', None)

        before_cache_fun_called = False
        files_built = 0
        files_skipped = 0
        files_failed = []

        for group_key in self.group_variations:
            start_index = self.group_output_samples[group_key] * out_variation
            end_index = self.group_output_samples[group_key] * (out_variation + 1) - 1

            start_variation = start_index // len(self.group_indices[group_key])
            end_variation = end_index // len(self.group_indices[group_key])

            variations = self.group_variations[group_key]
            needed_variations = [(x % variations) for x in range(start_variation, end_variation + 1, 1)]

            items_to_build = []
            validate_total = len(needed_variations) * len(self.group_indices[group_key])

            with tqdm(total=validate_total, smoothing=0.1, desc='validating cache') as bar:
                for in_variation in needed_variations:
                    for group_index, in_index in enumerate(self.group_indices[group_key]):
                        # Trust-mode early skip: avoid upstream pipeline calls
                        # (_get_source_path triggers crop_resolution upstream,
                        # which can do per-image I/O on slow cloud storage)
                        if skip_validation:
                            cached_fp = index_to_filepath.get(in_index)
                            if cached_fp is not None and cached_fp in self.cache_index['entries']:
                                if in_index not in self._source_path_cache:
                                    self._source_path_cache[in_index] = cached_fp
                                files_skipped += 1
                                bar.update(1)
                                continue

                        filepath = self._get_source_path(in_variation, in_index)
                        if filepath is None:
                            bar.update(1)
                            continue

                        filepath = os.path.normpath(filepath)
                        if in_index not in self._source_path_cache:
                            self._source_path_cache[in_index] = filepath
                        resolution = self._get_resolution_string(in_variation, in_index)

                        entry = self.cache_index['entries'].get(filepath)
                        if entry is not None:
                            if skip_validation:
                                files_skipped += 1
                                bar.update(1)
                                continue
                            status = self._validate_entry(filepath, entry, resolution, variations)
                            if status == 'valid':
                                files_skipped += 1
                                bar.update(1)
                                continue
                            with self._index_lock:
                                old_hash = entry.get('hash')
                                if old_hash:
                                    self._remove_from_hash_index(old_hash, filepath)
                                del self.cache_index['entries'][filepath]

                        items_to_build.append((filepath, group_key, in_variation, in_index, group_index, variations, resolution))
                        bar.update(1)

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

                    if filepath not in self.cache_index['entries']:
                        if self._try_dedup(filepath, file_hash, resolution, mtime):
                            entry = self.cache_index['entries'][filepath]
                            all_present = all(
                                os.path.isfile(self._real_pt_path(entry['cache_file'], v))
                                for v in range(variations)
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
        self.cache_index['schema'] = required_schema
        self.cache_index['schema_method'] = SCHEMA_METHOD
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
                real_path = self._real_pt_path(cache_entry['cache_file'], variation)
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
            pt = self._real_pt_path(entry['cache_file'], 0)
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

                real_cache_path = self._real_pt_path(cache_entry['cache_file'], variation)

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

        entries = index.get('entries', {})

        dead_filepaths = [fp for fp in entries if not os.path.isfile(fp)]

        referenced_cache_files = set()
        for fp, entry in entries.items():
            if fp not in dead_filepaths:
                cf = entry.get('cache_file', '')
                for v in range(1, 100):
                    pt = os.path.join(cache_dir, f"{cf}_{v}.pt")
                    if os.path.isfile(pt):
                        referenced_cache_files.add(os.path.normpath(pt))
                    else:
                        break

        orphan_count = 0
        orphan_bytes = 0

        for entry in os.scandir(cache_dir):
            if entry.name.endswith('.pt') and entry.is_file():
                if os.path.normpath(entry.path) not in referenced_cache_files:
                    orphan_count += 1
                    orphan_bytes += entry.stat().st_size

        return {'orphan_count': orphan_count, 'orphan_bytes': orphan_bytes}

    @staticmethod
    def gc_clean(cache_dir: str):
        cache_path = os.path.join(cache_dir, 'cache.json')
        if not os.path.isfile(cache_path):
            return

        with open(cache_path, 'r') as f:
            index = json.load(f)

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
                    cf = entry.get('cache_file', '')
                    for v in range(1, 100):
                        pt = os.path.join(cache_dir, f"{cf}_{v}.pt")
                        if os.path.isfile(pt):
                            os.remove(pt)
                        else:
                            break

        referenced_cache_files = set()
        for entry in entries.values():
            cf = entry.get('cache_file', '')
            for v in range(1, 100):
                pt = os.path.join(cache_dir, f"{cf}_{v}.pt")
                if os.path.isfile(pt):
                    referenced_cache_files.add(os.path.normpath(pt))
                else:
                    break

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
