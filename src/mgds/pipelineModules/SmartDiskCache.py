import concurrent
import hashlib
import json
import math
import os
import shutil
import threading
from typing import Any, Callable

import torch
import xxhash
from tqdm import tqdm

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.SingleVariationRandomAccessPipelineModule import SingleVariationRandomAccessPipelineModule

CACHE_VERSION = 1


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
            modeltype: str = "",
            source_path_in_name: str | None = None,
            sourceless: bool = False,
    ):
        super(SmartDiskCache, self).__init__()
        self.cache_dir = cache_dir

        self.split_names = [] if split_names is None else split_names
        self.aggregate_names = [] if aggregate_names is None else aggregate_names

        self.variations_in_name = variations_in_name
        self.balancing_in_name = balancing_in_name
        self.balancing_strategy_in_name = balancing_strategy_in_name
        self.variations_group_in_names = \
            [variations_group_in_name] if isinstance(variations_group_in_name, str) else variations_group_in_name

        self.group_enabled_in_name = group_enabled_in_name

        self.before_cache_fun = (lambda: None) if before_cache_fun is None else before_cache_fun

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
        return self.split_names + self.aggregate_names

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

        if os.path.exists(cache_path):
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        if os.path.exists(tmp_path):
            try:
                with open(tmp_path, 'r') as f:
                    data = json.load(f)
                shutil.move(tmp_path, cache_path)
                return data
            except (json.JSONDecodeError, OSError):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        return {"version": CACHE_VERSION, "entries": {}, "hash_index": {}}

    def _save_cache_index(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = self._get_cache_json_path()
        tmp_path = cache_path + '.tmp'
        bak_path = cache_path + '.bak'

        with self._index_lock:
            with open(tmp_path, 'w') as f:
                json.dump(self.cache_index, f, indent=2)

        if os.path.exists(cache_path):
            try:
                shutil.copy2(cache_path, bak_path)
            except OSError:
                pass

        shutil.move(tmp_path, cache_path)

    def _flush_cache_index(self, count: int):
        if count > 0 and count % 50 == 0:
            self._save_cache_index()

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

    def _validate_entry(self, filepath: str, entry: dict, in_variation: int, in_index: int, variations: int) -> str:
        if entry['modeltype'] != self.modeltype:
            raise RuntimeError(
                f"Cache modeltype mismatch for '{filepath}': "
                f"cached as '{entry['modeltype']}', current model is '{self.modeltype}'. "
                f"Delete the cache directory or use a separate cache_dir for this model type."
            )

        resolution = self._get_resolution_string(in_variation, in_index)
        if resolution and entry.get('resolution') and resolution != entry['resolution']:
            return 'resolution_changed'

        for v in range(variations):
            if not os.path.isfile(self._pt_path(entry['cache_file'], v)):
                return 'missing_pt'

        try:
            current_mtime = os.path.getmtime(filepath)
        except OSError:
            return 'rebuild'

        if current_mtime == entry['mtime']:
            return 'valid'

        file_hash = self._hash_file(filepath)
        if file_hash == entry['hash']:
            entry['mtime'] = current_mtime
            return 'valid'

        return 'content_changed'

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
            pt_path = self._pt_path(cache_file, v)
            if os.path.isfile(pt_path):
                continue

            cache_data = {}
            with torch.no_grad():
                for name in self.split_names:
                    cache_data[name] = self.__clone_for_cache(self._get_previous_item(v, name, in_index))
                for name in self.aggregate_names:
                    cache_data[name] = self.__clone_for_cache(self._get_previous_item(v, name, in_index))
            cache_data['__cache_version'] = CACHE_VERSION
            cache_data['__modeltype'] = self.modeltype

            tmp_path = pt_path + f'.{os.getpid()}.{threading.get_ident()}.tmp'
            torch.save(cache_data, os.path.realpath(tmp_path))
            shutil.move(tmp_path, pt_path)

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
            return self._get_previous_item(in_variation, self.source_path_in_name, in_index)
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
            group_indices = {'': [in_index for in_index in range(self._get_previous_length(first_previous_name))]}
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
            pt_path = self._pt_path(entry['cache_file'], 0)
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

    def __refresh_cache_sourceless(self, out_variation: int):
        if not self.variations_initialized:
            self.__init_sourceless()

    def __refresh_cache(self, out_variation: int):
        if not self.variations_initialized:
            self.__init_variations()
        self.__reshuffle_samples(out_variation)

        self.cache_index = self._load_cache_index()
        os.makedirs(self.cache_dir, exist_ok=True)

        before_cache_fun_called = False
        files_built = 0
        files_skipped = 0
        files_failed = []

        all_input_files = set()

        for group_key in self.group_variations.keys():
            start_index = self.group_output_samples[group_key] * out_variation
            end_index = self.group_output_samples[group_key] * (out_variation + 1) - 1

            start_variation = start_index // len(self.group_indices[group_key])
            end_variation = end_index // len(self.group_indices[group_key])

            variations = self.group_variations[group_key]
            needed_variations = [(x % variations) for x in range(start_variation, end_variation + 1, 1)]

            items_to_build = []

            for in_variation in needed_variations:
                for group_index, in_index in enumerate(self.group_indices[group_key]):
                    filepath = self._get_source_path(in_variation, in_index)
                    if filepath is None:
                        continue

                    filepath = os.path.normpath(filepath)
                    all_input_files.add((filepath, group_key, in_variation, in_index, group_index, variations))

                    entry = self.cache_index['entries'].get(filepath)
                    if entry is not None:
                        status = self._validate_entry(filepath, entry, in_variation, in_index, variations)
                        if status == 'valid':
                            files_skipped += 1
                            continue
                        if status in ('resolution_changed', 'missing_pt', 'content_changed'):
                            with self._index_lock:
                                old_hash = entry.get('hash')
                                if old_hash:
                                    self._remove_from_hash_index(old_hash, filepath)
                                del self.cache_index['entries'][filepath]
                            items_to_build.append((filepath, group_key, in_variation, in_index, group_index, variations))
                        elif status == 'rebuild':
                            items_to_build.append((filepath, group_key, in_variation, in_index, group_index, variations))
                    else:
                        items_to_build.append((filepath, group_key, in_variation, in_index, group_index, variations))

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

            with tqdm(total=len(unique_items), smoothing=0.1, desc='caching') as bar:
                def fn(filepath, group_key, in_variation, in_index, group_index, variations, current_device):
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

                    resolution = self._get_resolution_string(in_variation, in_index)

                    if filepath not in self.cache_index['entries']:
                        if self._try_dedup(filepath, file_hash, resolution, mtime):
                            entry = self.cache_index['entries'][filepath]
                            all_present = all(
                                os.path.isfile(self._pt_path(entry['cache_file'], v))
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

                fs = (self._state.executor.submit(
                    fn, filepath, group_key, in_variation, in_index, group_index, variations, current_device)
                      for (filepath, group_key, in_variation, in_index, group_index, variations)
                      in unique_items)
                build_count = 0
                for i, f in enumerate(concurrent.futures.as_completed(fs)):
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
                    build_count += 1
                    if build_count % 250 == 0:
                        self._torch_gc()
                    self._flush_cache_index(build_count)
                    bar.update(1)

        self._save_cache_index()

        total = files_built + files_skipped + len(files_failed)
        if total > 0:
            print(f"SmartDiskCache: Cached {files_built}/{total} files. {files_skipped} reused from cache. {len(files_failed)} failed.")
        if files_failed:
            for fp, reason in files_failed[:10]:
                print(f"  {fp}: {reason}")
            if len(files_failed) > 10:
                print(f"  ... and {len(files_failed) - 10} more")

    def start(self, out_variation: int):
        if self.sourceless:
            self.__refresh_cache_sourceless(out_variation)
        else:
            self.__refresh_cache(out_variation)

    def get_item(self, index: int, requested_name: str = None) -> dict:
        group_key, in_variation, group_index, in_index = self.__get_input_index(self.current_variation, index)

        if self.sourceless:
            filepath = self._sourceless_filepaths[group_index]
        else:
            filepath = self._get_source_path(in_variation, in_index)

        if filepath is not None:
            if not self.sourceless:
                filepath = os.path.normpath(filepath)
            cache_entry = self.cache_index['entries'].get(filepath)

            if cache_entry is not None:
                variation = in_variation % self.group_variations[group_key]
                cache_path = self._pt_path(cache_entry['cache_file'], variation)

                cached = torch.load(os.path.realpath(cache_path), weights_only=False, map_location=self.pipeline.device)

                item = {}
                for name in self.split_names + self.aggregate_names:
                    if name in cached:
                        item[name] = cached[name]
                return item

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
        for fp, entry in entries.items():
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
        shutil.move(tmp_path, cache_path)
