import concurrent
import concurrent.futures
import hashlib
import json
import math
import os
import pathlib
from typing import Any, Callable

import torch
from tqdm import tqdm

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.SingleVariationRandomAccessPipelineModule import SingleVariationRandomAccessPipelineModule
from mgds.util.PersistentCacheState import PersistentCacheState
from mgds.util.FileUtil import safe_write_json_file

def _cache_debug_print(*message: str):
    print(*message)

def _clone_for_cache(item: Any) -> Any:
    if isinstance(item, torch.Tensor):
        return item.clone()
    else:
        return item

def _build_cache_data(pipeline: PipelineModule,
                      persistent_cache: PersistentCacheState,
                      split_names: list[str],
                      aggregate_names: list[str],
                      group_index: int,
                      in_index: int,
                      in_variation: int,
                      current_thread_cuda_device: int) -> tuple[int, dict]:
    # preserve current device for multi-GPU, which is thread-local in torch:
    if torch.cuda.is_available() and current_thread_cuda_device is not None:
        torch.cuda.set_device(current_thread_cuda_device)

    with torch.no_grad():
        split_item = {}
        for name in split_names:
            split_item[name] = _clone_for_cache(pipeline._get_previous_item(in_variation, name, in_index))
        persistent_cache.save_split_item(group_index, split_item)
        del split_item

        aggregate_item = {}
        for name in aggregate_names:
            aggregate_item[name] = _clone_for_cache(pipeline._get_previous_item(in_variation, name, in_index))
        return group_index, aggregate_item


class DiskCache(
    PipelineModule,
    SingleVariationRandomAccessPipelineModule,
):
    def __init__(
            self,
            cache_dir: str,
            persistent_key_in_name: str | None = None,
            split_names: list[str] | None = None,
            aggregate_names: list[str] | None = None,
            variations_in_name: str | None = None,
            balancing_in_name: str | None = None,
            balancing_strategy_in_name: str | None = None,
            variations_group_in_name: str | list[str] | None = None,
            group_enabled_in_name: str | None = None,
            group_name_in_name: str | None = None,
            before_cache_fun: Callable[[], None] | None = None,
            allow_unsafe_types: bool = False,
            delete_stale_cache_files: bool = False,
    ):
        super(DiskCache, self).__init__()
        self.cache_dir = cache_dir

        self.persistent_key_in_name = persistent_key_in_name

        self.split_names = [] if split_names is None else split_names
        self.aggregate_names = [] if aggregate_names is None else aggregate_names

        self.variations_in_name = variations_in_name
        self.balancing_in_name = balancing_in_name
        self.balancing_strategy_in_name = balancing_strategy_in_name
        self.variations_group_in_names = \
            [variations_group_in_name] if isinstance(variations_group_in_name, str) else variations_group_in_name

        self.group_enabled_in_name = group_enabled_in_name
        self.group_name_in_name = group_name_in_name

        self.before_cache_fun = (lambda: None) if before_cache_fun is None else before_cache_fun

        self.allow_unsafe_types = allow_unsafe_types
        self.delete_stale_cache_files = delete_stale_cache_files

        self.group_variations = {}
        self.group_indices = {}
        self.group_output_samples = {}
        self.variations_initialized = False

    def length(self) -> int:
        if not self.variations_initialized:
            name = self.split_names[0] if len(self.split_names) > 0 else self.aggregate_names[0]
            return self._get_previous_length(name)
        else:
            return sum(x for x in self.group_output_samples.values())

    def get_inputs(self) -> list[str]:
        return self.split_names + self.aggregate_names \
            + [self.variations_in_name] if self.variations_in_name else [] \
            + [self.balancing_in_name] if self.variations_in_name else [] \
            + [self.balancing_strategy_in_name] if self.variations_in_name else [] \
            + self.variations_group_in_names if self.variations_in_name else [] \
            + [self.group_enabled_in_name] if self.variations_in_name else [] \
            + [self.persistent_key_in_name] if self.persistent_key_in_name else [] \
            + [self.group_name_in_name] if self.group_name_in_name else []

    def get_outputs(self) -> list[str]:
        return self.split_names + self.aggregate_names

    def __string_key(self, data: list[Any]) -> str:
        json_data = json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(',', ':'), indent=None)
        return hashlib.sha256(json_data.encode('utf-8')).hexdigest()

    def __init_variations(self):
        """
        Prepares variations before caching starts. Each index is sorted into a group.

        Data is written into three variables.
            self.group_variations, mapping group keys to the number of variations of that group
            self.group_indices, mapping group keys to a list of input indices contained in the group
            self.group_output_samples, mapping group keys to the number of indices in the cache output for each group
        """
        if self.variations_in_name is not None:
            group_variations: dict[str, int] = {}
            group_indices: dict[str, list] = {}
            group_balancing: dict[str, int] = {}
            group_balancing_strategy: dict[str, str] = {}

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

        self.aggregate_cache = {}
        self.persistent_cache = {}

        self.group_variations = group_variations
        self.group_indices = group_indices
        self.group_output_samples = group_output_samples

        self.variations_initialized = True

    def __get_group_index_for_variation(self,
                                        group_key: str,
                                        sample_index: int):
        """
        Gets the `group_index` we should return for the sample index, for the current variation.
        
        Attempts to maximize the number of unique sample indices used across variations when possible,
        such as when `len(group_indices) > group_output_samples and group_variations > 1`.). In this
        case, unique samples will be used until they, or the number of variations allowed, are exhausted.
        
        However, this can lead to earlier samples being repeated up to one time more than later
        samples. If this is an issue, it can be avoided by keeping `group_output_samples` either a
        whole multiple of `group_indices` with any number of variations, or with any number of
        `group_output_samples` with only one variation set.
        
        :param group_key: The key for the group currently being cached.
        :type group_key: str
        
        :param sample_index: The number of the current sample in this group for this epoch/variation.
        :type sample_index: int
        
        :return: The `group_index` value to use for the provided `group_key` and `sample_index`.
        :rtype: int
        """
        assert sample_index >= 0

        current_cache_variation = self.current_variation % self.group_variations[group_key]
        num_samples_per_variation = self.group_output_samples[group_key]
        variation_offset = num_samples_per_variation * current_cache_variation
        actual_num_samples = len(self.group_indices[group_key])
        return (sample_index + variation_offset) % actual_num_samples

    def __get_persistent_cache_group_mapping(self, group_key):
        """
        Creates a mapping between the active samples for the current variation, and their unique
        persistent key. The persistent key will only actually be persistent across runs if the
        `persistent_key_in_name` value is set. Otherwise, the cache is only valid until a sample is
        added, removed, or changed in this group.
        
        :param group_key: The key for the group currently being cached.
        :type group_key: str
        
        :return: A dictionary of all active `group_index` values and their unique persistent key.
        :rtype: dict
        """
        group_index_mappings = {}

        if self.persistent_key_in_name is None:
            print('Warning: Using unstable cache file mappings.')

        current_group_indices = self.group_indices[group_key]
        num_out_samples = self.group_output_samples[group_key]
        for sample_index in range(num_out_samples):
            group_index = self.__get_group_index_for_variation(group_key, sample_index)

            # Depending on the number of `group_output_samples` and `group_indices`, it is possible
            # (and common) for multiple samples to map to the same `group_index``. Skip pulling this
            # data multiple times in these cases.
            if group_index in group_index_mappings:
                continue

            if self.persistent_key_in_name is not None:
                # Map each `group_index` to a persistent key--a value that will always be the
                # same for the file (or the file's data in a perfect world), so that our
                # persistent cache has a stable value to reference this file by.
                sample_label = self._get_previous_item(self.current_variation,
                                                       self.persistent_key_in_name,
                                                       current_group_indices[group_index])
            else:
                # As we don't have a persistent key to map each group_index to the file that it
                # represents, we fall-back to the legacy unstable behavior by mapping each
                # `group_index` it to a string of itself. This is not "safe", in that any
                # inclusions/removals to the files on disk will disrupt this order the next time
                # we run. It is much better to use the code path above this one, but this path
                # exists for legacy callers that have not been updated.
                sample_label = str(group_index)
            group_index_mappings[group_index] = sample_label

        return group_index_mappings

    def __generate_cache_future_tasks(self,
                                      group_key: str,
                                      aggregate_cache: dict[int, dict],
                                      persistent_cache_state: PersistentCacheState):
        """
        Creates and yields cache-generation futures for each cache entry missing from the current
        group.
        
        :param group_key: The key for the group currently being cached.
        :type group_key: str
        
        :param aggregate_cache: The aggregate cache for the current group.
        :type group_key: dict[int, dict]
        
        :param persistent_cache_state: The persistent cache state for the current group.
        :type group_key: PersistentCacheState
        
        :return: Generator that yields async task futures that build the missing cache entries.
        :rtype: Generator[Future[tuple[int, dict]], Any, None]
        """
        current_thread_cuda_device = torch.cuda.current_device() if torch.cuda.is_available() else None

        num_out_samples = self.group_output_samples[group_key]
        for sample_index in range(num_out_samples):
            group_index = self.__get_group_index_for_variation(group_key, sample_index)
            if group_index in aggregate_cache:
                continue

            aggregate_cache[group_index] = None
            yield self._state.executor.submit(_build_cache_data,
                                              self,
                                              persistent_cache_state,
                                              self.split_names,
                                              self.aggregate_names,
                                              group_index,
                                              self.group_indices[group_key][group_index],
                                              self.current_variation,
                                              current_thread_cuda_device)

    def __get_cache_group_label(self, group_key: str) -> str:
        """
        Gets a human-friendly name for the specific cache group.
        
        :param group_key: The key for the group currently being cached.
        :type group_key: str
        
        :return: The name of the current group, or the key for the current group if the name is
            unknown.
        :rtype: str
        """
        group_name = (self._get_previous_item(self.current_variation,
                                                self.group_name_in_name,
                                                self.group_indices[group_key][0])
                      if self.group_name_in_name is not None else
                      group_key)
        if group_name == group_key:
            return group_key
        else:
            return f'{group_name} ({group_key[0:7]}...{group_key[-7:]})'

    def __get_cache_dir_path(self, group_key: str) -> pathlib.Path:
        """
        Gets the Path for the for the specified group for the current variation.
        
        :param group_key: The key for the group currently being cached.
        :type group_key: str
        
        :return: The Path for the specified group's cache directory.
        :rtype: pathlib.Path
        """
        num_variations = self.group_variations[group_key]
        variation_folder = f"variation-{str(self.current_variation % num_variations)}"
        return pathlib.Path(self.cache_dir) / group_key / variation_folder

    def __refresh_cache(self):
        if not self.variations_initialized:
            self.__init_variations()

        before_cache_fun_called = False
        for group_key, num_variations in self.group_variations.items():
            # Build our list of `group_index` to persistent_key mappings. This is also a complete
            # list of every `group_index` we need to make sure we have/create a cache file for.
            persistent_cache_group_mappings = self.__get_persistent_cache_group_mapping(group_key)

            cache_path = self.__get_cache_dir_path(group_key)
            persistent_cache_state_path = cache_path / 'cache_state.json'

            # Load Persistent Cache State
            persistent_cache_state = None
            if persistent_cache_state_path.is_file():
                try:
                    with open(persistent_cache_state_path, 'r') as persistent_cache_reader:
                        persistent_cache_dict = json.load(persistent_cache_reader)
                    persistent_cache_state = PersistentCacheState.from_dict(cache_path, persistent_cache_dict)
                except ValueError as ex:
                    print(f'Could not load "{persistent_cache_state_path.name}" due to error: {str(ex)}.'
                            ' Cache will be rebuilt.')
            if persistent_cache_state is None:
                persistent_cache_state = PersistentCacheState(cache_path,
                                                              aggregate_filename='aggregate.pt')
            self.persistent_cache[group_key] = persistent_cache_state

            # Give our persistent cache state a list of the current run's set of samples.
            persistent_cache_state.build_cache_file_mappings(persistent_cache_group_mappings,
                                                             remove_stale_cache=self.delete_stale_cache_files)

            # Resave our cache state if we are using persistent file mappings
            if self.persistent_key_in_name is not None:
                persistent_cache_dict = persistent_cache_state.to_dict()
                safe_write_json_file(persistent_cache_dict, persistent_cache_state_path)

            # Load our aggregate items cache
            aggregate_cache: dict = persistent_cache_state.get_aggregate_cache(self.pipeline.device,
                                                                               self.allow_unsafe_types,
                                                                               validate_against_split_items=True)
            if aggregate_cache is None:
                aggregate_cache = {}
            self.aggregate_cache[group_key] = aggregate_cache

            # If we have uncached items, cache them now
            total_uncached_files = sum(1 for key in persistent_cache_group_mappings.keys() if key not in aggregate_cache)
            if total_uncached_files > 0:
                if not before_cache_fun_called and self.before_cache_fun is not None:
                    before_cache_fun_called = True
                    self.before_cache_fun()

                total_files = len(persistent_cache_group_mappings)
                total_cached_files = total_files - total_uncached_files
                caching_text = f'caching {self.__get_cache_group_label(group_key)}'
                with tqdm(initial=total_cached_files, total=total_files, smoothing=0.1, desc=caching_text) as bar:
                    fs = self.__generate_cache_future_tasks(group_key,
                                                            aggregate_cache,
                                                            persistent_cache_state)
                    for num_tasks_complete, future in enumerate(concurrent.futures.as_completed(fs)):
                        try:
                            group_index, index_aggregate_data = future.result()
                            aggregate_cache[group_index] = index_aggregate_data
                        except:
                            self._state.executor.shutdown(wait=True, cancel_futures=True)
                            raise

                        # Periodically flush our torch cache and save our aggregate cache to disk.
                        if num_tasks_complete % 200 == 0:
                            self._torch_gc()
                            persistent_cache_state.save_aggregate_cache(aggregate_cache)
                        bar.update(1)

                    persistent_cache_state.save_aggregate_cache(aggregate_cache)

    def __get_input_index(self, out_index: int) -> tuple[str, int]:
        offset = 0
        for group_key, group_output_samples in self.group_output_samples.items():
            if out_index >= offset + group_output_samples:
                offset += group_output_samples
                continue

            local_index = out_index - offset
            group_index = local_index % len(self.group_indices[group_key])

            return group_key, group_index

        raise IndexError(f'Could not find index for variation {out_index}.')

    def start(self, out_variation: int):
        self.__refresh_cache()

    def get_item(self, index: int, requested_name: str = None) -> dict:
        item = {}

        group_key, input_index = self.__get_input_index(index)

        group_index = self.__get_group_index_for_variation(group_key, input_index)

        if requested_name is None or requested_name in self.aggregate_names:
            aggregate_item: dict[str, Any] = self.aggregate_cache[group_key][group_index]
            item = aggregate_item.copy()

        if requested_name is None or requested_name in self.split_names:
            persistent_cache: PersistentCacheState = self.persistent_cache[group_key]
            split_item: dict[str, Any] = \
                persistent_cache.get_split_item(group_index,
                                                self.pipeline.device,
                                                allow_unsafe_types=self.allow_unsafe_types)
            for name in self.split_names:
                item[name] = split_item[name]

        return item
