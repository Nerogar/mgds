import concurrent
import hashlib
import json
import math
import os
from typing import Any, Callable

import torch
from tqdm import tqdm

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.SingleVariationRandomAccessPipelineModule import SingleVariationRandomAccessPipelineModule


class DiskCache(
    PipelineModule,
    SingleVariationRandomAccessPipelineModule,
):
    def __init__(
            self,
            cache_dir: str,
            split_names: list[str] | None = None,
            aggregate_names: list[str] | None = None,
            variations_in_name: str | None = None,
            repeats_in_name: str | None = None,
            variations_group_in_name: str | list[str] | None = None,
            group_enabled_in_name: str | None = None,
            before_cache_fun: Callable[[], None] | None = None,
    ):
        super(DiskCache, self).__init__()
        self.cache_dir = cache_dir

        self.split_names = [] if split_names is None else split_names
        self.aggregate_names = [] if aggregate_names is None else aggregate_names

        self.variations_in_name = variations_in_name
        self.repeats_in_name = repeats_in_name
        self.variations_group_in_names = \
            [variations_group_in_name] if isinstance(variations_group_in_name, str) else variations_group_in_name

        self.group_enabled_in_name = group_enabled_in_name

        self.before_cache_fun = before_cache_fun

        self.group_variations = {}
        self.group_indices = {}
        self.group_output_samples = {}
        self.variations_initialized = False
        self.executor = concurrent.futures.ThreadPoolExecutor(8)

        if len(self.split_names) + len(self.aggregate_names) == 0:
            raise ValueError('No cache items supplied')

    def length(self) -> int:
        if not self.variations_initialized:
            name = self.split_names[0] if len(self.split_names) > 0 else self.aggregate_names[0]
            return self._get_previous_length(name)
        else:
            return sum(x for x in self.group_output_samples.values())

    def get_inputs(self) -> list[str]:
        return self.split_names + self.aggregate_names \
            + [self.variations_in_name] if self.variations_in_name else [] \
            + [self.repeats_in_name] if self.repeats_in_name else [] \
            + self.variations_group_in_names if self.repeats_in_name else [] \
            + [self.group_enabled_in_name] if self.repeats_in_name else []

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
            group_variations = {}
            group_indices = {}
            group_repeats = {}

            for in_index in range(self._get_previous_length(self.variations_in_name)):
                if self.group_enabled_in_name and not self._get_previous_item(0, self.group_enabled_in_name, in_index):
                    continue

                variations = self._get_previous_item(0, self.variations_in_name, in_index)
                repeats = self._get_previous_item(0, self.repeats_in_name, in_index)
                group_key = self.__string_key(
                    [self._get_previous_item(0, name, in_index) for name in self.variations_group_in_names]
                )

                if group_key not in group_variations:
                    group_variations[group_key] = variations

                if group_key not in group_indices:
                    group_indices[group_key] = []
                group_indices[group_key].append(in_index)

                if group_key not in group_repeats:
                    group_repeats[group_key] = repeats

            group_output_samples = {}
            for group_key, repeats in group_repeats.items():
                num = int(math.floor(len(group_indices[group_key]) * repeats))
                group_output_samples[group_key] = num
        else:
            first_previous_name = self.split_names[0] if len(self.split_names) > 0 else self.aggregate_names[0]

            group_variations = {'': 1}
            group_indices = {'': [in_index for in_index in range(self._get_previous_length(first_previous_name))]}
            group_output_samples = {'': len(group_indices[''])}

        self.aggregate_cache = {}

        self.group_variations = group_variations
        self.group_indices = group_indices
        self.group_output_samples = group_output_samples

        self.variations_initialized = True

    def __get_cache_dir(self, group_key: str, in_variation: int) -> str:
        variations = self.group_variations[group_key]
        return os.path.join(self.cache_dir, group_key, "variation-" + str(in_variation % variations))

    def __is_caching_done(self, group_key: str, in_variation: int):
        cache_dir = self.__get_cache_dir(group_key, in_variation)

        cache_exists = False
        caching_done = False

        if os.path.isdir(cache_dir):
            with os.scandir(cache_dir) as path_iter:
                cache_exists = any(path_iter)

            aggregate_path = os.path.join(cache_dir, 'aggregate.pt')
            caching_done = os.path.exists(aggregate_path) and os.path.isfile(aggregate_path)

        return cache_exists and caching_done

    def __refresh_cache(self, out_variation: int):
        if not self.variations_initialized:
            self.__init_variations()

        self.aggregate_cache = {}
        for group_key, variations in self.group_variations.items():
            self.aggregate_cache[group_key] = [None for _ in range(variations)]

        before_cache_fun_called = False
        for group_key in self.group_variations.keys():
            start_index = self.group_output_samples[group_key] * out_variation
            end_index = self.group_output_samples[group_key] * (out_variation + 1) - 1

            start_variation = start_index // len(self.group_indices[group_key])
            end_variation = end_index // len(self.group_indices[group_key])

            variations = self.group_variations[group_key]
            for in_variation in [(x % variations) for x in range(start_variation, end_variation + 1, 1)]:
                cache_dir = self.__get_cache_dir(group_key, in_variation)
                if not self.__is_caching_done(group_key, in_variation):
                    if not before_cache_fun_called and self.before_cache_fun is not None:
                        before_cache_fun_called = True
                        self.before_cache_fun()

                    os.makedirs(cache_dir, exist_ok=True)

                    aggregate_cache_ = []
                    aggregate_cache = []
                    fs = []

                    for group_index, in_index in enumerate(tqdm(self.group_indices[group_key], desc='caching')):
                        if in_index % 100 == 0:
                            for f in concurrent.futures.as_completed(fs):
                                aggregate_cache_.append(f.result())
                            fs = []
                            self._torch_gc()

                        def fn(group_index, in_index, in_variation):
                            split_item = {}
                            aggregate_item = {}

                            for name in self.split_names:
                                split_item[name] = self._get_previous_item(in_variation, name, in_index)
                            for name in self.aggregate_names:
                                aggregate_item[name] = self._get_previous_item(in_variation, name, in_index)

                            torch.save(split_item, os.path.realpath(os.path.join(cache_dir, str(group_index) + '.pt')))
                            return (group_index, aggregate_item)
                            #aggregate_cache.append(aggregate_item)
                        fs.append(self.executor.submit(fn, group_index, in_index, in_variation))

                    for f in concurrent.futures.as_completed(fs):
                        aggregate_cache_.append(f.result())
                    aggregate_cache_.sort(key=lambda x: x[0])
                    aggregate_cache = list(map(lambda x: x[1], aggregate_cache_))
                    fs = []
                    torch.save(aggregate_cache, os.path.realpath(os.path.join(cache_dir, 'aggregate.pt')))

                if self.aggregate_cache[group_key][in_variation] is None:
                    self.aggregate_cache[group_key][in_variation] = \
                        torch.load(os.path.realpath(os.path.join(cache_dir, 'aggregate.pt')))

    def __get_input_index(self, out_variation: int, out_index: int) -> (str, int, int):
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

    def start(self, out_variation: int):
        self.__refresh_cache(out_variation)

    def get_item(self, index: int, requested_name: str = None) -> dict:
        item = {}

        group_key, in_variation, group_index, in_index = self.__get_input_index(self.current_variation, index)

        aggregate_item = self.aggregate_cache[group_key][in_variation][group_index]

        if requested_name in self.aggregate_names:
            for name in self.aggregate_names:
                item[name] = aggregate_item[name]

        elif requested_name in self.split_names:
            cache_path = os.path.join(self.__get_cache_dir(group_key, in_variation), str(group_index) + '.pt')
            split_item = torch.load(os.path.realpath(cache_path))

            for name in self.split_names:
                item[name] = split_item[name]

        return item
