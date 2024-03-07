import gc
from abc import ABCMeta, abstractmethod
from contextlib import ExitStack
from random import Random

import torch

from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule
from mgds.pipelineModuleTypes.SerialPipelineModule import SerialPipelineModule
from mgds.pipelineModuleTypes.SingleVariationRandomAccessPipelineModule import SingleVariationRandomAccessPipelineModule


class PipelineModule(metaclass=ABCMeta):
    pipeline: 'LoadingPipeline'

    __base_seed: int
    __module_index: int

    __variation_cache_index: int
    __item_cache_index: int
    __item_cache: dict
    __length_cache: int

    def __init__(self):
        super(PipelineModule, self).__init__()
        self.clear_item_cache()

    def init(self, pipeline: 'LoadingPipeline', base_seed: int, module_index: int):
        self.pipeline = pipeline

        self.__base_seed = base_seed
        self.__module_index = module_index

    def clear_item_cache(self):
        self.__variation_cache_index = -1
        self.__item_cache_index = -1
        self.__item_cache = {}
        self.__length_cache = -1

    def __raise_variation_error(
            self,
            module: 'PipelineModule',
            name: str,
            current_variation: int,
            requested_variation: int,
    ):
        raise Exception(f"wrong variation requested by {self} from {str(module)}, name: {name}, current_variation: {current_variation}, requested_variation: {requested_variation}")

    def _get_previous_item(self, variation: int, name: str, index: int):
        split_name = name.split('.')
        item_name = split_name[0]
        path_names = split_name[1::]

        item = None

        for previous_module_index in range(self.__module_index - 1, -1, -1):
            module = self.pipeline.modules[previous_module_index]
            if item_name in module.get_outputs():
                # item is cached
                if module.__variation_cache_index == variation \
                        and module.__item_cache_index == index \
                        and item_name in module.__item_cache.keys():
                    item = module.__item_cache[item_name]

                # the wrong index is cached, clear cache and recalculate
                elif module.__variation_cache_index != variation \
                        or module.__item_cache_index != index:
                    if isinstance(module, RandomAccessPipelineModule):
                        item = module.get_item(variation, index, item_name)
                    if isinstance(module, SingleVariationRandomAccessPipelineModule):
                        if variation != module.current_variation:
                            self.__raise_variation_error(module, name, module.current_variation, variation)
                        item = module.get_item(index, item_name)
                    if isinstance(module, SerialPipelineModule):
                        if variation != module.current_variation:
                            self.__raise_variation_error(module, name, module.current_variation, variation)
                        item = module.get_item(index, item_name)
                    module.__variation_cache_index = variation
                    module.__item_cache_index = index
                    module.__item_cache = item
                    item = item[item_name]

                # the item is cached and the index is correct, but the item_name is not part of the cache
                # recalculate and add to the cache
                elif item_name not in module.__item_cache.keys():
                    if isinstance(module, RandomAccessPipelineModule):
                        item = module.get_item(variation, index, item_name)
                    if isinstance(module, SingleVariationRandomAccessPipelineModule):
                        if variation != module.current_variation:
                            self.__raise_variation_error(module, name, module.current_variation, variation)
                        item = module.get_item(index, item_name)
                    if isinstance(module, SerialPipelineModule):
                        if variation != module.current_variation:
                            self.__raise_variation_error(module, name, module.current_variation, variation)
                        item = module.get_item(index, item_name)
                    module.__item_cache.update(item)
                    item = item[item_name]

                # if the item was found, break the loop
                # else, fall through to a previous module and try again
                if item is not None:
                    break

        for path_name in path_names:
            if path_name in item:
                item = item[path_name]
            else:
                item = None
                break

        return item

    def _get_previous_length(self, name: str):
        split_name = name.split('.')
        item_name = split_name[0]
        path_names = split_name[1::]

        for previous_module_index in range(self.__module_index - 1, -1, -1):
            module = self.pipeline.modules[previous_module_index]
            if item_name in module.get_outputs():
                if module.__length_cache < 0:
                    module.__length_cache = module.length()
                return module.__length_cache

    def _get_previous_meta(self, variation: int, name: str):
        for previous_module_index in range(self.__module_index - 1, -1, -1):
            module = self.pipeline.modules[previous_module_index]
            if name in module.get_outputs():
                if isinstance(module, RandomAccessPipelineModule):
                    return module.get_meta(variation, name)
                if isinstance(module, SingleVariationRandomAccessPipelineModule):
                    if variation != module.current_variation:
                        self.__raise_variation_error(module, name, module.current_variation, variation)
                    return module.get_meta(name)
                if isinstance(module, SerialPipelineModule):
                    if variation != module.current_variation:
                        self.__raise_variation_error(module, name, module.current_variation, variation)
                    return module.get_meta(name)

    def _all_contexts(self, autocast_contexts: list[torch.autocast | None]) -> ExitStack:
        stack = ExitStack()
        for context in autocast_contexts:
            stack.enter_context(context)
        return stack

    def _get_rand(self, variation: int, index: int = -1) -> Random:
        seed = hash((self.__base_seed, self.__module_index, variation, index))
        return Random(seed)

    def _torch_gc(self):
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    @abstractmethod
    def length(self) -> int:
        pass

    @abstractmethod
    def get_inputs(self) -> list[str]:
        pass

    @abstractmethod
    def get_outputs(self) -> list[str]:
        pass
