import gc
import threading
from abc import ABCMeta, abstractmethod
from concurrent import futures
from contextlib import ExitStack
from random import Random

import torch

from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule
from mgds.pipelineModuleTypes.SerialPipelineModule import SerialPipelineModule
from mgds.pipelineModuleTypes.SingleVariationRandomAccessPipelineModule import SingleVariationRandomAccessPipelineModule


class PipelineState:
    """Container for state shared amongst all pipeline modules in a pipeline.

    Each element must either be thread-safe itself or have a thread-safe way
    for it to be accessed.
    """
    # Executor that any pipeline module will use to fan out concurrency.
    # Defaults to 2x CPUs.
    executor: futures.Executor

    def __init__(self, max_threads: int|None = None):
        self.executor = futures.ThreadPoolExecutor(max_threads)


class PipelineModule(metaclass=ABCMeta):
    pipeline: 'LoadingPipeline'

    __base_seed: int
    __module_index: int
    _state: PipelineState

    def __init__(self):
        super(PipelineModule, self).__init__()
        self.clear_item_cache()

    def init(self, pipeline: 'LoadingPipeline', base_seed: int, module_index: int, state: PipelineState):
        self.pipeline = pipeline

        self.__base_seed = base_seed
        self.__module_index = module_index
        self._state = state

    # Bits for handling the cache.
    class Cache(threading.local):
        def __init__(self):
            self.variation_cache_index = -1
            self.item_cache_index = -1
            self.item_cache = {}
            self.length_cache = -1

    def clear_item_cache(self):
        self.__local_cache = self.Cache()

    def __raise_variation_error(
            self,
            module: 'PipelineModule',
            name: str,
            current_variation: int,
            requested_variation: int,
    ):
        raise Exception(f"wrong variation requested by {self} from {str(module)}, name: {name}, current_variation: {current_variation}, requested_variation: {requested_variation}")

    def __raise_index_error(
            self,
            module: 'PipelineModule',
            name: str,
            current_index: int,
            requested_index: int,
    ):
        raise Exception(f"wrong index requested by {self} from {str(module)}, name: {name}, current_index: {current_index}, requested_index: {requested_index}")

    def _get_previous_item(self, variation: int, name: str, index: int):
        split_name = name.split('.')
        item_name = split_name[0]
        path_names = split_name[1::]

        item = None

        for previous_module_index in range(self.__module_index - 1, -1, -1):
            module = self.pipeline.modules[previous_module_index]
            if item_name in module.get_outputs():
                # item is cached
                if module.__local_cache.variation_cache_index == variation \
                        and module.__local_cache.item_cache_index == index \
                        and item_name in module.__local_cache.item_cache.keys():
                    item = module.__local_cache.item_cache[item_name]

                # the wrong index is cached, clear cache and recalculate
                elif module.__local_cache.variation_cache_index != variation \
                        or module.__local_cache.item_cache_index != index:
                    if isinstance(module, RandomAccessPipelineModule):
                        item = module.get_item(variation, index, item_name)
                    if isinstance(module, SingleVariationRandomAccessPipelineModule):
                        if variation != module.current_variation:
                            self.__raise_variation_error(module, name, module.current_variation, variation)
                        item = module.get_item(index, item_name)
                    if isinstance(module, SerialPipelineModule):
                        if variation != module.current_variation:
                            self.__raise_variation_error(module, name, module.current_variation, variation)
                        if index != module.current_index:
                            if index == 0:
                                self.pipeline.reset_serial_modules_before(previous_module_index)
                            else:
                                self.__raise_index_error(module, name, module.current_index, index)
                        item = module.get_next_item()
                        module.current_index += 1
                    module.__local_cache.variation_cache_index = variation
                    module.__local_cache.item_cache_index = index
                    module.__local_cache.item_cache = item
                    item = item[item_name]

                # the item is cached and the index is correct, but the item_name is not part of the cache
                # recalculate and add to the cache
                elif item_name not in module.__local_cache.item_cache.keys():
                    if isinstance(module, RandomAccessPipelineModule):
                        item = module.get_item(variation, index, item_name)
                    if isinstance(module, SingleVariationRandomAccessPipelineModule):
                        if variation != module.current_variation:
                            self.__raise_variation_error(module, name, module.current_variation, variation)
                        item = module.get_item(index, item_name)
                    module.__local_cache.item_cache.update(item)
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
                if module.__local_cache.length_cache < 0:
                    if isinstance(module, SerialPipelineModule):
                        module.__local_cache.length_cache = module.approximate_length()
                    else:
                        module.__local_cache.length_cache = module.length()
                return module.__local_cache.length_cache

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

    def _has_previous_next(self, name: str):
        for previous_module_index in range(self.__module_index - 1, -1, -1):
            module = self.pipeline.modules[previous_module_index]
            if name in module.get_outputs():
                if isinstance(module, SerialPipelineModule):
                    return module.has_next()
                else:
                    return True


    def _all_contexts(self, autocast_contexts: list[torch.autocast | None]) -> ExitStack:
        stack = ExitStack()
        for context in autocast_contexts:
            stack.enter_context(context)
        return stack

    def _get_rand(self, variation: int, index: int = -1) -> Random:
        seed = hash((self.__base_seed, self.__module_index, variation, index))
        return Random(seed)

    def _torch_gc(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    @abstractmethod
    def get_inputs(self) -> list[str]:
        pass

    @abstractmethod
    def get_outputs(self) -> list[str]:
        pass
