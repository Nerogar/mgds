from typing import Iterator

import torch

from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule
from mgds.pipelineModuleTypes.SerialPipelineModule import SerialPipelineModule
from mgds.pipelineModuleTypes.SingleVariationRandomAccessPipelineModule import SingleVariationRandomAccessPipelineModule


class LoadingPipeline(Iterator):
    device: torch.device
    modules: list[PipelineModule]
    __output_module: SerialPipelineModule
    __current_epoch: int
    __current_index: int
    __last_initialized_epoch: int
    __batch_size: int
    __initial_epoch: int
    __initial_index: int
    __mark_reset_serial_modules: bool

    def __init__(
            self,
            device: torch.device,
            modules: list[PipelineModule],
            batch_size: int,
            seed: int,
            state: PipelineState,
            initial_epoch: int = 0,
            initial_index: int = 0
    ):
        self.device = device
        self.modules = list(filter(lambda x: x is not None, self.__flatten(modules)))
        for module in self.modules:
            if type(module).__name__ == 'OutputPipelineModule':
                self.__output_module = module

        for index, module in enumerate(self.modules):
            module.init(self, seed, index, state)

        self.__batch_size = batch_size
        self.__initial_epoch = initial_epoch
        self.__initial_index = initial_index - (initial_index % batch_size)

        self.__current_epoch = initial_epoch - 1
        self.__last_initialized_epoch = -1

        self.__mark_reset_serial_modules = False

    def __flatten(self, data: list | object) -> list:
        if isinstance(data, list):
            new_list = []
            for x in [self.__flatten(x) for x in data]:
                new_list.extend(x)
            return new_list
        else:
            return [data]

    def approximate_length(self) -> int:
        """
        Returns the approximated length of a full epoch.
        The number might not be exact, because the length can change between epochs.
        """
        return max(0, self.__output_module.approximate_length())

    def reset_serial_modules_before(self, module_index: int):
        with torch.no_grad():
            for module in self.modules[:module_index + 1]:
                # At the start of each epoch, the previous cache is cleared.
                # This prevents duplicating samples when training on single images.
                module.clear_item_cache()

                if isinstance(module, SerialPipelineModule):
                    if self.__current_epoch == self.__initial_epoch:
                        module.current_index = self.__initial_index
                        module.start(self.__current_epoch, self.__initial_index)
                    else:
                        module.current_index = 0
                        module.start(self.__current_epoch, 0)

    def start_next_epoch(self):
        self.__current_epoch += 1

        with torch.no_grad():
            for module in self.modules:
                # At the start of each epoch, the previous cache is cleared.
                # This prevents duplicating samples when training on single images.
                module.clear_item_cache()

                if isinstance(module, RandomAccessPipelineModule):
                    if not module.started:
                        module.start(self.__current_epoch)
                    module.started = True
                elif isinstance(module, SingleVariationRandomAccessPipelineModule):
                    module.current_variation = self.__current_epoch
                    module.start(self.__current_epoch)
                elif isinstance(module, SerialPipelineModule):
                    module.current_variation = self.__current_epoch
                    if self.__current_epoch == self.__initial_epoch:
                        module.current_index = self.__initial_index
                        module.start(self.__current_epoch, self.__initial_index)
                    else:
                        module.current_index = 0
                        module.start(self.__current_epoch, 0)

        self.__last_initialized_epoch = self.__current_epoch

        self.__mark_reset_serial_modules = True

    def __next__(self):
        if self.__mark_reset_serial_modules:
            self.__mark_reset_serial_modules = False
            self.reset_serial_modules_before(len(self.modules) - 1)

        if not self.__output_module.has_next():
            raise StopIteration

        with torch.no_grad():
            item = self.__output_module.get_next_item()
            self.__output_module.current_index += 1
            return item
