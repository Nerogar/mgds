import torch

from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule
from mgds.pipelineModuleTypes.SerialPipelineModule import SerialPipelineModule
from mgds.pipelineModuleTypes.SingleVariationRandomAccessPipelineModule import SingleVariationRandomAccessPipelineModule


class LoadingPipeline:
    device: torch.device
    modules: list[PipelineModule]
    __output_module: 'OutputPipelineModule'
    __current_epoch: int
    __last_initialized_epoch: int
    __batch_size: int
    __initial_epoch: int
    __initial_epoch_sample: int

    def __init__(
            self,
            device: torch.device,
            modules: list[PipelineModule],
            batch_size: int,
            seed: int,
            state: PipelineState,
            initial_epoch: int = 0,
            initial_epoch_sample: int = 0
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
        self.__initial_epoch_sample = initial_epoch_sample - (initial_epoch_sample % batch_size)

        self.__current_epoch = initial_epoch - 1
        self.__last_initialized_epoch = -1

    def __flatten(self, data: list | object) -> list:
        if isinstance(data, list):
            new_list = []
            for x in [self.__flatten(x) for x in data]:
                new_list.extend(x)
            return new_list
        else:
            return [data]

    def length(self) -> int:
        """
        Returns the exact length of a current epoch. This number can change between epochs.
        """
        if self.__current_epoch == self.__initial_epoch:
            # for the initial epoch, initial_epoch_sample defines the amount of samples to skip
            return max(0, self.__output_module.length() - self.__initial_epoch_sample)
        else:
            return self.__output_module.length()

    def approximate_length(self) -> int:
        """
        Returns an approximated length of a full epoch.
        The number may not be exact, because the length can change between epochs.
        """
        return max(0, self.__output_module.length())

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
                    module.start(self.__current_epoch)

        self.__last_initialized_epoch = self.__current_epoch

    def get_item(self, index: int) -> dict:
        # for the initial epoch, initial_epoch_sample defines the amount of samples to skip
        if self.__current_epoch == self.__initial_epoch:
            index += self.__initial_epoch_sample

        with torch.no_grad():
            return self.__output_module.get_item(index)
