import random
from abc import abstractmethod, ABCMeta
from random import Random

import torch
from torch.utils.data import DataLoader, Dataset


class PipelineModule(metaclass=ABCMeta):
    pipeline: 'LoadingPipeline'
    base_seed: int
    module_index: int

    item_cache_index: int
    item_cache: dict
    length_cache: int

    def __init__(self):
        self.clear_item_cache()

    def init(self, pipeline: 'LoadingPipeline', base_seed: int, module_index: int):
        self.pipeline = pipeline
        self.base_seed = base_seed
        self.module_index = module_index

    def clear_item_cache(self):
        self.item_cache_index = -1
        self.item_cache = {}
        self.length_cache = -1

    def get_previous_item(self, name: str, index: int):
        for previous_module_index in range(self.module_index - 1, -1, -1):
            module = self.pipeline.modules[previous_module_index]
            if name in module.get_outputs():
                if module.item_cache_index == index:
                    return module.item_cache[name]
                else:
                    item = module.get_item(index)
                    module.item_cache_index = index
                    module.item_cache = item
                    return item[name]

    def get_previous_length(self, name: str):
        for previous_module_index in range(self.module_index - 1, -1, -1):
            module = self.pipeline.modules[previous_module_index]
            if name in module.get_outputs():
                if module.length_cache < 0:
                    module.length_cache = module.length()
                return module.length_cache

    def get_rand(self, index: int = -1) -> Random:
        seed = hash((self.base_seed, self.module_index, self.pipeline.current_epoch, index))
        return Random(seed)

    @abstractmethod
    def length(self) -> int:
        pass

    @abstractmethod
    def get_inputs(self) -> list[str]:
        pass

    @abstractmethod
    def get_outputs(self) -> list[str]:
        pass

    def preprocess(self):
        pass

    @abstractmethod
    def get_item(self, index: int) -> dict:
        pass


class ConceptPipelineModule(PipelineModule):
    def __init__(self, concepts: list[dict]):
        super(ConceptPipelineModule, self).__init__()
        self.concepts = concepts

    def length(self) -> int:
        return len(self.concepts)

    def get_inputs(self) -> list[str]:
        return []

    def get_outputs(self) -> list[str]:
        return ['concept']

    def get_item(self, index: int) -> dict:
        return {
            'concept': self.concepts[index]
        }


class LoadingPipeline:
    device: torch.device
    concepts: list[dict]
    modules: list[PipelineModule]
    current_epoch: int

    def __init__(self, device: torch.device, concepts: list[dict], modules: list, seed: int):
        self.device = device
        self.concepts = concepts
        self.modules = list(filter(lambda x: x is not None, self.__flatten(modules)))
        self.cached_length = None

        self.modules.insert(0, ConceptPipelineModule(self.concepts))
        for index, module in enumerate(self.modules):
            module.init(self, seed, index)

        self.current_epoch = -1

    def __flatten(self, data: list | object) -> list:
        if isinstance(data, list):
            new_list = []
            for x in [self.__flatten(x) for x in data]:
                new_list.extend(x)
            return new_list
        else:
            return [data]

    def length(self) -> int:
        if not self.cached_length:
            self.cached_length = self.modules[-1].length()

        return self.cached_length

    def start(self):
        """
        Called after initializing the pipeline.
        Can be used to add caching or other logic that should run once.
        """

        for module_index in range(len(self.modules)):
            module = self.modules[module_index]
            module.preprocess()

    def get_item(self, index: int) -> dict:

        module_index = len(self.modules) - 1
        last_module = self.modules[module_index]

        return last_module.get_item(index)

    def start_next_epoch(self):
        self.current_epoch += 1

        # At the start of each epoch, the previous cache is cleared.
        # This prevents duplicating samples when training on single images.
        for module in self.modules:
            module.clear_item_cache()


class TrainDataSet(Dataset):
    device: torch.device
    loading_pipeline: LoadingPipeline

    def __init__(
            self,
            device: torch.device,
            concepts: [dict],
            definition: [PipelineModule],
            seed: int = 42
    ):
        self.device = device
        seed = (random.randint(-((1 << 30) - 1), (1 << 30) - 1) if seed == -1 else seed)
        self.loading_pipeline = LoadingPipeline(device, concepts, definition, seed=seed)

        self.loading_pipeline.start()

    def __len__(self):
        return self.loading_pipeline.length()

    def __getitem__(self, index):
        return self.loading_pipeline.get_item(index)

    def start_next_epoch(self):
        self.loading_pipeline.start_next_epoch()


class TrainDataLoader(DataLoader):
    def __init__(self, dataset: TrainDataSet, batch_size):
        super(TrainDataLoader, self).__init__(dataset, batch_size=batch_size, drop_last=True)
