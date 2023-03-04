from abc import abstractmethod, ABCMeta

import torch
from torch.utils.data import DataLoader, Dataset


class PipelineModule(metaclass=ABCMeta):
    pipeline: 'LoadingPipeline'
    module_index: int
    previous_item_cache_index: int
    previous_item_cache: dict
    previous_length_cache: dict

    def __init__(self):
        self.clear_previous_cache()

    def init(self, pipeline: 'LoadingPipeline', module_index: int):
        self.pipeline = pipeline
        self.module_index = module_index

    def clear_previous_cache(self):
        self.previous_item_cache_index = -1
        self.previous_item_cache = {}
        self.previous_length_cache = {}

    def get_previous_item(self, name: str, index: int):
        if self.previous_item_cache_index != index:
            self.clear_previous_cache()
            self.previous_item_cache_index = index

        if name not in self.previous_item_cache:
            for previous_module_index in range(self.module_index - 1, -1, -1):
                module = self.pipeline.modules[previous_module_index]
                if name in module.get_outputs():
                    item = module.get_item(index)
                    self.previous_item_cache[name] = item[name]
                    break

        return self.previous_item_cache[name]

    def get_previous_length(self, name: str):
        if name not in self.previous_length_cache:
            for previous_module_index in range(self.module_index - 1, -1, -1):
                module = self.pipeline.modules[previous_module_index]
                if name in module.get_outputs():
                    self.previous_length_cache[name] = module.length()
                    break

        return self.previous_length_cache[name]

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

    def __init__(self, device: torch.device, concepts: list[dict], modules: list[PipelineModule]):
        self.device = device
        self.concepts = concepts
        self.modules = list(filter(lambda x: x is not None, modules))
        self.cached_length = None

        self.modules.insert(0, ConceptPipelineModule(self.concepts))

        for index, module in enumerate(self.modules):
            module.init(self, index)

    def length(self):
        if not self.cached_length:
            self.cached_length = self.modules[-1].length()

        return self.cached_length

    def start(self):
        """
        Called after initializing the pipeline.
        Can be used to add caching or other logic that should run once.
        """

        for module_index in range(1, len(self.modules)):
            module = self.modules[module_index]
            module.preprocess()

    def get_item(self, index: int) -> dict:
        # At the start of each epoch, the previous cache is cleared.
        # This prevents duplicating samples when training on single images.
        if index == 0:
            for module in self.modules:
                module.clear_previous_cache()

        module_index = len(self.modules) - 1
        last_module = self.modules[module_index]

        return last_module.get_item(index)


class TrainDataSet(Dataset):
    device: torch.device
    loading_pipeline: LoadingPipeline

    def __init__(
            self,
            device: torch.device,
            concepts: [dict],
            definition: [PipelineModule],
    ):
        self.device = device
        self.loading_pipeline = LoadingPipeline(device, concepts, definition)

        self.loading_pipeline.start()

    def __len__(self):
        return self.loading_pipeline.length()

    def __getitem__(self, index):
        return self.loading_pipeline.get_item(index)


class TrainDataLoader(DataLoader):
    def __init__(self, dataset: TrainDataSet, batch_size):
        super(TrainDataLoader, self).__init__(dataset, batch_size=batch_size, drop_last=True)
