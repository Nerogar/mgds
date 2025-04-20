import random

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from mgds.ConceptPipelineModule import ConceptPipelineModule
from mgds.LoadingPipeline import LoadingPipeline
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.SettingsPipelineModule import SettingsPipelineModule


class MGDS(IterableDataset):
    device: torch.device
    batch_size: int
    loading_pipeline: LoadingPipeline

    def __init__(
            self,
            device: torch.device,
            concepts: list[dict],
            settings: dict,
            definition: [PipelineModule],
            batch_size: int, #global batch size
            state: PipelineState,
            seed: int = 42,
            initial_epoch: int = 0,
            initial_epoch_sample: int = 0
    ):
        self.device = device
        self.batch_size = batch_size
        seed = (random.randint(-(1 << 30), 1 << 30) if seed == -1 else seed)
        self.loading_pipeline = LoadingPipeline(
            device,
            [ConceptPipelineModule(concepts), SettingsPipelineModule(settings)] + definition,
            batch_size,
            seed,
            state,
            initial_epoch,
            initial_epoch_sample,
        )

    def __iter__(self):
        return self.loading_pipeline

    def approximate_length(self) -> int:
        return self.loading_pipeline.approximate_length() // self.batch_size

    def start_next_epoch(self):
        self.loading_pipeline.start_next_epoch()


class TrainDataLoader(DataLoader):
    def __init__(self, dataset: MGDS, batch_size):
        super(TrainDataLoader, self).__init__(dataset, batch_size=batch_size, drop_last=True)
