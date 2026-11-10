import random

from mgds.ConceptPipelineModule import ConceptPipelineModule
from mgds.LoadingPipeline import LoadingPipeline
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModules.TierAspectBatchSorting import TierAspectBatchSorting
from mgds.SettingsPipelineModule import SettingsPipelineModule

import torch
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data._utils.collate import default_collate


def _find_batch_plan(definition) -> TierAspectBatchSorting | None:
    if isinstance(definition, TierAspectBatchSorting):
        return definition
    if isinstance(definition, list):
        for item in definition:
            found = _find_batch_plan(item)
            if found is not None:
                return found
    return None


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
            batch_size: int, #local batch size
            state: PipelineState,
            seed: int = 42,
            initial_epoch: int = 0,
            initial_epoch_sample: int = 0
    ):
        self.device = device
        self.batch_size = batch_size
        # A TierAspectBatchSorting module in the definition makes the batch
        # size per micro-batch a property of the epoch plan: the dataset then
        # yields pre-collated batches (see __iter__) and the resume offset is
        # kept in samples without rounding to a fixed batch size.
        self.batch_plan: TierAspectBatchSorting | None = _find_batch_plan(definition)
        # Collate used for planned batches; the owning data loader may replace it.
        self.collate_fn = default_collate
        seed = (random.randint(-(1 << 30), 1 << 30) if seed == -1 else seed)
        self.loading_pipeline = LoadingPipeline(
            device,
            [ConceptPipelineModule(concepts), SettingsPipelineModule(settings)] + definition,
            1 if self.batch_plan is not None else batch_size,
            seed,
            state,
            initial_epoch,
            initial_epoch_sample,
        )

    def __iter__(self):
        if self.batch_plan is None:
            return self.loading_pipeline
        return self.__planned_batches()

    def __planned_batches(self):
        """Yield the epoch's micro-batches as collated dicts, each of the size
        the plan assigns. Starts at the plan's batch boundary at (or before)
        the pipeline's resume offset; a partial trailing batch is dropped."""
        plan = self.batch_plan
        pipeline = self.loading_pipeline
        start = pipeline.epoch_start_index()
        batch_index = plan.batch_index_at_sample(start)
        aligned = plan.sample_offset(batch_index)
        if aligned != start:
            pipeline.set_epoch_start_index(aligned)
        for size in plan.micro_batch_sizes[batch_index:]:
            items = []
            for _ in range(size):
                try:
                    items.append(next(pipeline))
                except StopIteration:
                    return
            yield self.collate_fn(items)

    def approximate_length(self) -> int:
        """Batches per epoch (micro-batches under a batch plan)."""
        if self.batch_plan is not None:
            return self.batch_plan.batch_count
        return self.loading_pipeline.approximate_length() // self.batch_size

    def sample_offset_for_batches(self, batches: int) -> int:
        """Samples consumed by the first ``batches`` batches of the current epoch."""
        if self.batch_plan is not None:
            return self.batch_plan.sample_offset(batches)
        return max(0, int(batches)) * self.batch_size

    def start_next_epoch(self):
        self.loading_pipeline.start_next_epoch()


class TrainDataLoader(DataLoader):
    def __init__(self, dataset: MGDS, batch_size):
        if getattr(dataset, "batch_plan", None) is not None:
            # The dataset yields whole batches; auto-batching is off.
            super().__init__(dataset, batch_size=None)
        else:
            super().__init__(dataset, batch_size=batch_size, drop_last=True)
