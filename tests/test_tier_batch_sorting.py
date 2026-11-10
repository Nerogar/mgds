"""TierAspectBatchSorting: per-tier micro-batch x accumulation windows, one tier
per window, deterministic layout, and MGDS yielding the planned batches with an
exact mid-epoch resume."""

from random import Random

from mgds.MGDS import MGDS, TrainDataLoader
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModules.TierAspectBatchSorting import TierAspectBatchSorting, tier_for_resolution
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch

import pytest

TABLE = {768: (4, 1), 1024: (2, 2), 1536: (1, 4)}


class SortingHarness(TierAspectBatchSorting):
    def __init__(self, resolutions, table=TABLE):
        super().__init__(resolution_in_name="crop_resolution", names=["row"], tier_batches=table)
        self.resolutions = resolutions
        self.current_variation = 0

    def _get_previous_length(self, name):
        return len(self.resolutions)

    def _get_previous_item(self, variation, name, index):
        if name == "crop_resolution":
            return self.resolutions[index]
        return index

    def _get_rand(self, variation, index=None):
        return Random(variation)


def _windows_of(sorting):
    """(resolution, [sample indices]) per planned window, in epoch order."""
    out = []
    position = 0
    for batch_size, accumulation in sorting.windows:
        rows = sorting.index_list[position : position + batch_size * accumulation]
        position += batch_size * accumulation
        out.append(({sorting.resolutions[row] for row in rows}, rows))
    return out


def test_tier_for_resolution_takes_the_nearest_tier():
    tiers = [768, 1024, 1280, 1536]
    assert tier_for_resolution((768, 768), tiers) == 768
    assert tier_for_resolution((640, 960), tiers) == 768  # 784 nominal, the 768 tier's 2:3 bucket
    assert tier_for_resolution((832, 1216), tiers) == 1024
    assert tier_for_resolution((1280, 1280), tiers) == 1280
    assert tier_for_resolution((1216, 1856), tiers) == 1536


def test_every_window_is_one_bucket_with_its_tiers_batch_shape():
    resolutions = [(768, 768)] * 10 + [(1024, 1024)] * 9 + [(1536, 1536)] * 5 + [(1216, 832)] * 4
    sorting = SortingHarness(resolutions)

    sorting.start(0)

    # 768: 10 // 4 = 2 windows (8 rows, 2 dropped); 1024x1024: 9 // 4 = 2 windows (8 rows);
    # 1536: 5 // 4 = 1 window (4 rows); 1216x832 (1024 tier): 4 // 4 = 1 window.
    assert sorting.window_count == 6
    assert sorting.length() == 8 + 8 + 4 + 4
    assert sorting.batch_count == 2 * 1 + 2 * 2 + 1 * 4 + 1 * 2
    assert sorting.batch_offsets[-1] == sorting.length()
    assert sum(sorting.micro_batch_sizes) == sorting.length()
    assert len(sorting.window_tiers) == sorting.window_count
    for (resolutions_in_window, rows), tier in zip(_windows_of(sorting), sorting.window_tiers, strict=True):
        assert len(resolutions_in_window) == 1
        (resolution,) = resolutions_in_window
        assert tier == tier_for_resolution(resolution, sorting.tiers)
        batch_size, accumulation = TABLE[tier]
        assert len(rows) == batch_size * accumulation
    stats = sorting.tier_stats
    assert stats[768] == {"samples": 10, "windows": 2, "dropped": 2, "batch_size": 4, "accumulation": 1}
    assert stats[1024]["samples"] == 13 and stats[1024]["windows"] == 3 and stats[1024]["dropped"] == 1
    assert stats[1536] == {"samples": 5, "windows": 1, "dropped": 1, "batch_size": 1, "accumulation": 4}


def test_layout_is_deterministic_per_epoch_and_differs_between_epochs():
    resolutions = [(768, 768)] * 12 + [(1024, 1024)] * 8
    first, second, third = SortingHarness(resolutions), SortingHarness(resolutions), SortingHarness(resolutions)
    first.start(0)
    second.start(0)
    third.start(1)
    assert first.index_list == second.index_list
    assert first.windows == second.windows
    assert first.index_list != third.index_list


def test_sample_offsets_round_trip_batch_indices():
    sorting = SortingHarness([(768, 768)] * 8 + [(1024, 1024)] * 8 + [(1536, 1536)] * 4)
    sorting.start(3)
    for batch_index in range(sorting.batch_count + 1):
        offset = sorting.sample_offset(batch_index)
        assert sorting.batch_index_at_sample(offset) == batch_index
    # An offset inside a batch maps to the batch that contains it.
    assert sorting.batch_index_at_sample(sorting.sample_offset(2) + 0) == 2
    if sorting.micro_batch_sizes[2] > 1:
        assert sorting.batch_index_at_sample(sorting.sample_offset(2) + 1) == 2


def test_rejects_an_empty_or_non_positive_table():
    with pytest.raises(ValueError):
        TierAspectBatchSorting("crop_resolution", ["row"], {})
    with pytest.raises(ValueError):
        TierAspectBatchSorting("crop_resolution", ["row"], {1024: (0, 1)})


class _Rows(PipelineModule, RandomAccessPipelineModule):
    """A source of rows with a latent whose shape follows its bucket."""

    def __init__(self, resolutions):
        super().__init__()
        self.resolutions = resolutions

    def length(self):
        return len(self.resolutions)

    def get_inputs(self):
        return []

    def get_outputs(self):
        return ["crop_resolution", "latent", "row"]

    def get_item(self, variation, index, requested_name=None):
        h, w = self.resolutions[index]
        return {"crop_resolution": (h, w), "latent": torch.zeros(1, h // 64, w // 64), "row": index}


def _dataset(resolutions, initial_epoch_sample=0, initial_epoch=0):
    definition = [
        _Rows(resolutions),
        TierAspectBatchSorting("crop_resolution", ["latent", "row"], TABLE),
        OutputPipelineModule(["latent", "row"]),
    ]
    return MGDS(
        torch.device("cpu"),
        [],
        {},
        definition,
        batch_size=4,
        state=PipelineState(1),
        seed=7,
        initial_epoch=initial_epoch,
        initial_epoch_sample=initial_epoch_sample,
    )


def test_mgds_yields_the_planned_variable_size_batches():
    resolutions = [(768, 768)] * 9 + [(1024, 1024)] * 7 + [(1536, 1536)] * 4
    ds = _dataset(resolutions)
    ds.start_next_epoch()
    plan = ds.batch_plan
    assert plan is not None
    assert ds.approximate_length() == plan.batch_count

    batches = list(TrainDataLoader(ds, 4))

    assert [len(batch["row"]) for batch in batches] == plan.micro_batch_sizes
    edge_for_size = {4: 768 // 64, 2: 1024 // 64, 1: 1536 // 64}
    for batch, size in zip(batches, plan.micro_batch_sizes, strict=True):
        assert tuple(batch["latent"].shape) == (size, 1, edge_for_size[size], edge_for_size[size])
    rows = [int(row) for batch in batches for row in batch["row"]]
    assert rows == plan.index_list


def test_mgds_resumes_at_a_planned_batch_boundary():
    resolutions = [(768, 768)] * 9 + [(1024, 1024)] * 7 + [(1536, 1536)] * 4
    full = _dataset(resolutions)
    full.start_next_epoch()
    all_batches = list(TrainDataLoader(full, 4))
    plan = full.batch_plan

    resume_batch = 3
    resumed = _dataset(resolutions, initial_epoch_sample=plan.sample_offset(resume_batch))
    resumed.start_next_epoch()
    tail = list(TrainDataLoader(resumed, 4))

    assert resumed.batch_plan.index_list == plan.index_list
    assert [[int(r) for r in b["row"]] for b in tail] == [
        [int(r) for r in b["row"]] for b in all_batches[resume_batch:]
    ]
    assert resumed.sample_offset_for_batches(resume_batch) == plan.sample_offset(resume_batch)


def test_mgds_rounds_a_misaligned_resume_offset_down_to_a_batch_boundary():
    resolutions = [(768, 768)] * 9 + [(1024, 1024)] * 7 + [(1536, 1536)] * 4
    full = _dataset(resolutions)
    full.start_next_epoch()
    all_batches = list(TrainDataLoader(full, 4))
    plan = full.batch_plan
    # Pick a batch of more than one sample and start one sample into it.
    resume_batch = next(index for index, size in enumerate(plan.micro_batch_sizes) if size > 1)

    resumed = _dataset(resolutions, initial_epoch_sample=plan.sample_offset(resume_batch) + 1)
    resumed.start_next_epoch()
    tail = list(TrainDataLoader(resumed, 4))

    assert [[int(r) for r in b["row"]] for b in tail] == [
        [int(r) for r in b["row"]] for b in all_batches[resume_batch:]
    ]
