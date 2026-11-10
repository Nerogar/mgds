"""Aspect-bucket batch sorting with a micro-batch size and accumulation per
resolution tier.

``AspectBatchSorting`` cuts every bucket into batches of one size. With
adaptive resolution a run trains several tiers at once (1536 next to 768),
and one batch size cannot fit the big tier and fill the GPU on the small one.
This module gives every tier its own ``micro_batch_size x accumulation_steps``
and lays the epoch out in *windows*: one window is one optimizer step, all of
its micro-batches come from one bucket (so from one tier), and the windows
are shuffled -- the per-tier twin of the "one kind per optimizer step" rule.

The plan (window order, micro-batch sizes, sample offsets) is a pure function
of the bucket contents and the per-epoch seed, so it is deterministic and a
resume can be wound to any batch boundary (see ``sample_offset``). ``MGDS``
reads the plan to yield pre-collated variable-size batches.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Mapping

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.SingleVariationRandomAccessPipelineModule import SingleVariationRandomAccessPipelineModule


def tier_for_resolution(resolution, tiers) -> int:
    """The tier (nominal edge, e.g. 1024) a bucket ``(h, w)`` belongs to:
    the nearest tier to ``sqrt(h * w)``. Quantized buckets stray from their
    tier's nominal size by a few percent at most (the 768 tier holds 640x960,
    784 nominal) while neighbouring tiers are 25 % apart, so nearest is exact
    for every ladder in use; ``tier_collisions`` in OneTrainer checks a ladder."""
    nominal = math.sqrt(int(resolution[-2]) * int(resolution[-1]))
    return min(tiers, key=lambda tier: (abs(tier - nominal), tier))


class TierAspectBatchSorting(
    PipelineModule,
    SingleVariationRandomAccessPipelineModule,
):
    def __init__(
        self,
        resolution_in_name: str,
        names: list[str],
        tier_batches: Mapping[int, tuple[int, int]],
    ):
        """``tier_batches`` maps a tier to ``(micro_batch_size, accumulation_steps)``.
        Every tier a bucket can land on must be present (see ``tier_for_resolution``)."""
        super().__init__()
        self.resolution_in_name = resolution_in_name
        self.names = names
        self.tier_batches: dict[int, tuple[int, int]] = {}
        for tier, (batch_size, accumulation) in dict(tier_batches).items():
            tier, batch_size, accumulation = int(tier), int(batch_size), int(accumulation)
            if tier <= 0 or batch_size <= 0 or accumulation <= 0:
                raise ValueError(f"tier batch table entry {tier}: {batch_size}x{accumulation} must be positive")
            self.tier_batches[tier] = (batch_size, accumulation)
        if not self.tier_batches:
            raise ValueError("tier batch table is empty")
        self.tiers = sorted(self.tier_batches)

        self.bucket_dict: dict = {}
        self.index_list: list[int] = []
        # One entry per optimizer step: (micro_batch_size, micro_batch_count).
        self.windows: list[tuple[int, int]] = []
        # The tier of each window, parallel to ``windows``.
        self.window_tiers: list[int] = []
        # One entry per micro-batch, in epoch order.
        self.micro_batch_sizes: list[int] = []
        # Cumulative sample offsets; len == batch_count + 1.
        self.batch_offsets: list[int] = [0]
        # tier -> {"samples", "windows", "dropped", "batch_size", "accumulation"}
        self.tier_stats: dict[int, dict[str, int]] = {}

    def length(self) -> int:
        return len(self.index_list)

    def get_inputs(self) -> list[str]:
        return [self.resolution_in_name] + self.names

    def get_outputs(self) -> list[str]:
        return self.names

    # -- plan accessors -----------------------------------------------------

    @property
    def batch_count(self) -> int:
        return len(self.micro_batch_sizes)

    @property
    def window_count(self) -> int:
        return len(self.windows)

    def sample_offset(self, batch_index: int) -> int:
        """Samples before micro-batch ``batch_index`` (clamped to the plan)."""
        return self.batch_offsets[min(max(0, int(batch_index)), self.batch_count)]

    def batch_index_at_sample(self, sample_offset: int) -> int:
        """The micro-batch that starts at or before ``sample_offset``."""
        return min(max(0, bisect_right(self.batch_offsets, int(sample_offset)) - 1), self.batch_count)

    def window_batch_size(self, resolution) -> tuple[int, int]:
        return self.tier_batches[tier_for_resolution(resolution, self.tiers)]

    def describe(self) -> str:
        parts = []
        for tier in sorted(self.tier_stats):
            stats = self.tier_stats[tier]
            parts.append(
                f"{tier}: {stats['samples']} samples -> {stats['windows']} steps of "
                f"{stats['batch_size']}x{stats['accumulation']} (dropped {stats['dropped']})"
            )
        return "; ".join(parts) if parts else "no buckets"

    # -- epoch layout ---------------------------------------------------------

    def __sort_resolutions(self):
        self.bucket_dict = {}
        for index in range(self._get_previous_length(self.resolution_in_name)):
            resolution = self._get_previous_item(self.current_variation, self.resolution_in_name, index)
            self.bucket_dict.setdefault(resolution, []).append(index)

    def __layout(self, variation: int) -> list[int]:
        rand = self._get_rand(variation)
        bucket_dict = {key: value.copy() for key, value in self.bucket_dict.items()}

        # Shuffled list of windows as (resolution, window index within the bucket).
        windows = []
        window_samples = {}
        for bucket_key, bucket in bucket_dict.items():
            batch_size, accumulation = self.window_batch_size(bucket_key)
            window_samples[bucket_key] = batch_size * accumulation
            windows.extend((bucket_key, i) for i in range(len(bucket) // window_samples[bucket_key]))
        rand.shuffle(windows)

        for bucket in bucket_dict.values():
            rand.shuffle(bucket)

        self.tier_stats = {}
        for bucket_key, bucket in bucket_dict.items():
            batch_size, accumulation = self.window_batch_size(bucket_key)
            tier = tier_for_resolution(bucket_key, self.tiers)
            stats = self.tier_stats.setdefault(
                tier,
                {"samples": 0, "windows": 0, "dropped": 0, "batch_size": batch_size, "accumulation": accumulation},
            )
            complete = len(bucket) // window_samples[bucket_key]
            stats["samples"] += len(bucket)
            stats["windows"] += complete
            stats["dropped"] += len(bucket) - complete * window_samples[bucket_key]
            del bucket[complete * window_samples[bucket_key] :]

        index_list = []
        self.windows = []
        self.window_tiers = []
        self.micro_batch_sizes = []
        self.batch_offsets = [0]
        for bucket_key, window_index in windows:
            batch_size, accumulation = self.window_batch_size(bucket_key)
            samples = window_samples[bucket_key]
            index_list.extend(bucket_dict[bucket_key][window_index * samples : (window_index + 1) * samples])
            self.windows.append((batch_size, accumulation))
            self.window_tiers.append(tier_for_resolution(bucket_key, self.tiers))
            for _ in range(accumulation):
                self.micro_batch_sizes.append(batch_size)
                self.batch_offsets.append(self.batch_offsets[-1] + batch_size)
        return index_list

    def start(self, variation: int):
        self.__sort_resolutions()
        self.index_list = self.__layout(variation)

    def get_item(self, index: int, requested_name: str = None) -> dict:
        index = self.index_list[index]
        return {name: self._get_previous_item(self.current_variation, name, index) for name in self.names}
