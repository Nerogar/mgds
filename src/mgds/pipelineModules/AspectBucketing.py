import itertools
import math
from random import Random
from typing import Any

import numpy as np

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class AspectBucketing(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            quantization: int,
            resolution_in_name: str,
            target_resolution_in_name: str,
            enable_target_resolutions_override_in_name: str,
            target_resolutions_override_in_name: str,
            target_frames_in_name: str,
            frame_dim_enabled: bool,
            scale_resolution_out_name: str,
            crop_resolution_out_name: str,
            possible_resolutions_out_name: str,
    ):
        super(AspectBucketing, self).__init__()

        self.quantization = quantization
        self.resolution_in_name = resolution_in_name

        self.target_resolutions_in_name = target_resolution_in_name
        self.enable_target_resolutions_override_in_name = enable_target_resolutions_override_in_name
        self.target_resolutions_override_in_name = target_resolutions_override_in_name
        self.target_frames_in_name = target_frames_in_name
        self.frame_dim_enabled = frame_dim_enabled

        self.scale_resolution_out_name = scale_resolution_out_name
        self.crop_resolution_out_name = crop_resolution_out_name
        self.possible_resolutions_out_name = possible_resolutions_out_name

        self.bucket_resolutions = {}
        self.bucket_aspects = {}
        self.flattened_possible_resolutions = []

    def length(self) -> int:
        return self._get_previous_length(self.resolution_in_name)

    def get_inputs(self) -> list[str]:
        return [
            self.resolution_in_name,
            self.target_resolutions_in_name,
            self.enable_target_resolutions_override_in_name,
            self.target_resolutions_override_in_name,
            self.target_frames_in_name,
        ]

    def get_outputs(self) -> list[str]:
        return [self.scale_resolution_out_name, self.crop_resolution_out_name, self.possible_resolutions_out_name]

    def __quantize_resolution(self, resolution: tuple[float, float], quantization: int) -> tuple[int, int]:
        return (
            round(resolution[0] / quantization) * quantization,
            round(resolution[1] / quantization) * quantization,
        )

    def __create_automatic_buckets(
            self,
            target_resolutions: list[int],
    ) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[float]]]:
        # all possible target aspect ratios
        all_possible_input_aspects = [
            (1.0, 1.0),
            (1.0, 1.25),
            (1.0, 1.5),
            (1.0, 1.75),
            (1.0, 2.0),
            (1.0, 2.5),
            (1.0, 3.0),
            (1.0, 3.5),
            (1.0, 4.0),
        ]

        possible_resolutions = {}
        possible_aspects = {}

        for target_resolution in target_resolutions:
            # normalize to the same pixel count
            new_resolutions = [(
                h / math.sqrt(h * w) * target_resolution,
                w / math.sqrt(h * w) * target_resolution
            ) for (h, w) in all_possible_input_aspects]

            # add inverted dimensions
            new_resolutions = new_resolutions + [(w, h) for (h, w) in new_resolutions]

            # quantization
            new_resolutions = [self.__quantize_resolution(resolution, self.quantization) for resolution in
                               new_resolutions]

            # remove duplicates
            new_resolutions = list(set(new_resolutions))

            # add to lists
            possible_resolutions[target_resolution] = new_resolutions
            possible_aspects[target_resolution] = [h / w for (h, w) in new_resolutions]

        return possible_resolutions, possible_aspects

    def __get_bucket(self, rand: Random, h: int, w: int, target_resolution: int) -> tuple[int, int]:
        aspect = h / w
        bucket_index = np.argmin(abs(self.bucket_aspects[target_resolution] - aspect))
        return self.bucket_resolutions[target_resolution][bucket_index]

    def get_meta(self, variation: int, name: str) -> Any:
        if name == self.possible_resolutions_out_name:
            return self.flattened_possible_resolutions
        else:
            return None

    def start(self, variation: int):
        possible_target_resolutions = set()
        possible_fixed_resolutions = set()
        possible_frames = {1}

        for index in range(self._get_previous_length(self.target_resolutions_in_name)):
            resolutions = self._get_previous_item(variation, self.target_resolutions_in_name, index)
            if 'x' in resolutions and ',' not in resolutions:
                res = resolutions.strip().split('x')
                possible_fixed_resolutions.add(
                    self.__quantize_resolution(
                        (int(res[1]), int(res[0])), self.quantization
                    )
                )
            else:
                possible_target_resolutions |= set([int(res.strip()) for res in resolutions.split(',')])

        if self.target_resolutions_override_in_name is not None:
            for index in range(self._get_previous_length(self.target_resolutions_override_in_name)):
                resolutions = self._get_previous_item(variation, self.target_resolutions_override_in_name, index)
                if 'x' in resolutions and ',' not in resolutions:
                    res = resolutions.strip().split('x')
                    possible_fixed_resolutions.add(
                        self.__quantize_resolution(
                            (int(res[1]), int(res[0])), self.quantization
                        )
                    )
                else:
                    possible_target_resolutions |= set([int(res.strip()) for res in resolutions.split(',')])

        for index in range(self._get_previous_length(self.target_frames_in_name)):
            frames = self._get_previous_item(variation, self.target_resolutions_override_in_name, index)
            possible_frames.add(int(frames))

        self.bucket_resolutions, self.bucket_aspects = \
            self.__create_automatic_buckets(list(possible_target_resolutions))

        self.flattened_possible_resolutions = list(
            set(sum(self.bucket_resolutions.values(), [])) | possible_fixed_resolutions
        )
        if self.frame_dim_enabled:
            self.flattened_possible_resolutions = \
                list((f, *r) for f, r in itertools.product(possible_frames, self.flattened_possible_resolutions))

        self.bucket_aspects = {k: np.array(v) for (k, v) in self.bucket_aspects.items()}

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        rand = self._get_rand(variation, index)
        resolution = self._get_previous_item(variation, self.resolution_in_name, index)
        target_resolutions = self._get_previous_item(variation, self.target_resolutions_in_name, index)

        if self.enable_target_resolutions_override_in_name is not None:
            enable_resolution_override = self._get_previous_item(
                variation, self.enable_target_resolutions_override_in_name, index)
            if enable_resolution_override:
                target_resolutions = self._get_previous_item(variation, self.target_resolutions_override_in_name, index)

        if 'x' in target_resolutions and ',' not in target_resolutions:
            res = target_resolutions.strip().split('x')
            target_resolution = self.__quantize_resolution(
                (int(res[1]), int(res[0])), self.quantization
            )
        else:
            target_resolutions = [int(res.strip()) for res in target_resolutions.split(',')]

            target_resolution = rand.choice(target_resolutions)
            target_resolution = self.__get_bucket(rand, resolution[-2], resolution[-1], target_resolution)

        aspect = resolution[-2] / resolution[-1]
        target_aspect = target_resolution[-2] / target_resolution[-1]

        if aspect > target_aspect:
            scale = target_resolution[-1] / resolution[-1]
            scale_resolution = (
                *resolution[:-2],
                round(resolution[-2] * scale),
                target_resolution[-1]
            )
        else:
            scale = target_resolution[-2] / resolution[-2]
            scale_resolution = (
                *resolution[:-2],
                target_resolution[-2],
                round(resolution[-1] * scale)
            )

        target_resolution = (
            *resolution[:-2],
            *target_resolution
        )

        return {
            self.scale_resolution_out_name: scale_resolution,
            self.crop_resolution_out_name: target_resolution,
        }
