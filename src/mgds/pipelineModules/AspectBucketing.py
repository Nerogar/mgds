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

        self.scale_resolution_out_name = scale_resolution_out_name
        self.crop_resolution_out_name = crop_resolution_out_name
        self.possible_resolutions_out_name = possible_resolutions_out_name

        self.possible_resolutions = {}
        self.possible_aspects = {}
        self.flattened_possible_resolutions = []

    def length(self) -> int:
        return self._get_previous_length(self.resolution_in_name)

    def get_inputs(self) -> list[str]:
        return [
            self.resolution_in_name,
            self.target_resolutions_in_name,
            self.enable_target_resolutions_override_in_name,
            self.target_resolutions_override_in_name,
        ]

    def get_outputs(self) -> list[str]:
        return [self.scale_resolution_out_name, self.crop_resolution_out_name, self.possible_resolutions_out_name]

    @staticmethod
    def __create_buckets(target_resolutions: list[int], quantization: int) -> (np.ndarray, np.ndarray):
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
            new_resolutions = [(
                round(h / quantization) * quantization,
                round(w / quantization) * quantization,
            ) for (h, w) in new_resolutions]

            # remove duplicates
            new_resolutions = list(set(new_resolutions))

            # add to lists
            possible_resolutions[target_resolution] = new_resolutions
            possible_aspects[target_resolution] = [h / w for (h, w) in new_resolutions]

        return possible_resolutions, possible_aspects

    def __get_bucket(self, rand: Random, h: int, w: int, target_resolution: int):
        aspect = h / w
        bucket_index = np.argmin(abs(self.possible_aspects[target_resolution] - aspect))
        return self.possible_resolutions[target_resolution][bucket_index]

    def get_meta(self, variation: int, name: str) -> Any:
        if name == self.possible_resolutions_out_name:
            return self.flattened_possible_resolutions
        else:
            return None

    def start(self, variation: int):
        possible_target_resolutions = set()

        for index in range(self._get_previous_length(self.target_resolutions_in_name)):
            resolutions = self._get_previous_item(variation, self.target_resolutions_in_name, index)
            if isinstance(resolutions, int):
                possible_target_resolutions.add(resolutions)
            elif isinstance(resolutions, str):
                possible_target_resolutions |= set([int(res.strip()) for res in resolutions.split(',')])

        if self.target_resolutions_override_in_name is not None:
            for index in range(self._get_previous_length(self.target_resolutions_override_in_name)):
                resolutions = self._get_previous_item(variation, self.target_resolutions_override_in_name, index)
                if isinstance(resolutions, int):
                    possible_target_resolutions.add(resolutions)
                elif isinstance(resolutions, str):
                    possible_target_resolutions |= set([int(res.strip()) for res in resolutions.split(',')])

        self.possible_resolutions, self.possible_aspects = \
            self.__create_buckets(list(possible_target_resolutions), self.quantization)
        self.flattened_possible_resolutions = list(set(sum(self.possible_resolutions.values(), [])))
        self.possible_aspects = {k: np.array(v) for (k, v) in self.possible_aspects.items()}

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        rand = self._get_rand(variation, index)
        resolution = self._get_previous_item(variation, self.resolution_in_name, index)
        target_resolutions = self._get_previous_item(variation, self.target_resolutions_in_name, index)

        if self.enable_target_resolutions_override_in_name is not None:
            enable_resolution_override = self._get_previous_item(
                variation, self.enable_target_resolutions_override_in_name, index)
            target_resolutions = self._get_previous_item(variation, self.target_resolutions_override_in_name, index)
            if enable_resolution_override:
                target_resolutions = [int(res.strip()) for res in target_resolutions.split(',')]

        target_resolution = rand.choice(target_resolutions)
        target_resolution = self.__get_bucket(rand, resolution[0], resolution[1], target_resolution)

        aspect = resolution[0] / resolution[1]
        target_aspect = target_resolution[0] / target_resolution[1]

        if aspect > target_aspect:
            scale = target_resolution[1] / resolution[1]
            scale_resolution = (
                round(resolution[0] * scale),
                target_resolution[1]
            )
        else:
            scale = target_resolution[0] / resolution[0]
            scale_resolution = (
                target_resolution[0],
                round(resolution[1] * scale)
            )

        return {
            self.scale_resolution_out_name: scale_resolution,
            self.crop_resolution_out_name: target_resolution,
        }
