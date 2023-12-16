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
            target_resolution: int | list[int],
            quantization: int,
            resolution_in_name: str,
            scale_resolution_out_name: str,
            crop_resolution_out_name: str,
            possible_resolutions_out_name: str,
    ):
        super(AspectBucketing, self).__init__()

        self.target_resolutions = [target_resolution] if isinstance(target_resolution, int) else target_resolution
        self.quantization = quantization

        self.resolution_in_name = resolution_in_name

        self.scale_resolution_out_name = scale_resolution_out_name
        self.crop_resolution_out_name = crop_resolution_out_name
        self.possible_resolutions_out_name = possible_resolutions_out_name

        self.possible_resolutions, self.possible_aspects = self.__create_buckets(self.target_resolutions,
                                                                                 self.quantization)
        self.flattened_possible_resolutions = list(set(sum(self.possible_resolutions, [])))
        self.possible_aspects = [np.array(x) for x in self.possible_aspects]

    def length(self) -> int:
        return self._get_previous_length(self.resolution_in_name)

    def get_inputs(self) -> list[str]:
        return [self.resolution_in_name]

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

        possible_resolutions = []
        possible_aspects = []

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
            possible_resolutions.append(new_resolutions)
            possible_aspects.append([h / w for (h, w) in new_resolutions])

        return possible_resolutions, possible_aspects

    def __get_bucket(self, rand: Random, h: int, w: int):
        resolution_index = rand.randint(0, len(self.possible_resolutions) - 1)

        aspect = h / w
        bucket_index = np.argmin(abs(self.possible_aspects[resolution_index] - aspect))
        return self.possible_resolutions[resolution_index][bucket_index]

    def get_meta(self, variation: int, name: str) -> Any:
        if name == self.possible_resolutions_out_name:
            return self.flattened_possible_resolutions
        else:
            return None

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        rand = self._get_rand(variation, index)
        resolution = self._get_previous_item(variation, self.resolution_in_name, index)

        target_resolution = self.__get_bucket(rand, resolution[0], resolution[1])

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
