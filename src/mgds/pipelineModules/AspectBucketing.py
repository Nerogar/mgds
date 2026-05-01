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

    def __quantize_resolution(self, resolution: tuple[float|int, float|int], quantization: int) -> tuple[int, int]:
        return (
            round(resolution[0] / quantization) * quantization,
            round(resolution[1] / quantization) * quantization,
        )

    def __create_automatic_buckets(
            self,
            target_resolutions: list[int],
    ) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[float]]]:

        possible_resolutions = {}
        possible_aspects = {}

        for target_resolution in target_resolutions:
            # normalize to the same pixel count
            new_resolutions = [(
                h / math.sqrt(h * w) * target_resolution,
                w / math.sqrt(h * w) * target_resolution
            ) for (h, w) in self.all_possible_input_aspects]

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
        possible_target_resolutions: set[int]  = set()
        possible_fixed_resolutions: set[tuple[int, int]]  = set()
        possible_frames = {1}

        resolutions_warned_about_rounding: set[tuple[int, int]] = set()
        def _add_resolution_if_new(resolution: int|tuple[int, int], quantization: int):
            resolution_2d = (resolution, resolution) if isinstance(resolution, int) else resolution
            quantized_resolution = self.__quantize_resolution(resolution_2d, quantization)

            # Warn the user if we are rounding their preferred resolution.
            if quantized_resolution != resolution_2d and resolution not in resolutions_warned_about_rounding:
                resolutions_warned_about_rounding.add(resolution)
                print(f'Warning: Resolution {resolution_2d[1]}x{resolution_2d[0]}'
                      f' rounded to {quantized_resolution[1]}x{quantized_resolution[0]}'
                      f' because image model requires multiples of {quantization}.')

            if isinstance(resolution, int):
                # We are a bucketable resolution, store single dimension input value
                possible_target_resolutions.add(resolution)
            else:
                # We are a fixed-dimension resolution, store quantized value
                possible_fixed_resolutions.add(quantized_resolution)

        # Default resolution(s)
        for index in range(self._get_previous_length(self.target_resolutions_in_name)):
            resolutions = self._get_previous_item(variation, self.target_resolutions_in_name, index)
            if 'x' in resolutions and ',' not in resolutions:
                res = resolutions.split('x', 1)
                _add_resolution_if_new((int(res[1].strip()), int(res[0].strip())), self.quantization)
            else:
                for res in resolutions.split(','):
                    _add_resolution_if_new(int(res.strip()), self.quantization)

        # Resolution override(s)
        if (self.target_resolutions_override_in_name is not None and
            self.enable_target_resolutions_override_in_name is not None
        ):
            for index in range(self._get_previous_length(self.target_resolutions_override_in_name)):
                enable_resolution_override = self._get_previous_item(
                    variation, self.enable_target_resolutions_override_in_name, index)
                if enable_resolution_override:
                    override_resolutions = self._get_previous_item(variation, self.target_resolutions_override_in_name, index)
                    if 'x' in override_resolutions and ',' not in override_resolutions:
                        res = override_resolutions.split('x', 1)
                        _add_resolution_if_new((int(res[1].strip()), int(res[0].strip())), self.quantization)
                    else:
                        for res in override_resolutions.split(','):
                            _add_resolution_if_new(int(res.strip()), self.quantization)

        for index in range(self._get_previous_length(self.target_frames_in_name)):
            frames = self._get_previous_item(variation, self.target_frames_in_name, index)
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

        if (self.enable_target_resolutions_override_in_name is not None and 
            self._get_previous_item(variation, self.enable_target_resolutions_override_in_name, index)
        ):
            # Use override resolution(s)
            target_resolutions = self._get_previous_item(variation, self.target_resolutions_override_in_name, index)
        else:
            # Use base resolution(s)
            target_resolutions = self._get_previous_item(variation, self.target_resolutions_in_name, index)

        if 'x' in target_resolutions and ',' not in target_resolutions:
            # Get quantized resolution from a fixed resolution
            res = target_resolutions.split('x', 1)
            target_resolution = self.__quantize_resolution(
                (int(res[1].strip()), int(res[0].strip())), self.quantization
            )
        else:
            # Get quantized resolution bucket from a random single-dim resolution
            target_resolution_list = [int(res.strip())
                                      for res in target_resolutions.split(',')]
            random_resolution = rand.choice(target_resolution_list)
            target_resolution = self.__get_bucket(rand, resolution[-2], resolution[-1], random_resolution)

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
