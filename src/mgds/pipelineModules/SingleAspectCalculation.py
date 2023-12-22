from typing import Any

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class SingleAspectCalculation(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            resolution_in_name: str,
            target_resolution_in_name: str,
            enable_target_resolutions_override_in_name: str,
            target_resolutions_override_in_name: str,
            scale_resolution_out_name: str,
            crop_resolution_out_name: str,
            possible_resolutions_out_name: str
    ):
        super(SingleAspectCalculation, self).__init__()

        self.resolution_in_name = resolution_in_name

        self.target_resolutions_in_name = target_resolution_in_name
        self.enable_target_resolutions_override_in_name = enable_target_resolutions_override_in_name
        self.target_resolutions_override_in_name = target_resolutions_override_in_name

        self.scale_resolution_out_name = scale_resolution_out_name
        self.crop_resolution_out_name = crop_resolution_out_name
        self.possible_resolutions_out_name = possible_resolutions_out_name

        self.possible_target_resolutions = []

    def length(self) -> int:
        return self._get_previous_length(self.resolution_in_name)

    def get_inputs(self) -> list[str]:
        return [self.resolution_in_name]

    def get_outputs(self) -> list[str]:
        return [self.scale_resolution_out_name, self.crop_resolution_out_name, self.possible_resolutions_out_name]

    def get_meta(self, variation: int, name: str) -> Any:
        if name == self.possible_resolutions_out_name:
            return [(x, x) for x in self.possible_target_resolutions]
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

        self.possible_target_resolutions = list(possible_target_resolutions)

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
        target_resolution = (target_resolution, target_resolution)

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
