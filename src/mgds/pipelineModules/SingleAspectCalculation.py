from typing import Any

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class SingleAspectCalculation(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            target_resolution: int | list[int],
            resolution_in_name: str,
            scale_resolution_out_name: str,
            crop_resolution_out_name: str,
            possible_resolutions_out_name: str
    ):
        super(SingleAspectCalculation, self).__init__()

        self.target_resolutions = [target_resolution] if isinstance(target_resolution, int) else target_resolution

        self.resolution_in_name = resolution_in_name

        self.scale_resolution_out_name = scale_resolution_out_name
        self.crop_resolution_out_name = crop_resolution_out_name
        self.possible_resolutions_out_name = possible_resolutions_out_name

    def length(self) -> int:
        return self._get_previous_length(self.resolution_in_name)

    def get_inputs(self) -> list[str]:
        return [self.resolution_in_name]

    def get_outputs(self) -> list[str]:
        return [self.scale_resolution_out_name, self.crop_resolution_out_name, self.possible_resolutions_out_name]

    def get_meta(self, variation: int, name: str) -> Any:
        if name == self.possible_resolutions_out_name:
            return [(x, x) for x in self.target_resolutions]
        else:
            return None

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        rand = self._get_rand(variation, index)
        resolution = self._get_previous_item(variation, self.resolution_in_name, index)

        resolution_index = rand.randint(0, len(self.target_resolutions) - 1)
        target_resolution = (self.target_resolutions[resolution_index], self.target_resolutions[resolution_index])

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
