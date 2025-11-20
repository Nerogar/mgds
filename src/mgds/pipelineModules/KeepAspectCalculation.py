from typing import Any

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class KeepAspectCalculation(
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
            quantization: int,
    ):
        super(KeepAspectCalculation, self).__init__()

        self.resolution_in_name = resolution_in_name

        self.target_resolutions_in_name = target_resolution_in_name
        self.enable_target_resolutions_override_in_name = enable_target_resolutions_override_in_name
        self.target_resolutions_override_in_name = target_resolutions_override_in_name

        self.scale_resolution_out_name = scale_resolution_out_name
        self.crop_resolution_out_name = crop_resolution_out_name

        self.quantization = quantization


    def length(self) -> int:
        return self._get_previous_length(self.resolution_in_name)

    def get_inputs(self) -> list[str]:
        return [self.resolution_in_name]

    def get_outputs(self) -> list[str]:
        return [self.scale_resolution_out_name, self.crop_resolution_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        rand = self._get_rand(variation, index)
        resolution = self._get_previous_item(variation, self.resolution_in_name, index)
        target_resolutions = self._get_previous_item(variation, self.target_resolutions_in_name, index)

        if self.enable_target_resolutions_override_in_name is not None:
            enable_resolution_override = self._get_previous_item(
                variation, self.enable_target_resolutions_override_in_name, index)
            if enable_resolution_override:
                target_resolutions = self._get_previous_item(variation, self.target_resolutions_override_in_name, index)

        target_resolutions = [int(res.strip()) for res in target_resolutions.split(',')]

        target_resolution = rand.choice(target_resolutions)
        s = target_resolution / ((resolution[0] * resolution[1]) ** 0.5)
        s = min(s, 1.0) #only downscale
        scale_resolution = (round(resolution[0] * s), round(resolution[1] * s))

        target_resolution=(
            scale_resolution[0] - (scale_resolution[0] % self.quantization),
            scale_resolution[1] - (scale_resolution[1] % self.quantization)
        )

        print("-----")
        print("input: ", resolution)
        print("scale: ", scale_resolution)
        print("crop: ", target_resolution)

        return {
            self.scale_resolution_out_name: scale_resolution,
            self.crop_resolution_out_name: target_resolution,
        }
