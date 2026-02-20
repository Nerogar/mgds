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
            possible_resolutions_out_name: str,
            quantization: int|None = None,
    ):
        super(SingleAspectCalculation, self).__init__()

        self.resolution_in_name = resolution_in_name

        self.target_resolutions_in_name = target_resolution_in_name
        self.enable_target_resolutions_override_in_name = enable_target_resolutions_override_in_name
        self.target_resolutions_override_in_name = target_resolutions_override_in_name

        self.scale_resolution_out_name = scale_resolution_out_name
        self.crop_resolution_out_name = crop_resolution_out_name
        self.possible_resolutions_out_name = possible_resolutions_out_name

        self.possible_target_resolutions: list[tuple[int, int]] = []

        self.quantization = quantization

    def length(self) -> int:
        return self._get_previous_length(self.resolution_in_name)

    def get_inputs(self) -> list[str]:
        return [self.resolution_in_name]

    def get_outputs(self) -> list[str]:
        return [self.scale_resolution_out_name, self.crop_resolution_out_name, self.possible_resolutions_out_name]

    def get_meta(self, variation: int, name: str) -> Any:
        if name == self.possible_resolutions_out_name:
            return self.possible_target_resolutions.copy()
        else:
            return None

    def __quantize_resolution(self, resolution: tuple[int, int], quantization: int|None) -> tuple[int, int]:
        if quantization is None:
            return resolution

        return (
            round(resolution[0] / quantization) * quantization,
            round(resolution[1] / quantization) * quantization,
        )

    def start(self, variation: int):
        possible_target_resolutions: set[tuple[int, int]] = set()

        for index in range(self._get_previous_length(self.target_resolutions_in_name)):
            resolutions = self._get_previous_item(variation, self.target_resolutions_in_name, index)
            if isinstance(resolutions, int):
                possible_target_resolutions.add(
                    self.__quantize_resolution((resolutions, resolutions), self.quantization)
                )
            elif isinstance(resolutions, str):
                if 'x' in resolutions and ',' not in resolutions:
                    res = resolutions.strip().split('x')
                    possible_target_resolutions.add(
                        self.__quantize_resolution(
                            (int(res[1].strip()), int(res[0].strip())), self.quantization
                        )
                    )
                else:
                    for res in resolutions.split(',')
                        res = int(res.strip())
                        possible_target_resolutions.add(
                            self.__quantize_resolution((res, res), self.quantization)
                        )

        if self.target_resolutions_override_in_name is not None:
            for index in range(self._get_previous_length(self.target_resolutions_override_in_name)):
                resolutions = self._get_previous_item(variation, self.target_resolutions_override_in_name, index)
                if isinstance(resolutions, int):
                    possible_target_resolutions.add(
                        self.__quantize_resolution((resolutions, resolutions), self.quantization)
                    )
                elif isinstance(resolutions, str):
                    if 'x' in resolutions and ',' not in resolutions:
                        res = resolutions.strip().split('x')
                        possible_target_resolutions.add(
                            self.__quantize_resolution(
                                (int(res[1].strip()), int(res[0].strip())), self.quantization
                            )
                        )
                    else:
                        for res in resolutions.split(',')
                            res = int(res.strip())
                            possible_target_resolutions.add(
                                self.__quantize_resolution((res, res), self.quantization)
                            )

        self.possible_target_resolutions = list(possible_target_resolutions)

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
            target_resolutions = [int(res.strip())
                                  for res in target_resolutions.split(',')]

            target_resolution = rand.choice(target_resolutions)
            target_resolution = self.__quantize_resolution(
                (target_resolution, target_resolution), self.quantization
            )

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
