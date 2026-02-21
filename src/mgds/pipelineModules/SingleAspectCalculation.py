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

        self.quantization = quantization if quantization is not None else 1

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

    def __quantize_resolution(self,
                              resolution: int|tuple[int, int],
                              quantization: int) -> tuple[int, int]:
        if isinstance(resolution, int):
            resolution = (resolution, resolution)

        if quantization == 1:
            return resolution

        return (
            round(resolution[0] / quantization) * quantization,
            round(resolution[1] / quantization) * quantization,
        )

    def start(self, variation: int):
        possible_target_resolutions: set[tuple[int, int]] = set()
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

            possible_target_resolutions.add(quantized_resolution)

        # Default resolution
        for index in range(self._get_previous_length(self.target_resolutions_in_name)):
            resolutions = self._get_previous_item(variation, self.target_resolutions_in_name, index)
            if isinstance(resolutions, int):
                _add_resolution_if_new(resolutions, self.quantization)
            elif isinstance(resolutions, str):
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
                    resolutions = self._get_previous_item(variation, self.target_resolutions_override_in_name, index)
                    if isinstance(resolutions, int):
                        _add_resolution_if_new(resolutions, self.quantization)
                    elif isinstance(resolutions, str):
                        if 'x' in resolutions and ',' not in resolutions:
                            res = resolutions.split('x', 1)
                            _add_resolution_if_new((int(res[1].strip()), int(res[0].strip())), self.quantization)
                        else:
                            for res in resolutions.split(','):
                                _add_resolution_if_new(int(res.strip()),self.quantization)

        self.possible_target_resolutions = list(possible_target_resolutions)

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
            # Get quantized resolution from a random single-dim resolution
            target_resolution_list = [int(res.strip()) 
                                      for res in target_resolutions.split(',')]
            random_resolution = rand.choice(target_resolution_list)
            target_resolution = self.__quantize_resolution(random_resolution, self.quantization)

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
