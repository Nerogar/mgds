import torch
from torchvision.transforms import functional, InterpolationMode

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class RandomRotate(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            names: [str],
            enabled_in_name: str,
            fixed_enabled_in_name: str,
            max_angle_in_name: str,
    ):
        super(RandomRotate, self).__init__()
        self.names = names
        self.enabled_in_name = enabled_in_name
        self.fixed_enabled_in_name = fixed_enabled_in_name
        self.max_angle_in_name = max_angle_in_name

    def length(self) -> int:
        return self._get_previous_length(self.names[0])

    def get_inputs(self) -> list[str]:
        return self.names + [self.enabled_in_name] + [self.fixed_enabled_in_name] + [self.max_angle_in_name]

    def get_outputs(self) -> list[str]:
        return self.names

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        enabled = self._get_previous_item(variation, self.enabled_in_name, index)
        fixed_enabled = self._get_previous_item(variation, self.fixed_enabled_in_name, index)
        max_angle = self._get_previous_item(variation, self.max_angle_in_name, index)

        rand = self._get_rand(variation, index)
        item = {}

        if enabled:
            angle = rand.uniform(-max_angle, max_angle)
        elif fixed_enabled:
            angle = max_angle
        else:
            angle = 0.0

        for name in self.names:
            previous_item = self._get_previous_item(variation, name, index)
            if enabled or fixed_enabled:
                orig_dtype = previous_item.dtype
                if orig_dtype == torch.bfloat16:
                    previous_item = previous_item.to(dtype=torch.float32)
                previous_item = functional.rotate(previous_item, angle, interpolation=InterpolationMode.BILINEAR)
                previous_item = previous_item.to(dtype=orig_dtype)

            item[name] = previous_item

        return item
