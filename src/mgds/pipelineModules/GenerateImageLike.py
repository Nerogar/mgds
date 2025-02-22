import torch

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class GenerateImageLike(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            image_in_name: str,
            image_out_name: str,
            color: float | int | tuple[float, float, float],
            range_min: float,
            range_max: float,
    ):
        super(GenerateImageLike, self).__init__()
        self.image_in_name = image_in_name
        self.image_out_name = image_out_name
        if isinstance(color, int | float):
            self.color = [color]
        else:
            self.color = color

        self.range_min = range_min
        self.range_max = range_max

    def length(self) -> int:
        return self._get_previous_length(self.image_in_name)

    def get_inputs(self) -> list[str]:
        return []

    def get_outputs(self) -> list[str]:
        return [self.image_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        original_image = self._get_previous_item(variation, self.image_in_name, index)

        image_tensor = torch.tensor(self.color, device=self.pipeline.device, dtype=torch.float32) / 255
        while image_tensor.ndim < original_image.ndim:
            image_tensor = image_tensor.unsqueeze(1)
        image_tensor = image_tensor.expand((-1, *original_image.shape[1:]))

        image_tensor = image_tensor.to(dtype=original_image.dtype)
        image_tensor = image_tensor * (self.range_max - self.range_min) + self.range_min

        return {
            self.image_out_name: image_tensor
        }
