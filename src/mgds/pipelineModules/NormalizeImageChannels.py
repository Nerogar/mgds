
from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torchvision


class NormalizeImageChannels(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            image_in_name: str, image_out_name: str,
            mean: tuple[float, float, float],
            std: tuple[float, float, float],
    ):
        super().__init__()
        self.image_in_name = image_in_name
        self.image_out_name = image_out_name
        self.mean = mean
        self.std = std

        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)

    def length(self) -> int:
        return self._get_previous_length(self.image_in_name)

    def get_inputs(self) -> list[str]:
        return [self.image_in_name]

    def get_outputs(self) -> list[str]:
        return [self.image_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        image = self._get_previous_item(variation, self.image_in_name, index)

        image = self.normalize(image)

        return {
            self.image_out_name: image
        }
