from torchvision import transforms

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class ScaleImage(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(self, in_name: str, out_name: str, factor: float):
        super(ScaleImage, self).__init__()
        self.in_name = in_name
        self.out_name = out_name
        self.factor = factor

    def length(self) -> int:
        return self._get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        image = self._get_previous_item(variation, self.in_name, index)

        size = (round(image.shape[1] * self.factor), round(image.shape[2] * self.factor))

        t = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        ])

        image = t(image)

        return {
            self.out_name: image
        }
