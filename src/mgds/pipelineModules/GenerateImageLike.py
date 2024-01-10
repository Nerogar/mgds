from PIL import Image
from torchvision import transforms

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
            channels: int = 3,
    ):
        super(GenerateImageLike, self).__init__()
        self.image_in_name = image_in_name
        self.image_out_name = image_out_name
        self.color = color

        self.range_min = range_min
        self.range_max = range_max

        if channels == 3 and isinstance(color, tuple):
            self.mode = 'RGB'
        elif channels == 1 and (isinstance(color, float) or isinstance(color, int)):
            self.mode = 'L'
        else:
            raise ValueError('Only 1 and 3 channels are supported.')

    def length(self) -> int:
        return self._get_previous_length(self.image_in_name)

    def get_inputs(self) -> list[str]:
        return []

    def get_outputs(self) -> list[str]:
        return [self.image_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        original_image = self._get_previous_item(variation, self.image_in_name, index)

        image = Image.new(mode=self.mode, size=(original_image.shape[2], original_image.shape[1]), color=self.color)

        t = transforms.ToTensor()
        image_tensor = t(image).to(device=self.pipeline.device)

        image_tensor = image_tensor * (self.range_max - self.range_min) + self.range_min

        return {
            self.image_out_name: image_tensor
        }
