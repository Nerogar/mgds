import torch
from PIL import Image
from torchvision import transforms

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class LoadImage(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            path_in_name: str,
            image_out_name: str,
            range_min: float,
            range_max: float,
            channels: int = 3,
            dtype: torch.dtype | None = None,
    ):
        super(LoadImage, self).__init__()
        self.path_in_name = path_in_name
        self.image_out_name = image_out_name

        self.range_min = range_min
        self.range_max = range_max

        self.dtype = dtype

        if channels == 3:
            self.mode = 'RGB'
        elif channels == 1:
            self.mode = 'L'
        else:
            raise ValueError('Only 1 and 3 channels are supported.')

    def length(self) -> int:
        return self._get_previous_length(self.path_in_name)

    def get_inputs(self) -> list[str]:
        return [self.path_in_name]

    def get_outputs(self) -> list[str]:
        return [self.image_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        path = self._get_previous_item(variation, self.path_in_name, index)

        try:
            image = Image.open(path)
            image = image.convert(self.mode)

            t = transforms.ToTensor()
            image_tensor = t(image).to(device=self.pipeline.device)

            if self.dtype:
                image_tensor = image_tensor.to(dtype=self.dtype)

            image_tensor = image_tensor * (self.range_max - self.range_min) + self.range_min
        except FileNotFoundError:
            image_tensor = None
        except:
            print("could not load image, it might be corrupted: " + path)
            raise

        return {
            self.image_out_name: image_tensor
        }
