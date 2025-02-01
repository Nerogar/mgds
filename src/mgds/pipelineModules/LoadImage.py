import os.path

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image

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
            supported_extensions: set[str],
            channels: int = 3,
            dtype: torch.dtype | None = None,
    ):
        super(LoadImage, self).__init__()
        self.path_in_name = path_in_name
        self.image_out_name = image_out_name

        self.range_min = range_min
        self.range_max = range_max

        self.supported_extensions = supported_extensions

        self.dtype = dtype

        if channels == 3:
            self.mode = ImageReadMode.RGB
            self.pillow_mode = 'RGB'
        elif channels == 1:
            self.mode = ImageReadMode.GRAY
            self.pillow_mode = 'L'
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

        # temporary workaround for a torchvision memory leak https://github.com/pytorch/vision/issues/8710
        ext = os.path.splitext(path)[1]
        if ext.lower() in self.supported_extensions:
            if path.endswith('.webp'):
                image_tensor = self._load_pillow(path)
            else:
                try:
                    image_tensor = read_image(path, self.mode, apply_exif_orientation=True) \
                        .to(device=self.pipeline.device)

                    if self.dtype:
                        image_tensor = image_tensor.to(dtype=self.dtype)

                    # Transform 0 - 255 to 0-1
                    image_tensor = image_tensor / 255
                    image_tensor = image_tensor * (self.range_max - self.range_min) + self.range_min
                except FileNotFoundError:
                    image_tensor = None
                except RuntimeError:
                    # Torch builtins couldn't load it because not all image formats and
                    # variations there are supported. Fall back to pillow.
                    image_tensor = self._load_pillow(path)
                except:
                    print("could not load image, it might be corrupted: " + path)
                    raise
        else:
            image_tensor = None

        return {
            self.image_out_name: image_tensor
        }

    def _load_pillow(self, path):
        """Fallback method to load the image through pillow.

        MUCH slower, because the tensor transform holds the GIL.
        """
        try:
            image = Image.open(path)
            image = ImageOps.exif_transpose(image)
            image = image.convert(self.pillow_mode)

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

        return image_tensor
