import os.path

from mgds.PipelineModule import PipelineModule, SkipSampleError
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image

import numpy as np
from PIL import Image, ImageOps


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
            skip_corrupt: bool = True,
    ):
        super().__init__()
        self.path_in_name = path_in_name
        self.image_out_name = image_out_name

        self.range_min = range_min
        self.range_max = range_max

        self.supported_extensions = supported_extensions

        self.dtype = dtype
        self.skip_corrupt = skip_corrupt

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

        ext = os.path.splitext(path)[1]
        if ext.lower() in self.supported_extensions:
            try:
                image_tensor = read_image(path, self.mode, apply_exif_orientation=True) \
                    .to(device=self.pipeline.device)

                # Normalize integer dtypes (e.g. uint8 -> 255, uint16 -> 65535) in float32 before
                # casting to self.dtype so uint16 max (65535) does not overflow float16 (65504).
                if image_tensor.is_floating_point():
                    image_tensor = image_tensor.to(dtype=torch.float32)
                else:
                    image_max = torch.iinfo(image_tensor.dtype).max
                    image_tensor = image_tensor.to(dtype=torch.float32) / image_max

                image_tensor = image_tensor * (self.range_max - self.range_min) + self.range_min

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
            except Exception as e:
                if self.skip_corrupt:
                    print("could not load image, it might be corrupted, skipping: " + path)
                    raise SkipSampleError(f"could not load image: {path}") from e
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
            if image.mode.startswith("I;16"):
                # PIL's convert('L'/'RGB') clamps 16-bit grayscale values >= 255, and ToTensor()
                # reinterprets I;16 as signed int16 (wrapping >=32768 negative).
                image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0)
                image_tensor = image_tensor.to(device=self.pipeline.device) / 65535.0
                if self.pillow_mode == 'RGB':
                    image_tensor = image_tensor.repeat(3, 1, 1)
            else:
                image = image.convert(self.pillow_mode)
                image_tensor = t(image).to(device=self.pipeline.device, dtype=torch.float32)

            image_tensor = image_tensor * (self.range_max - self.range_min) + self.range_min

            if self.dtype:
                image_tensor = image_tensor.to(dtype=self.dtype)

            image_tensor = image_tensor * (self.range_max - self.range_min) + self.range_min
        except FileNotFoundError:
            image_tensor = None
        except Exception as e:
            if self.skip_corrupt:
                print("could not load image, it might be corrupted, skipping: " + path)
                raise SkipSampleError(f"could not load image: {path}") from e
            print("could not load image, it might be corrupted: " + path)
            raise

        return image_tensor
