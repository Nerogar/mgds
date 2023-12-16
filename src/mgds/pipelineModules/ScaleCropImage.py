from torchvision import transforms
from torchvision.transforms import functional

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class ScaleCropImage(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self, image_in_name: str,
            scale_resolution_in_name: str,
            crop_resolution_in_name: str,
            enable_crop_jitter_in_name: str,
            image_out_name: str,
            crop_offset_out_name: str,
    ):
        super(ScaleCropImage, self).__init__()
        self.image_in_name = image_in_name
        self.scale_resolution_in_name = scale_resolution_in_name
        self.crop_resolution_in_name = crop_resolution_in_name
        self.enable_crop_jitter_in_name = enable_crop_jitter_in_name
        self.image_out_name = image_out_name
        self.crop_offset_out_name = crop_offset_out_name

    def length(self) -> int:
        return self._get_previous_length(self.image_in_name)

    def get_inputs(self) -> list[str]:
        return [self.image_in_name, self.scale_resolution_in_name, self.crop_resolution_in_name]

    def get_outputs(self) -> list[str]:
        return [self.image_out_name, self.crop_offset_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        rand = self._get_rand(variation, index)
        image = self._get_previous_item(variation, self.image_in_name, index)
        scale_resolution = self._get_previous_item(variation, self.scale_resolution_in_name, index)
        crop_resolution = self._get_previous_item(variation, self.crop_resolution_in_name, index)
        enable_crop_jitter = self._get_previous_item(variation, self.enable_crop_jitter_in_name, index)

        resize = transforms.Resize(scale_resolution, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
        image = resize(image)

        if enable_crop_jitter:
            y_offset = rand.randint(0, scale_resolution[0] - crop_resolution[0])
            x_offset = rand.randint(0, scale_resolution[1] - crop_resolution[1])
        else:
            y_offset = (scale_resolution[0] - crop_resolution[0]) // 2
            x_offset = (scale_resolution[1] - crop_resolution[1]) // 2

        crop_offset = (y_offset, x_offset)
        image = functional.crop(image, y_offset, x_offset, crop_resolution[0], crop_resolution[1])

        return {
            self.crop_offset_out_name: crop_offset,
            self.image_out_name: image
        }
