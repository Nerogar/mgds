from torchvision import transforms
from torchvision.transforms import functional

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class ScaleCropImage(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            names: list[str],
            scale_resolution_in_name: str,
            crop_resolution_in_name: str,
            enable_crop_jitter_in_name: str,
            crop_offset_out_name: str,
    ):
        super(ScaleCropImage, self).__init__()
        self.names = names
        self.scale_resolution_in_name = scale_resolution_in_name
        self.crop_resolution_in_name = crop_resolution_in_name
        self.enable_crop_jitter_in_name = enable_crop_jitter_in_name
        self.crop_offset_out_name = crop_offset_out_name

    def length(self) -> int:
        return self._get_previous_length(self.names[0])

    def get_inputs(self) -> list[str]:
        return self.names + [self.scale_resolution_in_name, self.crop_resolution_in_name]

    def get_outputs(self) -> list[str]:
        return self.names + [self.crop_offset_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        rand = self._get_rand(variation, index)
        scale_resolution = self._get_previous_item(variation, self.scale_resolution_in_name, index)
        crop_resolution = self._get_previous_item(variation, self.crop_resolution_in_name, index)
        enable_crop_jitter = self._get_previous_item(variation, self.enable_crop_jitter_in_name, index)

        resize = transforms.Resize(
            scale_resolution[-2:],
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )

        item = {}

        if enable_crop_jitter:
            y_offset = rand.randint(0, scale_resolution[-2] - crop_resolution[-2])
            x_offset = rand.randint(0, scale_resolution[-1] - crop_resolution[-1])
        else:
            y_offset = (scale_resolution[-2] - crop_resolution[-2]) // 2
            x_offset = (scale_resolution[-1] - crop_resolution[-1]) // 2
        crop_offset = (y_offset, x_offset)

        for name in self.names:
            image = self._get_previous_item(variation, name, index)

            if image is not None:
                image = resize(image)
                image = functional.crop(image, y_offset, x_offset, crop_resolution[-2], crop_resolution[-1])

            item[name] = image

        item[self.crop_offset_out_name] = crop_offset

        return item
