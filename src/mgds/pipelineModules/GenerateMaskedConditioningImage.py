from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class GenerateMaskedConditioningImage(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            image_in_name: str,
            mask_in_name: str,
            image_out_name: str,
            image_range_min: float,
            image_range_max: float,
    ):
        super(GenerateMaskedConditioningImage, self).__init__()
        self.image_in_name = image_in_name
        self.mask_in_name = mask_in_name
        self.image_out_name = image_out_name

        self.image_range_min = image_range_min
        self.image_range_max = image_range_max

    def length(self) -> int:
        return self._get_previous_length(self.image_in_name)

    def get_inputs(self) -> list[str]:
        return [self.image_in_name, self.mask_in_name]

    def get_outputs(self) -> list[str]:
        return [self.image_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        image = self._get_previous_item(variation, self.image_in_name, index)
        mask = self._get_previous_item(variation, self.mask_in_name, index)

        image_midpoint = (self.image_range_max - self.image_range_min) / 2.0

        conditioning_image = (image * (1 - mask)) + (mask * image_midpoint)

        return {
            self.image_out_name: conditioning_image
        }
