from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class RescaleImageChannels(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            image_in_name: str, image_out_name: str,
            in_range_min: float, in_range_max: float,
            out_range_min: float, out_range_max: float,
    ):
        super(RescaleImageChannels, self).__init__()
        self.image_in_name = image_in_name
        self.image_out_name = image_out_name
        self.in_range_min = in_range_min
        self.in_range_max = in_range_max
        self.out_range_min = out_range_min
        self.out_range_max = out_range_max

    def length(self) -> int:
        return self._get_previous_length(self.image_in_name)

    def get_inputs(self) -> list[str]:
        return [self.image_in_name]

    def get_outputs(self) -> list[str]:
        return [self.image_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        image = self._get_previous_item(variation, self.image_in_name, index)

        image = (image - self.in_range_min) \
                * ((self.out_range_max - self.out_range_min) / (self.in_range_max - self.in_range_min)) \
                + self.out_range_min

        return {
            self.image_out_name: image
        }
