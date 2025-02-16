from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class CalcAspect(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(self, image_in_name: str, resolution_out_name: str):
        super(CalcAspect, self).__init__()
        self.image_in_name = image_in_name
        self.resolution_out_name = resolution_out_name

    def length(self) -> int:
        return self._get_previous_length(self.image_in_name)

    def get_inputs(self) -> list[str]:
        return [self.image_in_name]

    def get_outputs(self) -> list[str]:
        return [self.resolution_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        image = self._get_previous_item(variation, self.image_in_name, index)

        resolution = tuple(image.shape[1:])  # cuts off channel dimension

        return {
            self.resolution_out_name: resolution,
        }
