import os

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class GetFilename(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            path_in_name: str,
            filename_out_name: str,
            include_extension: bool,
    ):
        super(GetFilename, self).__init__()
        self.path_in_name = path_in_name
        self.filename_out_name = filename_out_name
        self.include_extension = include_extension

    def length(self) -> int:
        return self._get_previous_length(self.path_in_name)

    def get_inputs(self) -> list[str]:
        return [self.path_in_name]

    def get_outputs(self) -> list[str]:
        return [self.filename_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        path = self._get_previous_item(variation, self.path_in_name, index)

        filename = os.path.basename(path)
        if not self.include_extension:
            filename = os.path.splitext(filename)[0]

        return {
            self.filename_out_name: filename
        }
