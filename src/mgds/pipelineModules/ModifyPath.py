import os

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class ModifyPath(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(self, in_name: str, out_name: str, postfix: str, extension: str):
        super(ModifyPath, self).__init__()

        self.in_name = in_name
        self.out_name = out_name

        self.postfix = postfix
        self.extension = extension

        self.extra_paths = []

    def length(self) -> int:
        return self._get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def start(self, variation: int):
        for index in range(self._get_previous_length(self.in_name)):
            image_path = self._get_previous_item(variation, self.in_name, index)

            image_name = os.path.splitext(image_path)[0]
            extra_path = image_name + self.postfix + self.extension

            self.extra_paths.append(extra_path)

    def get_item(self, variation: int, index: int, requested_name: str = None) -> (str, object):
        return {
            self.out_name: self.extra_paths[index],
        }
