import os

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class LoadMultipleTexts(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(self, path_in_name: str, texts_out_name: str):
        super(LoadMultipleTexts, self).__init__()
        self.path_in_name = path_in_name
        self.texts_out_name = texts_out_name

    def length(self) -> int:
        return self._get_previous_length(self.path_in_name)

    def get_inputs(self) -> list[str]:
        return [self.path_in_name]

    def get_outputs(self) -> list[str]:
        return [self.texts_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        path = self._get_previous_item(variation, self.path_in_name, index)

        texts = []
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                texts = [line.strip() for line in f]
                f.close()

        texts = list(filter(lambda text: text != "", texts))

        if len(texts) == 0:
            texts = [""]

        return {
            self.texts_out_name: texts
        }
