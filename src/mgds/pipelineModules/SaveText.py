import os
from typing import Callable

from tqdm import tqdm

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.SerialPipelineModule import SerialPipelineModule


class SaveText(
    PipelineModule,
    SerialPipelineModule,
):
    def __init__(
            self,
            text_in_name: str,
            original_path_in_name: str,
            path: str,
            before_save_fun: Callable[[], None] | None = None,
    ):
        super(SaveText, self).__init__()
        self.text_in_name = text_in_name
        self.original_path_in_name = original_path_in_name
        self.path = path
        self.before_save_fun = before_save_fun

    def length(self) -> int:
        return self._get_previous_length(self.text_in_name)

    def get_inputs(self) -> list[str]:
        return [self.text_in_name, self.original_path_in_name]

    def get_outputs(self) -> list[str]:
        return []

    def start(self, variation: int):
        path = os.path.join(self.path, "epoch-" + str(variation))
        if not os.path.exists(path):
            os.makedirs(path)

        if self.before_save_fun is not None:
            self.before_save_fun()

        for index in tqdm(range(self._get_previous_length(self.original_path_in_name)),
                          desc='writing debug text for \'' + self.text_in_name + '\''):
            text = self._get_previous_item(variation, self.text_in_name, index)
            original_path = self._get_previous_item(variation, self.original_path_in_name, index)
            name = os.path.basename(original_path)
            name, ext = os.path.splitext(name)

            with open(os.path.join(path, str(index) + '-' + name + '-' + self.text_in_name + '.txt'), "w") as f:
                f.write(text)

    def get_item(self, index: int, requested_name: str = None) -> dict:
        return {}
