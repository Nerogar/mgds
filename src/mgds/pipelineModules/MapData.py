from typing import Any, Callable

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class MapData(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            in_name: str,
            out_name: str,
            map_fn: Callable[[Any], Any],
    ):
        super(MapData, self).__init__()
        self.in_name = in_name
        self.out_name = out_name
        self.map_fn = map_fn

    def length(self) -> int:
        return self._get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        data = self._get_previous_item(variation, self.in_name, index)

        data = self.map_fn(data)

        return {
            self.out_name: data,
        }
