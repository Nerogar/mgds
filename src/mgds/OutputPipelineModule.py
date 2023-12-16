from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.SerialPipelineModule import SerialPipelineModule


class OutputPipelineModule(
    PipelineModule,
    SerialPipelineModule,
):
    def __init__(self, names: list[str]):
        super(OutputPipelineModule, self).__init__()
        self.names = names

    def length(self) -> int:
        return self._get_previous_length(self.names[0])

    def get_inputs(self) -> list[str]:
        return self.names

    def get_outputs(self) -> list[str]:
        return self.names

    def get_item(self, index: int, requested_name: str = None) -> dict:
        item = {}

        for name in self.names:
            item[name] = self._get_previous_item(self.current_variation, name, index)

        return item
