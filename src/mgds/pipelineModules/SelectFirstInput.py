from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class SelectFirstInput(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(self, in_names: list[str], out_name: str):
        super(SelectFirstInput, self).__init__()
        self.in_names = in_names
        self.out_name = out_name

    def length(self) -> int:
        return self._get_previous_length(self.in_names[0])

    def get_inputs(self) -> list[str]:
        return self.in_names

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:

        for in_name in self.in_names:
            out = self._get_previous_item(variation, in_name, index)

            if out is not None:
                return {
                    self.out_name: out
                }

        return {
            self.out_name: None
        }
