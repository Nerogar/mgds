from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.SerialPipelineModule import SerialPipelineModule


class OutputPipelineModule(
    PipelineModule,
    SerialPipelineModule,
):
    def __init__(
            self,
            names: list[str | tuple[str, str]],
    ):
        super(OutputPipelineModule, self).__init__()
        self.input_names = []
        self.output_names = []
        for name in names:
            if isinstance(name, tuple):
                self.input_names.append(name[0])
                self.output_names.append(name[1])
            else:
                self.input_names.append(name)
                self.output_names.append(name)

    def approximate_length(self) -> int:
        return self._get_previous_length(self.input_names[0])

    def get_inputs(self) -> list[str]:
        return self.input_names

    def get_outputs(self) -> list[str]:
        return self.output_names

    def get_next_item(self) -> dict:
        item = {}

        for input_name, output_name in zip(self.input_names, self.output_names):
            if self._get_previous_length(input_name) <= self.current_index:
                raise StopIteration

            item[output_name] = self._get_previous_item(self.current_variation, input_name, self.current_index)

        # filter out None values
        item = {k: v for k, v in item.items() if v is not None}

        return item

    def has_next(self) -> bool:
        return self._has_previous_next(self.input_names[0])
