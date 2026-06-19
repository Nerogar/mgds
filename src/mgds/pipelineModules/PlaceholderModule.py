from mgds.PipelineModule import PipelineModule


class PlaceholderModule(PipelineModule):
    def get_inputs(self) -> list[str]:
        return []

    def get_outputs(self) -> list[str]:
        return []
