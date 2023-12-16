from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class SettingsPipelineModule(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(self, settings: dict):
        super(SettingsPipelineModule, self).__init__()
        self.settings = settings

    def length(self) -> int:
        return 1

    def get_inputs(self) -> list[str]:
        return []

    def get_outputs(self) -> list[str]:
        return ['settings']

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return {
            'settings': self.settings
        }
