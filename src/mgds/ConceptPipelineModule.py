from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class ConceptPipelineModule(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(self, concepts: list[dict]):
        super(ConceptPipelineModule, self).__init__()
        self.concepts = concepts

    def length(self) -> int:
        return len(self.concepts)

    def get_inputs(self) -> list[str]:
        return []

    def get_outputs(self) -> list[str]:
        return ['concept']

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return {
            'concept': self.concepts[index]
        }
