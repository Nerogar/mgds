from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class SelectRandomText(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(self, texts_in_name: str, text_out_name: str):
        super().__init__()
        self.texts_in_name = texts_in_name
        self.text_out_name = text_out_name

    def length(self) -> int:
        return self._get_previous_length(self.texts_in_name)

    def get_inputs(self) -> list[str]:
        return [self.texts_in_name]

    def get_outputs(self) -> list[str]:
        return [self.text_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        rand = self._get_rand(variation, index)
        texts = self._get_previous_item(variation, self.texts_in_name, index)

        text = texts if isinstance(texts, str) else rand.choice(texts)

        return {
            self.text_out_name: text
        }
