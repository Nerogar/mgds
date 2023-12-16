from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class ReplaceText(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            text_in_name: str,
            text_out_name: str,
            old_text: str,
            new_text: str
    ):
        super(ReplaceText, self).__init__()
        self.text_in_name = text_in_name
        self.text_out_name = text_out_name
        self.old_text = old_text
        self.new_text = new_text

    def length(self) -> int:
        return self._get_previous_length(self.text_in_name)

    def get_inputs(self) -> list[str]:
        return [self.text_in_name]

    def get_outputs(self) -> list[str]:
        return [self.text_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        text = self._get_previous_item(variation, self.text_in_name, index)

        text = text.replace(self.old_text, self.new_text)

        return {
            self.text_out_name: text
        }
