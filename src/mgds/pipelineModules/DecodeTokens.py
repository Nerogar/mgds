from transformers import CLIPTokenizer

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class DecodeTokens(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(self, in_name: str, out_name: str, tokenizer: CLIPTokenizer):
        super(DecodeTokens, self).__init__()
        self.in_name = in_name
        self.out_name = out_name
        self.tokenizer = tokenizer

    def length(self) -> int:
        return self._get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        tokens = self._get_previous_item(variation, self.in_name, index)

        text = self.tokenizer.decode(
            token_ids=tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return {
            self.out_name: text
        }
