from transformers import CLIPTokenizer, T5Tokenizer, T5TokenizerFast, GemmaTokenizer, LlamaTokenizer

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class ImageToVideo(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            in_name: str,
            out_name: str,
    ):
        super(ImageToVideo, self).__init__()
        self.in_name = in_name
        self.out_name = out_name

    def length(self) -> int:
        return self._get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        tensor = self._get_previous_item(variation, self.in_name, index)

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)

        return {
            self.out_name: tensor,
        }
