from transformers import CLIPTokenizer, T5Tokenizer, T5TokenizerFast, GemmaTokenizer, LlamaTokenizer, Qwen2Tokenizer

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class Tokenize(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            in_name: str,
            tokens_out_name: str,
            mask_out_name: str,
            tokenizer: CLIPTokenizer | T5Tokenizer | T5TokenizerFast | GemmaTokenizer | LlamaTokenizer | Qwen2Tokenizer,
            max_token_length: int | None,
            format_text: str | None = None,
            additional_format_text_tokens: int | None = None,
            expand_mask: int = 0,
            padding: str = 'max_length',
    ):
        super(Tokenize, self).__init__()
        self.in_name = in_name
        self.tokens_out_name = tokens_out_name
        self.mask_out_name = mask_out_name
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        self.format_text = format_text
        self.additional_format_text_tokens = additional_format_text_tokens
        self.expand_mask = expand_mask
        self.padding = padding

    def length(self) -> int:
        return self._get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.tokens_out_name, self.mask_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        text = self._get_previous_item(variation, self.in_name, index)

        if self.format_text is not None:
            text = self.format_text.format(text)
            tokenizer_output = self.tokenizer(
                text,
                padding=self.padding,
                truncation=True,
                max_length=self.max_token_length + self.additional_format_text_tokens,
                return_tensors="pt",
            )
        else:
            tokenizer_output = self.tokenizer(
                text,
                padding=self.padding,
                truncation=True,
                max_length=self.max_token_length,
                return_tensors="pt",
            )

        tokens = tokenizer_output.input_ids.to(self.pipeline.device)
        mask = tokenizer_output.attention_mask.to(self.pipeline.device)

        tokens = tokens.squeeze(dim=0)
        mask = mask.squeeze(dim=0)

        #unmask n tokens:
        if self.expand_mask > 0:
            masked_idx = (mask == 0).nonzero(as_tuple=True)[0]
            mask[masked_idx[:self.expand_mask]] = 1 #dtype is long

        return {
            self.tokens_out_name: tokens,
            self.mask_out_name: mask,
        }
