from contextlib import nullcontext

import torch
from transformers import LlamaModel

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class EncodeLlamaText(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            tokens_in_name: str,
            tokens_attention_mask_in_name: str | None,
            hidden_state_out_name: str,
            tokens_attention_mask_out_name: str | None,
            text_encoder: LlamaModel,
            hidden_state_output_index: int | None = None,
            crop_start: int | None = None,
            autocast_contexts: list[torch.autocast | None] = None,
            dtype: torch.dtype | None = None,
    ):
        super(EncodeLlamaText, self).__init__()
        self.tokens_in_name = tokens_in_name
        self.tokens_attention_mask_in_name = tokens_attention_mask_in_name
        self.hidden_state_out_name = hidden_state_out_name
        self.tokens_attention_mask_out_name = tokens_attention_mask_out_name
        self.text_encoder = text_encoder
        self.hidden_state_output_index = hidden_state_output_index
        self.crop_start = crop_start

        self.autocast_contexts = [nullcontext()] if autocast_contexts is None else autocast_contexts
        self.dtype = dtype

    def length(self) -> int:
        return self._get_previous_length(self.tokens_in_name)

    def get_inputs(self) -> list[str]:
        return [self.tokens_in_name, self.tokens_attention_mask_in_name]

    def get_outputs(self) -> list[str]:
        return [self.hidden_state_out_name, self.tokens_attention_mask_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        tokens = self._get_previous_item(variation, self.tokens_in_name, index)
        tokens = tokens.unsqueeze(0)

        if self.tokens_attention_mask_in_name is not None:
            tokens_attention_mask = self._get_previous_item(variation, self.tokens_attention_mask_in_name, index)
            tokens_attention_mask = tokens_attention_mask.unsqueeze(0)
        else:
            tokens_attention_mask = None

        with self._all_contexts(self.autocast_contexts):
            if tokens_attention_mask is not None and self.dtype:
                tokens_attention_mask = tokens_attention_mask.to(dtype=self.dtype)

            text_encoder_output = self.text_encoder(
                tokens,
                attention_mask=tokens_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = text_encoder_output.hidden_states
        hidden_states = [hidden_state.squeeze() for hidden_state in hidden_states]
        hidden_state = hidden_states[self.hidden_state_output_index]
        tokens_attention_mask = tokens_attention_mask.squeeze()

        if self.crop_start is not None:
            hidden_state = hidden_state[self.crop_start:]
            tokens_attention_mask = tokens_attention_mask[self.crop_start:]

        return {
            self.hidden_state_out_name: hidden_state,
            self.tokens_attention_mask_out_name: tokens_attention_mask,
        }
