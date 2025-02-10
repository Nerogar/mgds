from contextlib import nullcontext

import torch
from transformers import Gemma2Model

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class EncodeGemmaText(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            tokens_in_name: str,
            tokens_attention_mask_in_name: str | None,
            hidden_state_out_name: str,
            text_encoder: Gemma2Model,
            add_layer_norm: bool,
            hidden_state_output_index: int | None = None,
            autocast_contexts: list[torch.autocast | None] = None,
            dtype: torch.dtype | None = None,
    ):
        super(EncodeGemmaText, self).__init__()
        self.tokens_in_name = tokens_in_name
        self.tokens_attention_mask_in_name = tokens_attention_mask_in_name
        self.hidden_state_out_name = hidden_state_out_name
        self.text_encoder = text_encoder
        self.add_layer_norm = add_layer_norm
        self.hidden_state_output_index = hidden_state_output_index

        self.autocast_contexts = [nullcontext()] if autocast_contexts is None else autocast_contexts
        self.dtype = dtype

    def length(self) -> int:
        return self._get_previous_length(self.tokens_in_name)

    def get_inputs(self) -> list[str]:
        return [self.tokens_in_name, self.tokens_attention_mask_in_name]

    def get_outputs(self) -> list[str]:
        return [self.hidden_state_out_name]

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
                use_cache=False,
            )

        hidden_states = text_encoder_output.hidden_states

        hidden_states = [hidden_state.squeeze(dim=0) for hidden_state in hidden_states]

        hidden_state = hidden_states[self.hidden_state_output_index]

        if self.hidden_state_output_index != -1 and self.add_layer_norm:
            with self._all_contexts(self.autocast_contexts):
                final_layer_norm = self.text_encoder.norm
                hidden_state = final_layer_norm(
                    hidden_state
                )

        return {
            self.hidden_state_out_name: hidden_state,
        }
