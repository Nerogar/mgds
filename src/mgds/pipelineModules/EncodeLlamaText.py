from contextlib import nullcontext

import torch
from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule
from transformers import LlamaModel


class EncodeLlamaText(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            tokens_name: str,
            tokens_attention_mask_in_name: str | None,
            hidden_state_out_name: str,
            tokens_attention_mask_out_name: str | None,
            text_encoder: LlamaModel,
            hidden_state_output_index: int | None = None,
            output_all_hidden_states: bool = False,
            all_hidden_state_output_indices: list[int] | None = None,
            crop_start: int | None = None,
            autocast_contexts: list[torch.autocast | None] = None,
            dtype: torch.dtype | None = None,
    ):
        super(EncodeLlamaText, self).__init__()
        self.tokens_name = tokens_name
        self.tokens_attention_mask_in_name = tokens_attention_mask_in_name
        self.hidden_state_out_name = hidden_state_out_name
        self.tokens_attention_mask_out_name = tokens_attention_mask_out_name
        self.text_encoder = text_encoder
        self.hidden_state_output_index = hidden_state_output_index
        self.output_all_hidden_states = output_all_hidden_states
        self.max_hidden_state_output_index = max(all_hidden_state_output_indices) \
            if all_hidden_state_output_indices is not None else None
        self.crop_start = crop_start

        self.autocast_contexts = [nullcontext()] if autocast_contexts is None else autocast_contexts
        self.dtype = dtype

    def length(self) -> int:
        return self._get_previous_length(self.tokens_name)

    def get_inputs(self) -> list[str]:
        return [self.tokens_name, self.tokens_attention_mask_in_name]

    def get_outputs(self) -> list[str]:
        return [self.tokens_name, self.hidden_state_out_name, self.tokens_attention_mask_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        tokens = self._get_previous_item(variation, self.tokens_name, index)
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

        tokens = tokens.squeeze()
        hidden_states = text_encoder_output.hidden_states
        hidden_states = [hidden_state.squeeze(dim=0) for hidden_state in hidden_states]
        if self.output_all_hidden_states:
            hidden_state = hidden_states[1:self.max_hidden_state_output_index + 2]
        else:
            hidden_state = hidden_states[self.hidden_state_output_index]
        tokens_attention_mask = tokens_attention_mask.squeeze(dim=0)

        if self.crop_start is not None:
            tokens = tokens[self.crop_start:]

            if self.output_all_hidden_states:
                hidden_state = [t[self.crop_start:] for t in hidden_state]
            else:
                hidden_state = hidden_state[self.crop_start:]

            tokens_attention_mask = tokens_attention_mask[self.crop_start:]

        return {
            self.tokens_name: tokens,
            self.hidden_state_out_name: hidden_state,
            self.tokens_attention_mask_out_name: tokens_attention_mask,
        }
