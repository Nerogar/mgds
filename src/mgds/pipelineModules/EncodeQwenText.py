from contextlib import nullcontext

import torch
from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule
from transformers import Qwen2_5_VLForConditionalGeneration


class EncodeQwenText(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            tokens_name: str,
            tokens_attention_mask_in_name: str | None,
            hidden_state_out_name: str,
            tokens_attention_mask_out_name: str | None,
            text_encoder: Qwen2_5_VLForConditionalGeneration,
            hidden_state_output_index: int | None = None,
            crop_start: int | None = None,
            autocast_contexts: list[torch.autocast | None] = None,
            dtype: torch.dtype | None = None,
    ):
        super(EncodeQwenText, self).__init__()
        self.tokens_name = tokens_name
        self.tokens_attention_mask_in_name = tokens_attention_mask_in_name
        self.hidden_state_out_name = hidden_state_out_name
        self.tokens_attention_mask_out_name = tokens_attention_mask_out_name
        self.text_encoder = text_encoder
        self.hidden_state_output_index = hidden_state_output_index
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
            text_encoder_output = self.text_encoder(
                tokens,
                attention_mask=tokens_attention_mask.to(dtype=self.dtype),
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

        tokens = tokens.squeeze()
        hidden_state = text_encoder_output.hidden_states[self.hidden_state_output_index].squeeze(dim=0)
        tokens_attention_mask = tokens_attention_mask.squeeze(dim=0)

        if self.crop_start is not None:
            tokens = tokens[self.crop_start:]
            tokens_attention_mask = tokens_attention_mask[self.crop_start:]
            #set masked state to 0 should not make a difference, but the reference implementation in diffusers also does that:
            hidden_state = hidden_state[self.crop_start:] * tokens_attention_mask.unsqueeze(dim=-1)

        return {
            self.tokens_name: tokens,
            self.hidden_state_out_name: hidden_state,
            self.tokens_attention_mask_out_name: tokens_attention_mask,
        }
