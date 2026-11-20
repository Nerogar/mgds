from contextlib import nullcontext

import torch
from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3ForCausalLM, Qwen3VLModel


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
            text_encoder: Qwen2_5_VLForConditionalGeneration | Qwen3ForCausalLM | Qwen3VLModel,
            hidden_state_output_index: int | list[int],
            crop_start: int | None = None,
            autocast_contexts: list[torch.autocast | None] = None,
            dtype: torch.dtype | None = None,
            cumsum_position_ids: bool = False,
    ):
        super(EncodeQwenText, self).__init__()
        self.tokens_name = tokens_name
        self.tokens_attention_mask_in_name = tokens_attention_mask_in_name
        self.hidden_state_out_name = hidden_state_out_name
        self.tokens_attention_mask_out_name = tokens_attention_mask_out_name
        self.text_encoder = text_encoder
        self.hidden_state_indexes = hidden_state_output_index if isinstance(hidden_state_output_index, list) else [hidden_state_output_index]
        self.crop_start = crop_start
        # Krea 2 needs positions to skip the mid-template padding block (suffix tokens continue
        # right after the real prompt tokens instead of after the padding); Qwen-Image leaves
        # this False and uses the encoder's default position_ids.
        self.cumsum_position_ids = cumsum_position_ids

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
            position_ids = None
            if self.cumsum_position_ids:
                position_ids = (tokens_attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

            text_encoder_output = self.text_encoder(
                tokens,
                attention_mask=tokens_attention_mask.to(dtype=self.dtype),
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

        hidden_state = torch.cat([text_encoder_output.hidden_states[k] for k in self.hidden_state_indexes], dim=-1)
        tokens = tokens.squeeze(dim=0)
        hidden_state = hidden_state.squeeze(dim=0)
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
