from contextlib import nullcontext

import torch
from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule
from transformers import Gemma3ForConditionalGeneration, Gemma4UnifiedForConditionalGeneration


class EncodeGemma3Text(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            tokens_name: str,
            tokens_attention_mask_in_name: str | None,
            hidden_state_out_name: str,
            tokens_attention_mask_out_name: str | None,
            # nothing here is specific to Gemma 3: the module only asks for hidden states and stacks them, so
            # the Gemma 4 encoder LTX 2.5 ships goes through the same body under the same class name.
            text_encoder: Gemma3ForConditionalGeneration | Gemma4UnifiedForConditionalGeneration,
            hidden_state_output_index: int | list[int] | None = None,
            crop_start: int | None = None,
            autocast_contexts: list[torch.autocast | None] = None,
            dtype: torch.dtype | None = None,
    ):
        super(EncodeGemma3Text, self).__init__()
        self.tokens_name = tokens_name
        self.tokens_attention_mask_in_name = tokens_attention_mask_in_name
        self.hidden_state_out_name = hidden_state_out_name
        self.tokens_attention_mask_out_name = tokens_attention_mask_out_name
        self.text_encoder = text_encoder
        self.hidden_state_indexes = None if hidden_state_output_index is None \
            else (hidden_state_output_index if isinstance(hidden_state_output_index, list) else [hidden_state_output_index])
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

        if self.hidden_state_indexes is None:
            hidden_state = torch.stack(text_encoder_output.hidden_states, dim=-1).flatten(2, 3)
        else:
            hidden_state = torch.cat([text_encoder_output.hidden_states[k] for k in self.hidden_state_indexes], dim=-1)

        if self.dtype is not None:
            hidden_state = hidden_state.to(dtype=self.dtype)

        tokens = tokens.squeeze(dim=0)
        hidden_state = hidden_state.squeeze(dim=0)
        tokens_attention_mask = tokens_attention_mask.squeeze(dim=0)

        if self.crop_start is not None:
            tokens = tokens[self.crop_start:]
            tokens_attention_mask = tokens_attention_mask[self.crop_start:]
            hidden_state = hidden_state[self.crop_start:] * tokens_attention_mask.unsqueeze(dim=-1)

        return {
            self.tokens_name: tokens,
            self.hidden_state_out_name: hidden_state,
            self.tokens_attention_mask_out_name: tokens_attention_mask,
        }
