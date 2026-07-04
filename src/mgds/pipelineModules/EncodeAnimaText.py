from contextlib import nullcontext

import torch
from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule
from transformers import Qwen3ForCausalLM

from diffusers import AnimaTextConditioner


class EncodeAnimaText(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            tokens_name: str,
            tokens_attention_mask_name: str,
            t5_tokens_name: str,
            t5_tokens_attention_mask_name: str,
            hidden_state_out_name: str,
            text_encoder: Qwen3ForCausalLM,
            text_conditioner: AnimaTextConditioner,
            autocast_contexts: list[torch.autocast | None] = None,
            dtype: torch.dtype | None = None,
    ):
        super(EncodeAnimaText, self).__init__()
        self.tokens_name = tokens_name
        self.tokens_attention_mask_name = tokens_attention_mask_name
        self.t5_tokens_name = t5_tokens_name
        self.t5_tokens_attention_mask_name = t5_tokens_attention_mask_name
        self.hidden_state_out_name = hidden_state_out_name
        self.text_encoder = text_encoder
        self.text_conditioner = text_conditioner

        self.autocast_contexts = [nullcontext()] if autocast_contexts is None else autocast_contexts
        self.dtype = dtype

    def length(self) -> int:
        return self._get_previous_length(self.tokens_name)

    def get_inputs(self) -> list[str]:
        return [self.tokens_name, self.tokens_attention_mask_name, self.t5_tokens_name, self.t5_tokens_attention_mask_name]

    def get_outputs(self) -> list[str]:
        return [self.hidden_state_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        tokens = self._get_previous_item(variation, self.tokens_name, index).unsqueeze(0)
        tokens_mask = self._get_previous_item(variation, self.tokens_attention_mask_name, index).unsqueeze(0)
        t5_tokens = self._get_previous_item(variation, self.t5_tokens_name, index).unsqueeze(0)
        t5_tokens_mask = self._get_previous_item(variation, self.t5_tokens_attention_mask_name, index).unsqueeze(0)

        with self._all_contexts(self.autocast_contexts):
            qwen_hidden = self.text_encoder(
                tokens,
                attention_mask=tokens_mask.float(),
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            ).last_hidden_state
            # zero out padding positions (mirrors AnimaTextEncoderStep in diffusers)
            qwen_hidden = qwen_hidden * tokens_mask.to(qwen_hidden).unsqueeze(-1)
            hidden_state = self.text_conditioner(
                source_hidden_states=qwen_hidden.to(dtype=self.text_conditioner.dtype),
                target_input_ids=t5_tokens,
                target_attention_mask=t5_tokens_mask,
                source_attention_mask=tokens_mask,
            )

        hidden_state = hidden_state.squeeze(dim=0)

        return {
            self.hidden_state_out_name: hidden_state,
        }
