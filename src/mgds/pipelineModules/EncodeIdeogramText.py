from contextlib import nullcontext

import torch
from diffusers import Ideogram4Pipeline
from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule
from transformers import Qwen3VLModel


class EncodeIdeogramText(
    PipelineModule,
    RandomAccessPipelineModule,
):
    # Ideogram's text conditioning is not a single hidden-state index: it is the interleaved concatenation of the raw
    # hidden states tapped at 13 Qwen3-VL decoder layers (53248 = 13 * 4096), with the final layer taken pre-norm.
    def __init__(
            self,
            tokens_name: str,
            tokens_attention_mask_in_name: str | None,
            hidden_state_out_name: str,
            tokens_attention_mask_out_name: str | None,
            text_encoder: Qwen3VLModel,
            autocast_contexts: list[torch.autocast | None] = None,
            dtype: torch.dtype | None = None,
    ):
        super(EncodeIdeogramText, self).__init__()
        self.tokens_name = tokens_name
        self.tokens_attention_mask_in_name = tokens_attention_mask_in_name
        self.hidden_state_out_name = hidden_state_out_name
        self.tokens_attention_mask_out_name = tokens_attention_mask_out_name
        self.text_encoder = text_encoder

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
        tokens_attention_mask = self._get_previous_item(variation, self.tokens_attention_mask_in_name, index)
        tokens_attention_mask = tokens_attention_mask.unsqueeze(0)

        with self._all_contexts(self.autocast_contexts):
            # Mirrors Ideogram4Pipeline.encode_prompt: text-only MRoPE shares the linear token position across all 3
            # axes, so a plain arange is the position_ids for the real tokens. This assumes right-padding (real
            # tokens first, positions starting at 0), which is what the Tokenize node must produce for Ideogram.
            position_ids = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0).expand(tokens.shape[0], -1)
            selected = Ideogram4Pipeline._get_text_encoder_hidden_states(self.text_encoder, tokens, tokens_attention_mask.to(dtype=self.dtype), position_ids)

            # Interleave by hidden dim (NOT torch.cat): stack -> (L, B, T, H), permute -> (B, T, H, L), reshape ->
            # (B, T, H*L). This grouped-by-hidden-dim order is what the transformer's llm_cond_proj expects.
            hidden_state = torch.stack(selected, dim=0).permute(1, 2, 3, 0).reshape(tokens.shape[0], tokens.shape[1], -1)

        tokens = tokens.squeeze(dim=0)
        hidden_state = hidden_state.squeeze(dim=0)
        tokens_attention_mask = tokens_attention_mask.squeeze(dim=0)

        return {
            self.tokens_name: tokens,
            self.hidden_state_out_name: hidden_state,
            self.tokens_attention_mask_out_name: tokens_attention_mask,
        }
