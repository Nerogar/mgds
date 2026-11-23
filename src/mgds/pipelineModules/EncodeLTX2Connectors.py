from contextlib import nullcontext

import torch
from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors


class EncodeLTX2Connectors(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            hidden_state_in_name: str,
            tokens_attention_mask_in_name: str,
            video_embeds_out_name: str,
            audio_embeds_out_name: str,
            connectors: LTX2TextConnectors,
            padding_side: str = "left",
            autocast_contexts: list[torch.autocast | None] = None,
            dtype: torch.dtype | None = None,
    ):
        super(EncodeLTX2Connectors, self).__init__()
        self.hidden_state_in_name = hidden_state_in_name
        self.tokens_attention_mask_in_name = tokens_attention_mask_in_name
        self.video_embeds_out_name = video_embeds_out_name
        self.audio_embeds_out_name = audio_embeds_out_name
        self.connectors = connectors
        self.padding_side = padding_side

        self.autocast_contexts = [nullcontext()] if autocast_contexts is None else autocast_contexts
        self.dtype = dtype

    def length(self) -> int:
        return self._get_previous_length(self.hidden_state_in_name)

    def get_inputs(self) -> list[str]:
        return [self.hidden_state_in_name, self.tokens_attention_mask_in_name]

    def get_outputs(self) -> list[str]:
        return [self.video_embeds_out_name, self.audio_embeds_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        hidden_state = self._get_previous_item(variation, self.hidden_state_in_name, index).unsqueeze(0)
        tokens_attention_mask = self._get_previous_item(variation, self.tokens_attention_mask_in_name, index).unsqueeze(0)

        with self._all_contexts(self.autocast_contexts):
            video_embeds, audio_embeds, connector_attention_mask = self.connectors(
                hidden_state.to(dtype=self.dtype) if self.dtype is not None else hidden_state,
                tokens_attention_mask,
                padding_side=self.padding_side,
            )

        # the connectors replace padded positions with learnable registers and return an all-attend mask, so
        # the mask carries no information and is dropped. _assert_async queues the check as a kernel instead
        # of reading the value back, so it costs no device sync.
        torch._assert_async(connector_attention_mask.all(), "connector attention mask is not all-True")

        if self.dtype is not None:
            video_embeds = video_embeds.to(dtype=self.dtype)
            audio_embeds = audio_embeds.to(dtype=self.dtype)

        return {
            self.video_embeds_out_name: video_embeds.squeeze(dim=0),
            self.audio_embeds_out_name: audio_embeds.squeeze(dim=0),
        }
