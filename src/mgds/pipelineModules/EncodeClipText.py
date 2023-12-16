from contextlib import nullcontext

import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class EncodeClipText(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            in_name: str,
            hidden_state_out_name: str,
            pooled_out_name: str | None,
            text_encoder: CLIPTextModel | CLIPTextModelWithProjection,
            add_layer_norm: bool,
            hidden_state_output_index: int | None = None,
            override_allow_mixed_precision: bool | None = None,
    ):
        super(EncodeClipText, self).__init__()
        self.in_name = in_name
        self.hidden_state_out_name = hidden_state_out_name
        self.pooled_out_name = pooled_out_name
        self.text_encoder = text_encoder
        self.add_layer_norm = add_layer_norm
        self.hidden_state_output_index = hidden_state_output_index
        self.override_allow_mixed_precision = override_allow_mixed_precision

    def length(self) -> int:
        return self._get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        if self.pooled_out_name:
            return [self.hidden_state_out_name, self.pooled_out_name]
        else:
            return [self.hidden_state_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        tokens = self._get_previous_item(variation, self.in_name, index)

        tokens = tokens.unsqueeze(0)

        allow_mixed_precision = self.pipeline.allow_mixed_precision if self.override_allow_mixed_precision is None \
            else self.override_allow_mixed_precision

        with torch.no_grad():
            with torch.autocast(self.pipeline.device.type, self.pipeline.dtype) if allow_mixed_precision \
                    else nullcontext():
                text_encoder_output = self.text_encoder(tokens, output_hidden_states=True, return_dict=True)

        hidden_states = text_encoder_output.hidden_states
        if self.pooled_out_name:
            pooled_state = text_encoder_output.text_embeds
        else:
            pooled_state = None

        hidden_states = [hidden_state.squeeze() for hidden_state in hidden_states]
        pooled_state = None if pooled_state is None else pooled_state.squeeze()

        hidden_state = hidden_states[self.hidden_state_output_index]

        if self.add_layer_norm:
            final_layer_norm = self.text_encoder.text_model.final_layer_norm
            hidden_state = final_layer_norm(
                hidden_state
            )

        return {
            self.hidden_state_out_name: hidden_state,
            self.pooled_out_name: pooled_state,
        }
