from contextlib import nullcontext

import torch
from lens.text_encoder import LensGptOssEncoder
from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


# This module is Lens-specific: it relies on LensGptOssEncoder.encode_layers() rather than the
# standard output_hidden_states=True API. The standard API cannot be used because transformers'
# @capture_outputs applies tie_last_hidden_states, which norms hidden_states[-1] — corrupting
# the last selected layer (GPT-OSS layer 23 is the final layer in the model).
# text_encoder.set_selected_layers() must have been called before this module is used;
# LensModelLoader does this immediately after loading the text encoder.
class EncodeLensText(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            tokens_name: str,
            tokens_attention_mask_in_name: str | None,
            hidden_state_out_name: str,
            tokens_attention_mask_out_name: str | None,
            text_encoder: LensGptOssEncoder,
            crop_start: int | None = None,
            autocast_contexts: list[torch.autocast | None] = None,
            dtype: torch.dtype | None = None,
    ):
        super(EncodeLensText, self).__init__()
        self.tokens_name = tokens_name
        self.tokens_attention_mask_in_name = tokens_attention_mask_in_name
        self.hidden_state_out_name = hidden_state_out_name
        self.tokens_attention_mask_out_name = tokens_attention_mask_out_name
        self.text_encoder = text_encoder
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
            layer_outputs = self.text_encoder.encode_layers(tokens, tokens_attention_mask)

        hidden_state = torch.cat(layer_outputs, dim=-1)
        tokens = tokens.squeeze(dim=0)
        hidden_state = hidden_state.squeeze(dim=0)
        tokens_attention_mask = tokens_attention_mask.squeeze(dim=0)

        if self.crop_start is not None:
            tokens = tokens[self.crop_start:]
            tokens_attention_mask = tokens_attention_mask[self.crop_start:]
            hidden_state = hidden_state[self.crop_start:]

        return {
            self.tokens_name: tokens,
            self.hidden_state_out_name: hidden_state,
            self.tokens_attention_mask_out_name: tokens_attention_mask,
        }
