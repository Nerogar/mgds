from contextlib import nullcontext

import torch
from diffusers import AutoencoderKL

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class DecodeVAE(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            in_name: str,
            out_name: str,
            vae: AutoencoderKL,
            autocast_contexts: list[torch.autocast | None] = None,
            dtype: torch.dtype | None = None,
    ):
        super(DecodeVAE, self).__init__()
        self.in_name = in_name
        self.out_name = out_name
        self.vae = vae

        self.autocast_contexts = [nullcontext()] if autocast_contexts is None else autocast_contexts
        self.dtype = dtype

    def length(self) -> int:
        return self._get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        latent_image = self._get_previous_item(variation, self.in_name, index)

        if self.dtype:
            latent_image = latent_image.to(dtype=self.dtype)

        with self._all_contexts(self.autocast_contexts):
            image = self.vae.decode(latent_image.unsqueeze(0)).sample
            image = image.clamp(-1, 1).squeeze(dim=0)

        return {
            self.out_name: image
        }
