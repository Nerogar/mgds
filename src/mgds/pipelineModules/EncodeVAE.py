from contextlib import nullcontext

import torch
from diffusers import AutoencoderKL

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class EncodeVAE(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            in_name: str,
            out_name: str,
            vae: AutoencoderKL,
            autocast_context: torch.autocast | None = None,
    ):
        super(EncodeVAE, self).__init__()
        self.in_name = in_name
        self.out_name = out_name
        self.vae = vae

        self.autocast_context = nullcontext() if autocast_context is None else autocast_context
        self.autocast_enabled = isinstance(self.autocast_context, torch.autocast)

    def length(self) -> int:
        return self._get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        image = self._get_previous_item(variation, self.in_name, index)

        if not self.autocast_enabled:
            image = image.to(dtype=self.vae.dtype)

        with self.autocast_context:
            latent_distribution = self.vae.encode(image.unsqueeze(0)).latent_dist

        return {
            self.out_name: latent_distribution
        }
