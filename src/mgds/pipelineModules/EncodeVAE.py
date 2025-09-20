from contextlib import nullcontext

import torch
from diffusers import AutoencoderKL, AutoencoderDC, AutoencoderKLQwenImage
from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import AutoencoderKLHunyuanVideo

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
            vae: AutoencoderKL | AutoencoderDC | AutoencoderKLHunyuanVideo | AutoencoderKLQwenImage,
            autocast_contexts: list[torch.autocast | None] = None,
            dtype: torch.dtype | None = None,
    ):
        super(EncodeVAE, self).__init__()
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
        image = self._get_previous_item(variation, self.in_name, index)

        if self.dtype:
            image = image.to(dtype=self.dtype)

        # With high concurrency, we can occasionally get memory spikes here
        # that prevent allocation even if we're nowhere near the memory limit
        # of the GPU. So try a few times.
        retries = 0
        while True:
            try:
                with self._all_contexts(self.autocast_contexts):
                    image = image.unsqueeze(dim=0) #add batch dimension
                    vae_output = self.vae.encode(image)
                    if hasattr(vae_output, "latent_dist"):
                        output = vae_output.latent_dist
                    if hasattr(vae_output, "latent"):
                        output = vae_output.latent.squeeze(dim=0)
                    break
            except RuntimeError:
                retries += 1
                if retries > 3:
                    raise

        return {
            self.out_name: output
        }
