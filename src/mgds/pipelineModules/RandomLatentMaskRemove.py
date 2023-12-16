from contextlib import nullcontext

import torch
from diffusers import AutoencoderKL

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class RandomLatentMaskRemove(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            latent_mask_name: str,
            latent_conditioning_image_name: str | None,
            possible_resolutions_in_name: str,
            replace_probability: float,
            vae: AutoencoderKL | None,
            override_allow_mixed_precision: bool | None = None,
    ):
        super(RandomLatentMaskRemove, self).__init__()
        self.latent_mask_name = latent_mask_name
        self.latent_conditioning_image_name = latent_conditioning_image_name
        self.possible_resolutions_in_name = possible_resolutions_in_name
        self.replace_probability = replace_probability
        self.vae = vae
        self.override_allow_mixed_precision = override_allow_mixed_precision

        self.inputs_outputs = [latent_mask_name]
        if latent_conditioning_image_name is not None:
            self.inputs_outputs.append(latent_conditioning_image_name)

        self.full_mask_cache = {}
        self.blank_conditioning_image_cache = {}

    def length(self) -> int:
        return self._get_previous_length(self.latent_mask_name)

    def get_inputs(self) -> list[str]:
        return self.inputs_outputs

    def get_outputs(self) -> list[str]:
        return self.inputs_outputs

    def start(self, variation: int):
        possible_resolutions = self._get_previous_meta(variation, self.possible_resolutions_in_name)

        allow_mixed_precision = self.pipeline.allow_mixed_precision if self.override_allow_mixed_precision is None \
            else self.override_allow_mixed_precision

        if self.latent_conditioning_image_name is not None:
            with torch.no_grad():
                with torch.autocast(self.pipeline.device.type, self.pipeline.dtype) if allow_mixed_precision \
                        else nullcontext():
                    for resolution in possible_resolutions:
                        blank_conditioning_image = torch.zeros(
                            resolution,
                            dtype=self.pipeline.dtype if allow_mixed_precision else self.vae.dtype,
                            device=self.pipeline.device
                        )
                        blank_conditioning_image = blank_conditioning_image\
                            .unsqueeze(0).unsqueeze(0).expand([-1, 3, -1, -1])
                        self.blank_conditioning_image_cache[resolution] = self.vae.encode(
                            blank_conditioning_image).latent_dist.mode().squeeze()

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        rand = self._get_rand(variation, index)
        latent_mask = self._get_previous_item(variation, self.latent_mask_name, index)
        latent_resolution = (latent_mask.shape[1], latent_mask.shape[2])
        resolution = (latent_mask.shape[1] * 8, latent_mask.shape[2] * 8)

        if latent_resolution not in self.full_mask_cache:
            self.full_mask_cache[latent_resolution] = torch.ones_like(latent_mask)

        replace = rand.random() < self.replace_probability

        if replace:
            latent_mask = self.full_mask_cache[latent_resolution]

        latent_conditioning_image = None
        if replace and self.latent_conditioning_image_name is not None:
            latent_conditioning_image = self.blank_conditioning_image_cache[resolution]
        elif not replace and self.latent_conditioning_image_name is not None:
            latent_conditioning_image = self._get_previous_item(variation, self.latent_conditioning_image_name, index)

        if self.latent_conditioning_image_name is not None:
            return {
                self.latent_mask_name: latent_mask,
                self.latent_conditioning_image_name: latent_conditioning_image,
            }
        else:
            return {
                self.latent_mask_name: latent_mask,
            }
