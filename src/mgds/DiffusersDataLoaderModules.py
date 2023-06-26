from contextlib import nullcontext

import torch
from diffusers import VQModel
from diffusers.models.autoencoder_kl import AutoencoderKL

from .MGDS import PipelineModule


class EncodeVAE(PipelineModule):
    def __init__(self, in_name: str, out_name: str, vae: AutoencoderKL):
        super(EncodeVAE, self).__init__()
        self.in_name = in_name
        self.out_name = out_name
        self.vae = vae

    def length(self) -> int:
        return self.get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def get_item(self, index: int, requested_name: str = None) -> dict:
        image = self.get_previous_item(self.in_name, index)

        image = image.to(device=image.device, dtype=self.pipeline.dtype)

        with torch.no_grad():
            with torch.autocast(self.pipeline.device.type) if self.pipeline.allow_mixed_precision else nullcontext():
                latent_distribution = self.vae.encode(image.unsqueeze(0)).latent_dist

        return {
            self.out_name: latent_distribution
        }


class EncodeMoVQ(PipelineModule):
    def __init__(
            self,
            in_name: str,
            out_name: str,
            movq: VQModel,
    ):
        super(EncodeMoVQ, self).__init__()
        self.in_name = in_name
        self.out_name = out_name
        self.movq = movq

    def length(self) -> int:
        return self.get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def get_item(self, index: int, requested_name: str = None) -> dict:
        image = self.get_previous_item(self.in_name, index)

        image = image.to(device=image.device, dtype=self.pipeline.dtype)

        with torch.no_grad():
            with torch.autocast(self.pipeline.device.type) if self.pipeline.allow_mixed_precision else nullcontext():
                latent_image = self.movq.encode(image.unsqueeze(0)).latents

        latent_image = latent_image.squeeze()

        return {
            self.out_name: latent_image,
        }


class SampleVAEDistribution(PipelineModule):
    def __init__(self, in_name: str, out_name: str, mode='mean'):
        super(SampleVAEDistribution, self).__init__()
        self.in_name = in_name
        self.out_name = out_name
        self.mode = mode

    def length(self) -> int:
        return self.get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def get_item(self, index: int, requested_name: str = None) -> dict:
        distribution = self.get_previous_item(self.in_name, index)

        if self.mode == 'sample':
            latent = distribution.sample()
        elif self.mode == 'mean':
            latent = distribution.mode()
        else:
            raise Exception('method not supported')

        latent = latent.squeeze()

        return {
            self.out_name: latent
        }


class RandomLatentMaskRemove(PipelineModule):
    def __init__(self, latent_mask_name: str, latent_conditioning_image_name: str, possible_resolutions_in_name: str, replace_probability: float, vae: AutoencoderKL):
        super(RandomLatentMaskRemove, self).__init__()
        self.latent_mask_name = latent_mask_name
        self.latent_conditioning_image_name = latent_conditioning_image_name
        self.possible_resolutions_in_name = possible_resolutions_in_name
        self.replace_probability = replace_probability
        self.vae = vae

        self.inputs_outputs = [latent_mask_name]
        if latent_conditioning_image_name is not None:
            self.inputs_outputs.append(latent_conditioning_image_name)

        self.full_mask_cache = {}
        self.blank_conditioning_image_cache = {}

    def length(self) -> int:
        return self.get_previous_length(self.latent_mask_name)

    def get_inputs(self) -> list[str]:
        return self.inputs_outputs

    def get_outputs(self) -> list[str]:
        return self.inputs_outputs

    def start(self):
        possible_resolutions = self.get_previous_meta(self.possible_resolutions_in_name)

        with torch.no_grad():
            with torch.autocast(self.pipeline.device.type) if self.pipeline.allow_mixed_precision else nullcontext():
                for resolution in possible_resolutions:
                    blank_conditioning_image = torch.zeros(resolution, dtype=self.pipeline.dtype, device=self.pipeline.device)
                    blank_conditioning_image = blank_conditioning_image.unsqueeze(0).unsqueeze(0).expand([-1, 3, -1, -1])
                    self.blank_conditioning_image_cache[resolution] = self.vae.encode(blank_conditioning_image).latent_dist.mode().squeeze()

    def get_item(self, index: int, requested_name: str = None) -> dict:
        rand = self._get_rand(index)
        latent_mask = self.get_previous_item(self.latent_mask_name, index)
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
            latent_conditioning_image = self.get_previous_item(self.latent_conditioning_image_name, index)

        if self.latent_conditioning_image_name is not None:
            return {
                self.latent_mask_name: latent_mask,
                self.latent_conditioning_image_name: latent_conditioning_image,
            }
        else:
            return {
                self.latent_mask_name: latent_mask,
            }
