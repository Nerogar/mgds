import torch
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

        image = image.to(device=image.device, dtype=self.vae.dtype)

        with torch.no_grad():
            latent_distribution = self.vae.encode(image.unsqueeze(0)).latent_dist

        return {
            self.out_name: latent_distribution
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
