import os

import torch
from diffusers.models.autoencoder_kl import AutoencoderKL
from torchvision import transforms
from tqdm import tqdm

from .TrainDataSet import PipelineModule


class SaveImage(PipelineModule):
    def __init__(self, image_in_name: str, original_path_in_name: str, path: str, in_range_min: float, in_range_max: float):
        super(SaveImage, self).__init__()
        self.image_in_name = image_in_name
        self.original_path_in_name = original_path_in_name
        self.path = path
        self.in_range_min = in_range_min
        self.in_range_max = in_range_max

    def length(self) -> int:
        return self.get_previous_length(self.image_in_name)

    def get_inputs(self) -> list[str]:
        return [self.image_in_name, self.original_path_in_name]

    def get_outputs(self) -> list[str]:
        return []

    def preprocess(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        for index in tqdm(range(self.get_previous_length(self.image_in_name)), desc='writing debug images for \'' + self.image_in_name + '\''):
            image_tensor = self.get_previous_item(self.image_in_name, index)
            original_path = self.get_previous_item(self.original_path_in_name, index)
            name = os.path.basename(original_path)
            name, ext = os.path.splitext(name)

            t = transforms.Compose([
                transforms.ToPILImage(),
            ])

            image_tensor = (image_tensor - self.in_range_min) / (self.in_range_max - self.in_range_min)

            image = t(image_tensor)
            image.save(os.path.join(self.path, name + '-' + self.image_in_name + ext))

    def get_item(self, index: int, requested_name: str = None) -> dict:
        return {}


class DecodeVAE(PipelineModule):
    def __init__(self, in_name: str, out_name: str, vae: AutoencoderKL):
        super(DecodeVAE, self).__init__()
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
        latent_image = self.get_previous_item(self.in_name, index)

        with torch.no_grad():
            image = self.vae.decode(latent_image.unsqueeze(0)).sample
            image = image.clamp(-1, 1).squeeze()

        return {
            self.out_name: image
        }
