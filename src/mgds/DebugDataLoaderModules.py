import os
from contextlib import nullcontext

import torch
from diffusers import VQModel
from diffusers.models.autoencoder_kl import AutoencoderKL
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTokenizer

from .MGDS import PipelineModule


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

    def start_next_epoch(self):
        path = os.path.join(self.path, "epoch-" + str(self.pipeline.current_epoch))
        if not os.path.exists(path):
            os.makedirs(path)

        for index in tqdm(range(self.get_previous_length(self.original_path_in_name)), desc='writing debug images for \'' + self.image_in_name + '\''):
            image_tensor = self.get_previous_item(self.image_in_name, index)
            original_path = self.get_previous_item(self.original_path_in_name, index)
            name = os.path.basename(original_path)
            name, ext = os.path.splitext(name)

            t = transforms.Compose([
                transforms.ToPILImage(),
            ])

            image_tensor = (image_tensor - self.in_range_min) / (self.in_range_max - self.in_range_min)

            image = t(image_tensor)
            image.save(os.path.join(path, name + '-' + self.image_in_name + ext))

    def get_item(self, index: int, requested_name: str = None) -> dict:
        return {}


class SaveText(PipelineModule):
    def __init__(self, text_in_name: str, original_path_in_name: str, path: str):
        super(SaveText, self).__init__()
        self.text_in_name = text_in_name
        self.original_path_in_name = original_path_in_name
        self.path = path

    def length(self) -> int:
        return self.get_previous_length(self.text_in_name)

    def get_inputs(self) -> list[str]:
        return [self.text_in_name, self.original_path_in_name]

    def get_outputs(self) -> list[str]:
        return []

    def start_next_epoch(self):
        path = os.path.join(self.path, "epoch-" + str(self.pipeline.current_epoch))
        if not os.path.exists(path):
            os.makedirs(path)

        for index in tqdm(range(self.get_previous_length(self.original_path_in_name)), desc='writing debug text for \'' + self.text_in_name + '\''):
            text = self.get_previous_item(self.text_in_name, index)
            original_path = self.get_previous_item(self.original_path_in_name, index)
            name = os.path.basename(original_path)
            name, ext = os.path.splitext(name)

            with open(os.path.join(path, name + '-' + self.text_in_name + '.txt'), "w") as f:
                f.write(text)

    def get_item(self, index: int, requested_name: str = None) -> dict:
        return {}


class DecodeVAE(PipelineModule):
    def __init__(
            self,
            in_name: str,
            out_name: str,
            vae: AutoencoderKL,
            override_allow_mixed_precision: bool | None = None,
    ):
        super(DecodeVAE, self).__init__()
        self.in_name = in_name
        self.out_name = out_name
        self.vae = vae
        self.override_allow_mixed_precision = override_allow_mixed_precision

    def length(self) -> int:
        return self.get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def get_item(self, index: int, requested_name: str = None) -> dict:
        latent_image = self.get_previous_item(self.in_name, index)

        allow_mixed_precision = self.pipeline.allow_mixed_precision if self.override_allow_mixed_precision is None \
            else self.override_allow_mixed_precision

        latent_image = latent_image if allow_mixed_precision else latent_image.to(self.vae.dtype)

        with torch.no_grad():
            with torch.autocast(self.pipeline.device.type, self.pipeline.dtype) if allow_mixed_precision \
                    else nullcontext():
                image = self.vae.decode(latent_image.unsqueeze(0)).sample
                image = image.clamp(-1, 1).squeeze()

        return {
            self.out_name: image
        }


class DecodeMoVQ(PipelineModule):
    def __init__(
            self,
            in_name: str,
            out_name: str,
            movq: VQModel,
            override_allow_mixed_precision: bool | None = None,
    ):
        super(DecodeMoVQ, self).__init__()
        self.in_name = in_name
        self.out_name = out_name
        self.movq = movq
        self.override_allow_mixed_precision = override_allow_mixed_precision

    def length(self) -> int:
        return self.get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def get_item(self, index: int, requested_name: str = None) -> dict:
        latent_image = self.get_previous_item(self.in_name, index)

        allow_mixed_precision = self.pipeline.allow_mixed_precision if self.override_allow_mixed_precision is None \
            else self.override_allow_mixed_precision

        latent_image = latent_image if allow_mixed_precision else latent_image.to(self.movq.dtype)

        with torch.no_grad():
            with torch.autocast(self.pipeline.device.type, self.pipeline.dtype) if allow_mixed_precision \
                    else nullcontext():
                image = self.movq.decode(latent_image.unsqueeze(0)).sample
                image = image.clamp(-1, 1).squeeze()

        return {
            self.out_name: image,
        }


class DecodeTokens(PipelineModule):
    def __init__(self, in_name: str, out_name: str, tokenizer: CLIPTokenizer):
        super(DecodeTokens, self).__init__()
        self.in_name = in_name
        self.out_name = out_name
        self.tokenizer = tokenizer

    def length(self) -> int:
        return self.get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def get_item(self, index: int, requested_name: str = None) -> dict:
        tokens = self.get_previous_item(self.in_name, index)

        text = self.tokenizer.decode(
            token_ids=tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return {
            self.out_name: text
        }
