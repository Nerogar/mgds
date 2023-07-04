import traceback
from contextlib import nullcontext

import torch
from PIL import Image
from transformers import DPTForDepthEstimation, DPTImageProcessor, CLIPTokenizer

from .MGDS import PipelineModule


class GenerateDepth(PipelineModule):
    def __init__(self, path_in_name: str, image_out_name: str, image_depth_processor: DPTImageProcessor, depth_estimator: DPTForDepthEstimation):
        super(GenerateDepth, self).__init__()
        self.path_in_name = path_in_name
        self.image_out_name = image_out_name
        self.image_depth_processor = image_depth_processor
        self.depth_estimator = depth_estimator

    def length(self) -> int:
        return self.get_previous_length(self.path_in_name)

    def get_inputs(self) -> list[str]:
        return [self.path_in_name]

    def get_outputs(self) -> list[str]:
        return [self.image_out_name]

    def get_item(self, index: int, requested_name: str = None) -> dict:
        path = self.get_previous_item(self.path_in_name, index)

        try:
            image = Image.open(path)
        except:
            print("could not load image, it might be missing or corrupted: " + path)
            raise

        image = image.convert('RGB')

        with torch.no_grad():
            with torch.autocast(self.pipeline.device.type) if self.pipeline.allow_mixed_precision else nullcontext():
                image = self.image_depth_processor(image, return_tensors="pt").pixel_values
                image = image.to(self.pipeline.device)
                depth = self.depth_estimator(image).predicted_depth

                depth_min = torch.amin(depth, dim=[1, 2], keepdim=True)
                depth_max = torch.amax(depth, dim=[1, 2], keepdim=True)
                depth = 2.0 * (depth - depth_min) / (depth_max - depth_min) - 1.0

        return {
            self.image_out_name: depth
        }


class Tokenize(PipelineModule):
    def __init__(
            self,
            in_name: str,
            tokens_out_name: str,
            mask_out_name: str,
            tokenizer: CLIPTokenizer,
            max_token_length: int,
    ):
        super(Tokenize, self).__init__()
        self.in_name = in_name
        self.tokens_out_name = tokens_out_name
        self.mask_out_name = mask_out_name
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length

    def length(self) -> int:
        return self.get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.tokens_out_name, self.mask_out_name]

    def get_item(self, index: int, requested_name: str = None) -> dict:
        text = self.get_previous_item(self.in_name, index)

        tokenizer_output = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_token_length,
            return_tensors="pt",
        )

        tokens = tokenizer_output.input_ids.to(self.pipeline.device)
        mask = tokenizer_output.attention_mask.to(self.pipeline.device)

        tokens = tokens.squeeze()

        return {
            self.tokens_out_name: tokens,
            self.mask_out_name: mask,
        }
