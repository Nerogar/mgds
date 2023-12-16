from contextlib import nullcontext

import torch
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class GenerateDepth(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            path_in_name: str,
            image_out_name: str,
            image_depth_processor: DPTImageProcessor,
            depth_estimator: DPTForDepthEstimation,
            override_allow_mixed_precision: bool | None = None,
    ):
        super(GenerateDepth, self).__init__()
        self.path_in_name = path_in_name
        self.image_out_name = image_out_name
        self.image_depth_processor = image_depth_processor
        self.depth_estimator = depth_estimator
        self.override_allow_mixed_precision = override_allow_mixed_precision

    def length(self) -> int:
        return self._get_previous_length(self.path_in_name)

    def get_inputs(self) -> list[str]:
        return [self.path_in_name]

    def get_outputs(self) -> list[str]:
        return [self.image_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        path = self._get_previous_item(variation, self.path_in_name, index)

        try:
            image = Image.open(path)
        except:
            print("could not load image, it might be missing or corrupted: " + path)
            raise

        image = image.convert('RGB')

        allow_mixed_precision = self.pipeline.allow_mixed_precision if self.override_allow_mixed_precision is None \
            else self.override_allow_mixed_precision

        with torch.no_grad():
            with torch.autocast(self.pipeline.device.type, self.pipeline.dtype) if allow_mixed_precision \
                    else nullcontext():
                image = self.image_depth_processor(image, return_tensors="pt").pixel_values
                image = image.to(self.pipeline.device)
                image = image if allow_mixed_precision else image.to(self.depth_estimator.dtype)
                depth = self.depth_estimator(image).predicted_depth

                depth_min = torch.amin(depth, dim=[1, 2], keepdim=True)
                depth_max = torch.amax(depth, dim=[1, 2], keepdim=True)
                depth = 2.0 * (depth - depth_min) / (depth_max - depth_min) - 1.0

        return {
            self.image_out_name: depth
        }
