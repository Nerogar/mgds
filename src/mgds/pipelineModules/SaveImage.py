import os
from typing import Callable

import torch
from torchvision import transforms
from tqdm import tqdm

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class SaveImage(
    PipelineModule,
    RandomAccessPipelineModule,
):

    def __init__(
            self,
            image_in_name: str,
            original_path_in_name: str,
            path: str,
            in_range_min: float,
            in_range_max: float,
            before_save_fun: Callable[[], None] | None = None,
    ):
        super(SaveImage, self).__init__()
        self.image_in_name = image_in_name
        self.original_path_in_name = original_path_in_name
        self.path = path
        self.in_range_min = in_range_min
        self.in_range_max = in_range_max
        self.before_save_fun = before_save_fun

    def approximate_length(self) -> int:
        return self._get_previous_length(self.image_in_name)

    def get_inputs(self) -> list[str]:
        return [self.image_in_name, self.original_path_in_name]

    def get_outputs(self) -> list[str]:
        return [self.image_in_name]

    def length(self) -> int:
        return 0

    def start(self, variation: int):
        path = os.path.join(self.path, "epoch-" + str(variation))
        if not os.path.exists(path):
            os.makedirs(path)

        if self.before_save_fun is not None:
            self.before_save_fun()

        for index in tqdm(range(self._get_previous_length(self.original_path_in_name)),
                          desc='writing debug images for \'' + self.image_in_name + '\''):
            image_tensor = self._get_previous_item(variation, self.image_in_name, index)
            original_path = self._get_previous_item(variation, self.original_path_in_name, index)
            name = os.path.basename(original_path)
            name, ext = os.path.splitext(name)

            t = transforms.Compose([
                transforms.ToPILImage(),
            ])

            image_tensor = (image_tensor - self.in_range_min) / (self.in_range_max - self.in_range_min)

            image = t(image_tensor.to(dtype=torch.float32))
            image.save(os.path.join(path, str(index) + '-' + name + '-' + self.image_in_name + ext))

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return {}
