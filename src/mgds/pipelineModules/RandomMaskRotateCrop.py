import math
from random import Random

import torch
from torch import Tensor
from torchvision.transforms import functional, InterpolationMode

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class RandomMaskRotateCrop(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            mask_name: str,
            additional_names: list[str],
            enabled_in_name: str,
            min_size: int,
            min_padding_percent: float,
            max_padding_percent: float,
            max_rotate_angle: float = 0
    ):
        super(RandomMaskRotateCrop, self).__init__()
        self.mask_name = mask_name
        self.additional_names = additional_names
        self.enabled_in_name = enabled_in_name
        self.min_size = min_size
        self.min_padding_percent = min_padding_percent
        self.max_padding_percent = max_padding_percent
        self.max_rotate_angle = max_rotate_angle

    def length(self) -> int:
        return self._get_previous_length(self.mask_name)

    def get_inputs(self) -> list[str]:
        return [self.mask_name] + self.additional_names

    def get_outputs(self) -> list[str]:
        return [self.mask_name] + self.additional_names

    @staticmethod
    def __get_masked_region(mask: Tensor) -> (int, int, int, int):
        # Find the first and last occurrence of a 1 in the mask by
        # 1. reducing the 2D image tensor to a 1D tensor
        # 2. multiplying the result by an ascending or descending sequence
        # 3. getting the max value of this sequence

        # y/height direction
        reduced_mask = (torch.amax(mask, dim=-1, keepdim=True) > 0.5).float()
        height = reduced_mask.shape[-2]

        ascending_sequence = torch.arange(0, height, 1, device=mask.device, dtype=mask.dtype).unsqueeze(1)
        while ascending_sequence.ndim < mask.ndim:
            ascending_sequence = ascending_sequence.unsqueeze(0)
        ascending_mask = reduced_mask * ascending_sequence

        descending_sequence = torch.arange(height, 0, -1, device=mask.device, dtype=mask.dtype).unsqueeze(1)
        while descending_sequence.ndim < mask.ndim:
            descending_sequence = descending_sequence.unsqueeze(0)
        descending_mask = reduced_mask * descending_sequence

        y_min = height - torch.max(descending_mask).item()
        y_max = torch.max(ascending_mask).item()

        # x/width direction
        reduced_mask = (torch.amax(mask, dim=-2, keepdim=True) > 0.5).float()
        width = reduced_mask.shape[-1]

        ascending_sequence = torch.arange(0, width, 1, device=mask.device, dtype=mask.dtype).unsqueeze(0)
        while ascending_sequence.ndim < mask.ndim:
            ascending_sequence = ascending_sequence.unsqueeze(0)
        ascending_mask = reduced_mask * ascending_sequence

        descending_sequence = torch.arange(width, 0, -1, device=mask.device, dtype=mask.dtype).unsqueeze(0)
        while descending_sequence.ndim < mask.ndim:
            descending_sequence = descending_sequence.unsqueeze(0)
        descending_mask = reduced_mask * descending_sequence

        x_min = width - torch.max(descending_mask).item()
        x_max = torch.max(ascending_mask).item()

        # safety check, if the found region is negative in size
        # this can happen if no mask exists
        if y_max < y_min or x_max < x_min:
            y_min = 0
            y_max = height
            x_min = 0
            x_max = width

        return y_min, y_max, x_min, x_max

    @staticmethod
    def __rotate(tensor: Tensor, center: list[int], angle: float) -> Tensor:
        orig_dtype = tensor.dtype
        if orig_dtype == torch.bfloat16:
            tensor = tensor.to(dtype=torch.float32)
        tensor = functional.rotate(tensor, angle, interpolation=InterpolationMode.BILINEAR, center=center)
        return tensor.to(dtype=orig_dtype)

    @staticmethod
    def __crop(tensor: Tensor, y_min: int, y_max: int, x_min: int, x_max: int) -> Tensor:
        return functional.crop(tensor, y_min, x_min, y_max - y_min, x_max - x_min)

    def __apply(self, rand: Random, mask: Tensor, item: dict[str, Tensor]):
        mask_height = mask.shape[-2]
        mask_width = mask.shape[-1]

        # get initial dimensions for rotation
        y_min, y_max, x_min, x_max = self.__get_masked_region(mask)
        y_center = (y_max + y_min) / 2
        x_center = (x_max + x_min) / 2

        # rotate
        angle = rand.uniform(-self.max_rotate_angle, self.max_rotate_angle)
        mask = self.__rotate(mask, [x_center, y_center], angle)

        for key in item.keys():
            item[key] = self.__rotate(item[key], [x_center, y_center], angle)

        # get dimensions for cropping
        y_min, y_max, x_min, x_max = self.__get_masked_region(mask)

        height = y_max - y_min
        width = x_max - x_min

        min_height = height / (1 - (self.min_padding_percent / 100))
        min_width = width / (1 - (self.min_padding_percent / 100))

        max_height = height / (1 - (self.max_padding_percent / 100))
        max_width = width / (1 - (self.max_padding_percent / 100))

        min_y_expand = (min_height - height) / 2
        min_x_expand = (min_width - width) / 2

        max_y_expand = (max_height - height) / 2
        max_x_expand = (max_width - width) / 2

        y_expand_top = rand.uniform(min_y_expand, max_y_expand)
        y_expand_bottom = rand.uniform(min_y_expand, max_y_expand)
        x_expand_left = rand.uniform(min_x_expand, max_x_expand)
        x_expand_right = rand.uniform(min_x_expand, max_x_expand)

        # stretch region
        y_min -= y_expand_top
        y_max += y_expand_bottom
        x_min -= x_expand_left
        x_max += x_expand_right

        # increase size of region in case it is smaller than self.min_size, while preserving the aspect ratio
        area = (y_max - y_min) * (x_max - x_min)
        min_area = self.min_size * self.min_size
        if area < min_area:
            scale = math.sqrt(min_area / area)
            y_expand = (scale - 1) * (y_max - y_min)
            x_expand = (scale - 1) * (x_max - x_min)
            y_min -= y_expand
            y_max += y_expand
            x_min -= x_expand
            x_max += x_expand

        # move the region back into the image bounds
        if y_min < 0:
            y_shift = -y_min
            y_min += y_shift
            y_max += y_shift
        if y_max > mask_height:
            y_shift = mask_height - y_max
            y_min += y_shift
            y_max += y_shift
        if x_min < 0:
            x_shift = -x_min
            x_min += x_shift
            x_max += x_shift
        if x_max > mask_width:
            x_shift = mask_width - x_max
            x_min += x_shift
            x_max += x_shift

        # crop to image bounds
        y_min = int(max(0, y_min))
        y_max = int(min(mask_height, y_max))
        x_min = int(max(0, x_min))
        x_max = int(min(mask_width, x_max))

        # apply crop
        mask = self.__crop(mask, y_min, y_max, x_min, x_max)
        for key in item.keys():
            item[key] = self.__crop(item[key], y_min, y_max, x_min, x_max)

        # add mask to return value
        item[self.mask_name] = mask

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        enabled = self._get_previous_item(variation, self.enabled_in_name, index)
        mask = self._get_previous_item(variation, self.mask_name, index)

        item = {}
        for name in self.additional_names:
            item[name] = self._get_previous_item(variation, name, index)

        if enabled:
            rand = self._get_rand(variation, index)

            self.__apply(rand, mask, item)
        else:
            item[self.mask_name] = mask

        return item
