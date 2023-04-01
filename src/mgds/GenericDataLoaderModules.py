import math
import os

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import functional, InterpolationMode
from tqdm import tqdm

from .MGDS import PipelineModule


class CollectPaths(PipelineModule):
    def __init__(
            self,
            concept_in_name: str, path_in_name: str, name_in_name: str,
            path_out_name: str, concept_out_name: str,
            extensions: [str], include_postfix: [str], exclude_postfix: [str]
    ):
        super(CollectPaths, self).__init__()

        self.concept_in_name = concept_in_name
        self.path_in_name = path_in_name
        self.name_in_name = name_in_name

        self.path_out_name = path_out_name
        self.concept_out_name = concept_out_name

        self.extensions = [extension.lower() for extension in extensions]
        self.include_postfix = include_postfix
        self.exclude_postfix = exclude_postfix

        self.image_paths = []
        self.concept_name = []
        self.concepts = {}

    def length(self) -> int:
        return len(self.image_paths)

    def get_inputs(self) -> list[str]:
        return [self.concept_in_name]

    def get_outputs(self) -> list[str]:
        return [self.path_out_name, self.concept_out_name]

    def start(self):
        for index in tqdm(range(self.get_previous_length(self.concept_in_name)), desc='enumerating sample paths'):
            concept = self.get_previous_item(self.concept_in_name, index)
            path = concept[self.path_in_name]
            concept_name = concept[self.name_in_name]

            file_names = [os.path.join(path, filename) for filename in os.listdir(path)]
            file_names = sorted(file_names)

            file_names = list(filter(lambda name: os.path.splitext(name)[1].lower() in self.extensions, file_names))

            if self.include_postfix:
                file_names = list(filter(lambda name: any(os.path.splitext(name)[0].endswith(postfix) for postfix in self.include_postfix), file_names))

            if self.exclude_postfix:
                file_names = list(filter(lambda name: not any(os.path.splitext(name)[0].endswith(postfix) for postfix in self.exclude_postfix), file_names))

            self.image_paths.extend(file_names)
            self.concept_name.extend([concept_name] * len(file_names))
            self.concepts[concept_name] = concept

    def get_item(self, index: int, requested_name: str = None) -> dict:
        return {
            self.path_out_name: self.image_paths[index],
            self.concept_out_name: self.concepts[self.concept_name[index]],
        }


class ModifyPath(PipelineModule):
    def __init__(self, in_name: str, out_name: str, postfix: [str], extension: [str]):
        super(ModifyPath, self).__init__()

        self.in_name = in_name
        self.out_name = out_name

        self.postfix = postfix
        self.extension = extension

        self.extra_paths = []

    def length(self) -> int:
        return self.get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def start(self):
        for index in range(self.get_previous_length(self.in_name)):
            image_path = self.get_previous_item(self.in_name, index)

            image_name = os.path.splitext(image_path)[0]
            extra_path = image_name + self.postfix + self.extension

            self.extra_paths.append(extra_path)

    def get_item(self, index: int, requested_name: str = None) -> (str, object):
        return {
            self.out_name: self.extra_paths[index],
        }


class CalcAspect(PipelineModule):
    def __init__(self, image_in_name: str, resolution_out_name: str):
        super(CalcAspect, self).__init__()
        self.image_in_name = image_in_name
        self.resolution_out_name = resolution_out_name

    def length(self) -> int:
        return self.get_previous_length(self.image_in_name)

    def get_inputs(self) -> list[str]:
        return [self.image_in_name]

    def get_outputs(self) -> list[str]:
        return [self.resolution_out_name]

    def get_item(self, index: int, requested_name: str = None) -> dict:
        image = self.get_previous_item(self.image_in_name, index)

        height, width = image.shape[1], image.shape[2]

        return {
            self.resolution_out_name: (height, width),
        }


class AspectBucketing(PipelineModule):
    def __init__(self, batch_size: int, target_resolution: int,
                 resolution_in_name: str,
                 scale_resolution_out_name: str, crop_resolution_out_name: str):
        super(AspectBucketing, self).__init__()

        self.batch_size = batch_size
        self.target_resolution = target_resolution

        self.resolution_in_name = resolution_in_name

        self.scale_resolution_out_name = scale_resolution_out_name
        self.crop_resolution_out_name = crop_resolution_out_name

        self.possible_resolutions, self.possible_aspects = self.create_buckets(target_resolution)

    def length(self) -> int:
        return self.get_previous_length(self.resolution_in_name)

    def get_inputs(self) -> list[str]:
        return [self.resolution_in_name]

    def get_outputs(self) -> list[str]:
        return [self.scale_resolution_out_name, self.crop_resolution_out_name]

    @staticmethod
    def create_buckets(target_resolution: int) -> (np.ndarray, np.ndarray):
        quantization = 8

        # all possible target aspect ratios
        possible_resolutions = np.array([
            (1.0, 1.0),
            (1.0, 1.25),
            (1.0, 1.5),
            (1.0, 1.75),
            (1.0, 2.0),
            (1.0, 2.0),
            (1.0, 2.5),
            (1.0, 3.0),
            (1.0, 3.5),
            (1.0, 4.0),
        ])

        # normalize to the same pixel count
        possible_resolutions = [(
            h / math.sqrt(h * w) * target_resolution,
            w / math.sqrt(h * w) * target_resolution
        ) for (h, w) in possible_resolutions]

        # add inverted dimensions
        possible_resolutions = possible_resolutions + [(w, h) for (h, w) in possible_resolutions]

        # quantization
        possible_resolutions = [(
            round(h / quantization) * quantization,
            round(w / quantization) * quantization,
        ) for (h, w) in possible_resolutions]

        # remove duplicates
        possible_resolutions = list(set(possible_resolutions))

        possible_aspects = np.array([h / w for (h, w) in possible_resolutions])
        return possible_resolutions, possible_aspects

    def get_bucket(self, h: int, w: int):
        aspect = h / w
        bucket_index = np.argmin(abs(self.possible_aspects - aspect))
        return self.possible_resolutions[bucket_index]

    def get_item(self, index: int, requested_name: str = None) -> dict:
        resolution = self.get_previous_item(self.resolution_in_name, index)

        target_resolution = self.get_bucket(resolution[0], resolution[1])

        aspect = resolution[0] / resolution[1]
        target_aspect = target_resolution[0] / target_resolution[1]

        if aspect > target_aspect:
            scale = target_resolution[1] / resolution[1]
            scale_resolution = (
                round(resolution[0] * scale),
                target_resolution[1]
            )
        else:
            scale = target_resolution[0] / resolution[0]
            scale_resolution = (
                target_resolution[0],
                round(resolution[1] * scale)
            )

        return {
            self.scale_resolution_out_name: scale_resolution,
            self.crop_resolution_out_name: target_resolution,
        }


class LoadImage(PipelineModule):
    def __init__(self, path_in_name: str, image_out_name: str, range_min: float, range_max: float, channels: int = 3):
        super(LoadImage, self).__init__()
        self.path_in_name = path_in_name
        self.image_out_name = image_out_name

        self.range_min = range_min
        self.range_max = range_max

        if channels == 3:
            self.mode = 'RGB'
        elif channels == 1:
            self.mode = 'L'
        else:
            raise ValueError('Only 1 and 3 channels are supported.')

    def length(self) -> int:
        return self.get_previous_length(self.path_in_name)

    def get_inputs(self) -> list[str]:
        return [self.path_in_name]

    def get_outputs(self) -> list[str]:
        return [self.image_out_name]

    def get_item(self, index: int, requested_name: str = None) -> dict:
        path = self.get_previous_item(self.path_in_name, index)

        image_tensor = Image.open(path)
        image_tensor = image_tensor.convert(self.mode)

        t = transforms.ToTensor()
        image_tensor = t(image_tensor).to(self.pipeline.device)

        image_tensor = image_tensor * (self.range_max - self.range_min) + self.range_min

        return {
            self.image_out_name: image_tensor
        }


class ScaleCropImage(PipelineModule):
    def __init__(self, image_in_name: str, scale_resolution_in_name: str, crop_resolution_in_name: str, image_out_name: str):
        super(ScaleCropImage, self).__init__()
        self.image_in_name = image_in_name
        self.scale_resolution_in_name = scale_resolution_in_name
        self.crop_resolution_in_name = crop_resolution_in_name
        self.image_out_name = image_out_name

    def length(self) -> int:
        return self.get_previous_length(self.image_in_name)

    def get_inputs(self) -> list[str]:
        return [self.image_in_name, self.scale_resolution_in_name, self.crop_resolution_in_name]

    def get_outputs(self) -> list[str]:
        return [self.image_out_name]

    def get_item(self, index: int, requested_name: str = None) -> dict:
        image = self.get_previous_item(self.image_in_name, index)
        scale_resolution = self.get_previous_item(self.scale_resolution_in_name, index)
        crop_resolution = self.get_previous_item(self.crop_resolution_in_name, index)

        t = transforms.Compose([
            transforms.Resize(scale_resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(crop_resolution),
        ])

        image = t(image)

        return {
            self.image_out_name: image
        }


class LoadText(PipelineModule):
    def __init__(self, path_in_name: str, text_out_name: str):
        super(LoadText, self).__init__()
        self.path_in_name = path_in_name
        self.text_out_name = text_out_name

    def length(self) -> int:
        return self.get_previous_length(self.path_in_name)

    def get_inputs(self) -> list[str]:
        return [self.path_in_name]

    def get_outputs(self) -> list[str]:
        return [self.text_out_name]

    def get_item(self, index: int, requested_name: str = None) -> dict:
        path = self.get_previous_item(self.path_in_name, index)

        text = ''
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                text = f.readline().strip()
                f.close()

        return {
            self.text_out_name: text
        }


class RandomFlip(PipelineModule):
    def __init__(self, names: [str]):
        super(RandomFlip, self).__init__()
        self.names = names

    def length(self) -> int:
        return self.get_previous_length(self.names[0])

    def get_inputs(self) -> list[str]:
        return self.names

    def get_outputs(self) -> list[str]:
        return self.names

    def get_item(self, index: int, requested_name: str = None) -> dict:
        rand = self._get_rand(index)
        item = {}

        check = rand.random()
        flip = check < 0.5

        for name in self.names:
            previous_item = self.get_previous_item(name, index)
            if flip:
                previous_item = functional.hflip(previous_item)
            item[name] = previous_item

        return item


class Downscale(PipelineModule):
    def __init__(self, in_name: str, out_name: str):
        super(Downscale, self).__init__()
        self.in_name = in_name
        self.out_name = out_name

    def length(self) -> int:
        return self.get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def get_item(self, index: int, requested_name: str = None) -> dict:
        image = self.get_previous_item(self.in_name, index)

        size = (int(image.shape[1] / 8), int(image.shape[2] / 8))

        t = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        ])

        image = t(image)

        return {
            self.out_name: image
        }


class DiskCache(PipelineModule):
    def __init__(self, cache_dir: str, split_names=None, aggregate_names=None):
        super(DiskCache, self).__init__()
        self.cache_dir = cache_dir
        self.split_names = [] if split_names is None else split_names
        self.aggregate_names = [] if aggregate_names is None else aggregate_names
        self.cache_length = None
        self.aggregate_cache = None

        if len(self.split_names) + len(self.aggregate_names) == 0:
            raise ValueError('No cache items supplied')

    def length(self) -> int:
        if not self.cache_length:
            name = self.split_names[0] if len(self.split_names) > 0 else self.aggregate_names[0]
            return self.get_previous_length(name)
        else:
            return self.cache_length

    def get_inputs(self) -> list[str]:
        return self.split_names + self.aggregate_names

    def get_outputs(self) -> list[str]:
        return self.split_names + self.aggregate_names

    def start(self):
        caching_done = False

        if os.path.isdir(self.cache_dir):
            with os.scandir(self.cache_dir) as path_iter:
                if any(path_iter):
                    caching_done = True

        if caching_done:
            if len(self.aggregate_names) > 0:
                self.aggregate_cache = torch.load(os.path.join(self.cache_dir, 'aggregate.pt'))
                length = len(self.aggregate_cache)
            else:
                length = len(os.listdir(self.cache_dir))
        else:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)

            length = self.length()
            self.aggregate_cache = []

            for index in tqdm(range(length), desc='caching'):
                split_item = {}
                aggregate_item = {}

                for name in self.split_names:
                    split_item[name] = self.get_previous_item(name, index)
                for name in self.aggregate_names:
                    aggregate_item[name] = self.get_previous_item(name, index)

                torch.save(split_item, os.path.join(self.cache_dir, str(index) + '.pt'))
                self.aggregate_cache.append(aggregate_item)

            torch.save(self.aggregate_cache, os.path.join(self.cache_dir, 'aggregate.pt'))

        self.cache_length = length

    def get_item(self, index: int, requested_name: str = None) -> dict:
        item = {}

        aggregate_item = self.aggregate_cache[index]
        for name in self.aggregate_names:
            item[name] = aggregate_item[name]

        if requested_name in self.split_names:
            split_item = torch.load(os.path.join(self.cache_dir, str(index) + '.pt'))

            for name in self.split_names:
                item[name] = split_item[name]

        return item


class AspectBatchSorting(PipelineModule):
    def __init__(self, resolution_in_name: str, names: [str], batch_size: int, sort_resolutions_for_each_epoch: bool):
        super(AspectBatchSorting, self).__init__()
        self.resolution_in_name = resolution_in_name
        self.names = names
        self.batch_size = batch_size
        self.sort_resolutions_for_each_epoch = sort_resolutions_for_each_epoch

        self.bucket_dict = {}
        self.index_list = []

    def length(self) -> int:
        return len(self.index_list)

    def get_inputs(self) -> list[str]:
        return [self.resolution_in_name] + self.names

    def get_outputs(self) -> list[str]:
        return self.names

    def shuffle(self) -> list[int]:
        rand = self._get_rand()

        bucket_dict = {key: value.copy() for (key, value) in self.bucket_dict.items()}

        # generate a shuffled list of batches in the format (resolution, batch index within resolution)
        batches = []
        for bucket_key in bucket_dict.keys():
            batch_count = int(len(bucket_dict[bucket_key]) / self.batch_size)
            batches.extend((bucket_key, i) for i in range(batch_count))
        rand.shuffle(batches)

        # for each bucket, generate a shuffled list of samples
        for bucket_key, bucket in bucket_dict.items():
            rand.shuffle(bucket)

        # drop images for full buckets
        for bucket_key in bucket_dict.keys():
            samples = bucket_dict[bucket_key]
            samples_to_drop = len(samples) % self.batch_size
            for i in range(samples_to_drop):
                print('dropping sample from bucket ' + str(bucket_key))
                samples.pop()

        # calculate the order of samples
        index_list = []
        for bucket_key, bucket_index in batches:
            for i in range(bucket_index * self.batch_size, (bucket_index + 1) * self.batch_size):
                index_list.append(bucket_dict[bucket_key][i])

        print(bucket_dict)
        print(index_list)

        return index_list

    def __sort_resolutions(self):
        resolutions = []
        for index in tqdm(range(self.get_previous_length(self.resolution_in_name)), desc='caching resolutions'):
            resolution = self.get_previous_item(self.resolution_in_name, index)

            resolution = resolution[0], resolution[1]
            resolutions.append(resolution)

        # sort samples into dict of lists, with key = resolution
        self.bucket_dict = {}
        for index, resolution in enumerate(resolutions):
            if resolution not in self.bucket_dict:
                self.bucket_dict[resolution] = []
            self.bucket_dict[resolution].append(index)

    def start(self):
        if not self.sort_resolutions_for_each_epoch:
            self.__sort_resolutions()

    def start_next_epoch(self):
        if self.sort_resolutions_for_each_epoch:
            self.__sort_resolutions()

        self.index_list = self.shuffle()

    def get_item(self, index: int, requested_name: str = None) -> dict:
        index = self.index_list[index]

        item = {}

        for name in self.names:
            item[name] = self.get_previous_item(name, index)

        return item


class GenerateMaskedConditioningImage(PipelineModule):
    def __init__(self, image_in_name: str, mask_in_name: str, image_out_name: str):
        super(GenerateMaskedConditioningImage, self).__init__()
        self.image_in_name = image_in_name
        self.mask_in_name = mask_in_name
        self.image_out_name = image_out_name

    def length(self) -> int:
        return self.get_previous_length(self.image_in_name)

    def get_inputs(self) -> list[str]:
        return [self.image_in_name, self.mask_in_name]

    def get_outputs(self) -> list[str]:
        return [self.image_out_name]

    def get_item(self, index: int, requested_name: str = None) -> dict:
        image = self.get_previous_item(self.image_in_name, index)
        mask = self.get_previous_item(self.mask_in_name, index)

        conditioning_image = image * (1 - mask)

        return {
            self.image_out_name: conditioning_image
        }


class RandomMaskRotateCrop(PipelineModule):
    def __init__(self, mask_name: str, additional_names: [str], min_size: int, min_padding_percent: float, max_padding_percent: float, max_rotate_angle: float = 0):
        super(RandomMaskRotateCrop, self).__init__()
        self.mask_name = mask_name
        self.additional_names = additional_names
        self.min_size = min_size
        self.min_padding_percent = min_padding_percent
        self.max_padding_percent = max_padding_percent
        self.max_rotate_angle = max_rotate_angle

    def length(self) -> int:
        return self.get_previous_length(self.mask_name)

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
        reduced_mask = (torch.amax(mask, dim=2, keepdim=True) > 0.5).float()
        height = reduced_mask.shape[1]

        ascending_sequence = torch.arange(0, height, 1, device=mask.device, dtype=mask.dtype).unsqueeze(0).unsqueeze(2)
        ascending_mask = reduced_mask * ascending_sequence

        descending_sequence = torch.arange(height, 0, -1, device=mask.device, dtype=mask.dtype).unsqueeze(0).unsqueeze(2)
        descending_mask = reduced_mask * descending_sequence

        y_min = height - torch.max(descending_mask).item()
        y_max = torch.max(ascending_mask).item()

        # x/width direction
        reduced_mask = (torch.amax(mask, dim=1, keepdim=True) > 0.5).float()
        width = reduced_mask.shape[2]

        ascending_sequence = torch.arange(0, width, 1, device=mask.device, dtype=mask.dtype).unsqueeze(0).unsqueeze(0)
        ascending_mask = reduced_mask * ascending_sequence

        descending_sequence = torch.arange(width, 0, -1, device=mask.device, dtype=mask.dtype).unsqueeze(0).unsqueeze(0)
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
        return functional.rotate(tensor, angle, interpolation=InterpolationMode.BILINEAR, center=center)

    @staticmethod
    def __crop(tensor: Tensor, y_min: int, y_max: int, x_min: int, x_max: int) -> Tensor:
        return functional.crop(tensor, y_min, x_min, y_max - y_min, x_max - x_min)

    def get_item(self, index: int, requested_name: str = None) -> dict:
        rand = self._get_rand(index)
        mask = self.get_previous_item(self.mask_name, index)

        mask_height = mask.shape[1]
        mask_width = mask.shape[2]

        item = {}

        for name in self.additional_names:
            item[name] = self.get_previous_item(name, index)

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

        return item


class RandomCircularMaskShrink(PipelineModule):
    def __init__(self, mask_name: str, shrink_probability: float, shrink_factor_min: float, shrink_factor_max: float = 1):
        super(RandomCircularMaskShrink, self).__init__()

        self.mask_name = mask_name
        self.shrink_probability = shrink_probability
        self.shrink_factor_min = shrink_factor_min
        self.shrink_factor_max = shrink_factor_max

    def length(self) -> int:
        return self.get_previous_length(self.mask_name)

    def get_inputs(self) -> list[str]:
        return [self.mask_name]

    def get_outputs(self) -> list[str]:
        return [self.mask_name]

    @staticmethod
    def __get_random_point_in_mask(mask: Tensor, seed: int) -> (int, int):
        # Generate a random point within a binary tensor by
        # 1. generating a new tensor with random uniform numbers in the same shape as the mask
        # 2. multiplying the mask with the random tensor
        # 3. using argmax, get the point with the highest random number within the masked region

        generator = torch.Generator(device=mask.device)
        generator.manual_seed(seed)
        random = torch.rand(size=mask.shape, generator=generator, dtype=mask.dtype, device=mask.device)
        random_mask = torch.flatten(random * mask)
        max_index_flat = torch.argmax(random_mask).item()
        max_index = (max_index_flat // random.shape[2], max_index_flat % random.shape[2])

        return max_index

    @staticmethod
    def __get_radial_gradient(mask: Tensor, center: (int, int)) -> Tensor:
        # Generate a radial gradient where each pixel is calculated as sqrt(x^2 + y^2) by
        # 1. calculating and squaring a gradient in each direction
        # 2. adding both gradients together
        # 3. taking the square root of the resulting gradient

        resolution = (mask.shape[1], mask.shape[2])
        top = -center[0]
        bottom = resolution[0] - center[0] - 1
        left = -center[1]
        right = resolution[1] - center[1] - 1

        vertical_gradient = torch.linspace(start=top, end=bottom, steps=resolution[0], dtype=mask.dtype, device=mask.device)
        vertical_gradient = vertical_gradient * vertical_gradient
        vertical_gradient = vertical_gradient.unsqueeze(1)
        vertical_gradient = vertical_gradient.expand(resolution)

        horizontal_gradient = torch.linspace(start=left, end=right, steps=resolution[1], dtype=mask.dtype, device=mask.device)
        horizontal_gradient = horizontal_gradient * horizontal_gradient
        horizontal_gradient = horizontal_gradient.unsqueeze(0)
        horizontal_gradient = horizontal_gradient.expand(resolution)

        circular_gradient = torch.sqrt(vertical_gradient + horizontal_gradient)
        circular_gradient = circular_gradient.unsqueeze(0)

        return circular_gradient

    @staticmethod
    def __get_disc_mask(radial_gradient: Tensor, radius) -> Tensor:
        return (radial_gradient <= radius).float()

    def get_item(self, index: int, requested_name: str = None) -> dict:
        rand = self._get_rand(index)
        mask = self.get_previous_item(self.mask_name, index)

        random_center = self.__get_random_point_in_mask(mask, rand.randint(0, 1 << 30))
        radial_gradient = self.__get_radial_gradient(mask, random_center)

        max_radius = (mask * radial_gradient).max().item()
        radius = rand.uniform(self.shrink_factor_min, self.shrink_factor_max) * max_radius

        disc_mask = self.__get_disc_mask(radial_gradient, radius)

        result_mask = mask * disc_mask

        return {
            self.mask_name: result_mask,
        }
