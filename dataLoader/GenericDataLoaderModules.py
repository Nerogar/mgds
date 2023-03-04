import math
import os
import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from dataLoader.TrainDataSet import PipelineModule


class CollectPaths(PipelineModule):
    def __init__(self, concept_in_name: str, path_out_name: str, concept_out_name: str, extensions: [str], include_postfix: [str], exclude_postfix: [str]):
        super(CollectPaths, self).__init__()

        self.concept_in_name = concept_in_name
        self.path_out_name = path_out_name
        self.concept_out_name = concept_out_name

        self.extensions = extensions
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

    def preprocess(self):
        for index in tqdm(range(self.get_previous_length(self.concept_in_name)), desc='enumerating sample paths'):
            concept = self.get_previous_item(self.concept_in_name, index)
            path = concept['path']
            concept_name = concept['name']

            file_names = [os.path.join(path, filename) for filename in os.listdir(path)]

            file_names = list(filter(lambda name: os.path.splitext(name)[1] in self.extensions, file_names))

            if self.include_postfix:
                file_names = list(filter(lambda name: any(os.path.splitext(name)[0].endswith(postfix) for postfix in self.include_postfix), file_names))

            if self.exclude_postfix:
                file_names = list(filter(lambda name: not any(os.path.splitext(name)[0].endswith(postfix) for postfix in self.exclude_postfix), file_names))

            self.image_paths.extend(file_names)
            self.concept_name.extend([concept_name] * len(file_names))
            self.concepts[concept_name] = concept

    def get_item(self, index: int) -> dict:
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

    def preprocess(self):
        for index in range(self.get_previous_length(self.in_name)):
            image_path = self.get_previous_item(self.in_name, index)

            image_name = os.path.splitext(image_path)[0]
            extra_path = image_name + self.postfix + self.extension

            self.extra_paths.append(extra_path)

    def get_item(self, index: int) -> (str, object):
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

    def get_item(self, index: int) -> dict:
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

        self.scale_resolutions = []
        self.crop_resolutions = []

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

    def preprocess(self):
        # calculate bucket for each sample
        for index in tqdm(range(self.get_previous_length(self.resolution_in_name)), desc='aspect ratio bucketing'):
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

            self.scale_resolutions.append(scale_resolution)
            self.crop_resolutions.append(target_resolution)

    def get_item(self, index: int) -> dict:
        return {
            self.scale_resolution_out_name: self.scale_resolutions[index],
            self.crop_resolution_out_name: self.crop_resolutions[index],
        }


class LoadImage(PipelineModule):
    def __init__(self, path_in_name: str, image_out_name: str, range_min: float, range_max: float):
        super(LoadImage, self).__init__()
        self.path_in_name = path_in_name
        self.image_out_name = image_out_name

        self.range_min = range_min
        self.range_max = range_max

    def length(self) -> int:
        return self.get_previous_length(self.path_in_name)

    def get_inputs(self) -> list[str]:
        return [self.path_in_name]

    def get_outputs(self) -> list[str]:
        return [self.image_out_name]

    def get_item(self, index: int) -> dict:
        path = self.get_previous_item(self.path_in_name, index)

        image_tensor = Image.open(path)
        image_tensor = image_tensor.convert('RGB')

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

    def get_item(self, index: int) -> dict:
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

    def get_item(self, index: int) -> dict:
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

    def get_item(self, index: int) -> dict:
        item = {}

        for name in self.names:
            item[name] = self.get_previous_item(name, index)

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

    def get_item(self, index: int) -> dict:
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
    def __init__(self, names: [str], cache_dir: str):
        super(DiskCache, self).__init__()
        self.names = names
        self.cache_dir = cache_dir
        self.cache_length = None

    def length(self) -> int:
        if not self.cache_length:
            return self.get_previous_length(self.names[0])
        else:
            return self.cache_length

    def get_inputs(self) -> list[str]:
        return self.names

    def get_outputs(self) -> list[str]:
        return self.names

    def preprocess(self):
        caching_done = False

        if os.path.isdir(self.cache_dir):
            with os.scandir(self.cache_dir) as path_iter:
                if any(path_iter):
                    caching_done = True

        if caching_done:
            length = len(os.listdir(self.cache_length))
        else:
            length = self.get_previous_length(self.names[0])

            for index in tqdm(range(length), desc='caching'):
                item = {}

                for name in self.names:
                    item[name] = self.get_previous_item(name, index)

                torch.save(item, os.path.join(self.cache_dir, str(index) + '.pt'))

        self.cache_length = length

    def get_item(self, index: int) -> dict:
        cache_item = torch.load(os.path.join(self.cache_dir, str(index) + '.pt'))

        item = {}

        for name in self.names:
            item[name] = cache_item[name]

        return item


class AspectBatchSorting(PipelineModule):
    def __init__(self, resolution_in_name: str, names: [str], batch_size: int):
        super(AspectBatchSorting, self).__init__()
        self.resolution_in_name = resolution_in_name
        self.names = names
        self.batch_size = batch_size

        self.bucket_dict = {}
        self.index_list = []

    def length(self) -> int:
        return len(self.index_list)

    def get_inputs(self) -> list[str]:
        return [self.resolution_in_name]

    def get_outputs(self) -> list[str]:
        return []

    def preprocess(self):
        resolutions = []
        for index in range(self.get_previous_length(self.resolution_in_name)):
            resolution = self.get_previous_item(self.resolution_in_name, index)

            resolution = resolution[0], resolution[1]
            resolutions.append(resolution)

        # sort buckets
        self.bucket_dict = {}
        for index, resolution in enumerate(resolutions):
            if resolution not in self.bucket_dict:
                self.bucket_dict[resolution] = []
            self.bucket_dict[resolution].append(index)

        # drop images for full buckets
        # TODO drop/duplicate in the shuffle function, so different samples are dropped in each epoch
        for bucket_key in self.bucket_dict.keys():
            samples = self.bucket_dict[bucket_key]
            samples_to_drop = len(samples) % self.batch_size
            for i in range(samples_to_drop):
                samples.pop()

        self.index_list = self.shuffle()

    def shuffle(self) -> list[int]:
        # generate a shuffled list of batches
        batches = []
        for bucket_key in self.bucket_dict.keys():
            batch_count = int(len(self.bucket_dict[bucket_key]) / self.batch_size)
            batches.extend((bucket_key, i) for i in range(batch_count))
        random.shuffle(batches)

        # for each bucket, generate a shuffled list of samples
        samples = {bucket_key: self.bucket_dict[bucket_key].copy() for bucket_key in self.bucket_dict.keys()}
        for sample_key in samples:
            random.shuffle(samples[sample_key])

        # calculate the order of samples
        index_list = []
        for bucket_key, index in batches:
            for i in range(index * self.batch_size, (index + 1) * self.batch_size):
                index_list.append(samples[bucket_key][i])

        print(batches)
        print(samples)
        print(index_list)

        return index_list

    def get_item(self, index: int) -> dict:
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

    def get_item(self, index: int) -> dict:
        image = self.get_previous_item(self.image_in_name, index)
        mask = self.get_previous_item(self.mask_in_name, index)

        conditioning_image = image * mask

        return {
            self.image_out_name: conditioning_image
        }
