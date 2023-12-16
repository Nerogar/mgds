import os.path

import torch
from diffusers import AutoencoderKL
from tqdm import tqdm
from transformers import DPTImageProcessor, DPTForDepthEstimation, CLIPTokenizer

from mgds.pipelineModules.AspectBatchSorting import AspectBatchSorting
from mgds.pipelineModules.AspectBucketing import AspectBucketing
from mgds.pipelineModules.CalcAspect import CalcAspect
from mgds.pipelineModules.CollectPaths import CollectPaths
from mgds.pipelineModules.DecodeTokens import DecodeTokens
from mgds.pipelineModules.DecodeVAE import DecodeVAE
from mgds.pipelineModules.DiskCache import DiskCache
from mgds.pipelineModules.EncodeVAE import EncodeVAE
from mgds.pipelineModules.GenerateDepth import GenerateDepth
from mgds.pipelineModules.GenerateImageLike import GenerateImageLike
from mgds.pipelineModules.GenerateMaskedConditioningImage import GenerateMaskedConditioningImage
from mgds.pipelineModules.LoadImage import LoadImage
from mgds.pipelineModules.LoadText import LoadText
from mgds.pipelineModules.ModifyPath import ModifyPath
from mgds.pipelineModules.RandomCircularMaskShrink import RandomCircularMaskShrink
from mgds.pipelineModules.RandomFlip import RandomFlip
from mgds.pipelineModules.RandomLatentMaskRemove import RandomLatentMaskRemove
from mgds.pipelineModules.RandomMaskRotateCrop import RandomMaskRotateCrop
from mgds.pipelineModules.SampleVAEDistribution import SampleVAEDistribution
from mgds.pipelineModules.SaveImage import SaveImage
from mgds.pipelineModules.SaveText import SaveText
from mgds.pipelineModules.ScaleCropImage import ScaleCropImage
from mgds.pipelineModules.ScaleImage import ScaleImage
from mgds.pipelineModules.ShuffleTags import ShuffleTags
from mgds.pipelineModules.Tokenize import Tokenize
from src.mgds.MGDS import MGDS, TrainDataLoader
from mgds.OutputPipelineModule import OutputPipelineModule

DEVICE = 'cuda'
DTYPE = torch.float32
BATCH_SIZE = 1


def test():
    depth_model_path = '..\\models\\diffusers-base\\sd-v2-0-depth'

    vae = AutoencoderKL.from_pretrained(os.path.join(depth_model_path, 'vae')).to(DEVICE)
    image_depth_processor = DPTImageProcessor.from_pretrained(os.path.join(depth_model_path, 'feature_extractor'))
    depth_estimator = DPTForDepthEstimation.from_pretrained(os.path.join(depth_model_path, 'depth_estimator')).to(DEVICE)
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(depth_model_path, 'tokenizer'))

    input_modules = [
        CollectPaths(concept_in_name='concept', path_in_name='path', name_in_name='name', path_out_name='image_path', concept_out_name='concept', extensions=['.png', '.jpg'], include_postfix=None, exclude_postfix=['-masklabel'], include_subdirectories_in_name='concept.include_subdirectories'),
        ModifyPath(in_name='image_path', out_name='mask_path', postfix='-masklabel', extension='.png'),
        ModifyPath(in_name='image_path', out_name='prompt_path', postfix='', extension='.txt'),
        LoadImage(path_in_name='image_path', image_out_name='image', range_min=-1.0, range_max=1.0),
        GenerateImageLike(image_in_name='image', image_out_name='mask', color=255, range_min=0, range_max=1, channels=1),
        LoadImage(path_in_name='mask_path', image_out_name='mask', range_min=0, range_max=1, channels=1),
        GenerateDepth(path_in_name='image_path', image_out_name='depth', image_depth_processor=image_depth_processor, depth_estimator=depth_estimator),
        RandomCircularMaskShrink(mask_name='mask', shrink_probability=1.0, shrink_factor_min=0.2, shrink_factor_max=1.0, enabled_in_name='concept.random_circular_crop'),
        RandomMaskRotateCrop(mask_name='mask', additional_names=['image', 'depth'], min_size=512, min_padding_percent=10, max_padding_percent=30, max_rotate_angle=20, enabled_in_name='concept.random_mask_rotate_crop'),
        CalcAspect(image_in_name='image', resolution_out_name='original_resolution'),
        AspectBucketing(target_resolution=[512, 768, 1024], resolution_in_name='original_resolution', scale_resolution_out_name='scale_resolution', crop_resolution_out_name='crop_resolution', possible_resolutions_out_name='possible_resolutions', quantization=8),
        ScaleCropImage(image_in_name='image', scale_resolution_in_name='scale_resolution', crop_resolution_in_name='crop_resolution', enable_crop_jitter_in_name='concept.enable_crop_jitter', image_out_name='image', crop_offset_out_name='crop_offset'),
        ScaleCropImage(image_in_name='mask', scale_resolution_in_name='scale_resolution', crop_resolution_in_name='crop_resolution', enable_crop_jitter_in_name='concept.enable_crop_jitter', image_out_name='mask', crop_offset_out_name='crop_offset'),
        ScaleCropImage(image_in_name='depth', scale_resolution_in_name='scale_resolution', crop_resolution_in_name='crop_resolution', enable_crop_jitter_in_name='concept.enable_crop_jitter', image_out_name='depth', crop_offset_out_name='crop_offset'),
        LoadText(path_in_name='prompt_path', text_out_name='prompt'),
        GenerateMaskedConditioningImage(image_in_name='image', mask_in_name='mask', image_out_name='conditioning_image', image_range_min=0, image_range_max=1),
        RandomFlip(names=['image', 'mask', 'depth', 'conditioning_image'], enabled_in_name='concept.random_flip'),
        EncodeVAE(in_name='image', out_name='latent_image_distribution', vae=vae),
        ScaleImage(in_name='mask', out_name='latent_mask', factor=1./8.),
        EncodeVAE(in_name='conditioning_image', out_name='latent_conditioning_image_distribution', vae=vae),
        ScaleImage(in_name='depth', out_name='latent_depth', factor=1./8.),
        ShuffleTags(text_in_name='prompt', enabled_in_name='concept.enable_tag_shuffling', delimiter_in_name='concept.tag_delimiter', keep_tags_count_in_name='concept.keep_tags_count', text_out_name='prompt'),
        Tokenize(in_name='prompt', tokens_out_name='tokens', mask_out_name='tokens_mask', max_token_length=77, tokenizer=tokenizer),
        DiskCache(cache_dir='cache', split_names=['latent_image_distribution', 'latent_mask', 'latent_conditioning_image_distribution', 'latent_depth', 'tokens'], aggregate_names=['crop_resolution', 'image_path'], variations_in_name='concept.variations', repeats_in_name='concept.repeats', variations_group_in_name='concept'),
        SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image', mode='mean'),
        SampleVAEDistribution(in_name='latent_conditioning_image_distribution', out_name='latent_conditioning_image', mode='mean'),
        RandomLatentMaskRemove(latent_mask_name='latent_mask', latent_conditioning_image_name='latent_conditioning_image', replace_probability=0.1, vae=vae, possible_resolutions_in_name='possible_resolutions')
    ]

    debug_modules = [
        DecodeVAE(in_name='latent_image', out_name='decoded_image', vae=vae),
        DecodeVAE(in_name='latent_conditioning_image', out_name='decoded_conditioning_image', vae=vae),
        DecodeTokens(in_name='tokens', out_name='decoded_text', tokenizer=tokenizer),
        SaveImage(image_in_name='decoded_image', original_path_in_name='image_path', path='debug', in_range_min=-1, in_range_max=1),
        SaveImage(image_in_name='mask', original_path_in_name='image_path', path='debug', in_range_min=0, in_range_max=1),
        SaveImage(image_in_name='decoded_conditioning_image', original_path_in_name='image_path', path='debug', in_range_min=-1, in_range_max=1),
        SaveText(text_in_name='decoded_text', original_path_in_name='image_path', path='debug'),
        # SaveImage(image_in_name='depth', original_path_in_name='image_path', path='debug', in_range_min=-1, in_range_max=1),
        # SaveImage(image_in_name='latent_mask', original_path_in_name='image_path', path='debug', in_range_min=0, in_range_max=1),
        # SaveImage(image_in_name='latent_depth', original_path_in_name='image_path', path='debug', in_range_min=-1, in_range_max=1),
    ]

    output_modules = [
        AspectBatchSorting(resolution_in_name='crop_resolution', names=['latent_image', 'latent_conditioning_image', 'latent_mask', 'latent_depth', 'tokens'], batch_size=BATCH_SIZE),
        OutputPipelineModule(names=['latent_image', 'latent_conditioning_image', 'latent_mask', 'latent_depth', 'tokens'])
    ]

    ds = MGDS(
        device=torch.device(DEVICE),
        dtype=torch.float32,
        allow_mixed_precision=False,
        concepts=[
            {
                'name': 'DS',
                'path': '..\\datasets\\dataset5-dark',
                'random_circular_crop': True,
                'random_mask_rotate_crop': True,
                'random_flip': True,
                'include_subdirectories': True,
                'enable_tag_shuffling': True,
                'tag_delimiter': ',',
                'keep_tags_count': 3,
                'variations': 3,
                'repeats': 1.5,
            },
            # {
            #     'name': 'DS4',
            #     'path': '..\\..\\datasets\\dataset4',
            #     'random_circular_crop': False,
            #     'random_mask_rotate_crop': False,
            #     'random_flip': False,
            # },
        ],
        settings={},
        definition=[
            input_modules,
            debug_modules,
            output_modules
        ],
        batch_size=BATCH_SIZE,
        seed=42,
        initial_epoch=0,
        initial_epoch_sample=10,
    )
    dl = TrainDataLoader(ds, batch_size=BATCH_SIZE)

    for epoch in range(10):
        ds.start_next_epoch()
        for batch in tqdm(dl):
            pass


if __name__ == '__main__':
    test()
