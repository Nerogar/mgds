import torch
from diffusers import AutoencoderKL
from tqdm import tqdm

from mgds.pipelineModules.AspectBatchSorting import AspectBatchSorting
from mgds.pipelineModules.AspectBucketing import AspectBucketing
from mgds.pipelineModules.CalcAspect import CalcAspect
from mgds.pipelineModules.CollectPaths import CollectPaths
from mgds.pipelineModules.DecodeVAE import DecodeVAE
from mgds.pipelineModules.EncodeVAE import EncodeVAE
from mgds.pipelineModules.LoadImage import LoadImage
from mgds.pipelineModules.ModifyPath import ModifyPath
from mgds.pipelineModules.SampleVAEDistribution import SampleVAEDistribution
from mgds.pipelineModules.SaveImage import SaveImage
from mgds.pipelineModules.ScaleCropImage import ScaleCropImage
from src.mgds.MGDS import MGDS, TrainDataLoader
from mgds.OutputPipelineModule import OutputPipelineModule

DEVICE = 'cuda'
DTYPE = torch.float32
BATCH_SIZE = 4


def test():
    base_model_path = '..\\..\\models\\diffusers-base\\sdxl-v0-9-base'
    # base_model_path = '..\\..\\models\\diffusers-base\\sd-v1-5-inpainting'

    vae = AutoencoderKL.from_pretrained(
        base_model_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to(DEVICE)

    input_modules = [
        CollectPaths(concept_in_name='concept', path_in_name='path', name_in_name='name', path_out_name='image_path', concept_out_name='concept', extensions=['.png', '.jpg'], include_postfix=None, exclude_postfix=['-masklabel'], include_subdirectories_in_name='concept.include_subdirectories'),
        ModifyPath(in_name='image_path', out_name='mask_path', postfix='-masklabel', extension='.png'),
        LoadImage(path_in_name='image_path', image_out_name='image', range_min=-1.0, range_max=1.0),
        CalcAspect(image_in_name='image', resolution_out_name='original_resolution'),
        AspectBucketing(target_resolution=512, resolution_in_name='original_resolution', scale_resolution_out_name='scale_resolution', crop_resolution_out_name='crop_resolution', possible_resolutions_out_name='possible_resolutions', quantization=64),
        ScaleCropImage(image_in_name='image', scale_resolution_in_name='scale_resolution', crop_resolution_in_name='crop_resolution', enable_crop_jitter_in_name='concept.enable_crop_jitter', image_out_name='image', crop_offset_out_name='crop_offset'),
        EncodeVAE(in_name='image', out_name='latent_image_distribution', vae=vae, override_allow_mixed_precision=False),
        SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image', mode='mean'),
    ]

    debug_modules = [
        DecodeVAE(in_name='latent_image', out_name='decoded_image', vae=vae, override_allow_mixed_precision=False),
        SaveImage(image_in_name='decoded_image', original_path_in_name='image_path', path='debug', in_range_min=-1, in_range_max=1),
    ]

    output_modules = [
        AspectBatchSorting(resolution_in_name='crop_resolution', names=['latent_image'], batch_size=BATCH_SIZE),
        OutputPipelineModule(names=['latent_image'])
    ]

    ds = MGDS(
        device=torch.device(DEVICE),
        concepts=[
            {
                'name': 'DS',
                'path': '..\\..\\datasets\\dataset',
                'random_circular_crop': True,
                'random_mask_rotate_crop': True,
                'random_flip': True,
                'include_subdirectories': True,
            },
            {
                'name': 'DS4',
                'path': '..\\..\\datasets\\dataset4',
                'random_circular_crop': False,
                'random_mask_rotate_crop': False,
                'random_flip': False,
            },
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
