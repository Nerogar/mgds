from dataLoader.DebugDataLoaderModules import SaveImage, DecodeVAE
from dataLoader.DiffusersDataLoaderModules import *
from dataLoader.GenericDataLoaderModules import *
from dataLoader.TrainDataSet import TrainDataSet, TrainDataLoader
from dataLoader.TransformersDataLoaderModules import *

DEVICE = 'cuda'
DTYPE = torch.float32
BATCH_SIZE = 4


def test():
    vae = AutoencoderKL.from_pretrained('F:\\StableTunerPip\\models\\v2-0-base\\vae').to(DEVICE)
    depth_estimator = DPTForDepthEstimation.from_pretrained('F:\\StableTunerPip\\models\\v2-0-depth\\depth_estimator').to(DEVICE)
    image_depth_processor = DPTImageProcessor.from_pretrained('F:\\StableTunerPip\\models\\v2-0-depth\\feature_extractor')
    tokenizer = CLIPTokenizer.from_pretrained('F:\\StableTunerPip\\models\\v2-0-depth\\tokenizer')

    ds = TrainDataSet(torch.device(DEVICE), [
        {'name': 'X', 'path': 'dataset'}
    ], [
        CollectPaths(concept_in_name='concept', path_out_name='image_path', concept_out_name='concept', extensions=['.png', '.jpg'], include_postfix=None, exclude_postfix=['-masklabel']),
        ModifyPath(in_name='image_path', out_name='mask_path', postfix='-masklabel', extension='.png'),
        ModifyPath(in_name='image_path', out_name='prompt_path', postfix='', extension='.txt'),
        LoadImage(path_in_name='image_path', image_out_name='image', range_min=-1.0, range_max=1.0),
        LoadImage(path_in_name='mask_path', image_out_name='mask', range_min=0, range_max=1),
        GenerateDepth(path_in_name='image_path', image_out_name='depth', image_depth_processor=image_depth_processor, depth_estimator=depth_estimator),
        CalcAspect(image_in_name='image', resolution_out_name='original_resolution'),
        AspectBucketing(batch_size=BATCH_SIZE, target_resolution=512, resolution_in_name='original_resolution', scale_resolution_out_name='scale_resolution', crop_resolution_out_name='crop_resolution'),
        ScaleCropImage(image_in_name='image', scale_resolution_in_name='scale_resolution', crop_resolution_in_name='crop_resolution', image_out_name='image'),
        ScaleCropImage(image_in_name='mask', scale_resolution_in_name='scale_resolution', crop_resolution_in_name='crop_resolution', image_out_name='mask'),
        ScaleCropImage(image_in_name='depth', scale_resolution_in_name='scale_resolution', crop_resolution_in_name='crop_resolution', image_out_name='depth'),
        LoadText(path_in_name='prompt_path', text_out_name='prompt'),
        GenerateMaskedConditioningImage(image_in_name='image', mask_in_name='mask', image_out_name='conditioning_image'),
        RandomFlip(names=['image', 'mask', 'depth', 'conditioning_image']),
        EncodeVAE(in_name='image', out_name='latent_image_distribution', vae=vae),
        Downscale(in_name='mask', out_name='latent_mask'),
        EncodeVAE(in_name='conditioning_image', out_name='latent_conditioning_image_distribution', vae=vae),
        Downscale(in_name='depth', out_name='latent_depth'),
        Tokenize(in_name='prompt', out_name='tokens', tokenizer=tokenizer),
        DiskCache(names=['latent_image_distribution', 'latent_mask', 'latent_conditioning_image_distribution', 'latent_depth', 'tokens'], cache_dir='cache'),
        SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image', mode='mean'),
        SampleVAEDistribution(in_name='latent_conditioning_image_distribution', out_name='latent_conditioning_image', mode='mean'),
        AspectBatchSorting(resolution_in_name='crop_resolution', names=['latent_image', 'latent_conditioning_image', 'latent_mask', 'latent_depth', 'tokens'], batch_size=BATCH_SIZE),

        # debut modules
        SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image_debug', mode='mean'),
        SampleVAEDistribution(in_name='latent_conditioning_image_distribution', out_name='latent_conditioning_image_debug', mode='mean'),
        DecodeVAE(in_name='latent_image_debug', out_name='decoded_image', vae=vae),
        DecodeVAE(in_name='latent_conditioning_image_debug', out_name='decoded_conditioning_image', vae=vae),
        SaveImage(image_in_name='decoded_image', original_path_in_name='image_path', postfix='', path='debug', in_range_min=-1, in_range_max=1),
        SaveImage(image_in_name='mask', original_path_in_name='image_path', postfix='mask', path='debug', in_range_min=0, in_range_max=1),
        SaveImage(image_in_name='decoded_conditioning_image', original_path_in_name='image_path', postfix='conditioning', path='debug', in_range_min=-1, in_range_max=1),
        SaveImage(image_in_name='depth', original_path_in_name='image_path', postfix='depth', path='debug', in_range_min=-1, in_range_max=1),
        SaveImage(image_in_name='latent_mask', original_path_in_name='image_path', postfix='mask', path='debug', in_range_min=0, in_range_max=1),
        SaveImage(image_in_name='latent_depth', original_path_in_name='image_path', postfix='depth', path='debug', in_range_min=-1, in_range_max=1),
    ])
    dl = TrainDataLoader(ds, batch_size=BATCH_SIZE)

    for batch in tqdm(dl):
        pass


if __name__ == '__main__':
    test()
