import os
from typing import Callable

import av
import torch
from PIL import Image
from tqdm import tqdm

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


# SaveImage funnels everything through ToPILImage, which can't take the [C, F, H, W] tensors the video
# dataloaders carry; this writes them as h264 instead. The clip is written at the frame rate the caller
# names rather than the source file's, so a source at a different rate shows up here as motion running
# too slow or too fast. Single-frame clips (still images widened to video) are written as PNG.
class SaveVideo(
    PipelineModule,
    RandomAccessPipelineModule,
):

    def __init__(
            self,
            video_in_name: str,
            original_path_in_name: str,
            path: str,
            in_range_min: float,
            in_range_max: float,
            fps: int,
            before_save_fun: Callable[[], None] | None = None,
    ):
        super(SaveVideo, self).__init__()
        self.video_in_name = video_in_name
        self.original_path_in_name = original_path_in_name
        self.path = path
        self.in_range_min = in_range_min
        self.in_range_max = in_range_max
        self.fps = fps
        self.before_save_fun = before_save_fun

    def approximate_length(self) -> int:
        return self._get_previous_length(self.video_in_name)

    def get_inputs(self) -> list[str]:
        return [self.video_in_name, self.original_path_in_name]

    def get_outputs(self) -> list[str]:
        return [self.video_in_name]

    def length(self) -> int:
        return 0

    def start(self, variation: int):
        path = os.path.join(self.path, "epoch-" + str(variation))
        if not os.path.exists(path):
            os.makedirs(path)

        if self.before_save_fun is not None:
            self.before_save_fun()

        for index in tqdm(range(self._get_previous_length(self.original_path_in_name)),
                          desc='writing debug videos for \'' + self.video_in_name + '\''):
            video_tensor = self._get_previous_item(variation, self.video_in_name, index)
            original_path = self._get_previous_item(variation, self.original_path_in_name, index)
            name, _ = os.path.splitext(os.path.basename(original_path))
            name = str(index) + '-' + name + '-' + self.video_in_name

            # [C, F, H, W] in the module's input range -> [F, H, W, C] uint8, the layout av wants
            video_tensor = (video_tensor.to(dtype=torch.float32) - self.in_range_min) \
                           / (self.in_range_max - self.in_range_min)
            frames = video_tensor.clamp(0, 1).mul(255).round().to(dtype=torch.uint8) \
                .permute(1, 2, 3, 0).cpu().numpy()

            if frames.shape[0] == 1:
                Image.fromarray(frames[0]).save(os.path.join(path, name + '.png'))
                continue

            with av.open(os.path.join(path, name + '.mp4'), 'w') as container:
                stream = container.add_stream('libx264', rate=self.fps)
                stream.options = {'crf': '0'}
                stream.height = frames.shape[1]
                stream.width = frames.shape[2]
                stream.pix_fmt = 'yuv444p'

                for frame in frames:
                    for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format='rgb24')):
                        container.mux(packet)

                for packet in stream.encode():
                    container.mux(packet)

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return {}
