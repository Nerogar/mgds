import os

import av
import torch
from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

AV_TIME_BASE = 1_000_000

class LoadVideo(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            path_in_name: str,
            target_frame_count_in_name: str,
            video_out_name: str,
            range_min: float,
            range_max: float,
            target_frame_rate: float,
            supported_extensions: set[str],
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.path_in_name = path_in_name
        self.target_frame_count_in_name = target_frame_count_in_name
        self.video_out_name = video_out_name

        self.range_min = range_min
        self.range_max = range_max

        self.target_frame_rate = target_frame_rate

        self.supported_extensions = supported_extensions

        self.dtype = dtype

        self.duration_cache = {}

    def length(self) -> int:
        return self._get_previous_length(self.path_in_name)

    def get_inputs(self) -> list[str]:
        return [self.path_in_name]

    def get_outputs(self) -> list[str]:
        return [self.video_out_name]

    def __get_duration(self, path: str) -> tuple[int, float]:
        container = av.open(path)

        video_stream = container.streams.video[0]

        frame_rate = video_stream.base_rate.numerator / video_stream.base_rate.denominator
        if video_stream.frames > 0:
            frame_count = video_stream.frames
        elif container.duration > 0:
            frame_count = int(container.duration / AV_TIME_BASE * frame_rate)
        elif 'DURATION' in video_stream.metadata:
            metadata_duration_frames = [frame_rate, frame_rate * 60, frame_rate * 60 * 60]
            metadata_duration = reversed([float(x) for x in video_stream.metadata['DURATION'].split(':')])
            frame_count = int(sum(d * f for d, f in zip(metadata_duration, metadata_duration_frames, strict=False)))
        else:
            print(f"could not find length of video, falling back to full decode {path}")
            # fall back to counting frames (and hope the user didn't specify a long video)
            decoded = container.decode(video=0)

            frame_count = sum(1 for _ in decoded)

        return frame_count, frame_rate

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        rand = self._get_rand(variation, index)
        path = self._get_previous_item(variation, self.path_in_name, index)
        target_frame_count = self._get_previous_item(variation, self.target_frame_count_in_name, index)

        target_frame_count = int(target_frame_count)

        ext = os.path.splitext(path)[1]
        if ext.lower() in self.supported_extensions:
            try:
                if path not in self.duration_cache:
                    self.duration_cache[path] = self.__get_duration(path)

                frame_count, frame_rate = self.duration_cache[path]

                duration = (frame_count - 1) / frame_rate
                target_duration = (target_frame_count - 1) / self.target_frame_rate

                start_offset = rand.uniform(0, duration - target_duration)
                start_offset_frame = int(start_offset * frame_rate)

                container = av.open(path)
                decoded = container.decode(video=0)

                # skip initial frames # TODO: better seeking
                for _ in range(start_offset_frame):
                    decoded.__next__()

                frames = []
                while len(frames) < target_frame_count:
                    frame = next(decoded, None)
                    if frame is None:
                        frames.append(frames[-1])
                    else:
                        frames.append(torch.from_numpy(frame.to_rgb().to_ndarray()).movedim(2, 0))

                video_tensor = torch.stack(frames, dim=1)
                del frames

                if self.dtype:
                    video_tensor = video_tensor.to(dtype=self.dtype)
                else:
                    video_tensor = video_tensor.to(dtype=torch.float32)

                # Transform 0 - 255 to 0-1
                video_tensor = video_tensor / 255
                video_tensor = video_tensor * (self.range_max - self.range_min) + self.range_min

                video_tensor = video_tensor.to(device=self.pipeline.device)

            except FileNotFoundError:
                video_tensor = None
            except:
                print("could not load video, it might be corrupted: " + path)
                raise
        else:
            video_tensor = None

        return {
            self.video_out_name: video_tensor
        }
