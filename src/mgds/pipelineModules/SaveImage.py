import os

from torchvision import transforms
from tqdm import tqdm

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.SerialPipelineModule import SerialPipelineModule


class SaveImage(
    PipelineModule,
    SerialPipelineModule,
):
    def __init__(
            self,
            image_in_name: str,
            original_path_in_name: str,
            path: str,
            in_range_min: float,
            in_range_max: float,
    ):
        super(SaveImage, self).__init__()
        self.image_in_name = image_in_name
        self.original_path_in_name = original_path_in_name
        self.path = path
        self.in_range_min = in_range_min
        self.in_range_max = in_range_max

    def length(self) -> int:
        return self._get_previous_length(self.image_in_name)

    def get_inputs(self) -> list[str]:
        return [self.image_in_name, self.original_path_in_name]

    def get_outputs(self) -> list[str]:
        return [self.image_in_name]

    def start(self, variation: int):
        path = os.path.join(self.path, "epoch-" + str(variation))
        if not os.path.exists(path):
            os.makedirs(path)

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

            image = t(image_tensor)
            image.save(os.path.join(path, str(index) + '-' + name + '-' + self.image_in_name + ext))

    def get_item(self, index: int, requested_name: str = None) -> dict:
        return {}
