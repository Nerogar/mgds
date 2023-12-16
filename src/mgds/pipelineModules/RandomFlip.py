from torchvision.transforms import functional

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class RandomFlip(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            names: [str],
            enabled_in_name: str
    ):
        super(RandomFlip, self).__init__()
        self.names = names
        self.enabled_in_name = enabled_in_name

    def length(self) -> int:
        return self._get_previous_length(self.names[0])

    def get_inputs(self) -> list[str]:
        return self.names

    def get_outputs(self) -> list[str]:
        return self.names

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        enabled = self._get_previous_item(variation, self.enabled_in_name, index)

        rand = self._get_rand(variation, index)
        item = {}

        check = rand.random()
        flip = enabled and check < 0.5

        for name in self.names:
            previous_item = self._get_previous_item(variation, name, index)
            if flip:
                previous_item = functional.hflip(previous_item)
            item[name] = previous_item

        return item
