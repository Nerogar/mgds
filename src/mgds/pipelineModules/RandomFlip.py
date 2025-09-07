from torchvision.transforms import functional

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class RandomFlip(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            names: list[str],
            enabled_in_name: str,
            fixed_enabled_in_name: str,
    ):
        super(RandomFlip, self).__init__()
        self.names = names
        self.enabled_in_name = enabled_in_name
        self.fixed_enabled_in_name = fixed_enabled_in_name

    def length(self) -> int:
        return self._get_previous_length(self.names[0])

    def get_inputs(self) -> list[str]:
        return self.names + [self.enabled_in_name] + [self.fixed_enabled_in_name]

    def get_outputs(self) -> list[str]:
        return self.names

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        enabled = self._get_previous_item(variation, self.enabled_in_name, index)
        fixed_enabled = self._get_previous_item(variation, self.fixed_enabled_in_name, index)

        rand = self._get_rand(variation, index)
        item = {}

        if enabled:
            check = rand.random()
            flip = check < 0.5
        elif fixed_enabled:
            flip = True
        else:
            flip = False

        for name in self.names:
            previous_item = self._get_previous_item(variation, name, index)
            if previous_item is not None:
                if flip:
                    previous_item = functional.hflip(previous_item)
            item[name] = previous_item

        return item
