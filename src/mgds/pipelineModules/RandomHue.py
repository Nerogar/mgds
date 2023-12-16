from torchvision.transforms import functional

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class RandomHue(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            names: [str],
            enabled_in_name: str,
            max_strength_in_name: str,
    ):
        super(RandomHue, self).__init__()
        self.names = names
        self.enabled_in_name = enabled_in_name
        self.max_strength_in_name = max_strength_in_name

    def length(self) -> int:
        return self._get_previous_length(self.names[0])

    def get_inputs(self) -> list[str]:
        return self.names

    def get_outputs(self) -> list[str]:
        return self.names

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        enabled = self._get_previous_item(variation, self.enabled_in_name, index)
        max_strength = self._get_previous_item(variation, self.max_strength_in_name, index)

        rand = self._get_rand(variation, index)
        item = {}

        strength = rand.uniform(-max_strength * 0.5, max_strength * 0.5)
        strength = max(-0.5, min(0.5, strength))

        for name in self.names:
            previous_item = self._get_previous_item(variation, name, index)
            if enabled:
                previous_item = functional.adjust_hue(previous_item, strength)
            item[name] = previous_item

        return item
