from torchvision.transforms import functional

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class RandomBrightness(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            names: [str],
            enabled_in_name: str,
            fixed_enabled_in_name: str,
            max_strength_in_name: str,
    ):
        super(RandomBrightness, self).__init__()
        self.names = names
        self.enabled_in_name = enabled_in_name
        self.fixed_enabled_in_name = fixed_enabled_in_name
        self.max_strength_in_name = max_strength_in_name

    def length(self) -> int:
        return self._get_previous_length(self.names[0])

    def get_inputs(self) -> list[str]:
        return self.names + [self.enabled_in_name] + [self.fixed_enabled_in_name] + [self.max_strength_in_name]

    def get_outputs(self) -> list[str]:
        return self.names

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        enabled = self._get_previous_item(variation, self.enabled_in_name, index)
        fixed_enabled = self._get_previous_item(variation, self.fixed_enabled_in_name, index)
        max_strength = self._get_previous_item(variation, self.max_strength_in_name, index)

        rand = self._get_rand(variation, index)
        item = {}

        if enabled:
            strength = rand.uniform(1 - max_strength, 1 + max_strength)
            strength = max(0.0, strength)
        elif fixed_enabled:
            strength = 1.0 + max_strength
        else:
            strength = 1.0

        for name in self.names:
            previous_item = self._get_previous_item(variation, name, index)
            if enabled or fixed_enabled:
                previous_item = functional.adjust_brightness(previous_item, strength)
            item[name] = previous_item

        return item
