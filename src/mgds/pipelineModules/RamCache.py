from tqdm import tqdm

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.SingleVariationRandomAccessPipelineModule import SingleVariationRandomAccessPipelineModule


class RamCache(
    PipelineModule,
    SingleVariationRandomAccessPipelineModule,
):
    def __init__(
            self,
            names: list[str] = None,
    ):
        super(RamCache, self).__init__()
        self.names = names
        self.cache_length = None
        self.cache = None

    def length(self) -> int:
        if not self.cache_length:
            return self._get_previous_length(self.names[0])
        else:
            return self.cache_length

    def get_inputs(self) -> list[str]:
        return self.names

    def get_outputs(self) -> list[str]:
        return self.names

    def start(self, variation: int):
        length = self.length()
        self.cache = []
        for index in tqdm(range(length), desc='caching'):
            if index % 100 == 0:
                self._torch_gc()

            item = {}

            for name in self.names:
                item[name] = self._get_previous_item(variation, name, index)

            self.cache.append(item)

        self.cache_length = length

    def get_item(self, index: int, requested_name: str = None) -> dict:
        return self.cache[index]
