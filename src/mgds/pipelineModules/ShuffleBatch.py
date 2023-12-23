from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.SingleVariationRandomAccessPipelineModule import SingleVariationRandomAccessPipelineModule


class ShuffleBatch(
    PipelineModule,
    SingleVariationRandomAccessPipelineModule,
):
    def __init__(self, names: [str], batch_size: int):
        super(ShuffleBatch, self).__init__()
        self.names = names
        self.batch_size = batch_size

        self.index_list = []

    def length(self) -> int:
        return len(self.index_list)

    def get_inputs(self) -> list[str]:
        return self.names

    def get_outputs(self) -> list[str]:
        return self.names

    def start(self, variation: int):
        rand = self._get_rand(variation)
        first_in_name = self.names[0]
        self.index_list = list(range(self._get_previous_length(first_in_name)))
        rand.shuffle(self.index_list)

    def get_item(self, index: int, requested_name: str = None) -> dict:
        index = self.index_list[index]

        item = {}

        for name in self.names:
            item[name] = self._get_previous_item(self.current_variation, name, index)

        return item
