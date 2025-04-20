from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.SingleVariationRandomAccessPipelineModule import SingleVariationRandomAccessPipelineModule


class DistributedSampler(
    PipelineModule,
    SingleVariationRandomAccessPipelineModule,
):
    def __init__(self, names: [str], local_batch_size: int, world_size: int, rank: int):
        super(DistributedSampler, self).__init__()
        self.names = names
        self.local_batch_size = local_batch_size
        self.world_size = world_size
        self.rank = rank


    def length(self) -> int:
        return self._get_previous_length(self.names[0]) // self.world_size

    def get_inputs(self) -> list[str]:
        return self.names

    def get_outputs(self) -> list[str]:
        return self.names

    def get_item(self, index: int, requested_name: str = None) -> dict:
        prev_index = (index * self.world_size) + self.rank
        item = {}
        for name in self.names:
            item[name] = self._get_previous_item(self.current_variation, name, prev_index)
        return item
