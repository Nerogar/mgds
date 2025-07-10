from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.SerialPipelineModule import SerialPipelineModule


class InlineDistributedSampler(
    PipelineModule,
    SerialPipelineModule,
):
    def __init__(self, names: [str], world_size: int, rank: int):
        super(InlineDistributedSampler, self).__init__()
        self.names = names
        self.world_size = world_size
        self.rank = rank

    def get_inputs(self) -> list[str]:
        return self.names

    def get_outputs(self) -> list[str]:
        return self.names

    def approximate_length(self) -> int:
        return self._get_previous_length(self.names[0]) // self.world_size

    def has_next(self) -> bool:
        return self._has_previous_next(self.names[0])

    
    def get_next_item(self) -> dict:
        prev_index = self.current_index * self.world_size + self.rank

        #discard all items < rank
        for i in range(self.current_index * self.world_size, prev_index):
            self._get_previous_item(self.current_variation, self.names[0], i)

        item = {}
        for name in self.names:
            item[name] = self._get_previous_item(self.current_variation, name, prev_index)

        #discard all items > rank
        for i in range(prev_index + 1, (self.current_index + 1) * self.world_size):
            self._get_previous_item(self.current_variation, self.names[0], i)

        return item
