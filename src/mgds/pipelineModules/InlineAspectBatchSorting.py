from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.SerialPipelineModule import SerialPipelineModule


class InlineAspectBatchSorting(
    PipelineModule,
    SerialPipelineModule,
):
    def __init__(
            self,
            resolution_in_name: str,
            names: [str],
            batch_size: int,
    ):
        super(InlineAspectBatchSorting, self).__init__()
        self.resolution_in_name = resolution_in_name
        self.names = names
        self.batch_size = batch_size

        self.bucket_dict = {}
        self.in_index_list = []
        self.next_in_cache_index = 0
        self.current_resolution = None
        self.__has_next = False

    def approximate_length(self) -> int:
        return len(self.in_index_list)

    def has_next(self) -> bool:
        return self.__has_next

    def get_inputs(self) -> list[str]:
        return [self.resolution_in_name] + self.names

    def get_outputs(self) -> list[str]:
        return self.names

    def __shuffle(self, variation: int) -> list[int]:
        rand = self._get_rand(variation)

        length = self._get_previous_length(self.resolution_in_name)
        index_list = list(range(length))
        rand.shuffle(index_list)
        return index_list

    def __fill_cache(self):
        """
        Fills the internal cache until one bucket is full
        """

        if self.current_resolution is not None:
            return

        while self.next_in_cache_index < len(self.in_index_list):
            in_index = self.in_index_list[self.next_in_cache_index]
            self.next_in_cache_index += 1

            item = {}
            for name in self.names:
                item[name] = self._get_previous_item(self.current_variation, name, in_index)

            if self.resolution_in_name in item:
                resolution = item[self.resolution_in_name]
            else:
                resolution = self._get_previous_item(self.current_variation, self.resolution_in_name, in_index)

            if resolution not in self.bucket_dict:
                self.bucket_dict[resolution] = []

            self.bucket_dict[resolution].append(item)

            if len(self.bucket_dict[resolution]) == self.batch_size:
                self.current_resolution = resolution
                self.__has_next = True
                return

        # not enough items to fill a bucket, signal end of iteration
        self.__has_next = False

    def start(self, variation: int, start_index: int):
        self.bucket_dict = {}
        self.in_index_list = self.__shuffle(variation)
        self.next_in_cache_index = start_index
        self.current_resolution = None
        self.__fill_cache()

    def get_next_item(self) -> dict:
        bucket = self.bucket_dict[self.current_resolution]
        item = bucket.pop(0)
        if len(bucket) == 0:
            self.current_resolution = None
            self.__fill_cache()

        return item
