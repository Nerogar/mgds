from tqdm import tqdm

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.SingleVariationRandomAccessPipelineModule import SingleVariationRandomAccessPipelineModule


class AspectBatchSorting(
    PipelineModule,
    SingleVariationRandomAccessPipelineModule,
):
    def __init__(self, resolution_in_name: str, names: [str], batch_size: int):
        super(AspectBatchSorting, self).__init__()
        self.resolution_in_name = resolution_in_name
        self.names = names
        self.batch_size = batch_size

        self.bucket_dict = {}
        self.index_list = []
        self.index_list = []

    def length(self) -> int:
        return len(self.index_list)

    def get_inputs(self) -> list[str]:
        return [self.resolution_in_name] + self.names

    def get_outputs(self) -> list[str]:
        return self.names

    def __shuffle(self) -> list[int]:
        rand = self._get_rand(self.current_variation)

        bucket_dict = {key: value.copy() for (key, value) in self.bucket_dict.items()}

        # generate a shuffled list of batches in the format (resolution, batch index within resolution)
        batches = []
        for bucket_key in bucket_dict.keys():
            batch_count = int(len(bucket_dict[bucket_key]) / self.batch_size)
            batches.extend((bucket_key, i) for i in range(batch_count))
        rand.shuffle(batches)

        # for each bucket, generate a shuffled list of samples
        for bucket_key, bucket in bucket_dict.items():
            rand.shuffle(bucket)

        # drop images for full buckets
        for bucket_key in bucket_dict.keys():
            samples = bucket_dict[bucket_key]
            samples_to_drop = len(samples) % self.batch_size
            for i in range(samples_to_drop):
                # print('dropping sample from bucket ' + str(bucket_key))
                samples.pop()

        # calculate the order of samples
        index_list = []
        for bucket_key, bucket_index in batches:
            for i in range(bucket_index * self.batch_size, (bucket_index + 1) * self.batch_size):
                index_list.append(bucket_dict[bucket_key][i])

        # print(bucket_dict)
        # print(index_list)

        return index_list

    def __sort_resolutions(self, variation: int):
        resolutions = []
        for index in tqdm(range(self._get_previous_length(self.resolution_in_name)), desc='caching resolutions'):
            resolution = self._get_previous_item(self.current_variation, self.resolution_in_name, index)

            resolution = resolution[0], resolution[1]
            resolutions.append(resolution)

        # sort samples into dict of lists, with key = resolution
        self.bucket_dict = {}
        for index, resolution in enumerate(resolutions):
            if resolution not in self.bucket_dict:
                self.bucket_dict[resolution] = []
            self.bucket_dict[resolution].append(index)

    def start(self, variation: int):
        self.__sort_resolutions(variation)

        self.index_list = self.__shuffle()

    def get_item(self, index: int, requested_name: str = None) -> dict:
        index = self.index_list[index]

        item = {}

        for name in self.names:
            item[name] = self._get_previous_item(self.current_variation, name, index)

        return item
