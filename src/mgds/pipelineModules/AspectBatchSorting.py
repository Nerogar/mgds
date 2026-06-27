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
        for index in range(self._get_previous_length(self.resolution_in_name)):
            resolution = self._get_previous_item(self.current_variation, self.resolution_in_name, index)

            resolutions.append(resolution)

        # sort samples into dict of lists, with key = resolution
        self.bucket_dict = {}
        for index, resolution in enumerate(resolutions):
            if resolution not in self.bucket_dict:
                self.bucket_dict[resolution] = []
            self.bucket_dict[resolution].append(index)

    def start(self, variation: int):
        # Consistency guard: every batched input must have the same number of
        # items as the resolution input. A mismatch means the upstream caches
        # disagree on sample count (e.g. the image cache and text cache were
        # built to different lengths, or a cache sync completed only partially),
        # which would otherwise surface as a cryptic index/unpack error deep
        # inside variation sorting. Fail early with an actionable message.
        resolution_length = self._get_previous_length(self.resolution_in_name)
        for name in self.names:
            name_length = self._get_previous_length(name)
            if name_length != resolution_length:
                raise RuntimeError(
                    f"AspectBatchSorting: input '{name}' has {name_length} items but "
                    f"resolution input '{self.resolution_in_name}' has {resolution_length}. "
                    f"The caches feeding this dataset are inconsistent - most often the "
                    f"image and text caches were built to different counts, or a cache "
                    f"sync completed only partially. Rebuild the cache (or finish syncing "
                    f"it) so every input has the same number of samples."
                )

        self.__sort_resolutions(variation)

        self.index_list = self.__shuffle()

    def get_item(self, index: int, requested_name: str = None) -> dict:
        index = self.index_list[index]

        item = {}

        for name in self.names:
            item[name] = self._get_previous_item(self.current_variation, name, index)

        return item
