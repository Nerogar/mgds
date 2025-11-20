from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.SerialPipelineModule import SerialPipelineModule
import torch


class BatchPadding(
    PipelineModule,
    SerialPipelineModule,
):
    def __init__(
            self,
            latent_name: str,
            mask_out_name: str,
            names: [str],
            batch_size: int,
    ):
        super(BatchPadding, self).__init__()
        self.latent_name = latent_name
        self.mask_out_name = mask_out_name
        self.names = names
        if self.latent_name in self.names: #TODO necessary?
            self.names.remove(self.latent_name)
        if self.mask_out_name in self.names:
            self.names.remove(self.mask_out_name)

        self.batch_size = batch_size

        self.in_index_list = []
        self.next_in_cache_index = 0
        self.batch = []

    def approximate_length(self) -> int:
        return len(self.in_index_list)

    def has_next(self) -> bool:
        return len(self.batch) > 0

    def get_inputs(self) -> list[str]:
        return self.names + [self.latent_name]

    def get_outputs(self) -> list[str]:
        return self.get_inputs() + [self.mask_out_name]

    def __shuffle(self, variation: int) -> list[int]:
        rand = self._get_rand(variation)

        length = self._get_previous_length(self.latent_name)
        index_list = list(range(length))
        rand.shuffle(index_list)
        return index_list

    def __fill_cache(self):
        max_shape = (0,0)
        self.batch = []
        for i in range(self.batch_size):
            if self.next_in_cache_index >= len(self.in_index_list):
                break

            in_index = self.in_index_list[self.next_in_cache_index]
            self.next_in_cache_index += 1

            latent = self._get_previous_item(self.current_variation, self.latent_name, in_index)
            item = {
                self.latent_name: latent
            }
            for name in self.names:
                item[name] = self._get_previous_item(self.current_variation, name, in_index)
            self.batch.append(item)

            max_shape = (max(max_shape[-2], latent.shape[-2]), max(max_shape[-1], latent.shape[-1]))

        for sample in self.batch:
            latent = sample[self.latent_name]
            padded_latent = torch.zeros(latent.shape[:-2] + max_shape, dtype=latent.dtype, device=latent.device)
            padded_latent[..., :latent.shape[-2], :latent.shape[-1]] = latent
            sample[self.latent_name] = padded_latent

            mask = torch.full(max_shape, False, dtype=torch.bool, device=latent.device)
            mask[:latent.shape[-2], :latent.shape[-1]] = True
            sample[self.mask_out_name] = mask

            #TODO pad conditioning images etc.

    def start(self, variation: int, start_index: int):
        self.in_index_list = self.__shuffle(variation)
        self.next_in_cache_index = start_index
        self.__fill_cache()

    def get_next_item(self) -> dict:
        item = self.batch.pop(0)
        if len(self.batch) == 0:
            self.__fill_cache()
        return item
