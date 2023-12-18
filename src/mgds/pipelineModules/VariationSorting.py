import hashlib
import json
import math
from typing import Any

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class VariationSorting(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            names: list[str],
            repeats_in_name: str | None = None,
            variations_group_in_name: str | list[str] | None = None,
    ):
        super(VariationSorting, self).__init__()

        self.names = names

        self.repeats_in_name = repeats_in_name
        self.variations_group_in_name = \
            [variations_group_in_name] if isinstance(variations_group_in_name, str) else variations_group_in_name

        self.variations_initialized = False

    def length(self) -> int:
        if not self.variations_initialized:
            return self._get_previous_length(self.names[0])
        else:
            return sum(x for x in self.group_output_samples.values())

    def get_inputs(self) -> list[str]:
        return self.names

    def get_outputs(self) -> list[str]:
        return self.names

    def __string_key(self, data: list[Any]) -> str:
        json_data = json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(',', ':'), indent=None)
        return hashlib.sha256(json_data.encode('utf-8')).hexdigest()

    def __init_variations(self):
        """
        Prepares variations before caching starts. Each index is sorted into a group.

        Data is written into three variables.
            self.group_variations, mapping group keys to the number of variations of that group
            self.group_indices, mapping group keys to a list of input indices contained in the group
            self.group_output_samples, mapping group keys to the number of indices in the output for each group
        """
        if self.repeats_in_name is not None:
            group_indices = {}
            group_repeats = {}

            for in_index in range(self._get_previous_length(self.repeats_in_name)):
                repeats = self._get_previous_item(0, self.repeats_in_name, in_index)
                group_key = self.__string_key(
                    [self._get_previous_item(0, name, in_index) for name in self.variations_group_in_name]
                )

                if group_key not in group_indices:
                    group_indices[group_key] = []
                if group_key not in group_repeats:
                    group_repeats[group_key] = repeats
                group_indices[group_key].append(in_index)

            group_output_samples = {}
            for group_key, repeats in group_repeats.items():
                num = int(math.floor(len(group_indices[group_key]) * repeats))
                group_output_samples[group_key] = num
        else:
            first_previous_name = self.names[0]

            group_indices = {'': [in_index for in_index in range(self._get_previous_length(first_previous_name))]}
            group_output_samples = {'': len(group_indices[''])}

        self.aggregate_cache = {}

        self.group_indices = group_indices
        self.group_output_samples = group_output_samples

        self.variations_initialized = True

    def __get_input_index(self, out_variation: int, out_index: int) -> (str, int, int):
        offset = 0
        for group_key, group_output_samples in self.group_output_samples.items():
            if out_index >= group_output_samples + offset:
                offset += group_output_samples
                continue

            local_index = (out_index - offset) + (out_variation * self.group_output_samples[group_key])
            in_variation = (local_index // len(self.group_indices[group_key]))
            group_index = local_index % len(self.group_indices[group_key])
            in_index = self.group_indices[group_key][group_index]

            return group_key, in_variation, group_index, in_index

    def start(self, variation: int):
        if not self.variations_initialized:
            self.__init_variations()


    def get_item(self, variation:int, index: int, requested_name: str = None) -> dict:
        group_key, in_variation, group_index, in_index = self.__get_input_index(variation, index)
        value = self._get_previous_item(variation, requested_name, in_index)
        return {
            requested_name: value
        }
