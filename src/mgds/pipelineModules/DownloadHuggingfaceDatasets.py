import os

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule
import huggingface_hub
from contextlib import suppress


class DownloadHuggingfaceDatasets(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            concept_in_name: str,
            path_in_name: str,
            enabled_in_name: str,
            concept_out_name: str,
    ):
        super(DownloadHuggingfaceDatasets, self).__init__()

        self.concept_in_name = concept_in_name
        self.enabled_in_name = enabled_in_name
        self.path_in_name = path_in_name

        self.concept_out_name = concept_out_name

        self.concepts = []

    def length(self) -> int:
        return len(self.concepts)

    def get_inputs(self) -> list[str]:
        return [self.concept_in_name]

    def get_outputs(self) -> list[str]:
        return [self.concept_out_name]

    def start(self, variation: int):
        for in_index in range(self._get_previous_length(self.concept_in_name)):
            concept = self._get_previous_item(variation, self.concept_in_name, in_index)
            enabled = concept[self.enabled_in_name]
            if enabled:
                path = concept[self.path_in_name]

                is_local = os.path.isdir(path)
                if not is_local:
                    with suppress(huggingface_hub.errors.HFValidationError):
                        hf_path = huggingface_hub.snapshot_download(
                            repo_id=path,
                            repo_type="dataset",
                            max_workers=16,
                        )
                        concept = concept.copy()
                        concept[self.path_in_name] = hf_path

                self.concepts.append(concept)

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return {
            self.concept_out_name: self.concepts[index],
        }
