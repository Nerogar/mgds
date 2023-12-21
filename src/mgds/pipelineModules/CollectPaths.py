import os

from tqdm import tqdm

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class CollectPaths(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            concept_in_name: str, path_in_name: str, include_subdirectories_in_name: str,
            path_out_name: str, concept_out_name: str,
            extensions: [str], include_postfix: [str], exclude_postfix: [str],
    ):
        super(CollectPaths, self).__init__()

        self.concept_in_name = concept_in_name
        self.path_in_name = path_in_name
        self.include_subdirectories_in_name = include_subdirectories_in_name

        self.path_out_name = path_out_name
        self.concept_out_name = concept_out_name

        self.extensions = [extension.lower() for extension in extensions]
        self.include_postfix = include_postfix
        self.exclude_postfix = exclude_postfix

        self.paths = []
        self.concepts = []

    def length(self) -> int:
        return len(self.paths)

    def get_inputs(self) -> list[str]:
        return [self.concept_in_name]

    def get_outputs(self) -> list[str]:
        return [self.path_out_name, self.concept_out_name]

    def __list_files(self, path: str, include_subdirectories: bool) -> list[str]:
        dir_list = [os.path.join(path, filename) for filename in os.listdir(path)]

        files = list(filter(os.path.isfile, dir_list))

        if include_subdirectories:
            sub_directories = list(filter(os.path.isdir, dir_list))
            for sub_directory in sub_directories:
                files.extend(self.__list_files(sub_directory, include_subdirectories))

        return files

    def start(self, variation: int):
        for index in tqdm(range(self._get_previous_length(self.concept_in_name)), desc='enumerating sample paths'):
            concept = self._get_previous_item(variation, self.concept_in_name, index)
            include_subdirectories = self._get_previous_item(variation, self.include_subdirectories_in_name, index)
            path = concept[self.path_in_name]

            file_names = sorted(self.__list_files(path, include_subdirectories))

            file_names = list(filter(lambda name: os.path.splitext(name)[1].lower() in self.extensions, file_names))

            if self.include_postfix:
                file_names = list(filter(
                    lambda name: any(os.path.splitext(name)[0].endswith(postfix) for postfix in self.include_postfix),
                    file_names))

            if self.exclude_postfix:
                file_names = list(filter(lambda name: not any(
                    os.path.splitext(name)[0].endswith(postfix) for postfix in self.exclude_postfix), file_names))

            self.paths.extend(file_names)
            self.concepts.extend([concept] * len(file_names))

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return {
            self.path_out_name: self.paths[index],
            self.concept_out_name: self.concepts[index],
        }
