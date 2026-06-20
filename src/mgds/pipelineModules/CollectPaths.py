import os

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

from tqdm import tqdm


class CollectPaths(
    PipelineModule,
    RandomAccessPipelineModule,
):
    FILE_ATTRIBUTE_REPARSE_POINT = 0x400

    def __init__(
            self,
            concept_in_name: str, path_in_name: str, include_subdirectories_in_name: str, enabled_in_name: str,
            path_out_name: str, concept_out_name: str,
            extensions: list[str], include_postfix: list[str] | None, exclude_postfix: list[str],
    ):
        super().__init__()

        self.concept_in_name = concept_in_name
        self.path_in_name = path_in_name
        self.include_subdirectories_in_name = include_subdirectories_in_name
        self.enabled_in_name = enabled_in_name

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

    def __directory_key(self, path: str) -> str:
        return os.path.normcase(os.path.realpath(path))

    def __is_reparse_point(self, entry: os.DirEntry) -> bool:
        try:
            attributes = getattr(entry.stat(follow_symlinks=False), 'st_file_attributes', 0)
            return bool(attributes & self.FILE_ATTRIBUTE_REPARSE_POINT)
        except OSError:
            return True

    def __list_files(
            self,
            path: str,
            include_subdirectories: bool,
            seen_directories: set[str] | None = None,
    ) -> list[str]:
        if seen_directories is None:
            seen_directories = set()

        directory_key = self.__directory_key(path)
        if directory_key in seen_directories:
            return []
        seen_directories.add(directory_key)

        files = []
        sub_directories = []

        with os.scandir(path) as entries:
            for entry in entries:
                if entry.name.startswith('.') and entry.is_dir(follow_symlinks=False):
                    continue

                if entry.is_file(follow_symlinks=False):
                    files.append(entry.path)
                elif include_subdirectories and entry.is_dir(follow_symlinks=False):
                    if entry.is_symlink() or self.__is_reparse_point(entry):
                        continue
                    sub_directories.append(entry.path)

        for sub_directory in sub_directories:
            files.extend(self.__list_files(sub_directory, include_subdirectories, seen_directories))

        return files

    def start(self, variation: int):
        progress = tqdm(range(self._get_previous_length(self.concept_in_name)), desc='enumerating sample paths')
        for in_index in progress:
            concept = self._get_previous_item(variation, self.concept_in_name, in_index)
            enabled = concept[self.enabled_in_name]
            if enabled:
                include_subdirectories = self._get_previous_item(variation, self.include_subdirectories_in_name, in_index)
                path = concept[self.path_in_name]
                progress.set_postfix_str(os.path.basename(path) or path, refresh=False)

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
        #note: index is not the same index as the index in previous nodes
        return {
            self.path_out_name: self.paths[index],
            self.concept_out_name: self.concepts[index],
        }
