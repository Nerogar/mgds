from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class ShuffleTags(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            text_in_name: str,
            enabled_in_name: str,
            delimiter_in_name: str,
            keep_tags_count_in_name: str,
            text_out_name: str,
    ):
        super(ShuffleTags, self).__init__()
        self.text_in_name = text_in_name
        self.enabled_in_name = enabled_in_name
        self.delimiter_in_name = delimiter_in_name
        self.keep_tags_count_in_name = keep_tags_count_in_name
        self.text_out_name = text_out_name

    def length(self) -> int:
        return self._get_previous_length(self.text_in_name)

    def get_inputs(self) -> list[str]:
        return [self.text_in_name, self.enabled_in_name, self.delimiter_in_name, self.keep_tags_count_in_name]

    def get_outputs(self) -> list[str]:
        return [self.text_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        text = self._get_previous_item(variation, self.text_in_name, index)
        delimiter = self._get_previous_item(variation, self.delimiter_in_name, index)
        keep_tags_count = self._get_previous_item(variation, self.keep_tags_count_in_name, index)
        enabled = self._get_previous_item(variation, self.enabled_in_name, index)
        rand = self._get_rand(variation, index)

        if enabled:
            tags = [tag.strip() for tag in text.split(delimiter)]
            keep_tags = tags[:keep_tags_count]
            shuffle_tags = tags[keep_tags_count:]
            rand.shuffle(shuffle_tags)
            tags = keep_tags + shuffle_tags

            text = ", ".join(tags)

        return {
            self.text_out_name: text
        }
