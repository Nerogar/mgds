from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class LoadText(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(self, path_in_name: str, text_out_name: str):
        super(LoadText, self).__init__()
        self.path_in_name = path_in_name
        self.text_out_name = text_out_name

    def length(self) -> int:
        return self._get_previous_length(self.path_in_name)

    def get_inputs(self) -> list[str]:
        return [self.path_in_name]

    def get_outputs(self) -> list[str]:
        return [self.text_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        path = self._get_previous_item(variation, self.path_in_name, index)

        try:
            with open(path, encoding='utf-8') as f:
                text = f.readline().strip()
                f.close()
        except FileNotFoundError:
            text = ''
        except:
            print("could not load text, it might be corrupted: " + path)
            raise

        return {
            self.text_out_name: text
        }
