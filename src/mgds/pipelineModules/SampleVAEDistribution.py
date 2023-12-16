from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class SampleVAEDistribution(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(self, in_name: str, out_name: str, mode='mean'):
        super(SampleVAEDistribution, self).__init__()
        self.in_name = in_name
        self.out_name = out_name
        self.mode = mode

    def length(self) -> int:
        return self._get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        distribution = self._get_previous_item(variation, self.in_name, index)

        if self.mode == 'sample':
            latent = distribution.sample()
        elif self.mode == 'mean':
            latent = distribution.mode()
        else:
            raise Exception('method not supported')

        latent = latent.squeeze()

        return {
            self.out_name: latent
        }
