from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch.nn.functional as F


class PadMaskedTokens(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            tokens_name: str,
            tokens_mask_name: str,
            hidden_state_name: str,
            max_length: int,
    ):
        super().__init__()
        self.tokens_name = tokens_name
        self.tokens_mask_name = tokens_mask_name
        self.hidden_state_name = hidden_state_name
        self.max_length = max_length

    def length(self) -> int:
        return self._get_previous_length(self.tokens_name)

    def get_inputs(self) -> list[str]:
        return [self.tokens_name, self.tokens_mask_name, self.hidden_state_name]

    def get_outputs(self) -> list[str]:
        return [self.tokens_name, self.tokens_mask_name, self.hidden_state_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        tokens = self._get_previous_item(variation, self.tokens_name, index)
        tokens_mask = self._get_previous_item(variation, self.tokens_mask_name, index)
        hidden_state = self._get_previous_item(variation, self.hidden_state_name, index)

        pad_length = self.max_length - tokens.shape[0]
        if pad_length < 0:
            # F.pad with a negative pad silently TRUNCATES — after
            # PruneMaskedTokens every incoming token is a real prompt token,
            # so a max_length below the tokenizer's effective cap would drop
            # real tokens (and their hidden states) without any warning.
            raise ValueError(
                f"PadMaskedTokens: sequence length {tokens.shape[0]} exceeds "
                f"max_length {self.max_length}; refusing to silently truncate "
                f"real tokens. Align max_length with the tokenizer's effective "
                f"maximum (including any additional format tokens)."
            )

        tokens = F.pad(tokens, (0, pad_length), value=0)
        tokens_mask = F.pad(tokens_mask, (0, pad_length), value=0)
        # hidden_state shape: [N, hidden_dim] -> [max_length, hidden_dim]
        hidden_state = F.pad(hidden_state, (0, 0, 0, pad_length), value=0.0)

        return {
            self.tokens_name: tokens,
            self.tokens_mask_name: tokens_mask,
            self.hidden_state_name: hidden_state,
        }
