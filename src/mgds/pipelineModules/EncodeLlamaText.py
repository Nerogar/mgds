from contextlib import nullcontext

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule
from mgds.TextEncoderBatching import BatchCollector, PendingEncode

import torch

from transformers import LlamaModel


class EncodeLlamaText(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            tokens_name: str,
            tokens_attention_mask_in_name: str | None,
            hidden_state_out_name: str,
            tokens_attention_mask_out_name: str | None,
            text_encoder: LlamaModel,
            hidden_state_output_index: int | None = None,
            output_all_hidden_states: bool = False,
            all_hidden_state_output_indices: list[int] | None = None,
            crop_start: int | None = None,
            autocast_contexts: list[torch.autocast | None] = None,
            dtype: torch.dtype | None = None,
            batch_collector: bool = False,
            max_batch_size: int = 8,
    ):
        super().__init__()
        self.tokens_name = tokens_name
        self.tokens_attention_mask_in_name = tokens_attention_mask_in_name
        self.hidden_state_out_name = hidden_state_out_name
        self.tokens_attention_mask_out_name = tokens_attention_mask_out_name
        self.text_encoder = text_encoder
        self.hidden_state_output_index = hidden_state_output_index
        self.output_all_hidden_states = output_all_hidden_states
        self.max_hidden_state_output_index = max(all_hidden_state_output_indices) \
            if all_hidden_state_output_indices is not None else None
        self.crop_start = crop_start

        self.autocast_contexts = [nullcontext()] if autocast_contexts is None else autocast_contexts
        self.dtype = dtype

        # Opt-in micro-batching across concurrent get_item callers (cache
        # build workers). Off by default: single-request behavior is exactly
        # the legacy bs=1 forward. Attention never crosses batch rows, so
        # per-sample results match bs=1 up to kernel-level float noise.
        self._collector = (
            BatchCollector(self._run_collected_batch, max_batch_size)
            if batch_collector and max_batch_size > 1
            else None
        )

    def length(self) -> int:
        return self._get_previous_length(self.tokens_name)

    def get_inputs(self) -> list[str]:
        return [self.tokens_name, self.tokens_attention_mask_in_name]

    def get_outputs(self) -> list[str]:
        return [self.tokens_name, self.hidden_state_out_name, self.tokens_attention_mask_out_name]

    def _forward(self, tokens: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
        """Single padded-batch forward; returns the full hidden_states tuple,
        each entry of shape (batch, seq, hidden)."""
        with self._all_contexts(self.autocast_contexts):
            text_encoder_output = self.text_encoder(
                tokens,
                attention_mask=mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        return text_encoder_output.hidden_states

    def _extract(self, hidden_states: tuple[torch.Tensor, ...], batch_index: int):
        """Per-request result: a list of per-layer rows in all-hidden-states
        mode (HiDream), or a single layer's row otherwise."""
        if self.output_all_hidden_states:
            return [hs[batch_index] for hs in hidden_states[1:self.max_hidden_state_output_index + 2]]
        return hidden_states[self.hidden_state_output_index][batch_index]

    def _run_collected_batch(self, batch: list[PendingEncode]):
        # Batching assumes uniform upstream sequence lengths (Tokenize pads
        # to a fixed max_token_length). Anything else falls back to per-item
        # encodes rather than crashing the cache build on torch.stack.
        if len({r.tokens.shape[0] for r in batch}) != 1 or not all(
            (r.mask is None) == (batch[0].mask is None) for r in batch
        ):
            for request in batch:
                request.result = self._encode_single(request.tokens, request.mask)
            return

        tokens = torch.stack([r.tokens for r in batch])
        mask = torch.stack([r.mask for r in batch]) if batch[0].mask is not None else None

        hidden_states = self._forward(tokens, mask)
        for i, request in enumerate(batch):
            request.result = self._extract(hidden_states, i)

    def _encode_single(self, tokens: torch.Tensor, mask: torch.Tensor | None):
        hidden_states = self._forward(
            tokens.unsqueeze(0),
            mask.unsqueeze(0) if mask is not None else None,
        )
        return self._extract(hidden_states, 0)

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        tokens = self._get_previous_item(variation, self.tokens_name, index)

        if self.tokens_attention_mask_in_name is not None:
            tokens_attention_mask = self._get_previous_item(variation, self.tokens_attention_mask_in_name, index)
        else:
            tokens_attention_mask = None

        if tokens_attention_mask is not None and self.dtype:
            tokens_attention_mask = tokens_attention_mask.to(dtype=self.dtype)

        if self._collector is not None:
            hidden_state = self._collector.encode(tokens, tokens_attention_mask)
        else:
            hidden_state = self._encode_single(tokens, tokens_attention_mask)

        tokens = tokens.squeeze()

        if self.crop_start is not None:
            tokens = tokens[self.crop_start:]

            if self.output_all_hidden_states:
                hidden_state = [t[self.crop_start:] for t in hidden_state]
            else:
                hidden_state = hidden_state[self.crop_start:]

            tokens_attention_mask = tokens_attention_mask[self.crop_start:]

        return {
            self.tokens_name: tokens,
            self.hidden_state_out_name: hidden_state,
            self.tokens_attention_mask_out_name: tokens_attention_mask,
        }
