from contextlib import nullcontext

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule
from mgds.TextEncoderBatching import BatchCollector, PendingEncode

import torch

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3ForCausalLM


class EncodeQwenText(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            tokens_name: str,
            tokens_attention_mask_in_name: str | None,
            hidden_state_out_name: str,
            tokens_attention_mask_out_name: str | None,
            text_encoder: Qwen2_5_VLForConditionalGeneration | Qwen3ForCausalLM,
            hidden_state_output_index: int | list[int],
            crop_start: int | None = None,
            autocast_contexts: list[torch.autocast | None] = None,
            dtype: torch.dtype | None = None,
            trim_padding: bool = False,
            batch_collector: bool = False,
            max_batch_size: int = 8,
    ):
        super().__init__()
        self.tokens_name = tokens_name
        self.tokens_attention_mask_in_name = tokens_attention_mask_in_name
        self.hidden_state_out_name = hidden_state_out_name
        self.tokens_attention_mask_out_name = tokens_attention_mask_out_name
        self.text_encoder = text_encoder
        self.hidden_state_indexes = hidden_state_output_index if isinstance(hidden_state_output_index, list) else [hidden_state_output_index]
        self.crop_start = crop_start

        self.autocast_contexts = [nullcontext()] if autocast_contexts is None else autocast_contexts
        self.dtype = dtype

        # When True, only the real-token prefix is run through the encoder;
        # the hidden state is zero re-padded back to the full sequence
        # length. With right padding and a causal LM the hidden states of
        # real tokens are mathematically unaffected by trailing pad tokens,
        # so this is exact for every position a downstream PruneMaskedTokens
        # keeps. Only enable when the padded positions of the hidden state
        # are discarded downstream (pruned or masked) — their values change
        # from "encoder output for pad tokens" to zeros.
        self.trim_padding = trim_padding

        # Opt-in micro-batching across concurrent get_item callers (cache
        # build workers). Off by default: single-request behavior is exactly
        # the legacy bs=1 forward.
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

    @staticmethod
    def _unpadded_length(mask: torch.Tensor) -> int | None:
        """Length of the real-token prefix, or None when the mask is not a
        contiguous prefix (left padding, expand_mask) — callers must then
        encode the full sequence to stay exact.
        """
        n = int(mask.sum().item())
        if n <= 0 or n >= mask.shape[0]:
            return None
        if bool(mask[:n].all().item()):
            return n
        return None

    def _forward(self, tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Single padded-batch forward; returns the concatenated hidden
        state selection of shape (batch, seq, hidden)."""
        with self._all_contexts(self.autocast_contexts):
            text_encoder_output = self.text_encoder(
                tokens,
                attention_mask=mask.to(dtype=self.dtype) if mask is not None else None,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        return torch.cat([text_encoder_output.hidden_states[k] for k in self.hidden_state_indexes], dim=-1)

    def _effective_length(self, tokens: torch.Tensor, mask: torch.Tensor | None) -> int:
        if self.trim_padding and mask is not None:
            n = self._unpadded_length(mask)
            if n is not None:
                return n
        return tokens.shape[0]

    def _run_collected_batch(self, batch: list[PendingEncode]):
        """Encode a batch of requests with one forward.

        Each sequence is sliced to the batch's longest effective length and
        stacked — every row keeps right padding for its own tail, and causal
        attention plus the attention mask guarantee the real-token rows match
        a bs=1 forward. Results are re-expanded to each request's original
        sequence length with zero rows for the trimmed tail.
        """
        # Batching assumes uniform upstream sequence lengths (Tokenize pads
        # to a fixed max_token_length). Anything else falls back to per-item
        # encodes rather than crashing the cache build on torch.stack.
        if len({r.tokens.shape[0] for r in batch}) != 1:
            for request in batch:
                request.result = self._encode_single(request.tokens, request.mask)
            return

        lengths = [self._effective_length(r.tokens, r.mask) for r in batch]
        max_len = max(lengths)

        tokens = torch.stack([r.tokens[:max_len] for r in batch])
        mask = torch.stack([r.mask[:max_len] for r in batch]) if all(r.mask is not None for r in batch) else None

        hidden = self._forward(tokens, mask)

        for i, request in enumerate(batch):
            seq_len = request.tokens.shape[0]
            n = lengths[i]
            if n < seq_len:
                # Zero everything past this request's OWN effective length —
                # not just past the batch max. Positions [n, max_len) hold
                # the encoder's outputs for this row's pad tokens; keeping
                # them would make the result (and any cache built from it)
                # depend on which longer requests shared the batch, where
                # _encode_single deterministically writes zeros.
                full = torch.zeros(
                    (seq_len, hidden.shape[-1]),
                    dtype=hidden.dtype,
                    device=hidden.device,
                )
                full[:n] = hidden[i][:n]
                request.result = full
            else:
                request.result = hidden[i]

    def _encode_single(self, tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        seq_len = tokens.shape[0]
        n = self._effective_length(tokens, mask)
        if n < seq_len:
            hidden = self._forward(
                tokens[:n].unsqueeze(0),
                mask[:n].unsqueeze(0) if mask is not None else None,
            ).squeeze(dim=0)
            full = torch.zeros((seq_len, hidden.shape[-1]), dtype=hidden.dtype, device=hidden.device)
            full[:n] = hidden
            return full
        return self._forward(
            tokens.unsqueeze(0),
            mask.unsqueeze(0) if mask is not None else None,
        ).squeeze(dim=0)

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        tokens = self._get_previous_item(variation, self.tokens_name, index)

        if self.tokens_attention_mask_in_name is not None:
            tokens_attention_mask = self._get_previous_item(variation, self.tokens_attention_mask_in_name, index)
        else:
            tokens_attention_mask = None

        if self._collector is not None:
            hidden_state = self._collector.encode(tokens, tokens_attention_mask)
        else:
            hidden_state = self._encode_single(tokens, tokens_attention_mask)

        if self.crop_start is not None:
            tokens = tokens[self.crop_start:]
            tokens_attention_mask = tokens_attention_mask[self.crop_start:]
            #set masked state to 0 should not make a difference, but the reference implementation in diffusers also does that:
            hidden_state = hidden_state[self.crop_start:] * tokens_attention_mask.unsqueeze(dim=-1)

        return {
            self.tokens_name: tokens,
            self.hidden_state_out_name: hidden_state,
            self.tokens_attention_mask_out_name: tokens_attention_mask,
        }
