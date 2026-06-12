"""
Batch-vs-single equivalence tests for EncodeMistralText and EncodeLlamaText.

Same invariant as the EncodeQwenText suite: with the batch collector enabled
and requests arriving concurrently, every sample's hidden states must match
the legacy bs=1 forward. EncodeMistralText only reads
``text_encoder(tokens, attention_mask=..., output_hidden_states=...)``, so a
tiny MistralForCausalLM stands in for the Mistral3 VLM wrapper.
"""

from concurrent.futures import ThreadPoolExecutor

from mgds.MGDS import MGDS
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModules.EncodeLlamaText import EncodeLlamaText
from mgds.pipelineModules.EncodeMistralText import EncodeMistralText
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch

import pytest

SEQ_LEN = 16
NUM_ITEMS = 8


class TokenSource(PipelineModule, RandomAccessPipelineModule):
    def __init__(self, tokens: list[torch.Tensor], masks: list[torch.Tensor]):
        super().__init__()
        self.tokens = tokens
        self.masks = masks

    def length(self) -> int:
        return len(self.tokens)

    def get_inputs(self) -> list[str]:
        return []

    def get_outputs(self) -> list[str]:
        return ["tokens", "tokens_mask"]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return {"tokens": self.tokens[index], "tokens_mask": self.masks[index]}


@pytest.fixture(scope="module")
def token_data():
    g = torch.Generator()
    g.manual_seed(1)
    tokens, masks = [], []
    for i in range(NUM_ITEMS):
        real_len = 3 + (i * 2) % (SEQ_LEN - 2)
        t = torch.randint(1, 128, (SEQ_LEN,), generator=g)
        t[real_len:] = 0
        m = torch.zeros(SEQ_LEN, dtype=torch.long)
        m[:real_len] = 1
        tokens.append(t)
        masks.append(m)
    return tokens, masks


@pytest.fixture(scope="module")
def tiny_mistral():
    from transformers import MistralConfig, MistralForCausalLM

    config = MistralConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return MistralForCausalLM(config).eval()


@pytest.fixture(scope="module")
def tiny_llama():
    from transformers import LlamaConfig, LlamaModel

    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return LlamaModel(config).eval()


def _build_pipeline(source_module, encode_module):
    output = OutputPipelineModule(names=["hidden"])
    MGDS(
        device=torch.device("cpu"),
        concepts=[{"name": "C", "path": "dummy"}],
        settings={},
        definition=[[source_module], [encode_module], [output]],
        batch_size=1,
        state=PipelineState(),
        seed=7,
    )
    return encode_module


def _concurrent_results(module, num_items):
    def fetch(i):
        with torch.no_grad():
            return i, module.get_item(0, i)["hidden"]

    with ThreadPoolExecutor(max_workers=num_items) as pool:
        return dict(pool.map(fetch, range(num_items)))


class TestMistralBatching:
    def test_concurrent_batched_matches_single(self, tiny_mistral, token_data):
        tokens, masks = token_data
        baseline = _build_pipeline(
            TokenSource(tokens, masks),
            EncodeMistralText(
                tokens_name="tokens",
                tokens_attention_mask_in_name="tokens_mask",
                hidden_state_out_name="hidden",
                tokens_attention_mask_out_name="tokens_mask",
                text_encoder=tiny_mistral,
                hidden_state_output_index=-2,
            ),
        )
        batched = _build_pipeline(
            TokenSource(tokens, masks),
            EncodeMistralText(
                tokens_name="tokens",
                tokens_attention_mask_in_name="tokens_mask",
                hidden_state_out_name="hidden",
                tokens_attention_mask_out_name="tokens_mask",
                text_encoder=tiny_mistral,
                hidden_state_output_index=-2,
                batch_collector=True,
                max_batch_size=4,
            ),
        )

        with torch.no_grad():
            refs = [baseline.get_item(0, i)["hidden"] for i in range(NUM_ITEMS)]
        results = _concurrent_results(batched, NUM_ITEMS)

        for i in range(NUM_ITEMS):
            assert results[i].shape == refs[i].shape
            torch.testing.assert_close(results[i], refs[i], atol=1e-5, rtol=1e-4)


class TestLlamaBatching:
    def test_concurrent_batched_matches_single_one_layer(self, tiny_llama, token_data):
        tokens, masks = token_data

        def make(**kwargs):
            return _build_pipeline(
                TokenSource(tokens, masks),
                EncodeLlamaText(
                    tokens_name="tokens",
                    tokens_attention_mask_in_name="tokens_mask",
                    hidden_state_out_name="hidden",
                    tokens_attention_mask_out_name="tokens_mask",
                    text_encoder=tiny_llama,
                    hidden_state_output_index=-2,
                    dtype=torch.float32,
                    **kwargs,
                ),
            )

        baseline = make()
        batched = make(batch_collector=True, max_batch_size=4)

        with torch.no_grad():
            refs = [baseline.get_item(0, i)["hidden"] for i in range(NUM_ITEMS)]
        results = _concurrent_results(batched, NUM_ITEMS)

        for i in range(NUM_ITEMS):
            torch.testing.assert_close(results[i], refs[i], atol=1e-5, rtol=1e-4)

    def test_concurrent_batched_matches_single_all_layers_with_crop(self, tiny_llama, token_data):
        """HiDream-style config: all hidden states as a list, plus the
        HunyuanVideo-style crop_start head slice."""
        tokens, masks = token_data

        def make(**kwargs):
            return _build_pipeline(
                TokenSource(tokens, masks),
                EncodeLlamaText(
                    tokens_name="tokens",
                    tokens_attention_mask_in_name="tokens_mask",
                    hidden_state_out_name="hidden",
                    tokens_attention_mask_out_name="tokens_mask",
                    text_encoder=tiny_llama,
                    output_all_hidden_states=True,
                    all_hidden_state_output_indices=[0, 1],
                    crop_start=2,
                    dtype=torch.float32,
                    **kwargs,
                ),
            )

        baseline = make()
        batched = make(batch_collector=True, max_batch_size=4)

        with torch.no_grad():
            refs = [baseline.get_item(0, i)["hidden"] for i in range(NUM_ITEMS)]
        results = _concurrent_results(batched, NUM_ITEMS)

        for i in range(NUM_ITEMS):
            assert isinstance(results[i], list) and len(results[i]) == len(refs[i])
            for layer_out, layer_ref in zip(results[i], refs[i], strict=True):
                assert layer_out.shape == layer_ref.shape
                torch.testing.assert_close(layer_out, layer_ref, atol=1e-5, rtol=1e-4)
