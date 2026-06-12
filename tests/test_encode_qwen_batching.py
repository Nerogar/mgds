"""
Equivalence tests for EncodeQwenText trim_padding and the batch collector.

Uses a tiny randomly-initialized Qwen3 on CPU/fp32. The invariant under test:
for every real-token position, trimmed and batched forwards produce the same
hidden states as the legacy padded bs=1 forward (causal attention + right
padding means trailing pads cannot influence earlier positions).
"""

import threading
from concurrent.futures import ThreadPoolExecutor

from mgds.MGDS import MGDS
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModules.EncodeQwenText import EncodeQwenText
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule
from mgds.TextEncoderBatching import BatchCollector as _BatchCollector

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
def tiny_qwen():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return Qwen3ForCausalLM(config).eval()


@pytest.fixture(scope="module")
def token_data():
    g = torch.Generator()
    g.manual_seed(1)
    tokens, masks = [], []
    for i in range(NUM_ITEMS):
        real_len = 3 + (i * 2) % (SEQ_LEN - 2)  # varied real lengths, < SEQ_LEN
        t = torch.randint(1, 128, (SEQ_LEN,), generator=g)
        t[real_len:] = 0  # pad id
        m = torch.zeros(SEQ_LEN, dtype=torch.long)
        m[:real_len] = 1
        tokens.append(t)
        masks.append(m)
    return tokens, masks


def _make_encode_module(tiny_qwen, token_data, **encode_kwargs) -> EncodeQwenText:
    tokens, masks = token_data
    source = TokenSource(tokens, masks)
    encode = EncodeQwenText(
        tokens_name="tokens",
        tokens_attention_mask_in_name="tokens_mask",
        hidden_state_out_name="hidden",
        tokens_attention_mask_out_name="tokens_mask",
        text_encoder=tiny_qwen,
        hidden_state_output_index=-2,
        dtype=torch.float32,
        **encode_kwargs,
    )
    output = OutputPipelineModule(names=["hidden"])
    MGDS(
        device=torch.device("cpu"),
        concepts=[{"name": "C", "path": "dummy"}],
        settings={},
        definition=[[source], [encode], [output]],
        batch_size=1,
        state=PipelineState(),
        seed=7,
    )
    return encode


def _real_length(mask: torch.Tensor) -> int:
    return int(mask.sum().item())


class TestTrimPadding:
    def test_trimmed_matches_padded_on_real_tokens(self, tiny_qwen, token_data):
        baseline = _make_encode_module(tiny_qwen, token_data)
        trimmed = _make_encode_module(tiny_qwen, token_data, trim_padding=True)
        _, masks = token_data

        for i in range(NUM_ITEMS):
            with torch.no_grad():
                ref = baseline.get_item(0, i)["hidden"]
                out = trimmed.get_item(0, i)["hidden"]
            n = _real_length(masks[i])
            assert out.shape == ref.shape
            torch.testing.assert_close(out[:n], ref[:n], atol=1e-5, rtol=1e-4)
            assert torch.all(out[n:] == 0), "trimmed tail must be zero-filled"

    def test_non_contiguous_mask_falls_back_to_full_forward(self, tiny_qwen, token_data):
        tokens, _ = token_data
        hole_mask = torch.ones(SEQ_LEN, dtype=torch.long)
        hole_mask[5] = 0  # not a contiguous prefix -> no trim possible
        module = _make_encode_module(tiny_qwen, ([tokens[0]], [hole_mask]), trim_padding=True)
        baseline = _make_encode_module(tiny_qwen, ([tokens[0]], [hole_mask]))
        with torch.no_grad():
            out = module.get_item(0, 0)["hidden"]
            ref = baseline.get_item(0, 0)["hidden"]
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


class TestBatchCollector:
    def test_concurrent_batched_matches_sequential(self, tiny_qwen, token_data):
        baseline = _make_encode_module(tiny_qwen, token_data)
        batched = _make_encode_module(
            tiny_qwen, token_data, trim_padding=True, batch_collector=True, max_batch_size=4
        )
        _, masks = token_data

        with torch.no_grad():
            refs = [baseline.get_item(0, i)["hidden"] for i in range(NUM_ITEMS)]

        def fetch(i):
            with torch.no_grad():
                return i, batched.get_item(0, i)["hidden"]

        with ThreadPoolExecutor(max_workers=NUM_ITEMS) as pool:
            results = dict(pool.map(fetch, range(NUM_ITEMS)))

        for i in range(NUM_ITEMS):
            n = _real_length(masks[i])
            torch.testing.assert_close(results[i][:n], refs[i][:n], atol=1e-5, rtol=1e-4)

    def test_collector_actually_batches_and_never_deadlocks(self):
        batch_sizes = []
        lock = threading.Lock()

        def run_batch(batch):
            with lock:
                batch_sizes.append(len(batch))
            for request in batch:
                request.result = request.tokens * 2

        collector = _BatchCollector(run_batch, max_batch_size=4, max_wait_seconds=0.2)

        def call(i):
            tokens = torch.full((4,), i)
            result = collector.encode(tokens, None)
            assert torch.equal(result, tokens * 2)
            return i

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(call, i) for i in range(32)]
            done = [f.result(timeout=30) for f in futures]

        assert sorted(done) == list(range(32))
        assert sum(batch_sizes) == 32
        assert max(batch_sizes) > 1, "expected at least one multi-request batch"

    def test_collector_propagates_errors_to_all_waiters(self):
        def run_batch(batch):
            raise RuntimeError("encoder exploded")

        collector = _BatchCollector(run_batch, max_batch_size=4, max_wait_seconds=0.05)

        def call(i):
            collector.encode(torch.zeros(2), None)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(call, i) for i in range(4)]
            for f in futures:
                with pytest.raises(RuntimeError, match="encoder exploded"):
                    f.result(timeout=30)
