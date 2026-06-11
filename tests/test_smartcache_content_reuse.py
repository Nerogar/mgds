"""
Tests for SmartDiskCache content-addressed caption reuse
(``content_key_in_name``).

The pipeline simulates the text-caching chain: a caption file with one
caption per line, a deterministic line picker (variation -> line), and a
counting "encoder" whose output is a pure function of the caption text.
Reuse correctness is then observable two ways: the encoder call count, and
the cached embedding matching f(text) exactly.
"""

import json
import os
import threading

from mgds.MGDS import MGDS
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModules.SmartDiskCache import NO_RESOLUTION_KEY, SmartDiskCache
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch

import xxhash

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _embedding_for(text: str) -> torch.Tensor:
    """Deterministic stand-in for a text encoder forward."""
    seed = xxhash.xxh64(text.encode("utf-8")).intdigest() % (2**31)
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(8, generator=g)


class CaptionSource(PipelineModule, RandomAccessPipelineModule):
    """Yields one caption file path + concept dict per index."""

    def __init__(self, paths: list[str], concepts_per_index: list[dict]):
        super().__init__()
        self.paths = paths
        self.concepts_per_index = concepts_per_index

    def length(self) -> int:
        return len(self.paths)

    def get_inputs(self) -> list[str]:
        return []

    def get_outputs(self) -> list[str]:
        return ["sample_prompt_path", "concept"]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return {
            "sample_prompt_path": self.paths[index],
            "concept": self.concepts_per_index[index],
        }


class PickLine(PipelineModule, RandomAccessPipelineModule):
    """Deterministic SelectRandomText stand-in: variation v -> line v % n."""

    def get_inputs(self) -> list[str]:
        return ["sample_prompt_path"]

    def get_outputs(self) -> list[str]:
        return ["prompt"]

    def length(self) -> int:
        return self._get_previous_length("sample_prompt_path")

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        path = self._get_previous_item(variation, "sample_prompt_path", index)
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f if line.strip()]
        return {"prompt": lines[variation % len(lines)]}


class CountingEncoder(PipelineModule, RandomAccessPipelineModule):
    """Maps prompt -> deterministic embedding, recording every call."""

    def __init__(self):
        super().__init__()
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def get_inputs(self) -> list[str]:
        return ["prompt"]

    def get_outputs(self) -> list[str]:
        return ["embedding"]

    def length(self) -> int:
        return self._get_previous_length("prompt")

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        text = self._get_previous_item(variation, "prompt", index)
        with self._lock:
            self.calls.append(text)
        return {"embedding": _embedding_for(text)}


def _write_caption_file(directory, name: str, lines: list[str]) -> str:
    path = os.path.join(str(directory), name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _build_text_pipeline(
    tmp_path,
    paths: list[str],
    *,
    text_variations: int = 3,
    concepts_per_index: list[dict] | None = None,
    content_key_in_name: str | None = "prompt",
):
    """CaptionSource -> PickLine -> CountingEncoder -> SmartDiskCache -> out.

    Returns (ds, encoder, prepare_calls). Build a fresh pipeline on the same
    tmp_path to simulate a process restart against a persistent cache dir.
    """
    if concepts_per_index is None:
        concept = {
            "name": "C",
            "text_variations": text_variations,
            "balancing": 1.0,
            "balancing_strategy": "REPEATS",
        }
        concepts_per_index = [concept] * len(paths)

    source = CaptionSource(paths, concepts_per_index)
    picker = PickLine()
    encoder = CountingEncoder()
    prepare_calls: list[int] = []

    cache = SmartDiskCache(
        cache_dir=str(tmp_path / "cache"),
        split_names=["embedding"],
        aggregate_names=[],
        variations_in_name="concept.text_variations",
        balancing_in_name="concept.balancing",
        balancing_strategy_in_name="concept.balancing_strategy",
        variations_group_in_name="concept.name",
        before_cache_fun=lambda: prepare_calls.append(1),
        modeltype="testmodel",
        source_path_in_name="sample_prompt_path",
        content_key_in_name=content_key_in_name,
    )

    output = OutputPipelineModule(names=["embedding"])

    ds = MGDS(
        device=torch.device("cpu"),
        concepts=[{"name": "C", "path": "dummy"}],
        settings={},
        definition=[[source, picker, encoder], [cache], [output]],
        batch_size=1,
        state=PipelineState(),
        seed=42,
    )
    return ds, encoder, prepare_calls


def _drain(ds):
    ds.start_next_epoch()
    return list(ds)


def _variant_pt_path(cache_dir: str, source_path: str, variation: int) -> str:
    """Resolve the on-disk .pt for (source file, variation) via cache.json."""
    with open(os.path.join(cache_dir, "cache.json"), "r") as f:
        index = json.load(f)
    entry = index["entries"][os.path.normpath(source_path)]
    cache_file = entry["variants"][NO_RESOLUTION_KEY]["cache_file"]
    return os.path.join(cache_dir, f"{cache_file}_{variation + 1}.pt")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestContentReuse:
    def test_initial_build_encodes_every_unique_line(self, tmp_path):
        paths = [
            _write_caption_file(tmp_path / "src", "a.txt", ["a zero", "a one", "a two"]),
            _write_caption_file(tmp_path / "src", "b.txt", ["b zero", "b one", "b two"]),
        ]
        ds, encoder, prepare_calls = _build_text_pipeline(tmp_path, paths)

        batches = _drain(ds)

        assert sorted(encoder.calls) == sorted(["a zero", "a one", "a two", "b zero", "b one", "b two"])
        assert len(prepare_calls) == 1
        served = [b["embedding"].squeeze(0) for b in batches]
        for expected_text in ["a zero", "b zero"]:  # epoch 0 -> variation 0 -> line 0
            assert any(torch.equal(s, _embedding_for(expected_text)) for s in served)

        # content_index.json is persisted and points at real .pt files
        with open(tmp_path / "cache" / "content_index.json", "r") as f:
            content_index = json.load(f)
        assert len(content_index["entries"]) == 6
        for pt_name in content_index["entries"].values():
            assert os.path.isfile(tmp_path / "cache" / pt_name)

    def test_edit_one_line_reencodes_only_that_line(self, tmp_path):
        lines_a = ["a zero", "a one", "a two"]
        path_a = _write_caption_file(tmp_path / "src", "a.txt", lines_a)
        path_b = _write_caption_file(tmp_path / "src", "b.txt", ["b zero", "b one", "b two"])

        ds, encoder, _ = _build_text_pipeline(tmp_path, [path_a, path_b])
        _drain(ds)
        assert len(encoder.calls) == 6

        # Edit line 1 of a.txt only; simulate a restart with a fresh pipeline.
        _write_caption_file(tmp_path / "src", "a.txt", ["a zero", "a one EDITED", "a two"])
        ds2, encoder2, _ = _build_text_pipeline(tmp_path, [path_a, path_b])
        _drain(ds2)

        assert encoder2.calls == ["a one EDITED"]

        # Every variation .pt of a.txt holds exactly f(current line text).
        for v, text in enumerate(["a zero", "a one EDITED", "a two"]):
            data = torch.load(_variant_pt_path(str(tmp_path / "cache"), path_a, v), weights_only=False)
            assert torch.equal(data["embedding"], _embedding_for(text)), f"variation {v} mismatch"

    def test_reorder_lines_zero_encodes_and_no_model_prepare(self, tmp_path):
        path_a = _write_caption_file(tmp_path / "src", "a.txt", ["a zero", "a one", "a two"])
        ds, encoder, _ = _build_text_pipeline(tmp_path, [path_a])
        _drain(ds)
        assert len(encoder.calls) == 3

        reordered = ["a two", "a zero", "a one"]
        _write_caption_file(tmp_path / "src", "a.txt", reordered)
        ds2, encoder2, prepare_calls2 = _build_text_pipeline(tmp_path, [path_a])
        _drain(ds2)

        assert encoder2.calls == []
        # All variations were content-addressed copies: the encoder prepare
        # hook (before_cache_fun) must never have fired.
        assert prepare_calls2 == []
        for v, text in enumerate(reordered):
            data = torch.load(_variant_pt_path(str(tmp_path / "cache"), path_a, v), weights_only=False)
            assert torch.equal(data["embedding"], _embedding_for(text)), f"variation {v} mismatch"

    def test_shared_line_across_different_files_encoded_once(self, tmp_path):
        path_a = _write_caption_file(tmp_path / "src", "a.txt", ["shared trigger line"])
        ds, encoder, _ = _build_text_pipeline(tmp_path, [path_a], text_variations=1)
        _drain(ds)
        assert encoder.calls == ["shared trigger line"]

        # Different file bytes (so whole-file dedup can't match), same line.
        path_b = _write_caption_file(tmp_path / "src", "b.txt", ["shared trigger line", "never picked"])
        ds2, encoder2, _ = _build_text_pipeline(tmp_path, [path_a, path_b], text_variations=1)
        _drain(ds2)

        assert encoder2.calls == []
        data = torch.load(_variant_pt_path(str(tmp_path / "cache"), path_b, 0), weights_only=False)
        assert torch.equal(data["embedding"], _embedding_for("shared trigger line"))

    def test_without_content_key_no_reuse(self, tmp_path):
        path_a = _write_caption_file(tmp_path / "src", "a.txt", ["shared trigger line"])
        ds, encoder, _ = _build_text_pipeline(tmp_path, [path_a], text_variations=1, content_key_in_name=None)
        _drain(ds)
        assert encoder.calls == ["shared trigger line"]

        path_b = _write_caption_file(tmp_path / "src", "b.txt", ["shared trigger line", "never picked"])
        ds2, encoder2, _ = _build_text_pipeline(
            tmp_path, [path_a, path_b], text_variations=1, content_key_in_name=None
        )
        _drain(ds2)

        assert encoder2.calls == ["shared trigger line"]

    def test_reuse_refreshes_concept_metadata(self, tmp_path):
        concept_a = {
            "name": "A",
            "text_variations": 1,
            "balancing": 1.0,
            "balancing_strategy": "REPEATS",
            "loss_weight": 1.0,
        }
        concept_b = dict(concept_a, name="B", loss_weight=2.5)

        path_a = _write_caption_file(tmp_path / "src", "a.txt", ["shared trigger line"])
        ds, _, _ = _build_text_pipeline(
            tmp_path, [path_a], text_variations=1, concepts_per_index=[concept_a]
        )
        _drain(ds)

        path_b = _write_caption_file(tmp_path / "src", "b.txt", ["shared trigger line", "never picked"])
        ds2, encoder2, _ = _build_text_pipeline(
            tmp_path,
            [path_a, path_b],
            text_variations=1,
            concepts_per_index=[concept_a, concept_b],
        )
        _drain(ds2)

        assert encoder2.calls == []
        data = torch.load(_variant_pt_path(str(tmp_path / "cache"), path_b, 0), weights_only=False)
        assert data["__concept_loss_weight"] == 2.5
        assert data["__concept_name"] == "B"

    def test_unchanged_files_keep_fast_validation(self, tmp_path):
        """Content addressing must not disturb the no-change fast paths:
        a second epoch and a restart with identical files encode nothing
        and rewrite no .pt files."""
        path_a = _write_caption_file(tmp_path / "src", "a.txt", ["a zero", "a one", "a two"])
        ds, encoder, _ = _build_text_pipeline(tmp_path, [path_a])
        _drain(ds)
        calls_after_build = len(encoder.calls)

        pt_files = [f for f in os.listdir(tmp_path / "cache") if f.endswith(".pt")]
        mtimes = {f: os.path.getmtime(tmp_path / "cache" / f) for f in pt_files}

        _drain(ds)  # second epoch, same session
        assert len(encoder.calls) == calls_after_build

        ds2, encoder2, prepare_calls2 = _build_text_pipeline(tmp_path, [path_a])
        _drain(ds2)  # restart, no changes
        assert encoder2.calls == []
        assert prepare_calls2 == []
        for f in pt_files:
            assert os.path.getmtime(tmp_path / "cache" / f) == mtimes[f], f"{f} was rewritten"
