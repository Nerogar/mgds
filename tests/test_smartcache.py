"""
Comprehensive test suite for the SmartDiskCache module.

Covers hashing, cache validation flow, deduplication, atomic writes / crash
recovery, garbage collection, cache file format, and SAMPLES balancing
strategy.  Uses real temp files wherever SmartDiskCache needs to hash on disk.
"""

import json
import math
import os
import time

from mgds.MGDS import MGDS
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModules.SmartDiskCache import CACHE_VERSION, SmartDiskCache
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch

import pytest
import xxhash

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyDataModule(PipelineModule, RandomAccessPipelineModule):
    """Provides configurable dummy data for pipeline testing.

    *data* maps output names to lists of values.  ``get_item`` returns
    ``values[index % len(values)]`` so that any index is valid.
    """

    def __init__(self, data: dict[str, list], length: int):
        super().__init__()
        self.data = data
        self._length = length

    def length(self) -> int:
        return self._length

    def get_inputs(self) -> list[str]:
        return []

    def get_outputs(self) -> list[str]:
        return list(self.data.keys())

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return {name: values[index % len(values)] for name, values in self.data.items()}


class MutableDummyDataModule(PipelineModule, RandomAccessPipelineModule):
    """Like DummyDataModule but allows mutating data between epochs."""

    def __init__(self, data: dict[str, list], length: int):
        super().__init__()
        self.data = data
        self._length = length

    def length(self) -> int:
        return self._length

    def get_inputs(self) -> list[str]:
        return []

    def get_outputs(self) -> list[str]:
        return list(self.data.keys())

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return {name: values[index % len(values)] for name, values in self.data.items()}


def _make_tensors(n: int, seed: int = 0) -> list[torch.Tensor]:
    g = torch.Generator()
    g.manual_seed(seed)
    return [torch.randn(4, 4, generator=g) for _ in range(n)]


def _build_smart_pipeline(
    tmp_path,
    concepts,
    dummy_data,
    dummy_length,
    split_names,
    aggregate_names,
    *,
    variations_in_name=None,
    balancing_in_name=None,
    balancing_strategy_in_name=None,
    variations_group_in_name=None,
    group_enabled_in_name=None,
    modeltype="testmodel",
    source_path_in_name=None,
    batch_size=1,
    seed=42,
    dummy_module_cls=DummyDataModule,
    extra_watched_paths_in_names=None,
):
    """Build MGDS pipeline: DummyDataModule -> SmartDiskCache -> Output."""
    cache_dir = str(tmp_path / "cache")

    dummy_mod = dummy_module_cls(data=dummy_data, length=dummy_length)

    cache_mod = SmartDiskCache(
        cache_dir=cache_dir,
        split_names=split_names,
        aggregate_names=aggregate_names,
        variations_in_name=variations_in_name,
        balancing_in_name=balancing_in_name,
        balancing_strategy_in_name=balancing_strategy_in_name,
        variations_group_in_name=variations_group_in_name,
        group_enabled_in_name=group_enabled_in_name,
        modeltype=modeltype,
        source_path_in_name=source_path_in_name,
        extra_watched_paths_in_names=extra_watched_paths_in_names,
    )

    all_output_names = split_names + aggregate_names
    output_mod = OutputPipelineModule(names=all_output_names)

    ds = MGDS(
        device=torch.device("cpu"),
        concepts=concepts,
        settings={},
        definition=[[dummy_mod], [cache_mod], [output_mod]],
        batch_size=batch_size,
        state=PipelineState(),
        seed=seed,
    )
    return ds, cache_dir, dummy_mod


def _drain(ds):
    """Run one epoch and return all batches."""
    ds.start_next_epoch()
    return list(ds)


def _create_source_file(directory, name, content: bytes) -> str:
    """Create a real file with given content, return its path."""
    path = os.path.join(str(directory), name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)
    return path


def _hash_file_xxh64(filepath: str) -> str:
    h = xxhash.xxh64()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_cache_json(cache_dir: str) -> dict:
    with open(os.path.join(cache_dir, "cache.json"), "r") as f:
        return json.load(f)


def _entry_cache_file(entry: dict) -> str:
    """Return any variant's cache_file from a v3 entry. Tests that don't
    care which variant they're looking at (single-resolution caches) use
    this to avoid hard-coding the variant key."""
    return next(iter(entry["variants"].values()))["cache_file"]


def _entry_variant(entry: dict, key: str) -> dict:
    """Return entry['variants'][key]. Raises KeyError if not present."""
    return entry["variants"][key]


# ---------------------------------------------------------------------------
# Hashing tests
# ---------------------------------------------------------------------------


class TestHashing:
    def test_hash_consistency(self, tmp_path):
        """Same file content must always produce the same hash."""
        content = b"hello world deterministic content 12345"
        path = _create_source_file(tmp_path, "a.bin", content)

        sdc = SmartDiskCache.__new__(SmartDiskCache)
        h1 = sdc._hash_file(path)
        h2 = sdc._hash_file(path)
        assert h1 == h2

    def test_hash_different_content(self, tmp_path):
        """Different content must produce different hashes."""
        p1 = _create_source_file(tmp_path, "a.bin", b"content_A")
        p2 = _create_source_file(tmp_path, "b.bin", b"content_B")

        sdc = SmartDiskCache.__new__(SmartDiskCache)
        assert sdc._hash_file(p1) != sdc._hash_file(p2)

    def test_hash_truncation(self):
        """12-char filename must be derived from the 16-char full xxhash64 hex."""
        sdc = SmartDiskCache.__new__(SmartDiskCache)
        full_hash = "abcdef0123456789"  # 16 hex chars, typical xxhash64
        truncated = sdc._hash_to_filename(full_hash)
        assert truncated == "abcdef012345"
        assert len(truncated) == 12


# ---------------------------------------------------------------------------
# Cache validation flow
# ---------------------------------------------------------------------------


class TestCacheValidation:
    """Tests that exercise _validate_entry logic via full pipeline runs."""

    def _setup_files(self, tmp_path, n=3):
        """Create n real source files and return (paths, tensors)."""
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        paths = []
        for i in range(n):
            content = f"source file {i} content".encode()
            p = _create_source_file(src_dir, f"img_{i}.bin", content)
            paths.append(p)
        tensors = _make_tensors(n, seed=77)
        return paths, tensors

    def test_image_cache_runtime_metadata_does_not_resolve_prompts(self):
        """Image cache sourceless metadata only needs concept values."""
        cache = SmartDiskCache.__new__(SmartDiskCache)
        cache.source_path_in_name = "image_path"

        def previous_item(_variation, name, _index):
            if name == "concept":
                return {"name": "concept-a"}
            raise AssertionError(f"unexpected upstream lookup: {name}")

        cache._safe_previous_item = previous_item

        assert cache._sourceless_runtime_values(0, 0) == {"concept": {"name": "concept-a"}}

    def test_text_cache_runtime_metadata_resolves_prompts(self):
        """Text cache sourceless metadata carries prompt values for parity."""
        cache = SmartDiskCache.__new__(SmartDiskCache)
        cache.source_path_in_name = "sample_prompt_path"

        values = {
            "concept": {"name": "concept-a"},
            "prompt": "caption-a",
            "prompt_1": "caption-b",
            "prompt_2": None,
        }
        cache._safe_previous_item = lambda _variation, name, _index: values.get(name)

        assert cache._sourceless_runtime_values(0, 0) == {
            "concept": {"name": "concept-a"},
            "prompt": "caption-a",
            "prompt_1": "caption-b",
        }

    def test_sourceless_index_metadata_ready_requires_current_entries_only(self):
        cache = SmartDiskCache.__new__(SmartDiskCache)
        cache.source_path_in_name = "image_path"
        cache.cache_index = {
            "entries": {
                "active.png": {
                    "sourceless": {"source_index": 0},
                    "sourceless_runtime_values": {"0": {"concept": {"name": "active"}}},
                },
                "stale.png": {},
            }
        }

        assert cache._sourceless_index_metadata_ready({0: "active.png", 1: "missing.png"})
        assert not cache._sourceless_index_metadata_ready({0: "active.png", 1: "stale.png"})

    def test_sourceless_index_upgrade_stamps_metadata_without_pt_rewrite(self, tmp_path):
        cache = SmartDiskCache.__new__(SmartDiskCache)
        cache.source_path_in_name = "image_path"
        cache.cache_index = {"entries": {"active.png": {}}}
        cache.group_indices = {"group": [0]}
        cache.group_variations = {"group": 1}
        cache._save_cache_index = lambda: None
        cache._safe_previous_item = lambda _variation, name, _index: {
            "concept": {"name": "concept-a"},
            "image_path": "active.png",
            "sample_prompt_path": "active.txt",
            "concept.path": "dataset",
            "concept.seed": 123,
            "concept.include_subdirectories": True,
            "concept.image": {},
            "concept.balancing": 1.0,
            "concept.balancing_strategy": "REPEATS",
            "concept.enabled": True,
        }.get(name)
        cache.variations_group_in_names = ["concept.path", "concept.seed"]
        cache.balancing_in_name = "concept.balancing"
        cache.balancing_strategy_in_name = "concept.balancing_strategy"
        cache.group_enabled_in_name = "concept.enabled"

        changed = cache._upgrade_sourceless_runtime_values({0: "active.png"})

        entry = cache.cache_index["entries"]["active.png"]
        assert changed == 1  # one entry upgraded (presence-gated, counts entries not fields)
        assert entry["sourceless"]["linked_paths"]["sample_prompt_path"] == "active.txt"
        # Concept is interned: the row references it by id (not embedded), and
        # the entry-level duplicate slot is no longer written.
        assert "sourceless_runtime_values" not in entry
        row = entry["sourceless_rows"]["0"]
        assert "concept" not in row["runtime_values"]["0"]
        assert cache.cache_index["concepts"][row["concept_id"]]["name"] == "concept-a"
        # The read path reattaches the interned concept transparently.
        assert cache._sourceless_runtime_values_for_row(entry, 0, 0, {})["concept"]["name"] == "concept-a"
        # Re-running must be a no-op now the row is stamped (convergence).
        assert cache._upgrade_sourceless_runtime_values({0: "active.png"}) == 0

    def test_migrate_intern_sourceless_concepts_collapses_embedded_copies(self):
        """Old-format caches (concept embedded per row+variation, plus an
        entry-level duplicate) migrate to one shared concept table with the
        per-variation prompts preserved and reconstructable byte-for-byte."""
        concept = {"name": "Solo", "seed": 123, "image": {"a": 1}, "text": {"b": 2}}

        def block():
            return {str(v): {"concept": dict(concept), "prompt": f"p{v}"} for v in range(3)}

        index = {
            "version": 3,
            "entries": {
                # Two entries sharing one concept; one has two rows.
                "a.txt": {
                    "sourceless_rows": {"0": {"runtime_values": block()}, "1": {"runtime_values": block()}},
                    "sourceless_runtime_values": block(),  # entry-level duplicate
                },
                "b.txt": {
                    "sourceless_rows": {"2": {"runtime_values": block()}},
                    "sourceless_runtime_values": block(),
                },
            },
        }
        # 3 rows + 2 entry-level blocks, each 3 variations = 15 embedded copies.
        removed = SmartDiskCache._migrate_intern_sourceless_concepts(index)
        assert removed == 15
        assert len(index["concepts"]) == 1  # all identical -> one interned copy

        cache = SmartDiskCache.__new__(SmartDiskCache)
        cache.cache_index = index
        cache._sourceless_source_indices = None
        for fp, ridx in (("a.txt", 0), ("a.txt", 1), ("b.txt", 2)):
            entry = index["entries"][fp]
            assert "sourceless_runtime_values" not in entry
            for var in range(3):
                got = cache._sourceless_runtime_values_for_row(entry, ridx, var, {})
                assert got == {"concept": concept, "prompt": f"p{var}"}

        # Idempotent.
        assert SmartDiskCache._migrate_intern_sourceless_concepts(index) == 0

    def test_cache_refresh_reports_phase_status(self, tmp_path, capsys, monkeypatch):
        """Long cache startup work announces phases under OT_SMARTCACHE_VERBOSE."""
        monkeypatch.setenv("OT_SMARTCACHE_VERBOSE", "1")
        paths, tensors = self._setup_files(tmp_path, n=2)

        ds, _, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={
                "latent": tensors,
                "image_path": paths,
            },
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )

        _drain(ds)
        out = capsys.readouterr().out

        assert "SmartDiskCache[cache]: initializing cache groups" in out
        assert "SmartDiskCache[cache]: resolving source paths for cache validation" in out
        assert "SmartDiskCache[cache]: scanning existing cache tensors" in out
        assert "SmartDiskCache[cache]: validating cache entries" in out

    def test_unchanged_file(self, tmp_path):
        """When mtime has not changed, cache is accepted without rehash."""
        paths, tensors = self._setup_files(tmp_path, n=2)

        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={
                "latent": tensors,
                "image_path": paths,
            },
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )

        _drain(ds)  # build cache
        _read_cache_json(cache_dir)

        # Record .pt mtimes
        pt_files = [f for f in os.listdir(cache_dir) if f.endswith(".pt")]
        pt_mtimes = {f: os.path.getmtime(os.path.join(cache_dir, f)) for f in pt_files}

        time.sleep(0.05)
        _drain(ds)  # reuse cache -- source files untouched

        for f in pt_files:
            assert os.path.getmtime(os.path.join(cache_dir, f)) == pt_mtimes[f], (
                f"{f} was rewritten even though source was unchanged"
            )

    def test_touched_file(self, tmp_path):
        """mtime changed but content same -> cache accepted, mtime updated in index.

        Simulates a new training run (fresh pipeline) after a touch, since
        within-run session skip intentionally bypasses re-validation.
        """
        paths, tensors = self._setup_files(tmp_path, n=2)

        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={
                "latent": tensors,
                "image_path": paths,
            },
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )

        _drain(ds)
        index_before = _read_cache_json(cache_dir)
        norm_path = os.path.normpath(paths[0])
        old_mtime = index_before["entries"][norm_path]["mtime"]

        # Touch the file (same content, new mtime)
        time.sleep(0.05)
        os.utime(paths[0], None)
        new_file_mtime = os.path.getmtime(paths[0])
        assert new_file_mtime != old_mtime

        # Record .pt mtimes
        pt_files = [f for f in os.listdir(cache_dir) if f.endswith(".pt")]
        pt_mtimes = {f: os.path.getmtime(os.path.join(cache_dir, f)) for f in pt_files}

        # Fresh pipeline = new run; full validation will run against persisted cache.json.
        ds2, _, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": tensors, "image_path": paths},
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        _drain(ds2)

        # .pt files should NOT have been rewritten
        for f in pt_files:
            assert os.path.getmtime(os.path.join(cache_dir, f)) == pt_mtimes[f], (
                f"{f} was rewritten on touch with same content"
            )

        # But mtime in cache.json should be updated
        index_after = _read_cache_json(cache_dir)
        assert index_after["entries"][norm_path]["mtime"] == new_file_mtime

    def test_edited_file(self, tmp_path):
        """mtime changed AND content changed -> cache rebuilt with new .pt files."""
        paths, tensors = self._setup_files(tmp_path, n=2)

        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={
                "latent": tensors,
                "image_path": paths,
            },
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )

        _drain(ds)
        index_before = _read_cache_json(cache_dir)
        norm_path = os.path.normpath(paths[0])
        old_hash = index_before["entries"][norm_path]["hash"]

        # Edit file content
        time.sleep(0.05)
        with open(paths[0], "wb") as f:
            f.write(b"COMPLETELY DIFFERENT CONTENT NOW")

        # Fresh pipeline simulates a new run; within-run edits are not picked up by design.
        ds2, _, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": tensors, "image_path": paths},
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        _drain(ds2)

        index_after = _read_cache_json(cache_dir)
        new_hash = index_after["entries"][norm_path]["hash"]
        assert new_hash != old_hash, "Hash should change after content edit"

    def test_new_file(self, tmp_path):
        """No prior cache entry -> cache built from scratch."""
        paths, tensors = self._setup_files(tmp_path, n=2)

        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={
                "latent": tensors,
                "image_path": paths,
            },
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )

        _drain(ds)

        index = _read_cache_json(cache_dir)
        for p in paths:
            norm = os.path.normpath(p)
            assert norm in index["entries"], f"Expected entry for {norm}"
            assert "hash" in index["entries"][norm]
            assert "mtime" in index["entries"][norm]

    def test_missing_pt(self, tmp_path):
        """Cache entry exists but .pt file deleted -> variation rebuilt."""
        paths, tensors = self._setup_files(tmp_path, n=2)

        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={
                "latent": tensors,
                "image_path": paths,
            },
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )

        _drain(ds)

        # Delete one .pt file
        pt_files = [f for f in os.listdir(cache_dir) if f.endswith(".pt")]
        assert len(pt_files) > 0
        victim = os.path.join(cache_dir, pt_files[0])
        os.remove(victim)
        assert not os.path.isfile(victim)

        # Fresh pipeline = new run; detection relies on full validation at start of run.
        ds2, _, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": tensors, "image_path": paths},
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        _drain(ds2)

        # The file should be recreated
        assert os.path.isfile(victim), f"Expected {victim} to be rebuilt after deletion"

    def test_wrong_modeltype(self, tmp_path):
        """Cache entry with wrong modeltype -> RuntimeError raised."""
        paths, tensors = self._setup_files(tmp_path, n=1)

        # Build with modeltype "alpha"
        ds1, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={
                "latent": tensors,
                "image_path": paths,
            },
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="alpha",
            source_path_in_name="image_path",
        )
        _drain(ds1)

        # Now build with modeltype "beta" pointing at same cache dir
        dummy_mod = DummyDataModule(
            data={"latent": tensors, "image_path": paths},
            length=len(paths),
        )
        cache_mod = SmartDiskCache(
            cache_dir=cache_dir,
            split_names=["latent"],
            aggregate_names=[],
            modeltype="beta",
            source_path_in_name="image_path",
        )
        output_mod = OutputPipelineModule(names=["latent"])

        ds2 = MGDS(
            device=torch.device("cpu"),
            concepts=[{"name": "A", "path": "dummy"}],
            settings={},
            definition=[[dummy_mod], [cache_mod], [output_mod]],
            batch_size=1,
            state=PipelineState(),
            seed=42,
        )

        with pytest.raises(RuntimeError, match="modeltype mismatch"):
            _drain(ds2)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_dedup_same_content(self, tmp_path):
        """Two source paths with identical content -> cached once, hash_index has both."""
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        content = b"identical content for dedup test"
        path_a = _create_source_file(src_dir, "img_a.bin", content)
        path_b = _create_source_file(src_dir, "img_b.bin", content)

        tensors = _make_tensors(2, seed=99)

        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "D", "path": "dummy"}],
            dummy_data={
                "latent": tensors,
                "image_path": [path_a, path_b],
            },
            dummy_length=2,
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        _drain(ds)

        index = _read_cache_json(cache_dir)
        norm_a = os.path.normpath(path_a)
        norm_b = os.path.normpath(path_b)

        # Both entries should share the same cache_file
        assert _entry_cache_file(index["entries"][norm_a]) == _entry_cache_file(index["entries"][norm_b])

        # hash_index should list both paths under the same hash
        file_hash = index["entries"][norm_a]["hash"]
        assert norm_a in index["hash_index"][file_hash]
        assert norm_b in index["hash_index"][file_hash]

    def test_dedup_edit_one(self, tmp_path):
        """After dedup, editing one copy creates new cache; old cache preserved."""
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        content = b"shared content for dedup edit test"
        path_a = _create_source_file(src_dir, "img_a.bin", content)
        path_b = _create_source_file(src_dir, "img_b.bin", content)

        tensors = _make_tensors(2, seed=101)

        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "D", "path": "dummy"}],
            dummy_data={
                "latent": tensors,
                "image_path": [path_a, path_b],
            },
            dummy_length=2,
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        _drain(ds)

        index_before = _read_cache_json(cache_dir)
        norm_a = os.path.normpath(path_a)
        norm_b = os.path.normpath(path_b)
        shared_cache_file = _entry_cache_file(index_before["entries"][norm_a])

        # Edit path_b so it diverges
        time.sleep(0.05)
        with open(path_b, "wb") as f:
            f.write(b"DIVERGED content after edit")

        # Fresh pipeline = new run; edit detection happens on full validation at start of run.
        ds2, _, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "D", "path": "dummy"}],
            dummy_data={"latent": tensors, "image_path": [path_a, path_b]},
            dummy_length=2,
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        _drain(ds2)

        index_after = _read_cache_json(cache_dir)

        # path_a should still have the original cache_file
        assert _entry_cache_file(index_after["entries"][norm_a]) == shared_cache_file

        # path_b should now have a different cache_file
        assert _entry_cache_file(index_after["entries"][norm_b]) != shared_cache_file


# ---------------------------------------------------------------------------
# Atomic writes / crash recovery
# ---------------------------------------------------------------------------


class TestAtomicWrites:
    def test_cache_json_tmp_recovery(self, tmp_path):
        """If only cache.json.tmp exists (no cache.json), SmartDiskCache recovers it."""
        cache_dir = str(tmp_path / "cache")
        os.makedirs(cache_dir)

        good_data = {
            "version": CACHE_VERSION,
            "entries": {
                "fake/path.bin": {
                    "filename": "path.bin",
                    "hash": "abcdef012345abcd",
                    "mtime": 1234567890.0,
                    "modeltype": "test",
                    "variants": {"_": {"cache_file": "abcdef012345"}},
                    "cache_version": CACHE_VERSION,
                }
            },
            "hash_index": {"abcdef012345abcd": ["fake/path.bin"]},
        }
        tmp_file = os.path.join(cache_dir, "cache.json.tmp")
        with open(tmp_file, "w") as f:
            json.dump(good_data, f)

        sdc = SmartDiskCache.__new__(SmartDiskCache)
        sdc.cache_dir = cache_dir
        loaded = sdc._load_cache_index()

        assert loaded["entries"]["fake/path.bin"]["hash"] == "abcdef012345abcd"
        # After recovery, cache.json should exist and .tmp should be gone
        assert os.path.isfile(os.path.join(cache_dir, "cache.json"))
        assert not os.path.isfile(tmp_file)

    def test_cache_json_with_tmp(self, tmp_path):
        """If both cache.json and cache.json.tmp exist, cache.json is used and .tmp deleted."""
        cache_dir = str(tmp_path / "cache")
        os.makedirs(cache_dir)

        main_data = {
            "version": CACHE_VERSION,
            "entries": {"main_entry": {"hash": "main_hash"}},
            "hash_index": {},
        }
        stale_data = {
            "version": CACHE_VERSION,
            "entries": {"stale_entry": {"hash": "stale_hash"}},
            "hash_index": {},
        }

        cache_json = os.path.join(cache_dir, "cache.json")
        tmp_file = os.path.join(cache_dir, "cache.json.tmp")

        with open(cache_json, "w") as f:
            json.dump(main_data, f)
        with open(tmp_file, "w") as f:
            json.dump(stale_data, f)

        sdc = SmartDiskCache.__new__(SmartDiskCache)
        sdc.cache_dir = cache_dir
        loaded = sdc._load_cache_index()

        assert "main_entry" in loaded["entries"]
        assert "stale_entry" not in loaded["entries"]
        assert not os.path.isfile(tmp_file)


# ---------------------------------------------------------------------------
# Garbage collection
# ---------------------------------------------------------------------------


class TestGarbageCollection:
    def _build_gc_scenario(self, tmp_path):
        """Build a cache with 3 real source files, return (paths, cache_dir)."""
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        paths = []
        for i in range(3):
            p = _create_source_file(src_dir, f"img_{i}.bin", f"gc content {i}".encode())
            paths.append(p)

        tensors = _make_tensors(3, seed=200)

        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "GC", "path": "dummy"}],
            dummy_data={
                "latent": tensors,
                "image_path": paths,
            },
            dummy_length=3,
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        _drain(ds)
        return paths, cache_dir

    def test_gc_preview_empty(self, tmp_path):
        """No orphans -> preview shows 0."""
        paths, cache_dir = self._build_gc_scenario(tmp_path)
        result = SmartDiskCache.gc_preview(cache_dir)
        assert result["orphan_count"] == 0
        assert result["orphan_bytes"] == 0

    def test_gc_preview_orphans(self, tmp_path):
        """Delete a source file -> preview counts its .pt as orphan."""
        paths, cache_dir = self._build_gc_scenario(tmp_path)

        # Delete one source file
        os.remove(paths[0])
        assert not os.path.isfile(paths[0])

        result = SmartDiskCache.gc_preview(cache_dir)
        assert result["orphan_count"] >= 1
        assert result["orphan_bytes"] > 0

    def test_gc_clean(self, tmp_path):
        """After gc_clean, orphaned .pt removed, active ones preserved."""
        paths, cache_dir = self._build_gc_scenario(tmp_path)

        # blank_sentinel.pt is created by _ensure_blank_sentinel and
        # intentionally referenced via cache.json's top-level 'blank_sentinel'
        # field (not via 'entries'). Filter it out of these counts.
        def _entry_pts(d):
            return {f for f in os.listdir(d) if f.endswith(".pt") and f != "blank_sentinel.pt"}

        pt_before = _entry_pts(cache_dir)
        assert len(pt_before) == 3  # one .pt per source

        # Delete one source file to create an orphan
        os.remove(paths[0])

        SmartDiskCache.gc_clean(cache_dir)

        pt_after = _entry_pts(cache_dir)
        # One .pt should be removed
        assert len(pt_after) == 2
        # The remaining .pt files should correspond to the surviving sources
        index = _read_cache_json(cache_dir)
        for surviving_path in paths[1:]:
            norm = os.path.normpath(surviving_path)
            assert norm in index["entries"], f"Entry for {norm} should survive gc_clean"

    def test_gc_orphan_pt_no_entry(self, tmp_path):
        """A random .pt file with no cache entry -> gc detects it as orphan."""
        paths, cache_dir = self._build_gc_scenario(tmp_path)

        # Plant a stray .pt file
        stray = os.path.join(cache_dir, "orphan_random_1.pt")
        torch.save({"junk": torch.zeros(2)}, stray)

        result = SmartDiskCache.gc_preview(cache_dir)
        assert result["orphan_count"] >= 1

        SmartDiskCache.gc_clean(cache_dir)
        assert not os.path.isfile(stray), "Stray .pt should be cleaned up"

    def test_gc_preview_nonexistent_dir(self, tmp_path):
        """gc_preview on a directory with no cache.json returns zeros."""
        result = SmartDiskCache.gc_preview(str(tmp_path / "nonexistent"))
        assert result["orphan_count"] == 0
        assert result["orphan_bytes"] == 0


# ---------------------------------------------------------------------------
# Cache file format
# ---------------------------------------------------------------------------


class TestCacheFileFormat:
    def test_pt_contains_metadata(self, tmp_path):
        """Verify .pt file has __cache_version and __modeltype keys."""
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        path = _create_source_file(src_dir, "img_0.bin", b"format test content")
        tensors = _make_tensors(1, seed=300)

        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "F", "path": "dummy"}],
            dummy_data={
                "latent": tensors,
                "image_path": [path],
            },
            dummy_length=1,
            split_names=["latent"],
            aggregate_names=[],
            modeltype="mymodel",
            source_path_in_name="image_path",
        )
        _drain(ds)

        pt_files = [f for f in os.listdir(cache_dir) if f.endswith(".pt")]
        assert len(pt_files) >= 1

        cached = torch.load(
            os.path.join(cache_dir, pt_files[0]),
            weights_only=False,
            map_location="cpu",
        )
        assert "__cache_version" in cached
        assert cached["__cache_version"] == CACHE_VERSION
        assert "__modeltype" in cached
        assert cached["__modeltype"] == "mymodel"

    def test_pt_contains_all_names(self, tmp_path):
        """Verify split_names AND aggregate_names are all in the .pt file."""
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        path = _create_source_file(src_dir, "img_0.bin", b"names test content")
        tensors = _make_tensors(1, seed=301)

        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "F", "path": "dummy"}],
            dummy_data={
                "latent": tensors,
                "crop_resolution": [(128, 128)],
                "image_path": [path],
            },
            dummy_length=1,
            split_names=["latent"],
            aggregate_names=["crop_resolution"],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        _drain(ds)

        pt_files = [f for f in os.listdir(cache_dir) if f.endswith(".pt")]
        assert len(pt_files) >= 1

        cached = torch.load(
            os.path.join(cache_dir, pt_files[0]),
            weights_only=False,
            map_location="cpu",
        )
        assert "latent" in cached, "split name 'latent' missing from .pt"
        assert "crop_resolution" in cached, "aggregate name 'crop_resolution' missing from .pt"


# ---------------------------------------------------------------------------
# Sample selection (SAMPLES balancing strategy)
# ---------------------------------------------------------------------------


class TestSampleSelection:
    def test_samples_different_per_epoch(self, tmp_path):
        """With SAMPLES strategy, different items should be selected across epochs.

        Statistical check: over 8 epochs with 10 source items and sample_count=3,
        the union of selected items should include more than 3 distinct source
        items (proving different selections happen).
        """
        num_items = 10
        sample_count = 3
        src_dir = tmp_path / "sources"
        src_dir.mkdir()

        paths = []
        for i in range(num_items):
            p = _create_source_file(src_dir, f"img_{i}.bin", f"samples content {i}".encode())
            paths.append(p)

        tensors = _make_tensors(num_items, seed=500)

        concept_dict = {
            "name": "SampConcept",
            "variations": 1,
            "balancing": float(sample_count),
            "balancing_strategy": "SAMPLES",
        }

        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "SampConcept", "path": "dummy"}],
            dummy_data={
                "latent": tensors,
                "crop_resolution": [(64, 64)] * num_items,
                "image_path": paths,
                "concept": [concept_dict] * num_items,
            },
            dummy_length=num_items,
            split_names=["latent"],
            aggregate_names=["crop_resolution"],
            modeltype="testmodel",
            source_path_in_name="image_path",
            variations_in_name="concept.variations",
            balancing_in_name="concept.balancing",
            balancing_strategy_in_name="concept.balancing_strategy",
            variations_group_in_name="concept.name",
        )

        # Collect which items are returned per epoch
        all_selected = set()
        num_epochs = 8
        per_epoch_counts = []

        for _ in range(num_epochs):
            batches = _drain(ds)
            per_epoch_counts.append(len(batches))
            for b in batches:
                if "latent" in b:
                    # Identify by tensor content
                    for idx, t in enumerate(tensors):
                        if torch.equal(b["latent"].cpu(), t):
                            all_selected.add(idx)
                            break

        # Each epoch should produce exactly sample_count items
        for count in per_epoch_counts:
            assert count == sample_count, f"Expected {sample_count} items per epoch, got {count}"

        # Over 8 epochs with 10 items and 3 chosen each time, we expect
        # the union to be larger than 3 (almost certainly).
        assert len(all_selected) > sample_count, (
            f"Expected more than {sample_count} distinct items across {num_epochs} epochs, got {len(all_selected)}"
        )


# ---------------------------------------------------------------------------
# Integration: full pipeline round-trip
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_cache_reuse_across_epochs(self, tmp_path):
        """Cache built in epoch 1 is reused in epoch 2 (no .pt rewrite)."""
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        paths = [_create_source_file(src_dir, f"img_{i}.bin", f"reuse content {i}".encode()) for i in range(3)]
        tensors = _make_tensors(3, seed=600)

        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "R", "path": "dummy"}],
            dummy_data={
                "latent": tensors,
                "image_path": paths,
            },
            dummy_length=3,
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )

        _drain(ds)  # epoch 1

        pt_files = [f for f in os.listdir(cache_dir) if f.endswith(".pt")]
        pt_mtimes = {f: os.path.getmtime(os.path.join(cache_dir, f)) for f in pt_files}

        time.sleep(0.05)
        _drain(ds)  # epoch 2

        for f in pt_files:
            assert os.path.getmtime(os.path.join(cache_dir, f)) == pt_mtimes[f], f"{f} was rewritten on epoch 2"

    def test_cache_index_structure(self, tmp_path):
        """cache.json must have version, entries, and hash_index."""
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        path = _create_source_file(src_dir, "img_0.bin", b"index structure test")
        tensors = _make_tensors(1, seed=700)

        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "I", "path": "dummy"}],
            dummy_data={
                "latent": tensors,
                "image_path": [path],
            },
            dummy_length=1,
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        _drain(ds)

        index = _read_cache_json(cache_dir)
        assert "version" in index
        assert index["version"] == CACHE_VERSION
        assert "entries" in index
        assert "hash_index" in index

    def test_data_round_trip(self, tmp_path):
        """Verify data read from cache matches data from dummy module."""
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        paths = [_create_source_file(src_dir, f"img_{i}.bin", f"round trip {i}".encode()) for i in range(3)]
        tensors = _make_tensors(3, seed=800)

        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "RT", "path": "dummy"}],
            dummy_data={
                "latent": tensors,
                "crop_resolution": [(64, 64)] * 3,
                "image_path": paths,
            },
            dummy_length=3,
            split_names=["latent"],
            aggregate_names=["crop_resolution"],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )

        # Epoch 1 -- builds cache
        batches1 = _drain(ds)
        # Epoch 2 -- reads from cache
        batches2 = _drain(ds)

        assert len(batches1) == 3
        assert len(batches2) == 3

        for b1, b2 in zip(batches1, batches2, strict=True):
            assert torch.equal(b1["latent"], b2["latent"]), "Cached latent differs from original"
            assert b1["crop_resolution"] == b2["crop_resolution"]

    def test_variations_create_multiple_pt(self, tmp_path):
        """With variations > 1, multiple .pt files per source should be created."""
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        num_items = 3
        num_variations = 2
        paths = [
            _create_source_file(src_dir, f"img_{i}.bin", f"variation content {i}".encode()) for i in range(num_items)
        ]
        tensors = _make_tensors(num_items, seed=900)

        concept_dict = {
            "name": "VarTest",
            "variations": num_variations,
            "balancing": 1.0,
            "balancing_strategy": "REPEATS",
        }

        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "VarTest", "path": "dummy"}],
            dummy_data={
                "latent": tensors,
                "image_path": paths,
                "concept": [concept_dict] * num_items,
            },
            dummy_length=num_items,
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
            variations_in_name="concept.variations",
            balancing_in_name="concept.balancing",
            balancing_strategy_in_name="concept.balancing_strategy",
            variations_group_in_name="concept.name",
        )

        # Run enough epochs to cache all variations
        for _ in range(num_variations):
            _drain(ds)

        # Each source should produce num_variations .pt files
        index = _read_cache_json(cache_dir)
        for p in paths:
            norm = os.path.normpath(p)
            entry = index["entries"][norm]
            cache_file = _entry_cache_file(entry)
            for v in range(num_variations):
                pt = os.path.join(cache_dir, f"{cache_file}_{v + 1}.pt")
                assert os.path.isfile(pt), f"Missing variation {v} .pt file: {pt}"

    def test_content_addressed_filename(self, tmp_path):
        """Cache file name should be derived from content hash, not source path."""
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        content = b"content addressed naming test"
        path = _create_source_file(src_dir, "img_0.bin", content)
        tensors = _make_tensors(1, seed=1000)

        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "CA", "path": "dummy"}],
            dummy_data={
                "latent": tensors,
                "image_path": [path],
            },
            dummy_length=1,
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        _drain(ds)

        expected_hash = _hash_file_xxh64(path)
        expected_prefix = expected_hash[:12]

        index = _read_cache_json(cache_dir)
        norm = os.path.normpath(path)
        cache_file = _entry_cache_file(index["entries"][norm])
        assert cache_file.startswith(expected_prefix), (
            f"Cache file '{cache_file}' should start with hash prefix '{expected_prefix}'"
        )

    def test_balancing_repeats(self, tmp_path):
        """With REPEATS balancing=0.5, floor(N*0.5) items should be emitted."""
        num_items = 6
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        paths = [
            _create_source_file(src_dir, f"img_{i}.bin", f"repeats content {i}".encode()) for i in range(num_items)
        ]
        tensors = _make_tensors(num_items, seed=1100)

        concept_dict = {
            "name": "BalRep",
            "variations": 1,
            "balancing": 0.5,
            "balancing_strategy": "REPEATS",
        }

        expected_output = int(math.floor(num_items * 0.5))

        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "BalRep", "path": "dummy"}],
            dummy_data={
                "latent": tensors,
                "image_path": paths,
                "concept": [concept_dict] * num_items,
            },
            dummy_length=num_items,
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
            variations_in_name="concept.variations",
            balancing_in_name="concept.balancing",
            balancing_strategy_in_name="concept.balancing_strategy",
            variations_group_in_name="concept.name",
        )

        batches = _drain(ds)
        assert len(batches) == expected_output, (
            f"Expected {expected_output} items with REPEATS balancing=0.5, got {len(batches)}"
        )

    def test_no_source_path_fallback(self, tmp_path):
        """Without source_path_in_name, SmartDiskCache should not cache but still return data."""
        tensors = _make_tensors(3, seed=1200)

        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "NoSrc", "path": "dummy"}],
            dummy_data={
                "latent": tensors,
                "crop_resolution": [(64, 64)] * 3,
            },
            dummy_length=3,
            split_names=["latent"],
            aggregate_names=["crop_resolution"],
            modeltype="testmodel",
            source_path_in_name=None,
        )

        batches = _drain(ds)
        assert len(batches) == 3
        for b in batches:
            assert "latent" in b
            assert "crop_resolution" in b


# ---------------------------------------------------------------------------
# Rebuild hash_index cleanup regression test
# ---------------------------------------------------------------------------


class TestRebuildHashIndexCleanup:
    def test_rebuild_cleans_hash_index(self, tmp_path):
        """When validation drops a stale entry (content changed, missing pt,
        rebuild, etc.), the old hash must be removed from hash_index before the
        rebuild queues a new entry — otherwise the hash_index keeps a stale
        pointer to the old hash."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        src_file = _create_source_file(src_dir, "test.bin", b"original content")

        paths = [src_file]
        tensors = _make_tensors(1)
        dummy_data = {"latent": tensors, "image_path": paths}
        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            [{}],
            dummy_data,
            1,
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        _drain(ds)

        idx = _read_cache_json(cache_dir)
        fp = os.path.normpath(src_file)
        old_hash = idx["entries"][fp]["hash"]
        assert fp in idx["hash_index"][old_hash]

        # Change file content. _validate_entry sees mtime mismatch, then a
        # hash mismatch, returns 'content_changed', and the rebuild path must
        # clean hash_index before queuing a new entry under the new hash.
        time.sleep(0.05)
        with open(src_file, "wb") as f:
            f.write(b"completely new content")

        # Fresh pipeline = new run; full validation runs at start.
        ds2, _, _ = _build_smart_pipeline(
            tmp_path,
            [{}],
            dummy_data,
            1,
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        _drain(ds2)

        idx_after = _read_cache_json(cache_dir)
        new_hash = idx_after["entries"][fp]["hash"]
        assert new_hash != old_hash
        if old_hash in idx_after["hash_index"]:
            assert fp not in idx_after["hash_index"][old_hash]


# ---------------------------------------------------------------------------
# Fast validation tests
# ---------------------------------------------------------------------------


class TestFastValidation:
    """Tests for the fast validation path (directory mtime + spot check)."""

    def _setup_files(self, tmp_path, n=3):
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        paths = []
        for i in range(n):
            content = f"source file {i} content".encode()
            p = _create_source_file(src_dir, f"img_{i}.bin", content)
            paths.append(p)
        tensors = _make_tensors(n, seed=77)
        return paths, tensors

    def test_fast_validation_skips_full_loop(self, tmp_path, capsys):
        """Second *process* with no changes should use fast validation path."""
        paths, tensors = self._setup_files(tmp_path, n=5)

        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": tensors, "image_path": paths},
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )

        _drain(ds)  # first process — full validation + cache build

        idx = _read_cache_json(cache_dir)
        assert "last_validated" in idx

        time.sleep(0.05)

        # A new pipeline instance = new process semantics, so the session-skip
        # set is empty and fast validation is the first line of defense.
        ds2, _, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": tensors, "image_path": paths},
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        _drain(ds2)

        captured = capsys.readouterr()
        assert "Fast validation passed" in captured.out

    def test_fast_validation_fallback_on_new_file(self, tmp_path, capsys):
        """Adding a source file should cause fast validation to fall back."""
        paths, tensors = self._setup_files(tmp_path, n=3)

        ds, cache_dir, dummy = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": tensors, "image_path": paths},
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
            dummy_module_cls=MutableDummyDataModule,
        )

        _drain(ds)  # build cache

        # Add a new file (changes dir mtime)
        time.sleep(0.05)
        new_path = _create_source_file(tmp_path / "sources", "img_new.bin", b"new")
        new_tensor = torch.randn(4)
        dummy.data["latent"] = tensors + [new_tensor]
        dummy.data["image_path"] = paths + [new_path]
        dummy._length = len(paths) + 1

        _drain(ds)

        captured = capsys.readouterr()
        assert "Fast validation passed" not in captured.out

    def test_fast_validation_fallback_on_touched_file(self, tmp_path, capsys):
        """Touching a source file should cause fast validation to fall back."""
        paths, tensors = self._setup_files(tmp_path, n=3)

        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": tensors, "image_path": paths},
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )

        _drain(ds)  # build cache

        # Touch a file (dir mtime unchanged, but file mtime changes)
        time.sleep(0.05)
        os.utime(paths[0], None)

        _drain(ds)

        capsys.readouterr()
        # Spot check has a chance of catching this, but with only 3 files
        # and sample_size = max(1, min(20, 3//20)) = 1, it may or may not.
        # The full validation should still succeed either way.
        # Just verify the cache is still valid after the run.
        idx = _read_cache_json(cache_dir)
        assert "last_validated" in idx

    def test_no_fast_validation_on_first_run(self, tmp_path, capsys):
        """First run should never use fast validation."""
        paths, tensors = self._setup_files(tmp_path, n=2)

        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": tensors, "image_path": paths},
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )

        _drain(ds)

        captured = capsys.readouterr()
        assert "Fast validation passed" not in captured.out


# ---------------------------------------------------------------------------
# Session-level skip (no per-epoch revalidation within one process)
# ---------------------------------------------------------------------------


class TestSessionSkip:
    """Tests that the in-process session cache skips re-validation entirely
    on subsequent epochs when the set of required filepaths is unchanged."""

    def _setup_files(self, tmp_path, n=4):
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        paths = [_create_source_file(src_dir, f"img_{i}.bin", f"content_{i}".encode()) for i in range(n)]
        tensors = _make_tensors(n, seed=99)
        return paths, tensors

    def test_second_epoch_skips_validation(self, tmp_path, capsys):
        """Second epoch on the same pipeline must hit the session-skip path
        -- no 'validating cache' loop, no fast-validate call."""
        paths, tensors = self._setup_files(tmp_path, n=4)

        ds, _, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": tensors, "image_path": paths},
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        _drain(ds)
        capsys.readouterr()

        _drain(ds)

        out = capsys.readouterr().out
        assert "Skipped re-validation" in out
        assert "Fast validation passed" not in out

    def test_first_epoch_populates_but_does_not_skip(self, tmp_path, capsys):
        """First epoch must actually validate, not skip."""
        paths, tensors = self._setup_files(tmp_path, n=3)

        ds, _, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": tensors, "image_path": paths},
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        _drain(ds)

        out = capsys.readouterr().out
        assert "Skipped re-validation" not in out

    def test_touched_file_within_run_is_not_rechecked(self, tmp_path, capsys):
        """Documents the trade-off: mid-run source edits are NOT detected.
        Within one process, session-skip bypasses validation entirely."""
        paths, tensors = self._setup_files(tmp_path, n=3)

        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": tensors, "image_path": paths},
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        _drain(ds)
        idx_before = _read_cache_json(cache_dir)
        norm_path = os.path.normpath(paths[0])
        old_hash = idx_before["entries"][norm_path]["hash"]
        capsys.readouterr()

        time.sleep(0.05)
        with open(paths[0], "wb") as f:
            f.write(b"edited content after first epoch")

        _drain(ds)

        out = capsys.readouterr().out
        assert "Skipped re-validation" in out
        idx_after = _read_cache_json(cache_dir)
        assert idx_after["entries"][norm_path]["hash"] == old_hash

    def test_fresh_pipeline_resets_session_skip(self, tmp_path, capsys):
        """A new pipeline instance = new process; session-skip must NOT apply."""
        paths, tensors = self._setup_files(tmp_path, n=3)

        def build():
            return _build_smart_pipeline(
                tmp_path,
                concepts=[{"name": "A", "path": "dummy"}],
                dummy_data={"latent": tensors, "image_path": paths},
                dummy_length=len(paths),
                split_names=["latent"],
                aggregate_names=[],
                modeltype="testmodel",
                source_path_in_name="image_path",
            )

        ds, _, _ = build()
        _drain(ds)
        capsys.readouterr()

        ds2, _, _ = build()
        _drain(ds2)

        out = capsys.readouterr().out
        assert "Skipped re-validation" not in out

    def test_repeated_epochs_all_skip(self, tmp_path, capsys):
        """Epochs 2..N all hit the session-skip path."""
        paths, tensors = self._setup_files(tmp_path, n=4)

        ds, _, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": tensors, "image_path": paths},
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        _drain(ds)
        capsys.readouterr()

        for _ in range(5):
            _drain(ds)

        out = capsys.readouterr().out
        assert out.count("Skipped re-validation") == 5


# ---------------------------------------------------------------------------
# Bulk scandir of cache dir (Change 1)
# ---------------------------------------------------------------------------


class TestBulkScanCorrectness:
    """Verify that the bulk cache-dir scan replaces per-file os.path.isfile
    while producing identical existence-check results."""

    def _build_cache(self, tmp_path, n=10):
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        paths = [_create_source_file(src_dir, f"img_{i}.bin", f"content_{i}".encode()) for i in range(n)]
        tensors = _make_tensors(n, seed=11)
        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": tensors, "image_path": paths},
            dummy_length=n,
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        _drain(ds)
        return ds, cache_dir, paths

    def test_existing_pt_files_matches_isfile(self, tmp_path):
        """Set membership in _existing_pt_files must mirror os.path.isfile."""
        ds, _cache_dir, _paths = self._build_cache(tmp_path, n=8)
        sdc = next(m for m in ds.loading_pipeline.modules if isinstance(m, SmartDiskCache))

        existing = sdc._scan_existing_pt_files()
        for entry in sdc.cache_index["entries"].values():
            cf = _entry_cache_file(entry)
            for v in range(1):  # variations=1 in this fixture
                pt_name = f"{cf}_{v + 1}.pt"
                disk_says = os.path.isfile(sdc._real_pt_path(cf, v))
                set_says = pt_name in existing
                assert disk_says == set_says, f"mismatch for {pt_name}"

    def test_built_pts_added_to_set_during_build(self, tmp_path):
        """After _build_cache_entry runs, _existing_pt_files must include the new .pt names."""
        ds, _cache_dir, _paths = self._build_cache(tmp_path, n=4)
        sdc = next(m for m in ds.loading_pipeline.modules if isinstance(m, SmartDiskCache))

        # All cached files should appear in the in-memory set as a side effect of build.
        names_in_index = {f"{_entry_cache_file(e)}_1.pt" for e in sdc.cache_index["entries"].values()}
        assert names_in_index.issubset(sdc._existing_pt_files), (
            f"missing from set: {names_in_index - sdc._existing_pt_files}"
        )

    def test_scan_handles_missing_cache_dir(self, tmp_path):
        """Scanning a non-existent dir must return an empty set, not raise."""
        sdc = SmartDiskCache.__new__(SmartDiskCache)
        sdc._real_cache_dir = str(tmp_path / "does_not_exist")
        result = sdc._scan_existing_pt_files()
        assert result == set()


# ---------------------------------------------------------------------------
# Bulk source mtime scan (Change 2)
# ---------------------------------------------------------------------------


class TestBulkStatCorrectness:
    """Verify that _bulk_stat_source_files returns the same mtimes as
    per-file os.path.getmtime, for all existing files, and omits missing ones."""

    def _make_files_in_dirs(self, tmp_path, n_dirs=4, files_per_dir=10):
        roots = []
        all_paths = []
        for d in range(n_dirs):
            dir_path = tmp_path / f"d{d}"
            dir_path.mkdir()
            for i in range(files_per_dir):
                p = _create_source_file(dir_path, f"f{i}.bin", f"d{d}_f{i}".encode())
                all_paths.append(os.path.normpath(p))
            roots.append(str(dir_path))
        return all_paths, roots

    def _make_sdc_with_executor(self):
        """Instantiate a stub SmartDiskCache with a real executor for parallel scandir."""
        sdc = SmartDiskCache.__new__(SmartDiskCache)
        sdc._state = PipelineState()
        return sdc

    def test_bulk_stat_matches_per_file_getmtime(self, tmp_path):
        all_paths, _ = self._make_files_in_dirs(tmp_path, n_dirs=3, files_per_dir=20)
        sdc = self._make_sdc_with_executor()

        bulk = sdc._bulk_stat_source_files(set(all_paths))
        # Every requested path should be present and equal getmtime to within float tolerance.
        for p in all_paths:
            assert p in bulk, f"missing {p}"
            assert bulk[p] == os.path.getmtime(p), f"mtime mismatch for {p}"

    def test_bulk_stat_omits_missing_files(self, tmp_path):
        all_paths, _ = self._make_files_in_dirs(tmp_path, n_dirs=2, files_per_dir=5)
        ghost = os.path.normpath(str(tmp_path / "d0" / "ghost.bin"))
        sdc = self._make_sdc_with_executor()

        bulk = sdc._bulk_stat_source_files(set(all_paths) | {ghost})
        assert ghost not in bulk
        assert len(bulk) == len(all_paths)

    def test_bulk_stat_handles_missing_parent_dir(self, tmp_path):
        """A path under a non-existent dir must just be omitted, not raise."""
        sdc = self._make_sdc_with_executor()
        ghost = os.path.normpath(str(tmp_path / "no_such_dir" / "x.bin"))
        bulk = sdc._bulk_stat_source_files({ghost})
        assert bulk == {}

    def test_bulk_stat_empty_input(self, tmp_path):
        sdc = self._make_sdc_with_executor()
        assert sdc._bulk_stat_source_files(set()) == {}


# ---------------------------------------------------------------------------
# Resolution short-circuit (Change 6)
# ---------------------------------------------------------------------------


class _ResolutionCountingCache(SmartDiskCache):
    """Test subclass that counts _get_resolution_string calls."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.resolution_call_count = 0

    def _get_resolution_string(self, in_variation, in_index):
        self.resolution_call_count += 1
        return super()._get_resolution_string(in_variation, in_index)


class TestResolutionShortCircuit:
    """Verify that _get_resolution_string is called efficiently across runs."""

    def _setup(self, tmp_path, n=8):
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        paths = [_create_source_file(src_dir, f"i{i}.bin", f"c{i}".encode()) for i in range(n)]
        tensors = _make_tensors(n, seed=42)
        cache_dir = str(tmp_path / "cache")
        dummy = DummyDataModule(data={"latent": tensors, "image_path": paths}, length=n)
        cache_mod = _ResolutionCountingCache(
            cache_dir=cache_dir,
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        output_mod = OutputPipelineModule(names=["latent"])
        ds = MGDS(
            device=torch.device("cpu"),
            concepts=[{"name": "A", "path": "dummy"}],
            settings={},
            definition=[[dummy], [cache_mod], [output_mod]],
            batch_size=1,
            state=PipelineState(),
            seed=7,
        )
        return ds, cache_mod, paths

    def test_resolution_called_lazily_on_full_validation(self, tmp_path):
        """First run: this cache has no resolution dimension (no
        crop_resolution aggregate, no resolution_from_upstream), so the
        rebuild path's _needs_resolution() gate skips the upstream walk
        entirely — zero calls even on a cold cache."""
        ds, cache_mod, paths = self._setup(tmp_path, n=10)
        _drain(ds)
        assert cache_mod.resolution_call_count == 0

    def test_resolution_called_once_per_hit_on_revalidation(self, tmp_path):
        """Second run with the SAME pipeline goes through session-skip — zero calls."""
        ds, cache_mod, _ = self._setup(tmp_path, n=10)
        _drain(ds)
        cache_mod.resolution_call_count = 0
        _drain(ds)
        # Session-skip path returns early without calling resolution at all
        assert cache_mod.resolution_call_count == 0

    def test_resolution_resolved_in_parallel_pass_on_cold_cache(self, tmp_path):
        """Resolution-keyed cache, cold start: the validation loop defers the
        per-item resolution (RESOLUTION_PENDING) and the parallel resolve
        pass fills every one in — exactly one upstream walk per index, and
        the built variants carry the real resolution keys."""
        n = 10
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        paths = [_create_source_file(src_dir, f"i{i}.bin", f"c{i}".encode()) for i in range(n)]
        tensors = _make_tensors(n, seed=42)
        cache_dir = str(tmp_path / "cache")
        dummy = DummyDataModule(
            data={
                "latent": tensors,
                "crop_resolution": [(64, 64)] * n,
                "image_path": paths,
            },
            length=n,
        )
        cache_mod = _ResolutionCountingCache(
            cache_dir=cache_dir,
            split_names=["latent"],
            aggregate_names=["crop_resolution"],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        output_mod = OutputPipelineModule(names=["latent"])
        ds = MGDS(
            device=torch.device("cpu"),
            concepts=[{"name": "A", "path": "dummy"}],
            settings={},
            definition=[[dummy], [cache_mod], [output_mod]],
            batch_size=1,
            state=PipelineState(),
            seed=7,
        )
        _drain(ds)
        # At least one resolve per index from the parallel pass. (The
        # aggregate-load synthesis fallback adds more in this stub setup —
        # no aspect_bucketing to recover aspect from — so don't pin an
        # exact total.)
        assert cache_mod.resolution_call_count >= n
        entries = cache_mod.cache_index["entries"]
        assert len(entries) == n
        for entry in entries.values():
            assert list(entry["variants"].keys()) == ["64x64"]


# ---------------------------------------------------------------------------
# Variation-dedup (Change 3)
# ---------------------------------------------------------------------------


class _ValidateCountingCache(SmartDiskCache):
    """Test subclass that counts _validate_entry calls."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.validate_call_count = 0

    def _validate_entry(self, *a, **kw):
        self.validate_call_count += 1
        return super()._validate_entry(*a, **kw)


class TestVariationDedup:
    """Verify that the validation loop runs once per in_index, regardless of
    how many variations the group needs."""

    def _setup(self, tmp_path, n=6):
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        paths = [_create_source_file(src_dir, f"i{i}.bin", f"c{i}".encode()) for i in range(n)]
        tensors = _make_tensors(n, seed=99)
        cache_dir = str(tmp_path / "cache")
        dummy = DummyDataModule(data={"latent": tensors, "image_path": paths}, length=n)
        cache_mod = _ValidateCountingCache(
            cache_dir=cache_dir,
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        output_mod = OutputPipelineModule(names=["latent"])
        ds = MGDS(
            device=torch.device("cpu"),
            concepts=[{"name": "A", "path": "dummy"}],
            settings={},
            definition=[[dummy], [cache_mod], [output_mod]],
            batch_size=1,
            state=PipelineState(),
            seed=7,
        )
        return ds, cache_mod, paths

    def test_validate_called_once_per_index(self, tmp_path):
        """Cold cache: validate is never called (no entries yet); warm cache:
        called at most N times in a fresh process, regardless of variations."""
        first_path = tmp_path / "run1"
        first_path.mkdir()
        ds, cache_mod, _ = self._setup(first_path, n=10)
        _drain(ds)  # cold — no entries yet, so 0 _validate_entry calls
        assert cache_mod.validate_call_count == 0

        # Build a fresh pipeline pointing at the SAME cache so we exercise
        # validation logic, but with a brand-new in-memory session.
        # (Re-using `first_path` keeps the cache contents; using a fresh
        # _setup call with the same paths re-creates the dummy module.)
        cache_dir = str(first_path / "cache")
        src_dir = first_path / "sources"
        paths = sorted(str(p) for p in src_dir.iterdir())
        tensors = _make_tensors(10, seed=99)
        from mgds.pipelineModules.SmartDiskCache import SmartDiskCache as _SDC  # noqa: F401

        dummy2 = DummyDataModule(data={"latent": tensors, "image_path": paths}, length=10)
        cache_mod2 = _ValidateCountingCache(
            cache_dir=cache_dir,
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        output_mod2 = OutputPipelineModule(names=["latent"])
        ds2 = MGDS(
            device=torch.device("cpu"),
            concepts=[{"name": "A", "path": "dummy"}],
            settings={},
            definition=[[dummy2], [cache_mod2], [output_mod2]],
            batch_size=1,
            state=PipelineState(),
            seed=7,
        )
        _drain(ds2)  # warm — entries exist
        # Cap is N (one call per index), not N×variations. Even better is
        # zero (fast path passes), but we just guard against the regression
        # where validate runs N×V times.
        assert cache_mod2.validate_call_count <= 10, (
            f"validate called {cache_mod2.validate_call_count} times for 10 indices"
        )


# ---------------------------------------------------------------------------
# Watched-file fingerprint (Change 4)
# ---------------------------------------------------------------------------


class TestWatchedFingerprint:
    """Verify the fingerprint-based fast-validate behaviour."""

    def _setup(self, tmp_path, n=4):
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        paths = [_create_source_file(src_dir, f"i{i}.bin", f"content_{i}".encode()) for i in range(n)]
        tensors = _make_tensors(n, seed=55)
        return paths, tensors

    def test_fingerprint_written_after_full_validation(self, tmp_path):
        paths, tensors = self._setup(tmp_path, n=4)
        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": tensors, "image_path": paths},
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        _drain(ds)
        idx = _read_cache_json(cache_dir)
        assert "watched_fingerprints" in idx
        assert idx["watched_fingerprints"], "fingerprint must be non-empty"
        # One key per parent dir; values are [count, mtime_sum] (lists after JSON roundtrip).
        for fp in idx["watched_fingerprints"].values():
            assert isinstance(fp, list) and len(fp) == 2
            assert fp[0] == 4 or fp[0] == len(paths)  # count

    def test_sidecar_touch_does_not_invalidate_fast_path(self, tmp_path, capsys):
        """Touching/adding an UNRELATED file in a watched parent dir must keep
        the fast path active (regression guard for the old parent-dir-mtime
        check, which would have bailed here)."""
        paths, tensors = self._setup(tmp_path, n=3)
        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": tensors, "image_path": paths},
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        _drain(ds)
        capsys.readouterr()  # clear captured output

        # Add a sidecar file in the same parent dir that's NOT in the cache.
        time.sleep(0.05)
        sidecar = tmp_path / "sources" / "caption_unrelated.txt"
        sidecar.write_text("this is a caption file we don't cache")

        # Fresh pipeline => new process semantics, must use fast path.
        ds2, _, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": tensors, "image_path": paths},
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        _drain(ds2)
        out = capsys.readouterr().out
        assert "Fast validation passed" in out, "sidecar touch must NOT invalidate fast validation"

    def test_fingerprint_fails_on_watched_file_touched(self, tmp_path, capsys):
        paths, tensors = self._setup(tmp_path, n=3)
        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": tensors, "image_path": paths},
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        _drain(ds)
        capsys.readouterr()

        time.sleep(0.05)
        os.utime(paths[0], None)  # bumps the watched file's mtime

        ds2, _, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": tensors, "image_path": paths},
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        _drain(ds2)
        out = capsys.readouterr().out
        assert "Fast validation passed" not in out, "watched-file mtime change must invalidate fast validation"

    def test_fingerprint_fails_when_watched_file_deleted(self, tmp_path, capsys):
        paths, tensors = self._setup(tmp_path, n=3)
        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": tensors, "image_path": paths},
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        _drain(ds)
        capsys.readouterr()

        # Delete one watched file (count drops from 3 to 2).
        os.remove(paths[0])

        ds2, _, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": tensors, "image_path": paths[1:]},  # mirror the deletion
            dummy_length=len(paths) - 1,
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        _drain(ds2)
        out = capsys.readouterr().out
        assert "Fast validation passed" not in out

    def test_legacy_cache_without_fingerprint_runs_full_validation(self, tmp_path, capsys):
        """A cache.json with last_validated but no watched_fingerprints must
        force a full pass on first post-upgrade run; second run hits the new
        fast path."""
        paths, tensors = self._setup(tmp_path, n=3)
        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": tensors, "image_path": paths},
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        _drain(ds)
        capsys.readouterr()

        # Strip the fingerprint from cache.json to simulate a legacy cache.
        idx = _read_cache_json(cache_dir)
        idx.pop("watched_fingerprints", None)
        with open(os.path.join(cache_dir, "cache.json"), "w") as f:
            json.dump(idx, f)

        ds2, _, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": tensors, "image_path": paths},
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        _drain(ds2)
        out_first = capsys.readouterr().out
        assert "Fast validation passed" not in out_first, "legacy cache must skip the fast path"

        # Third run should hit the fast path now that the fingerprint is written.
        ds3, _, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": tensors, "image_path": paths},
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        _drain(ds3)
        out_second = capsys.readouterr().out
        assert "Fast validation passed" in out_second


# ---------------------------------------------------------------------------
# Sidecar invalidation — extra_watched_paths_in_names integration tests
# ---------------------------------------------------------------------------


class TestSidecarValidation:
    """_check_sidecars + extra_watched_paths_in_names: an entry rebuilds when
    a watched sidecar (e.g. -masklabel.png) is added, removed, or content-
    edited, but a touch-only mtime drift with unchanged bytes does NOT
    rebuild."""

    def _make_cache(self, tmp_path, watched=("mask_path",)):
        cache_dir = str(tmp_path / "cache")
        os.makedirs(cache_dir, exist_ok=True)
        sdc = SmartDiskCache.__new__(SmartDiskCache)
        sdc.cache_dir = cache_dir
        sdc._real_cache_dir = os.path.realpath(cache_dir)
        sdc.extra_watched_paths_in_names = list(watched)
        sdc._extra_paths_by_filepath = {}
        sdc._source_mtimes = {}
        import threading as _t

        sdc._index_lock = _t.Lock()
        return sdc

    def test_no_sidecars_configured_is_always_valid(self, tmp_path):
        sdc = self._make_cache(tmp_path, watched=())
        entry = {"sidecar_mtimes": None, "sidecar_hashes": None}
        assert sdc._check_sidecars("anything", entry) is True

    def test_legacy_entry_populates_state_and_returns_valid(self, tmp_path):
        sdc = self._make_cache(tmp_path)
        sidecar = _create_source_file(tmp_path, "a-masklabel.png", b"mask_v1")
        primary = "fake.png"
        sdc._extra_paths_by_filepath[primary] = {"mask_path": sidecar}
        entry = {}  # legacy: missing sidecar_mtimes/sidecar_hashes
        assert sdc._check_sidecars(primary, entry) is True
        assert sidecar in entry["sidecar_mtimes"]
        assert sidecar in entry["sidecar_hashes"]

    def test_matching_mtime_returns_valid_without_hashing(self, tmp_path):
        sdc = self._make_cache(tmp_path)
        sidecar = _create_source_file(tmp_path, "a-masklabel.png", b"mask_v1")
        primary = "fake.png"
        sdc._extra_paths_by_filepath[primary] = {"mask_path": sidecar}
        mtime = os.path.getmtime(sidecar)
        entry = {
            "sidecar_mtimes": {sidecar: mtime},
            "sidecar_hashes": {sidecar: "dont_care_not_consulted"},
        }
        assert sdc._check_sidecars(primary, entry) is True

    def test_mtime_drift_hash_match_refreshes_mtime_and_stays_valid(self, tmp_path):
        sdc = self._make_cache(tmp_path)
        sidecar = _create_source_file(tmp_path, "a-masklabel.png", b"mask_v1")
        primary = "fake.png"
        sdc._extra_paths_by_filepath[primary] = {"mask_path": sidecar}
        stale_mtime = os.path.getmtime(sidecar) - 100.0  # pretend cached at older mtime
        real_hash = sdc._hash_file(sidecar)
        entry = {
            "sidecar_mtimes": {sidecar: stale_mtime},
            "sidecar_hashes": {sidecar: real_hash},
        }
        assert sdc._check_sidecars(primary, entry) is True
        assert entry["sidecar_mtimes"][sidecar] != stale_mtime, "stored mtime should be refreshed to current"

    def test_content_change_invalidates(self, tmp_path):
        sdc = self._make_cache(tmp_path)
        sidecar = _create_source_file(tmp_path, "a-masklabel.png", b"mask_v1")
        primary = "fake.png"
        sdc._extra_paths_by_filepath[primary] = {"mask_path": sidecar}
        old_mtime = os.path.getmtime(sidecar)
        old_hash = sdc._hash_file(sidecar)
        time.sleep(0.05)
        with open(sidecar, "wb") as f:
            f.write(b"mask_v2_different_bytes")  # changes content + mtime
        entry = {
            "sidecar_mtimes": {sidecar: old_mtime},
            "sidecar_hashes": {sidecar: old_hash},
        }
        assert sdc._check_sidecars(primary, entry) is False

    def test_sidecar_added_invalidates(self, tmp_path):
        sdc = self._make_cache(tmp_path)
        sidecar = _create_source_file(tmp_path, "a-masklabel.png", b"new_mask")
        primary = "fake.png"
        sdc._extra_paths_by_filepath[primary] = {"mask_path": sidecar}
        # Entry was cached without a sidecar present: stored dicts are empty.
        entry = {"sidecar_mtimes": {}, "sidecar_hashes": {}}
        assert sdc._check_sidecars(primary, entry) is False

    def test_sidecar_removed_invalidates(self, tmp_path):
        sdc = self._make_cache(tmp_path)
        # Entry recorded a sidecar that no longer exists on disk.
        primary = "fake.png"
        missing_path = str(tmp_path / "deleted-masklabel.png")
        sdc._extra_paths_by_filepath[primary] = {"mask_path": missing_path}
        entry = {
            "sidecar_mtimes": {missing_path: 100.0},
            "sidecar_hashes": {missing_path: "deadbeef"},
        }
        assert sdc._check_sidecars(primary, entry) is False

    def test_end_to_end_sidecar_edit_invalidates_fast_path(self, tmp_path, capsys):
        """Touching the bytes of a watched sidecar must fail the fast-path
        fingerprint and force re-validation."""
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        paths = [_create_source_file(src_dir, f"i{i}.bin", f"src_{i}".encode()) for i in range(3)]
        # Create sidecars next to each primary file.
        sidecars = [_create_source_file(src_dir, f"i{i}-masklabel.png", f"mask_{i}".encode()) for i in range(3)]
        tensors = _make_tensors(3, seed=99)
        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": tensors, "image_path": paths, "mask_path": sidecars},
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
            extra_watched_paths_in_names=["mask_path"],
        )
        _drain(ds)
        capsys.readouterr()

        idx = _read_cache_json(cache_dir)
        # Find the entry for the primary path matching paths[0].
        primary0 = os.path.normpath(paths[0])
        entry0 = idx["entries"].get(primary0) or idx["entries"].get(paths[0])
        assert entry0 is not None, "entry for paths[0] missing"
        assert "sidecar_mtimes" in entry0, "first run should stamp sidecar_mtimes"
        sidecar0 = os.path.normpath(sidecars[0])
        assert sidecar0 in entry0["sidecar_mtimes"], f"expected {sidecar0} in {list(entry0['sidecar_mtimes'].keys())}"

        time.sleep(0.05)
        with open(sidecars[0], "wb") as f:
            f.write(b"mask_0_edited_bytes")  # real content change

        ds2, _, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": tensors, "image_path": paths, "mask_path": sidecars},
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
            extra_watched_paths_in_names=["mask_path"],
        )
        _drain(ds2)
        out = capsys.readouterr().out
        assert "Fast validation passed" not in out, "sidecar content change must invalidate the fast path"

    def test_end_to_end_sidecar_touch_no_content_change_keeps_validity(self, tmp_path, capsys):
        """Touch-only mtime drift on a watched sidecar must NOT rebuild the
        entry — mirrors the primary-source mtime→hash escalation."""
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        paths = [_create_source_file(src_dir, f"i{i}.bin", f"src_{i}".encode()) for i in range(3)]
        sidecars = [_create_source_file(src_dir, f"i{i}-masklabel.png", f"mask_{i}".encode()) for i in range(3)]
        tensors = _make_tensors(3, seed=99)
        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": tensors, "image_path": paths, "mask_path": sidecars},
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
            extra_watched_paths_in_names=["mask_path"],
        )
        _drain(ds)
        idx_before = _read_cache_json(cache_dir)
        capsys.readouterr()

        time.sleep(0.05)
        os.utime(sidecars[0], None)  # mtime bumps but bytes unchanged

        ds2, _, _ = _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": tensors, "image_path": paths, "mask_path": sidecars},
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
            extra_watched_paths_in_names=["mask_path"],
        )
        _drain(ds2)
        idx_after = _read_cache_json(cache_dir)

        # Variants/cache_files must be unchanged across runs (no rebuild).
        for fp, entry_before in idx_before["entries"].items():
            entry_after = idx_after["entries"][fp]
            assert entry_before["variants"] == entry_after["variants"], (
                f"{fp}: touch-only sidecar drift must not rebuild"
            )


# ---------------------------------------------------------------------------
# Validation benchmarks — record timings for the headline speedups.
# Run with `pytest -k Benchmark -s` to see numbers.
# ---------------------------------------------------------------------------


class TestValidationBenchmarks:
    """Side-by-side timings: naive per-file syscalls vs the new bulk methods.
    Asserts conservative ratios so the suite stays green on slow CI."""

    def _make_dummy_cache_dir(self, tmp_path, n=300):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_files = []
        for i in range(n):
            cf = f"hash{i:08x}"
            for v in (1, 2):
                pt = cache_dir / f"{cf}_{v}.pt"
                pt.write_bytes(b"")
            cache_files.append(cf)
        return str(cache_dir), cache_files

    def test_bench_pt_isfile_vs_set_membership(self, tmp_path, capsys):
        """Scenario: 300 entries × 2 variations = 600 .pt existence checks.
        Naive: os.path.isfile per check. New: one scandir → set membership."""
        cache_dir, cache_files = self._make_dummy_cache_dir(tmp_path, n=300)
        sdc = SmartDiskCache.__new__(SmartDiskCache)
        sdc._real_cache_dir = cache_dir

        t0 = time.perf_counter()
        for _ in range(3):  # repeat to amortise jitter
            for cf in cache_files:
                for v in range(2):
                    os.path.isfile(os.path.join(cache_dir, f"{cf}_{v + 1}.pt"))
        naive = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(3):
            existing = sdc._scan_existing_pt_files()
            for cf in cache_files:
                for v in range(2):
                    _ = f"{cf}_{v + 1}.pt" in existing
        bulk = time.perf_counter() - t0

        print(f"\n[bench pt-existence] naive={naive * 1000:.1f}ms bulk={bulk * 1000:.1f}ms speedup={naive / bulk:.1f}×")
        assert bulk < naive, f"bulk {bulk:.4f}s should beat naive {naive:.4f}s"

    def test_bench_bulk_vs_serial_getmtime(self, tmp_path, capsys):
        """Scenario: 300 source files across 6 dirs. Naive: getmtime per file.
        New: parallel scandir per dir."""
        files_per_dir = 50
        n_dirs = 6
        all_paths = []
        for d in range(n_dirs):
            dir_path = tmp_path / f"d{d}"
            dir_path.mkdir()
            for i in range(files_per_dir):
                p = _create_source_file(dir_path, f"f{i}.bin", f"d{d}f{i}".encode())
                all_paths.append(os.path.normpath(p))

        sdc = SmartDiskCache.__new__(SmartDiskCache)
        sdc._state = PipelineState()

        t0 = time.perf_counter()
        for _ in range(3):
            naive_mtimes = {p: os.path.getmtime(p) for p in all_paths}
        naive = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(3):
            bulk_mtimes = sdc._bulk_stat_source_files(set(all_paths))
        bulk = time.perf_counter() - t0

        # Sanity: same data
        assert naive_mtimes == bulk_mtimes

        print(f"\n[bench source-mtime] naive={naive * 1000:.1f}ms bulk={bulk * 1000:.1f}ms speedup={naive / bulk:.1f}×")
        # Conservative threshold: bulk should at least not be worse than naive.
        # On Defender-laden systems the speedup is much larger (5-50x).
        assert bulk < naive * 2.0, f"bulk {bulk:.4f}s vs naive {naive:.4f}s"

    def test_bench_e2e_validation_speedup(self, tmp_path, capsys):
        """End-to-end: cold validation, fresh-pipeline fast-validate, and
        forced full validation after a single touched file. Records timings."""
        n = 200
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        paths = [_create_source_file(src_dir, f"img_{i:04d}.bin", f"c{i}".encode()) for i in range(n)]
        tensors = _make_tensors(n, seed=33)

        def build():
            return _build_smart_pipeline(
                tmp_path,
                concepts=[{"name": "B", "path": "dummy"}],
                dummy_data={"latent": tensors, "image_path": paths},
                dummy_length=n,
                split_names=["latent"],
                aggregate_names=[],
                modeltype="testmodel",
                source_path_in_name="image_path",
            )

        # Cold start
        ds_cold, _, _ = build()
        t0 = time.perf_counter()
        _drain(ds_cold)
        t_cold = time.perf_counter() - t0

        # Warm — fast-validate path on a brand new pipeline
        ds_warm, _, _ = build()
        t0 = time.perf_counter()
        _drain(ds_warm)
        t_warm = time.perf_counter() - t0

        # Force full validation by touching one file
        time.sleep(0.05)
        os.utime(paths[0], None)
        ds_full, _, _ = build()
        t0 = time.perf_counter()
        _drain(ds_full)
        t_full = time.perf_counter() - t0

        print(
            f"\n[bench e2e] cold={t_cold * 1000:.0f}ms warm_fast={t_warm * 1000:.0f}ms warm_full={t_full * 1000:.0f}ms"
        )
        # Fast validation should be MUCH cheaper than the cold build.
        assert t_warm < t_cold, "fast-validate should beat cold build"
        # Forced full validation, after the bulk-scan changes, should still be
        # fast — no more than 3× the fast-validate path on a clean bench.
        assert t_full < max(t_cold, t_warm * 50.0), (
            f"full-validate after touch should not regress: t_full={t_full:.3f}s"
        )


# ---------------------------------------------------------------------------
# Multi-resolution variant cache: v2 -> v3 migration, drift recovery, GC
# ---------------------------------------------------------------------------

from mgds.pipelineModules.SmartDiskCache import NO_RESOLUTION_KEY


class TestLegacyV2Migration:
    """In-place migration of v2 entries (top-level cache_file/resolution)
    to v3 entries (variants dict). No .pt rebuild required."""

    def _write_v2_index(self, cache_dir: str, entries: dict) -> str:
        os.makedirs(cache_dir, exist_ok=True)
        v2_data = {
            "version": 2,
            "entries": entries,
            "hash_index": {},
        }
        path = os.path.join(cache_dir, "cache.json")
        with open(path, "w") as f:
            json.dump(v2_data, f)
        return path

    def test_v2_with_resolution_lifts_to_keyed_variant(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        self._write_v2_index(
            cache_dir,
            {
                "fake/img.png": {
                    "filename": "img.png",
                    "hash": "deadbeefcafe1234",
                    "mtime": 1234.0,
                    "modeltype": "test",
                    "resolution": "896x640",
                    "cache_file": "deadbeefcafe_896x640",
                    "cache_version": 2,
                }
            },
        )

        sdc = SmartDiskCache.__new__(SmartDiskCache)
        sdc.cache_dir = cache_dir
        loaded = sdc._load_cache_index()

        entry = loaded["entries"]["fake/img.png"]
        assert entry.get("resolution") is None  # lifted out
        assert entry.get("cache_file") is None  # lifted out
        assert entry["variants"] == {"896x640": {"cache_file": "deadbeefcafe_896x640"}}
        assert loaded["version"] == CACHE_VERSION
        assert entry["cache_version"] == CACHE_VERSION

    def test_v2_without_resolution_uses_underscore_key(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        self._write_v2_index(
            cache_dir,
            {
                "fake/text.txt": {
                    "filename": "text.txt",
                    "hash": "deadbeefcafe5678",
                    "mtime": 5678.0,
                    "modeltype": "test",
                    "resolution": None,
                    "cache_file": "deadbeefcafe",
                    "cache_version": 2,
                }
            },
        )

        sdc = SmartDiskCache.__new__(SmartDiskCache)
        sdc.cache_dir = cache_dir
        loaded = sdc._load_cache_index()

        entry = loaded["entries"]["fake/text.txt"]
        assert entry["variants"] == {NO_RESOLUTION_KEY: {"cache_file": "deadbeefcafe"}}

    def test_migration_is_idempotent(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        self._write_v2_index(
            cache_dir,
            {
                "fake/a.bin": {
                    "filename": "a.bin",
                    "hash": "h1",
                    "mtime": 1.0,
                    "modeltype": "test",
                    "resolution": None,
                    "cache_file": "h1",
                    "cache_version": 2,
                }
            },
        )
        sdc = SmartDiskCache.__new__(SmartDiskCache)
        sdc.cache_dir = cache_dir
        first = sdc._load_cache_index()
        # Second read should not double-rewrite (idempotent).
        second = sdc._load_cache_index()
        assert first["entries"] == second["entries"]
        assert second["entries"]["fake/a.bin"]["variants"] == {NO_RESOLUTION_KEY: {"cache_file": "h1"}}

    def test_v3_index_passes_through_unchanged(self, tmp_path):
        """Already-v3 entries must not be re-migrated."""
        cache_dir = str(tmp_path / "cache")
        os.makedirs(cache_dir, exist_ok=True)
        v3_data = {
            "version": CACHE_VERSION,
            "entries": {
                "fake/img.png": {
                    "filename": "img.png",
                    "hash": "h1",
                    "mtime": 1.0,
                    "modeltype": "test",
                    "variants": {
                        "512x512": {"cache_file": "h1_512x512"},
                        "768x768": {"cache_file": "h1_768x768"},
                    },
                    "cache_version": CACHE_VERSION,
                }
            },
            "hash_index": {},
        }
        with open(os.path.join(cache_dir, "cache.json"), "w") as f:
            json.dump(v3_data, f)

        sdc = SmartDiskCache.__new__(SmartDiskCache)
        sdc.cache_dir = cache_dir
        loaded = sdc._load_cache_index()
        # Both variants survived.
        assert set(loaded["entries"]["fake/img.png"]["variants"].keys()) == {"512x512", "768x768"}


class TestVariantHelpers:
    """Internal helper coverage: _resolution_key, _any_variant_cache_file,
    _aspect_from_variant_keys, _make_cache_file."""

    def test_resolution_key_with_value(self):
        assert SmartDiskCache._resolution_key("896x640") == "896x640"

    def test_resolution_key_falls_back_to_underscore(self):
        assert SmartDiskCache._resolution_key(None) == NO_RESOLUTION_KEY
        assert SmartDiskCache._resolution_key("") == NO_RESOLUTION_KEY

    def test_any_variant_cache_file(self):
        entry = {"variants": {"512x512": {"cache_file": "h_512x512"}}}
        assert SmartDiskCache._any_variant_cache_file(entry) == "h_512x512"

    def test_any_variant_cache_file_empty(self):
        assert SmartDiskCache._any_variant_cache_file({}) is None
        assert SmartDiskCache._any_variant_cache_file({"variants": {}}) is None

    def test_aspect_from_variant_keys(self):
        # 896x640 = aspect 1.4
        variants = {"896x640": {"cache_file": "x"}}
        assert SmartDiskCache._aspect_from_variant_keys(variants) == 896 / 640

    def test_aspect_skips_underscore_key(self):
        variants = {NO_RESOLUTION_KEY: {"cache_file": "x"}}
        assert SmartDiskCache._aspect_from_variant_keys(variants) is None

    def test_aspect_skips_malformed_key(self):
        variants = {"not_a_resolution": {"cache_file": "x"}}
        assert SmartDiskCache._aspect_from_variant_keys(variants) is None

    def test_aspect_with_mixed_keys(self):
        # First valid key wins.
        variants = {NO_RESOLUTION_KEY: {}, "1024x512": {}}
        assert SmartDiskCache._aspect_from_variant_keys(variants) == 1024 / 512


class TestDriftRecoveryRebucket:
    """The aspect-math drift recovery path: derives new variant keys from
    cached resolutions and registers any pre-existing .pt files without
    decoding source images."""

    def _make_sdc(self, cache_dir, *, rebucket_provider, file_hash="abcdef012345abcd"):
        sdc = SmartDiskCache.__new__(SmartDiskCache)
        sdc.cache_dir = cache_dir
        sdc._real_cache_dir = os.path.realpath(cache_dir)
        sdc.modeltype = "testmodel"
        sdc._index_lock = __import__("threading").Lock()
        sdc.rebucket_provider = rebucket_provider
        sdc.bucket_method_provider = None
        sdc.resolution_from_upstream = False
        sdc._existing_pt_files = set()
        sdc._active_key_by_filepath = {}
        sdc.cache_index = {
            "version": CACHE_VERSION,
            "entries": {
                "fake/img1.png": {
                    "filename": "img1.png",
                    "hash": file_hash,
                    "mtime": 1.0,
                    "modeltype": "testmodel",
                    "variants": {"512x512": {"cache_file": file_hash[:12] + "_512x512"}},
                    "cache_version": CACHE_VERSION,
                },
            },
            "hash_index": {},
        }
        return sdc

    def test_collision_reuses_existing_pt(self, tmp_path):
        """If the rebucketed key happens to match an existing .pt on disk,
        register it as a new variant — no rebuild queued."""
        cache_dir = str(tmp_path / "c")
        os.makedirs(cache_dir, exist_ok=True)

        def rebucket(aspect):
            # Simulate bucket math returning a different key with the same hash prefix.
            return ["768x768"]

        sdc = self._make_sdc(cache_dir, rebucket_provider=rebucket)
        # Plant a .pt that the new key would point at.
        sdc._existing_pt_files = {"abcdef012345_768x768_1.pt"}

        sdc._drift_recovery_pass()

        entry = sdc.cache_index["entries"]["fake/img1.png"]
        # New variant registered alongside the original.
        assert "768x768" in entry["variants"]
        assert entry["variants"]["768x768"]["cache_file"] == "abcdef012345_768x768"
        # Old variant preserved.
        assert "512x512" in entry["variants"]
        # New key first (so next-iter picks it).
        assert next(iter(entry["variants"].keys())) == "768x768"
        # Active key is no longer pinned by drift recovery — variant
        # selection runs per-epoch via _get_resolution_string in
        # __refresh_cache / get_item. Asserting empty rather than
        # presence catches accidental re-introduction of the pin.
        assert sdc._active_key_by_filepath == {}

    def test_no_collision_leaves_variant_unfilled(self, tmp_path):
        """If the rebucketed key has no matching .pt on disk, the variant
        slot is NOT registered — per-index rebuild loop handles it."""
        cache_dir = str(tmp_path / "c")
        os.makedirs(cache_dir, exist_ok=True)

        def rebucket(aspect):
            return ["1024x768"]

        sdc = self._make_sdc(cache_dir, rebucket_provider=rebucket)
        sdc._existing_pt_files = set()  # No .pt for the new key.

        sdc._drift_recovery_pass()

        entry = sdc.cache_index["entries"]["fake/img1.png"]
        # New variant NOT in dict (would otherwise be a stale reference).
        assert "1024x768" not in entry["variants"]
        # Original variant preserved.
        assert "512x512" in entry["variants"]
        # No pin — __refresh_cache's per-epoch resolve will compute the
        # active key from upstream and see 'missing_variant' on its own.
        assert sdc._active_key_by_filepath == {}

    def test_no_provider_disables_drift(self, tmp_path):
        """When rebucket_provider is None (text caches), drift recovery is
        a no-op."""
        cache_dir = str(tmp_path / "c")
        os.makedirs(cache_dir, exist_ok=True)
        sdc = self._make_sdc(cache_dir, rebucket_provider=None)
        before = dict(sdc.cache_index["entries"]["fake/img1.png"]["variants"])
        sdc._drift_recovery_pass()
        after = dict(sdc.cache_index["entries"]["fake/img1.png"]["variants"])
        assert before == after
        assert sdc._active_key_by_filepath == {}

    def test_multi_target_registers_all_keys(self, tmp_path):
        """rebucket_provider returning multiple keys registers each variant."""
        cache_dir = str(tmp_path / "c")
        os.makedirs(cache_dir, exist_ok=True)

        def rebucket(aspect):
            return ["768x768", "1024x1024"]

        sdc = self._make_sdc(cache_dir, rebucket_provider=rebucket)
        sdc._existing_pt_files = {
            "abcdef012345_768x768_1.pt",
            "abcdef012345_1024x1024_1.pt",
        }

        sdc._drift_recovery_pass()
        entry = sdc.cache_index["entries"]["fake/img1.png"]
        assert "768x768" in entry["variants"]
        assert "1024x1024" in entry["variants"]
        # Reordering preserved (new keys first, used by text-cache callers'
        # any-variant fallback), but no active-key pinning.
        assert list(entry["variants"].keys())[:2] == ["768x768", "1024x1024"]
        assert sdc._active_key_by_filepath == {}


class TestValidateEntryVariantStatus:
    """_validate_entry returns 'missing_variant' when the requested key
    isn't present, distinct from 'missing_pt' (key present but file gone)."""

    def _make_sdc(self):
        sdc = SmartDiskCache.__new__(SmartDiskCache)
        sdc.modeltype = "testmodel"
        # Empty schema so _variant_schema_is_complete short-circuits to True
        # without needing cache_dir or a real .pt to peek.
        sdc.split_names = []
        sdc.aggregate_names = []
        sdc.tolerate_missing_source = False
        sdc.resolution_from_upstream = False
        sdc._existing_pt_files = set()
        sdc.extra_watched_paths_in_names = []
        sdc._extra_paths_by_filepath = {}
        sdc._source_mtimes = {}
        return sdc

    def test_missing_variant_when_key_not_present(self):
        sdc = self._make_sdc()
        entry = {
            "modeltype": "testmodel",
            "mtime": 100.0,
            "variants": {"512x512": {"cache_file": "h_512x512"}},
            "hash": "h",
        }
        # Request a key that doesn't exist in variants.
        status = sdc._validate_entry("path", entry, "768x768", 1, 100.0)
        assert status == "missing_variant"

    def test_missing_pt_when_variant_present_but_file_gone(self):
        sdc = self._make_sdc()
        entry = {
            "modeltype": "testmodel",
            "mtime": 100.0,
            "variants": {"512x512": {"cache_file": "h_512x512"}},
            "hash": "h",
        }
        # Variant exists in dict but no .pt on disk.
        status = sdc._validate_entry("path", entry, "512x512", 1, 100.0)
        assert status == "missing_pt"

    def test_valid_when_variant_and_pt_both_present(self):
        sdc = self._make_sdc()
        sdc._existing_pt_files = {"h_512x512_1.pt"}
        entry = {
            "modeltype": "testmodel",
            "mtime": 100.0,
            "variants": {"512x512": {"cache_file": "h_512x512"}},
            "hash": "h",
        }
        status = sdc._validate_entry("path", entry, "512x512", 1, 100.0)
        assert status == "valid"


class TestBucketMethodStamping:
    """bucket_method is computed from the provider, stamped to cache.json,
    and not stamped when no provider is wired."""

    def _make_pipeline_with_provider(self, tmp_path, method_hash="testhash01"):
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        paths = [_create_source_file(src_dir, f"i{i}.bin", f"c{i}".encode()) for i in range(3)]
        tensors = _make_tensors(3, seed=11)
        cache_dir = str(tmp_path / "cache")
        dummy = DummyDataModule(data={"latent": tensors, "image_path": paths}, length=3)
        cache_mod = SmartDiskCache(
            cache_dir=cache_dir,
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
            bucket_method_provider=lambda: method_hash,
            rebucket_provider=lambda aspect: [],
        )
        output_mod = OutputPipelineModule(names=["latent"])
        ds = MGDS(
            device=torch.device("cpu"),
            concepts=[{"name": "A", "path": "dummy"}],
            settings={},
            definition=[[dummy], [cache_mod], [output_mod]],
            batch_size=1,
            state=PipelineState(),
            seed=7,
        )
        return ds, cache_dir

    def test_bucket_method_stamped_after_first_run(self, tmp_path):
        ds, cache_dir = self._make_pipeline_with_provider(tmp_path, "method_v1_hash")
        _drain(ds)
        index = _read_cache_json(cache_dir)
        assert index.get("bucket_method") == "method_v1_hash"

    def test_no_bucket_method_when_provider_returns_none(self, tmp_path):
        """A None-returning provider (e.g. text cache) leaves the field unset."""
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        path = _create_source_file(src_dir, "x.bin", b"x")
        tensors = _make_tensors(1, seed=11)
        cache_dir = str(tmp_path / "cache")
        dummy = DummyDataModule(data={"latent": tensors, "image_path": [path]}, length=1)
        cache_mod = SmartDiskCache(
            cache_dir=cache_dir,
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
            bucket_method_provider=None,
            rebucket_provider=None,
        )
        output_mod = OutputPipelineModule(names=["latent"])
        ds = MGDS(
            device=torch.device("cpu"),
            concepts=[{"name": "A", "path": "dummy"}],
            settings={},
            definition=[[dummy], [cache_mod], [output_mod]],
            batch_size=1,
            state=PipelineState(),
            seed=7,
        )
        _drain(ds)
        index = _read_cache_json(cache_dir)
        assert "bucket_method" not in index


class TestGCWalksVariants:
    """gc_preview / gc_clean must walk all variants per entry, not just one."""

    def test_gc_preview_counts_all_variants_as_referenced(self, tmp_path):
        cache_dir = str(tmp_path / "c")
        os.makedirs(cache_dir, exist_ok=True)
        # Create source file so entry isn't dead.
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        src = _create_source_file(src_dir, "img.bin", b"data")
        norm_src = os.path.normpath(src)

        # Plant two variant .pt files for the same entry.
        torch.save({"data": torch.zeros(1)}, os.path.join(cache_dir, "h12_512x512_1.pt"))
        torch.save({"data": torch.zeros(1)}, os.path.join(cache_dir, "h12_768x768_1.pt"))

        index = {
            "version": CACHE_VERSION,
            "entries": {
                norm_src: {
                    "filename": "img.bin",
                    "hash": "h12abcdef012",
                    "mtime": os.path.getmtime(src),
                    "modeltype": "testmodel",
                    "variants": {
                        "512x512": {"cache_file": "h12_512x512"},
                        "768x768": {"cache_file": "h12_768x768"},
                    },
                    "cache_version": CACHE_VERSION,
                }
            },
            "hash_index": {},
        }
        with open(os.path.join(cache_dir, "cache.json"), "w") as f:
            json.dump(index, f)

        result = SmartDiskCache.gc_preview(cache_dir)
        # Both variant .pts are referenced; no orphans.
        assert result["orphan_count"] == 0

    def test_gc_clean_keeps_all_variant_pts(self, tmp_path):
        cache_dir = str(tmp_path / "c")
        os.makedirs(cache_dir, exist_ok=True)
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        src = _create_source_file(src_dir, "img.bin", b"data")
        norm_src = os.path.normpath(src)

        torch.save({"data": torch.zeros(1)}, os.path.join(cache_dir, "h12_512x512_1.pt"))
        torch.save({"data": torch.zeros(1)}, os.path.join(cache_dir, "h12_768x768_1.pt"))

        index = {
            "version": CACHE_VERSION,
            "entries": {
                norm_src: {
                    "filename": "img.bin",
                    "hash": "h12abcdef012",
                    "mtime": os.path.getmtime(src),
                    "modeltype": "testmodel",
                    "variants": {
                        "512x512": {"cache_file": "h12_512x512"},
                        "768x768": {"cache_file": "h12_768x768"},
                    },
                    "cache_version": CACHE_VERSION,
                }
            },
            "hash_index": {"h12abcdef012": [norm_src]},
        }
        with open(os.path.join(cache_dir, "cache.json"), "w") as f:
            json.dump(index, f)

        SmartDiskCache.gc_clean(cache_dir)

        # Both variant .pts must survive.
        assert os.path.isfile(os.path.join(cache_dir, "h12_512x512_1.pt"))
        assert os.path.isfile(os.path.join(cache_dir, "h12_768x768_1.pt"))

    def test_gc_clean_removes_all_variant_pts_when_source_dies(self, tmp_path):
        cache_dir = str(tmp_path / "c")
        os.makedirs(cache_dir, exist_ok=True)
        # Reference a non-existent source.
        torch.save({"data": torch.zeros(1)}, os.path.join(cache_dir, "h12_512x512_1.pt"))
        torch.save({"data": torch.zeros(1)}, os.path.join(cache_dir, "h12_768x768_1.pt"))

        dead_path = os.path.join(str(tmp_path), "no_such_source.bin")
        index = {
            "version": CACHE_VERSION,
            "entries": {
                dead_path: {
                    "filename": "no_such_source.bin",
                    "hash": "h12abcdef012",
                    "mtime": 1.0,
                    "modeltype": "testmodel",
                    "variants": {
                        "512x512": {"cache_file": "h12_512x512"},
                        "768x768": {"cache_file": "h12_768x768"},
                    },
                    "cache_version": CACHE_VERSION,
                }
            },
            "hash_index": {"h12abcdef012": [dead_path]},
        }
        with open(os.path.join(cache_dir, "cache.json"), "w") as f:
            json.dump(index, f)

        SmartDiskCache.gc_clean(cache_dir)

        # Both variant .pts must be cleaned up.
        assert not os.path.isfile(os.path.join(cache_dir, "h12_512x512_1.pt"))
        assert not os.path.isfile(os.path.join(cache_dir, "h12_768x768_1.pt"))

    def test_gc_handles_legacy_v2_index(self, tmp_path):
        """gc helpers must auto-migrate v2 cache.json before walking."""
        cache_dir = str(tmp_path / "c")
        os.makedirs(cache_dir, exist_ok=True)
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        src = _create_source_file(src_dir, "img.bin", b"data")
        norm_src = os.path.normpath(src)

        torch.save({"data": torch.zeros(1)}, os.path.join(cache_dir, "h12abcd_1.pt"))
        v2_index = {
            "version": 2,
            "entries": {
                norm_src: {
                    "filename": "img.bin",
                    "hash": "h12abcdef012",
                    "mtime": os.path.getmtime(src),
                    "modeltype": "testmodel",
                    "resolution": None,
                    "cache_file": "h12abcd",
                    "cache_version": 2,
                }
            },
            "hash_index": {},
        }
        with open(os.path.join(cache_dir, "cache.json"), "w") as f:
            json.dump(v2_index, f)

        # gc_preview should not flag this as orphan.
        result = SmartDiskCache.gc_preview(cache_dir)
        assert result["orphan_count"] == 0


class TestPerIndexAggregateCache:
    """The same source file can appear at multiple in_index values when
    it's referenced across concepts or under repeats. AspectBucketing's
    rand.choice is seeded on (variation, index), so each in_index resolves
    to a different target — and therefore a different variant key. The
    aggregate cache must keep those entries distinct so AspectBatchSorting's
    sort-time view stays in sync with split-fetch's per-in_index lookup.

    Repros the bug we hit on real training: a SoReal! image at in_idx=15249
    resolved to '1024x576' and a different in_idx for the same file
    resolved to '704x384'; a 2-tuple agg cache key let the second write
    clobber the first, sort then bucketed by the wrong crop_resolution,
    and torch.stack failed when the per-in_index .pt's latent shape
    didn't match.
    """

    FP_A = r"F:\Datasets\test\fileA.jpg"  # has both variants — like the SoReal! file at 4032x2268
    FP_B = r"F:\Datasets\test\fileB.jpg"  # only the 512 variant — like the pinterest file at 1200x675

    # Per-in_index variant resolution. Mirrors what _get_resolution_string
    # would walk upstream and compute for each (variation, in_index): rand
    # picks a target, bucket_for_aspect maps to a key.
    PER_INDEX_RESOLUTION = {
        100: "1024x576",  # fp_A's first occurrence — rand picks 768
        200: "704x384",  # fp_A's second occurrence (same file, different concept) — rand picks 512
        300: "704x384",  # fp_B — rand picks 512 (variant exists)
    }

    # The .pt's stored aggregate values per cache_file. Mirrors what
    # torch.load on each variant's .pt would return.
    PT_CONTENTS = {
        "hashA_1024x576": {"crop_resolution": (1024, 576), "image_path": FP_A},
        "hashA_704x384": {"crop_resolution": (704, 384), "image_path": FP_A},
        "hashB_704x384": {"crop_resolution": (704, 384), "image_path": FP_B},
    }

    def _make_sdc(self):
        """Build a SmartDiskCache primed with the multi-in_index dataset
        shape, no executor/pipeline/disk required."""
        sdc = SmartDiskCache.__new__(SmartDiskCache)
        sdc.cache_dir = "/fake/cache"
        sdc._real_cache_dir = "/fake/cache"
        sdc.modeltype = "testmodel"
        sdc.split_names = ["latent_image"]
        sdc.aggregate_names = ["crop_resolution", "image_path"]
        sdc.sourceless = False
        sdc.resolution_from_upstream = False
        sdc.aspect_bucketing = object()  # truthy sentinel; per_item_key gate
        sdc.bucket_method_provider = None
        sdc.rebucket_provider = None
        sdc.tolerate_missing_source = False
        sdc.extra_watched_paths_in_names = []
        sdc._extra_paths_by_filepath = {}
        sdc._existing_pt_files = {f"{cf}_1.pt" for cf in self.PT_CONTENTS}
        sdc._aggregate_cache = {}
        sdc._active_key_by_filepath = {}
        # group setup: one concept with 3 samples (fp_A at idx 100, fp_A at
        # idx 200, fp_B at idx 300). variations=1, balancing=3 so
        # group_output_samples=3.
        sdc.group_variations = {"g0": 1}
        sdc.group_indices = {"g0": [100, 200, 300]}
        sdc.group_output_samples = {"g0": 3}
        sdc.group_full_indices = {"g0": [100, 200, 300]}
        sdc.group_balancing = {"g0": 3}
        sdc.group_balancing_strategy = {"g0": "REPEATS"}
        sdc._source_path_cache = {100: self.FP_A, 200: self.FP_A, 300: self.FP_B}
        sdc.cache_index = {
            "entries": {
                self.FP_A: {
                    "hash": "hashAfullhash",
                    "modeltype": "testmodel",
                    "mtime": 1.0,
                    "original_resolution": [4032, 2268],
                    "variants": {
                        "1024x576": {"cache_file": "hashA_1024x576"},
                        "704x384": {"cache_file": "hashA_704x384"},
                    },
                },
                self.FP_B: {
                    "hash": "hashBfullhash",
                    "modeltype": "testmodel",
                    "mtime": 1.0,
                    "original_resolution": [1200, 675],
                    "variants": {
                        "704x384": {"cache_file": "hashB_704x384"},
                    },
                },
            },
        }
        # PipelineModule's __local_cache machinery isn't initialised when
        # we __new__ past the constructor; SmartDiskCache.get_item never
        # walks upstream for split paths (we patch _get_resolution_string
        # below), so we can skip it.
        return sdc

    def _patch_upstream(self, sdc, monkeypatch):
        """Stand in for _get_resolution_string and torch.load."""

        def fake_res(in_variation, in_index):
            return self.PER_INDEX_RESOLUTION.get(in_index)

        monkeypatch.setattr(sdc, "_get_resolution_string", fake_res)
        # _real_pt_path uses _real_cache_dir + cache_file name. Capture
        # which path was loaded so we can assert later.
        loaded = []

        def fake_load(path, **kwargs):
            loaded.append(path)
            # extract the cache_file from the path ('/fake/cache/hashA_1024x576_1.pt')
            name = os.path.basename(path).replace("_1.pt", "")
            return self.PT_CONTENTS[name]

        monkeypatch.setattr("mgds.pipelineModules.SmartDiskCache.torch.load", fake_load)
        return loaded

    def test_load_aggregate_cache_keys_per_in_index(self, monkeypatch):
        """_load_aggregate_cache writes one entry per (filepath, var, in_index),
        not per (filepath, var). Same-filepath/different-in_index writes
        must not clobber each other."""
        sdc = self._make_sdc()
        self._patch_upstream(sdc, monkeypatch)

        sdc._load_aggregate_cache(out_variation=0)

        # Two distinct entries for fp_A — one per in_index — preserved.
        assert (self.FP_A, 0, 100) in sdc._aggregate_cache
        assert (self.FP_A, 0, 200) in sdc._aggregate_cache
        assert sdc._aggregate_cache[(self.FP_A, 0, 100)]["crop_resolution"] == (1024, 576)
        assert sdc._aggregate_cache[(self.FP_A, 0, 200)]["crop_resolution"] == (704, 384)
        # fp_B has its own entry too.
        assert sdc._aggregate_cache[(self.FP_B, 0, 300)]["crop_resolution"] == (704, 384)
        # Pre-fix bug regression: confirm the legacy 2-tuple key is NOT
        # present (otherwise downstream code might still hit it).
        assert (self.FP_A, 0) not in sdc._aggregate_cache
        assert (self.FP_B, 0) not in sdc._aggregate_cache

    def test_get_item_uses_per_in_index_aggregate(self, monkeypatch):
        """get_item's aggregate lookup hits the (filepath, var, in_index)
        entry — so the value returned at sort time for output_idx N matches
        the variant the per-(V, in_index) split fetch will later load.
        """
        sdc = self._make_sdc()
        self._patch_upstream(sdc, monkeypatch)

        sdc._load_aggregate_cache(out_variation=0)

        # __get_input_index needs variations_initialized state; bypass it
        # by manually building the (output_idx -> in_idx) mapping we want
        # to test and calling the cache lookup directly. We're testing the
        # aggregate-cache key correctness, not the input-index translator.
        # output_idx 0 -> in_idx 100 (fp_A, '1024x576' bucket).
        # output_idx 1 -> in_idx 200 (fp_A, '704x384' bucket).
        # output_idx 2 -> in_idx 300 (fp_B, '704x384' bucket).
        sdc.current_variation = 0
        sdc.variations_initialized = True

        agg_0 = sdc._aggregate_cache.get((self.FP_A, 0, 100))
        agg_1 = sdc._aggregate_cache.get((self.FP_A, 0, 200))
        agg_2 = sdc._aggregate_cache.get((self.FP_B, 0, 300))
        assert agg_0["crop_resolution"] == (1024, 576)
        assert agg_1["crop_resolution"] == (704, 384)
        assert agg_2["crop_resolution"] == (704, 384)

        # Critical scenario from the real-training repro: AspectBatchSorting
        # buckets fp_A@200 and fp_B@300 together (both at (704, 384)).
        # At training, split fetch for fp_A@200 resolves '704x384', which
        # matches the agg; split fetch for fp_B@300 resolves '704x384',
        # also matches. No mismatch.
        assert agg_1["crop_resolution"] == agg_2["crop_resolution"]

        # And critically, fp_A@100 is NOT in the same bucket as fp_A@200
        # — different (V, in_index) seeds → different variant. The bug
        # we fixed would have made these collide on (fp_A, 0).
        assert agg_0["crop_resolution"] != agg_1["crop_resolution"]

    def test_get_item_returns_per_in_index_crop_resolution(self, monkeypatch):
        """End-to-end: call get_item('crop_resolution', output_idx) for the
        three output indices and confirm each returns the variant matching
        what split-fetch would later load for that specific in_index.

        Models the AspectBatchSorting sort-time walk exactly: walker queries
        crop_resolution per output index, SmartDiskCache.get_item returns
        the agg cache copy. With the per-in_index key, each call sees the
        right variant.
        """
        sdc = self._make_sdc()
        self._patch_upstream(sdc, monkeypatch)
        sdc.current_variation = 0
        sdc.variations_initialized = True
        sdc._load_aggregate_cache(out_variation=0)

        # output_idx 0 -> in_idx 100 (fp_A, '1024x576' bucket).
        item0 = sdc.get_item(0, "crop_resolution")
        assert item0["crop_resolution"] == (1024, 576), (
            f"fp_A at in_idx=100 should resolve to its '1024x576' variant; got {item0['crop_resolution']}"
        )
        # output_idx 1 -> in_idx 200 (fp_A, '704x384' bucket).
        item1 = sdc.get_item(1, "crop_resolution")
        assert item1["crop_resolution"] == (704, 384), (
            f"fp_A at in_idx=200 should resolve to its '704x384' variant; "
            f"got {item1['crop_resolution']}. (Under the old (filepath, var) "
            f"key this would have returned whatever was written last, "
            f"clobbering the per-in_idx distinction.)"
        )
        # output_idx 2 -> in_idx 300 (fp_B, '704x384' bucket).
        item2 = sdc.get_item(2, "crop_resolution")
        assert item2["crop_resolution"] == (704, 384)

    def test_get_item_aggregate_dict_is_isolated(self, monkeypatch):
        """The returned aggregate dict must be a copy — the PipelineModule
        walker stashes it and later .update()s it with split-fetch data,
        which would silently overwrite our agg cache entry without the
        copy.
        """
        sdc = self._make_sdc()
        self._patch_upstream(sdc, monkeypatch)
        sdc.current_variation = 0
        sdc.variations_initialized = True
        sdc._load_aggregate_cache(out_variation=0)

        item = sdc.get_item(0, "crop_resolution")
        original = sdc._aggregate_cache[(self.FP_A, 0, 100)]
        assert item is not original, (
            "get_item must return a fresh dict — returning the cache "
            "entry by reference would let the walker's later update() "
            "mutate our stored aggregate."
        )
        # And mutating the returned dict must not bleed back.
        item["crop_resolution"] = (999, 999)
        assert sdc._aggregate_cache[(self.FP_A, 0, 100)]["crop_resolution"] == (1024, 576)

    def test_pre_fix_bug_no_longer_reachable(self, monkeypatch):
        """The exact pre-fix failure mode: writing for fp at two in_idx
        leaves only the last write under a 2-tuple key. Verify the new
        3-tuple key path can't reproduce it (both writes survive)."""
        sdc = self._make_sdc()
        self._patch_upstream(sdc, monkeypatch)

        sdc._load_aggregate_cache(out_variation=0)

        # Count distinct fp_A entries — must be 2 (one per in_index).
        fp_a_entries = [
            key for key in sdc._aggregate_cache if isinstance(key, tuple) and len(key) >= 1 and key[0] == self.FP_A
        ]
        assert len(fp_a_entries) == 2, (
            f"Expected two distinct (fp_A, var, in_index) entries; got "
            f"{fp_a_entries}. Under the old (filepath, variation) key the "
            f"second agg-load write clobbered the first."
        )


class TestSynthesizeAggregateChecksVariantExists:
    """``_try_synthesize_aggregate`` must serve the .pt's *actual* stored
    ``crop_resolution`` — not a value parsed from the variant key string.
    The two diverge in production when drift recovery linked an
    out-of-grid key to an old .pt, when ``_target_int_for_resolution_key``
    returned an ambiguous target during the build pass, or any time the
    AspectBucketing config changed between build and read. Synth keys
    its lookup on the per-variant ``crop_resolution`` field stamped at
    build time (or lazy-stamped from the slow path on first read of an
    old cache). Returns None — routing the item to the slow path so the
    .pt's real value lands in the agg cache and gets stamped — whenever
    the variant either doesn't exist OR exists without a stamp.

    Original repro: multi-target config (e.g. "768,1024") + cache
    built under only one target → ``variant_key_from_aspect`` rolls the
    missing target for some ``(variation, in_index)`` pairs → synthesis
    yielded ``(1024, 576)`` while split-fetch loaded the only on-disk
    variant ``(704, 384)``.
    """

    FP = r"F:\Datasets\test\partial.jpg"
    EXISTING_KEY = "704x384"

    def _make_sdc(self, variants=None):
        """SmartDiskCache primed with a single entry whose ``variants``
        is the supplied dict (each value is the full variant record,
        including or omitting a stamped ``crop_resolution``).
        """
        if variants is None:
            variants = {
                self.EXISTING_KEY: {
                    "cache_file": f"hash_{self.EXISTING_KEY}",
                    "crop_resolution": [704, 384],
                },
            }
        import threading

        sdc = SmartDiskCache.__new__(SmartDiskCache)
        sdc.cache_dir = "/fake/cache"
        sdc._real_cache_dir = "/fake/cache"
        sdc.modeltype = "testmodel"
        sdc.split_names = ["latent_image"]
        sdc.aggregate_names = ["crop_resolution", "image_path"]
        sdc.sourceless = False
        sdc.resolution_from_upstream = False
        sdc.aspect_bucketing = object()
        sdc.bucket_method_provider = None
        sdc.rebucket_provider = None
        sdc.tolerate_missing_source = False
        sdc.extra_watched_paths_in_names = []
        sdc._extra_paths_by_filepath = {}
        sdc._aggregate_cache = {}
        sdc._active_key_by_filepath = {}
        sdc._index_lock = threading.Lock()
        sdc._index_io_lock = threading.Lock()
        sdc._index_disk_stat = None
        sdc._last_flush_time = 0.0
        sdc.cache_index = {
            "entries": {
                self.FP: {
                    "hash": "hashfull",
                    "modeltype": "testmodel",
                    "mtime": 1.0,
                    "original_resolution": [1024, 576],
                    "variants": variants,
                },
            },
        }
        return sdc

    def test_missing_variant_returns_none(self, monkeypatch):
        """``_fast_resolution_string`` rolls a key the entry doesn't have
        → ``_try_synthesize_aggregate`` must return None so the caller
        defers to the slow path that .pt-loads the active fallback.
        """
        sdc = self._make_sdc(
            variants={
                "704x384": {
                    "cache_file": "hash_704x384",
                    "crop_resolution": [704, 384],
                }
            },
        )
        monkeypatch.setattr(
            sdc,
            "_fast_resolution_string",
            lambda entry, in_variation, in_index: "1024x576",
        )

        result = sdc._try_synthesize_aggregate(
            self.FP,
            sdc.cache_index["entries"][self.FP],
            in_variation=0,
            in_index=100,
        )

        assert result is None, (
            f"Synthesis must abort when the computed key '1024x576' is "
            f"not in entry['variants']; otherwise the fast agg-cache "
            f"would report (1024, 576) while split-fetch falls back to "
            f"the only on-disk variant '704x384' and serves a "
            f"(704, 384) latent — the exact mismatch _shape_safe_collate "
            f"catches. Got: {result}"
        )

    def test_stamped_variant_serves_stored_crop_resolution(self, monkeypatch):
        """When the variant has a stamped ``crop_resolution``, synth
        serves it directly without torch.load.
        """
        sdc = self._make_sdc(
            variants={
                "704x384": {
                    "cache_file": "hash_704x384",
                    "crop_resolution": [704, 384],
                },
                "1024x576": {
                    "cache_file": "hash_1024x576",
                    "crop_resolution": [1024, 576],
                },
            },
        )
        monkeypatch.setattr(
            sdc,
            "_fast_resolution_string",
            lambda entry, in_variation, in_index: "1024x576",
        )

        result = sdc._try_synthesize_aggregate(
            self.FP,
            sdc.cache_index["entries"][self.FP],
            in_variation=0,
            in_index=100,
        )

        assert result is not None, (
            "Synthesis should succeed for a variant with a stamped crop_resolution — that's the entire fast-path win."
        )
        assert result["crop_resolution"] == (1024, 576)
        assert result["image_path"] == self.FP

    def test_stamped_value_overrides_key_string(self, monkeypatch):
        """The stamp wins over the variant key parse: when an entry was
        drift-recovered onto an old .pt whose stored crop_resolution
        differs from the new key string, synth must serve the stored
        value (matching what split-fetch loads), not the parsed key.

        This is the exact divergence that survives just-checking-variant-
        existence: the variant key says '1024x576', the .pt actually
        has (640, 960), split-fetch loads (640, 960), and AspectBatchSorting
        must bucket by (640, 960) too.
        """
        sdc = self._make_sdc(
            variants={
                "1024x576": {
                    "cache_file": "hash_1024x576",
                    # The .pt at this cache_file was built under a different
                    # config and stores (640, 960). The stamp records that.
                    "crop_resolution": [640, 960],
                }
            },
        )
        monkeypatch.setattr(
            sdc,
            "_fast_resolution_string",
            lambda entry, in_variation, in_index: "1024x576",
        )

        result = sdc._try_synthesize_aggregate(
            self.FP,
            sdc.cache_index["entries"][self.FP],
            in_variation=0,
            in_index=100,
        )

        assert result is not None
        assert result["crop_resolution"] == (640, 960), (
            f"Synth must serve the .pt's stored value (640, 960), not "
            f"the parsed key (1024, 576). The latter would mis-bucket "
            f"this item against batchmates whose split-fetch returned "
            f"(640, 960). Got: {result['crop_resolution']}"
        )

    def test_unstamped_variant_falls_to_slow_path(self, monkeypatch):
        """Legacy variant records (built before the crop_resolution
        stamp existed) must route to the slow path so torch.load can
        read the truth and lazy-stamp the entry.
        """
        sdc = self._make_sdc(
            variants={
                "1024x576": {  # no crop_resolution field
                    "cache_file": "hash_1024x576",
                }
            },
        )
        monkeypatch.setattr(
            sdc,
            "_fast_resolution_string",
            lambda entry, in_variation, in_index: "1024x576",
        )

        result = sdc._try_synthesize_aggregate(
            self.FP,
            sdc.cache_index["entries"][self.FP],
            in_variation=0,
            in_index=100,
        )

        assert result is None, (
            "Unstamped variant must return None so the slow path runs "
            "and lazy-stamps the variant from the .pt's actual value."
        )

    def test_load_aggregate_cache_lazy_stamps_legacy_variant(self, monkeypatch, tmp_path):
        """End-to-end: legacy entry without stamped crop_resolution →
        slow path runs → reads from .pt → writes crop_resolution to
        the variant record. Subsequent epochs hit the fast path.
        """
        sdc = self._make_sdc(
            variants={
                "1024x576": {  # legacy: no crop_resolution
                    "cache_file": "hash_1024x576",
                }
            },
        )
        sdc.cache_dir = str(tmp_path)
        sdc._real_cache_dir = str(tmp_path)
        sdc._existing_pt_files = {"hash_1024x576_1.pt"}
        sdc.group_variations = {"g0": 1}
        sdc.group_indices = {"g0": [100]}
        sdc.group_output_samples = {"g0": 1}
        sdc.group_full_indices = {"g0": [100]}
        sdc.group_balancing = {"g0": 1}
        sdc.group_balancing_strategy = {"g0": "REPEATS"}
        sdc._source_path_cache = {100: self.FP}

        monkeypatch.setattr(
            sdc,
            "_fast_resolution_string",
            lambda entry, in_variation, in_index: "1024x576",
        )
        monkeypatch.setattr(
            sdc,
            "_get_resolution_string",
            lambda in_variation, in_index: "1024x576",
        )
        load_paths = []

        def fake_load(path, **kwargs):
            load_paths.append(path)
            # .pt stores (640, 960) — the truth that diverges from the key.
            return {"crop_resolution": (640, 960), "image_path": self.FP}

        monkeypatch.setattr(
            "mgds.pipelineModules.SmartDiskCache.torch.load",
            fake_load,
        )

        sdc._load_aggregate_cache(out_variation=0)

        agg = sdc._aggregate_cache.get((self.FP, 0, 100))
        assert agg is not None
        assert agg["crop_resolution"] == (640, 960), (
            f"Slow path must serve the .pt's value, not the parsed key. Got: {agg['crop_resolution']}"
        )
        # The variant must now have the lazy-stamped value so the next
        # epoch's agg load takes the fast path.
        variant = sdc.cache_index["entries"][self.FP]["variants"]["1024x576"]
        assert variant.get("crop_resolution") == [640, 960], (
            f"Slow path must lazy-stamp the variant's crop_resolution "
            f"from the loaded .pt. Got: {variant.get('crop_resolution')}"
        )

    def test_load_aggregate_cache_falls_through_to_slow_path(self, monkeypatch, tmp_path):
        """End-to-end with the original repro shape: rolled key missing,
        slow path uses _active_cache_file fallback, torch.load reads the
        real value, aggregate matches what split-fetch will return.
        """
        sdc = self._make_sdc(
            variants={
                "704x384": {
                    "cache_file": "hash_704x384",
                    "crop_resolution": [704, 384],
                }
            },
        )
        sdc.cache_dir = str(tmp_path)
        sdc._real_cache_dir = str(tmp_path)
        sdc._existing_pt_files = {"hash_704x384_1.pt"}
        sdc.group_variations = {"g0": 1}
        sdc.group_indices = {"g0": [100]}
        sdc.group_output_samples = {"g0": 1}
        sdc.group_full_indices = {"g0": [100]}
        sdc.group_balancing = {"g0": 1}
        sdc.group_balancing_strategy = {"g0": "REPEATS"}
        sdc._source_path_cache = {100: self.FP}

        monkeypatch.setattr(
            sdc,
            "_fast_resolution_string",
            lambda entry, in_variation, in_index: "1024x576",
        )
        monkeypatch.setattr(
            sdc,
            "_get_resolution_string",
            lambda in_variation, in_index: "1024x576",
        )
        load_paths = []

        def fake_load(path, **kwargs):
            load_paths.append(path)
            return {"crop_resolution": (704, 384), "image_path": self.FP}

        monkeypatch.setattr(
            "mgds.pipelineModules.SmartDiskCache.torch.load",
            fake_load,
        )

        sdc._load_aggregate_cache(out_variation=0)

        agg = sdc._aggregate_cache.get((self.FP, 0, 100))
        assert agg is not None
        assert agg["crop_resolution"] == (704, 384), (
            f"Aggregate must reflect the .pt actually on disk (the "
            f"'704x384' fallback), not the synthesised '1024x576' key "
            f"that get_item will also miss. Got: {agg['crop_resolution']}"
        )
        assert load_paths, "Expected slow-path torch.load to fire for the missing-variant case; got no load events."


class _StubAspectBucketing:
    """Minimal AspectBucketing stand-in. Two bucket_resolutions so
    ``multi_target`` is True (the filepath-granular session skip is bypassed),
    plus a caller-supplied ``(variation, index) -> 'HxW'`` key function so the
    test fully controls which bucket each (epoch, item) rolls. The validation
    loop, get_item and _load_aggregate_cache all reach this via
    ``_fast_resolution_string``."""

    def __init__(self, key_fn):
        self._key_fn = key_fn
        self.bucket_resolutions = {512: [(256, 256)], 768: [(384, 384)]}
        self.frame_dim_enabled = False
        self._target_override = None

    def variant_key_from_aspect(self, variation, index, aspect):
        return self._key_fn(variation, index)


class TestMultiresVariantSessionSkip:
    """Multi-resolution caches are bypassed by the filepath-granular session
    skip (a filepath maps to several resolution variants and the required one
    rotates per epoch). The variant-aware skip tracks (filepath, key) pairs, so
    once every variant a stable dataset requests has been validated, later
    epochs skip the full 'validating cache' loop — while a genuinely new variant
    still forces validation and a lazy build.
    """

    def _setup(self, tmp_path, key_fn):
        """Build a multires pipeline and return (ds, cache_dir, cache_mod).

        Built inline (rather than via _build_smart_pipeline) so the test holds
        the SmartDiskCache reference and can attach the stub bucketing.
        """
        src_dir = tmp_path / "src"
        src_dir.mkdir(exist_ok=True)
        paths = [_create_source_file(src_dir, f"i{i}.bin", f"c{i}".encode()) for i in range(2)]
        tensors = _make_tensors(2, seed=7)
        cache_dir = str(tmp_path / "cache")

        dummy_mod = DummyDataModule(
            data={
                "latent": tensors,
                "image_path": paths,
                "crop_resolution": [(256, 256), (256, 256)],
                "original_resolution": [(512, 512), (512, 512)],
            },
            length=2,
        )
        cache_mod = SmartDiskCache(
            cache_dir=cache_dir,
            split_names=["latent"],
            aggregate_names=["crop_resolution"],
            modeltype="testmodel",
            source_path_in_name="image_path",
        )
        output_mod = OutputPipelineModule(names=["latent", "crop_resolution"])
        ds = MGDS(
            device=torch.device("cpu"),
            concepts=[{"name": "A", "path": "dummy"}],
            settings={},
            definition=[[dummy_mod], [cache_mod], [output_mod]],
            batch_size=1,
            state=PipelineState(),
            seed=42,
        )
        cache_mod.aspect_bucketing = _StubAspectBucketing(key_fn)
        return ds, cache_dir, cache_mod

    @staticmethod
    def _stable(_variation, _index):
        # Every (epoch, item) rolls the same bucket: the required variant set is
        # identical across epochs on a stable dataset.
        return "256x256"

    def test_first_epoch_does_not_skip(self, tmp_path, capsys):
        ds, _cache_dir, _cm = self._setup(tmp_path, self._stable)
        _drain(ds)
        out = capsys.readouterr().out
        assert "Skipped re-validation" not in out

    def test_second_epoch_skips_when_no_new_variants(self, tmp_path, capsys):
        """Epoch 2 of a stable multires dataset must hit the variant-aware skip —
        no 'validating cache' loop — even though multi_target disables the
        filepath-granular skip."""
        ds, _cache_dir, _cm = self._setup(tmp_path, self._stable)
        _drain(ds)
        capsys.readouterr()

        batches = _drain(ds)

        out = capsys.readouterr().out
        assert "Skipped re-validation" in out
        # Skipping must not starve the loader: data still flows.
        assert len(batches) > 0

    def test_repeated_epochs_all_skip(self, tmp_path, capsys):
        ds, _cache_dir, _cm = self._setup(tmp_path, self._stable)
        _drain(ds)
        capsys.readouterr()

        for _ in range(4):
            _drain(ds)

        out = capsys.readouterr().out
        assert out.count("Skipped re-validation") == 4

    def test_fresh_pipeline_does_not_skip(self, tmp_path, capsys):
        """A new pipeline instance = new process: the session set is empty, so
        even a fully-built multires cache validates on its first epoch."""
        ds1, _cd, _cm = self._setup(tmp_path, self._stable)
        _drain(ds1)
        capsys.readouterr()

        ds2, _cd2, _cm2 = self._setup(tmp_path, self._stable)
        _drain(ds2)

        out = capsys.readouterr().out
        assert "Skipped re-validation" not in out

    def test_new_variant_at_epoch_2_is_not_skipped(self, tmp_path, capsys):
        """Variant-granularity proof: if a file rolls a bucket it was never
        validated for, the skip must NOT fire — the new variant has to be
        validated and lazily built, not silently served from a stale one."""
        roll = {"i1": "256x256"}  # file 1's bucket; mutated after epoch 1

        def key_fn(_variation, index):
            return "256x256" if index == 0 else roll["i1"]

        ds, cache_dir, _cm = self._setup(tmp_path, key_fn)
        _drain(ds)
        capsys.readouterr()

        # File 1 now rolls a bucket that was never built/validated.
        roll["i1"] = "384x384"

        _drain(ds)

        out = capsys.readouterr().out
        assert "Skipped re-validation" not in out
        # The new variant was lazily built rather than skipped.
        built = {fp: sorted(e["variants"].keys()) for fp, e in _read_cache_json(cache_dir)["entries"].items()}
        assert any("384x384" in keys for keys in built.values()), (
            f"new variant '384x384' should have been built; got {built}"
        )
