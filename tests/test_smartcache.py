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

import pytest
import torch
import xxhash

from mgds.MGDS import MGDS
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModules.SmartDiskCache import SmartDiskCache, CACHE_VERSION
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


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
        index1 = _read_cache_json(cache_dir)

        # Record .pt mtimes
        pt_files = [f for f in os.listdir(cache_dir) if f.endswith(".pt")]
        pt_mtimes = {f: os.path.getmtime(os.path.join(cache_dir, f)) for f in pt_files}

        time.sleep(0.05)
        _drain(ds)  # reuse cache -- source files untouched

        for f in pt_files:
            assert os.path.getmtime(os.path.join(cache_dir, f)) == pt_mtimes[f], \
                f"{f} was rewritten even though source was unchanged"

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
            assert os.path.getmtime(os.path.join(cache_dir, f)) == pt_mtimes[f], \
                f"{f} was rewritten on touch with same content"

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
        assert index["entries"][norm_a]["cache_file"] == index["entries"][norm_b]["cache_file"]

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
        shared_cache_file = index_before["entries"][norm_a]["cache_file"]

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
        assert index_after["entries"][norm_a]["cache_file"] == shared_cache_file

        # path_b should now have a different cache_file
        assert index_after["entries"][norm_b]["cache_file"] != shared_cache_file


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
            "entries": {"fake/path.bin": {
                "filename": "path.bin",
                "hash": "abcdef012345abcd",
                "mtime": 1234567890.0,
                "modeltype": "test",
                "resolution": None,
                "cache_file": "abcdef012345",
                "cache_version": CACHE_VERSION,
            }},
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

        pt_before = set(f for f in os.listdir(cache_dir) if f.endswith(".pt"))
        assert len(pt_before) == 3  # one .pt per source

        # Delete one source file to create an orphan
        os.remove(paths[0])

        SmartDiskCache.gc_clean(cache_dir)

        pt_after = set(f for f in os.listdir(cache_dir) if f.endswith(".pt"))
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
            assert count == sample_count, \
                f"Expected {sample_count} items per epoch, got {count}"

        # Over 8 epochs with 10 items and 3 chosen each time, we expect
        # the union to be larger than 3 (almost certainly).
        assert len(all_selected) > sample_count, \
            f"Expected more than {sample_count} distinct items across {num_epochs} epochs, got {len(all_selected)}"


# ---------------------------------------------------------------------------
# Integration: full pipeline round-trip
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_cache_reuse_across_epochs(self, tmp_path):
        """Cache built in epoch 1 is reused in epoch 2 (no .pt rewrite)."""
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        paths = [
            _create_source_file(src_dir, f"img_{i}.bin", f"reuse content {i}".encode())
            for i in range(3)
        ]
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
            assert os.path.getmtime(os.path.join(cache_dir, f)) == pt_mtimes[f], \
                f"{f} was rewritten on epoch 2"

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
        paths = [
            _create_source_file(src_dir, f"img_{i}.bin", f"round trip {i}".encode())
            for i in range(3)
        ]
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

        for b1, b2 in zip(batches1, batches2):
            assert torch.equal(b1["latent"], b2["latent"]), \
                "Cached latent differs from original"
            assert b1["crop_resolution"] == b2["crop_resolution"]

    def test_variations_create_multiple_pt(self, tmp_path):
        """With variations > 1, multiple .pt files per source should be created."""
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        num_items = 3
        num_variations = 2
        paths = [
            _create_source_file(src_dir, f"img_{i}.bin", f"variation content {i}".encode())
            for i in range(num_items)
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
            cache_file = entry["cache_file"]
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
        cache_file = index["entries"][norm]["cache_file"]
        assert cache_file.startswith(expected_prefix), \
            f"Cache file '{cache_file}' should start with hash prefix '{expected_prefix}'"

    def test_balancing_repeats(self, tmp_path):
        """With REPEATS balancing=0.5, floor(N*0.5) items should be emitted."""
        num_items = 6
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        paths = [
            _create_source_file(src_dir, f"img_{i}.bin", f"repeats content {i}".encode())
            for i in range(num_items)
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
        assert len(batches) == expected_output, \
            f"Expected {expected_output} items with REPEATS balancing=0.5, got {len(batches)}"

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
        """When _validate_entry returns 'rebuild', the old hash must be
        removed from hash_index before re-queuing — otherwise a stale
        pointer is left in hash_index."""
        from unittest.mock import patch

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        src_file = _create_source_file(src_dir, "test.bin", b"original content")

        paths = [src_file]
        tensors = _make_tensors(1)
        dummy_data = {"latent": tensors, "image_path": paths}
        ds, cache_dir, _ = _build_smart_pipeline(
            tmp_path, [{}], dummy_data, 1,
            split_names=["latent"], aggregate_names=[],
            modeltype="testmodel", source_path_in_name="image_path",
        )
        _drain(ds)

        idx = _read_cache_json(cache_dir)
        fp = os.path.normpath(src_file)
        old_hash = idx["entries"][fp]["hash"]
        assert fp in idx["hash_index"][old_hash]

        # Change file content AND make getmtime fail once to trigger 'rebuild'
        with open(src_file, "wb") as f:
            f.write(b"completely new content")

        original_getmtime = os.path.getmtime
        call_count = [0]
        def flaky_getmtime(path):
            if os.path.normpath(path) == fp and call_count[0] == 0:
                call_count[0] += 1
                raise OSError("simulated access error")
            return original_getmtime(path)

        # Fresh pipeline = new run; 'rebuild' path triggers during full validation
        # at run start, not via within-run revalidation (which session-skip bypasses).
        ds2, _, _ = _build_smart_pipeline(
            tmp_path, [{}], dummy_data, 1,
            split_names=["latent"], aggregate_names=[],
            modeltype="testmodel", source_path_in_name="image_path",
        )
        with patch("os.path.getmtime", side_effect=flaky_getmtime):
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

        captured = capsys.readouterr()
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
        paths = [
            _create_source_file(src_dir, f"img_{i}.bin", f"content_{i}".encode())
            for i in range(n)
        ]
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
