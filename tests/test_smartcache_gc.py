"""SmartDiskCache and the cyclic garbage collector: the heap is frozen for the
build loops (so their forced collections skip the loaded index) and thawed
again afterwards, and the collector is paused while cache.json is parsed or
serialized."""

import gc
import json
import os
import threading
import weakref

from mgds.MGDS import MGDS
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModules.SmartDiskCache import SmartDiskCache, _gc_heap_frozen, _gc_paused
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch

import pytest


class _Rows(PipelineModule, RandomAccessPipelineModule):
    def __init__(self, paths: list[str]):
        super().__init__()
        self.paths = paths

    def length(self) -> int:
        return len(self.paths)

    def get_inputs(self) -> list[str]:
        return []

    def get_outputs(self) -> list[str]:
        return ["image_path", "latent"]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return {"image_path": self.paths[index], "latent": torch.full((2, 2), float(index))}


class _Node:
    """A GC-tracked object that can sit in a reference cycle."""

    def __init__(self):
        self.other = None


class _ProbeCache(SmartDiskCache):
    """Records the freeze count seen by every build and every augment write."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.build_freeze_counts: list[int] = []

    def _build_cache_entry(self, *args, **kwargs):
        self.build_freeze_counts.append(gc.get_freeze_count())
        return super()._build_cache_entry(*args, **kwargs)


def _sources(tmp_path, count: int) -> list[str]:
    paths = []
    for i in range(count):
        path = tmp_path / f"img_{i}.bin"
        path.write_bytes(f"source-{i}".encode())
        paths.append(str(path))
    return paths


def _pipeline(tmp_path, paths: list[str], split_names: list[str], cache_cls=_ProbeCache):
    cache = cache_cls(
        cache_dir=str(tmp_path / "cache"),
        split_names=split_names,
        aggregate_names=[],
        modeltype="testmodel",
        source_path_in_name="image_path",
    )
    ds = MGDS(
        device=torch.device("cpu"),
        concepts=[{}],
        settings={},
        definition=[[_Rows(paths)], [cache], [OutputPipelineModule(names=split_names)]],
        batch_size=1,
        state=PipelineState(),
        seed=42,
    )
    return ds, cache


@pytest.fixture(autouse=True)
def _thawed_heap():
    gc.unfreeze()
    yield
    gc.unfreeze()


def test_heap_is_frozen_inside_and_thawed_after():
    assert gc.get_freeze_count() == 0
    with _gc_heap_frozen():
        assert gc.get_freeze_count() > 0
    assert gc.get_freeze_count() == 0


def test_heap_is_thawed_when_the_loop_raises():
    with pytest.raises(RuntimeError), _gc_heap_frozen():
        raise RuntimeError("stopped")
    assert gc.get_freeze_count() == 0


def test_a_host_freeze_is_left_in_place():
    gc.freeze()
    host_frozen = gc.get_freeze_count()
    assert host_frozen > 0
    with _gc_heap_frozen():
        pass
    # Thawing would have dropped the count to 0 and undone the host's freeze.
    assert gc.get_freeze_count() >= host_frozen


def test_garbage_made_inside_is_still_collected():
    with _gc_heap_frozen():
        a, b = _Node(), _Node()
        a.other, b.other = b, a
        ref = weakref.ref(a)
        del a, b
        gc.collect()
        assert ref() is None


def test_a_cycle_frozen_alive_is_collected_after_the_thaw():
    a, b = _Node(), _Node()
    a.other, b.other = b, a
    ref = weakref.ref(a)
    with _gc_heap_frozen():
        del a, b
        gc.collect()
        # In the permanent generation: no collection looks at it.
        assert ref() is not None
    gc.collect()
    assert ref() is None


def test_build_loop_runs_frozen_and_leaves_the_heap_thawed(tmp_path):
    ds, cache = _pipeline(tmp_path, _sources(tmp_path, 4), ["latent"])
    ds.start_next_epoch()

    assert len(cache.build_freeze_counts) == 4
    assert all(count > 0 for count in cache.build_freeze_counts)
    assert gc.get_freeze_count() == 0
    assert len([name for name in os.listdir(tmp_path / "cache") if name.endswith(".pt")]) >= 4


def test_augment_loop_runs_frozen_and_leaves_the_heap_thawed(tmp_path):
    paths = _sources(tmp_path, 3)
    ds, _ = _pipeline(tmp_path, paths, ["latent"])
    ds.start_next_epoch()

    seen: list[int] = []

    class _WiderRows(_Rows):
        def get_outputs(self) -> list[str]:
            return ["image_path", "latent", "extra"]

        def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
            seen.append(gc.get_freeze_count())
            item = super().get_item(variation, index, requested_name)
            item["extra"] = torch.ones(2, 2)
            return item

    # Same cache dir, one more required name: the schema drift is repaired by
    # the augment loop, which computes "extra" through the upstream module.
    cache = SmartDiskCache(
        cache_dir=str(tmp_path / "cache"),
        split_names=["latent", "extra"],
        aggregate_names=[],
        modeltype="testmodel",
        source_path_in_name="image_path",
    )
    ds = MGDS(
        device=torch.device("cpu"),
        concepts=[{}],
        settings={},
        definition=[[_WiderRows(paths)], [cache], [OutputPipelineModule(names=["latent", "extra"])]],
        batch_size=1,
        state=PipelineState(),
        seed=42,
    )
    ds.start_next_epoch()

    assert seen and any(count > 0 for count in seen)
    assert gc.get_freeze_count() == 0
    first = next(iter(cache.cache_index["entries"].values()))
    cache_file = next(iter(first["variants"].values()))["cache_file"]
    assert "extra" in torch.load(os.path.join(tmp_path / "cache", f"{cache_file}_1.pt"), weights_only=False)


# ---------------------------------------------------------------------------
# Collector paused around the cache.json parse and dump
# ---------------------------------------------------------------------------


@pytest.fixture
def _collector_enabled():
    was_enabled = gc.isenabled()
    gc.enable()
    yield
    if was_enabled:
        gc.enable()
    else:
        gc.disable()


def test_pause_restores_an_enabled_collector(_collector_enabled):
    with _gc_paused():
        assert not gc.isenabled()
    assert gc.isenabled()


def test_pause_restores_after_an_error(_collector_enabled):
    with pytest.raises(ValueError), _gc_paused():
        raise ValueError("bad index")
    assert gc.isenabled()


def test_pause_leaves_a_disabled_collector_disabled(_collector_enabled):
    gc.disable()
    with _gc_paused():
        assert not gc.isenabled()
    assert not gc.isenabled()


def test_nested_pauses_resume_only_after_the_outermost(_collector_enabled):
    with _gc_paused():
        with _gc_paused():
            assert not gc.isenabled()
        assert not gc.isenabled()
    assert gc.isenabled()


def test_overlapping_pauses_in_two_threads_resume_after_the_last(_collector_enabled):
    inside = threading.Event()
    release = threading.Event()

    def hold():
        with _gc_paused():
            inside.set()
            release.wait(timeout=10)

    thread = threading.Thread(target=hold)
    thread.start()
    assert inside.wait(timeout=10)
    with _gc_paused():
        pass
    # The other thread is still inside its block.
    assert not gc.isenabled()
    release.set()
    thread.join(timeout=10)
    assert gc.isenabled()


def test_index_parse_and_dump_run_with_the_collector_paused(tmp_path, monkeypatch, _collector_enabled):
    ds, cache = _pipeline(tmp_path, _sources(tmp_path, 3), ["latent"])
    ds.start_next_epoch()

    states: dict[str, list[bool]] = {"load": [], "dumps": []}
    real_load, real_dumps = json.load, json.dumps

    def load(*args, **kwargs):
        states["load"].append(gc.isenabled())
        return real_load(*args, **kwargs)

    def dumps(*args, **kwargs):
        states["dumps"].append(gc.isenabled())
        return real_dumps(*args, **kwargs)

    monkeypatch.setattr(json, "load", load)
    monkeypatch.setattr(json, "dumps", dumps)

    before = cache._load_cache_index()
    cache._save_cache_index()
    after = cache._load_cache_index()

    assert states["load"] == [False, False]
    assert states["dumps"] == [False]
    assert gc.isenabled()
    assert after == before
    assert len(after["entries"]) == 3


def test_a_corrupt_index_still_falls_back_and_resumes_the_collector(tmp_path, _collector_enabled):
    ds, cache = _pipeline(tmp_path, _sources(tmp_path, 2), ["latent"])
    ds.start_next_epoch()
    index_path = os.path.join(tmp_path / "cache", "cache.json")
    good = cache._load_cache_index()
    os.replace(index_path, index_path + ".bak")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("{ not json")

    recovered = cache._load_cache_index()

    assert gc.isenabled()
    assert recovered["entries"].keys() == good["entries"].keys()


# ---------------------------------------------------------------------------
# One collection before the first freeze of a refresh
# ---------------------------------------------------------------------------


def _dead_cycle() -> weakref.ref:
    """A reference cycle that is already garbage but not collected yet."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        a, b = _Node(), _Node()
        a.other, b.other = b, a
        return weakref.ref(a)
    finally:
        if was_enabled:
            gc.enable()


def test_collect_first_releases_dead_garbage_instead_of_freezing_it():
    ref = _dead_cycle()
    with _gc_heap_frozen(collect_first=True):
        assert ref() is None


def test_without_collect_first_dead_garbage_waits_for_the_thaw():
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        ref = _dead_cycle()
        with _gc_heap_frozen():
            gc.collect()
            assert ref() is not None
        gc.collect()
        assert ref() is None
    finally:
        if was_enabled:
            gc.enable()


def test_only_the_first_loop_of_a_refresh_collects(tmp_path, monkeypatch):
    import mgds.pipelineModules.SmartDiskCache as sdc_module

    seen: list[bool] = []
    real = sdc_module._gc_heap_frozen

    def recording(collect_first: bool = False):
        seen.append(collect_first)
        return real(collect_first)

    monkeypatch.setattr(sdc_module, "_gc_heap_frozen", recording)

    cache = SmartDiskCache.__new__(SmartDiskCache)
    for _ in range(3):
        with cache._frozen_loop_heap():
            pass
    assert seen == [True, False, False]

    # A real refresh starts over: its first build loop collects again.
    seen.clear()
    ds, cache = _pipeline(tmp_path, _sources(tmp_path, 3), ["latent"])
    ds.start_next_epoch()
    assert seen == [True]
    assert gc.get_freeze_count() == 0
