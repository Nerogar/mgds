"""Sourceless start-up work: the aggregate values are loaded once when the
second load could only repeat the first, and the cache files' existence is
answered from one directory listing."""

import json
import os

from mgds.MGDS import MGDS
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModules.PlaceholderModule import PlaceholderModule
from mgds.pipelineModules.SmartDiskCache import SmartDiskCache
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch

import pytest

OUTPUT_NAMES = ["latent_image", "crop_resolution", "image_path"]


class _ImageSource(PipelineModule, RandomAccessPipelineModule):
    """Rows of one concept, so they share a variations group."""

    def __init__(self, image_paths: list[str], balancing_strategy: str, balancing: float):
        super().__init__()
        self.image_paths = image_paths
        self.concept = {
            "path": "concept-a",
            "image_variations": 1,
            "balancing": balancing,
            "balancing_strategy": balancing_strategy,
            "enabled": True,
        }

    def length(self) -> int:
        return len(self.image_paths)

    def get_inputs(self) -> list[str]:
        return []

    def get_outputs(self) -> list[str]:
        return ["image_path", "latent_image", "crop_resolution", "concept"]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return {
            "image_path": self.image_paths[index],
            "latent_image": torch.tensor([float(index)]),
            "crop_resolution": (512, 768) if index % 2 else (768, 512),
            "concept": self.concept,
        }


class _CountingCache(SmartDiskCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aggregate_loads: list[tuple[int, bool]] = []

    def _load_aggregate_cache(self, out_variation: int) -> bool:
        index_only = super()._load_aggregate_cache(out_variation)
        self.aggregate_loads.append((out_variation, index_only))
        return index_only


def _image_cache(cache_dir, *, sourceless: bool) -> _CountingCache:
    return _CountingCache(
        cache_dir=str(cache_dir),
        split_names=["latent_image"],
        aggregate_names=["crop_resolution", "image_path"],
        variations_in_name="concept.image_variations",
        balancing_in_name="concept.balancing",
        balancing_strategy_in_name="concept.balancing_strategy",
        variations_group_in_name=["concept.path"],
        group_enabled_in_name="concept.enabled",
        modeltype="testmodel",
        source_path_in_name="image_path",
        sourceless=sourceless,
    )


def _dataset(modules: list) -> MGDS:
    return MGDS(
        device=torch.device("cpu"),
        concepts=[{"name": "concept-a", "path": "concept-a"}],
        settings={},
        definition=[modules, OutputPipelineModule(names=OUTPUT_NAMES)],
        batch_size=1,
        state=PipelineState(),
        seed=42,
    )


def _build_cache(tmp_path, count: int = 4, balancing_strategy: str = "REPEATS", balancing: float = 1.0):
    """A cache built from sources; returns (cache dir, source paths)."""
    paths = []
    for i in range(count):
        path = tmp_path / "src" / f"img_{i}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"image-{i}".encode())
        paths.append(str(path))
    cache_dir = tmp_path / "cache" / "image"
    ds = _dataset([_ImageSource(paths, balancing_strategy, balancing), _image_cache(cache_dir, sourceless=False)])
    ds.start_next_epoch()
    return cache_dir, paths


def _sourceless(cache_dir):
    cache = _image_cache(cache_dir, sourceless=True)
    return _dataset([PlaceholderModule(), cache]), cache


def _rows(ds) -> list[tuple]:
    return sorted((os.path.basename(item["image_path"]), tuple(item["crop_resolution"])) for item in ds)


def test_first_start_loads_the_aggregate_values_once(tmp_path):
    cache_dir, paths = _build_cache(tmp_path)
    ds, cache = _sourceless(cache_dir)

    ds.start_next_epoch()

    assert cache.aggregate_loads == [(0, True)]
    kept = dict(cache._aggregate_cache)
    assert len(kept) == len(paths)
    # The load that was skipped would have produced exactly what was kept.
    cache._aggregate_cache = {}
    assert cache._load_aggregate_cache(0) is True
    assert cache._aggregate_cache == kept
    assert _rows(ds) == [(f"img_{i}.png", (512, 768) if i % 2 else (768, 512)) for i in range(len(paths))]

    # Later epochs load once each, as before.
    cache.aggregate_loads.clear()
    ds.start_next_epoch()
    assert cache.aggregate_loads == [(1, True)]


def test_a_samples_group_still_reloads_after_its_reshuffle(tmp_path):
    cache_dir, _ = _build_cache(tmp_path, count=5, balancing_strategy="SAMPLES", balancing=2)
    ds, cache = _sourceless(cache_dir)

    ds.start_next_epoch()

    # The init load covers every row; the reshuffle then keeps 2 of the 5.
    assert cache.aggregate_loads == [(0, True), (0, True)]
    assert len(cache._aggregate_cache) == 2
    assert len(_rows(ds)) == 2


def test_a_load_that_read_cache_files_is_followed_by_the_reload(tmp_path):
    cache_dir, paths = _build_cache(tmp_path)
    # A cache from before crop_resolution was stamped on the variants: the
    # first load reads it from the cache files and stamps the index, and the
    # second then serves every row from those stamps.
    index_path = os.path.join(cache_dir, "cache.json")
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    for entry in index["entries"].values():
        for variant in entry["variants"].values():
            del variant["crop_resolution"]
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f)

    ds, cache = _sourceless(cache_dir)
    ds.start_next_epoch()

    assert cache.aggregate_loads == [(0, False), (0, True)]
    with open(index_path, encoding="utf-8") as f:
        stamped = json.load(f)
    assert all(
        variant.get("crop_resolution")
        for entry in stamped["entries"].values()
        for variant in entry["variants"].values()
    )
    assert len(_rows(ds)) == len(paths)


def test_a_start_at_another_variation_than_the_init_loaded_reloads(tmp_path):
    cache_dir, paths = _build_cache(tmp_path)
    _, cache = _sourceless(cache_dir)

    # Started directly: current_variation is still -1, so the init loads
    # variation 0 while the start asks for 1.
    cache.start(1)

    assert cache.aggregate_loads == [(0, True), (1, True)]
    assert len(cache._aggregate_cache) == len(paths)


def test_no_aggregate_names_means_no_aggregate_load(tmp_path):
    paths = [str(tmp_path / "a.txt")]
    with open(paths[0], "w", encoding="utf-8") as f:
        f.write("caption")

    class _TextSource(_ImageSource):
        def get_outputs(self) -> list[str]:
            return ["image_path", "latent_image", "concept"]

    def text_cache(sourceless: bool) -> _CountingCache:
        return _CountingCache(
            cache_dir=str(tmp_path / "cache" / "text"),
            split_names=["latent_image"],
            aggregate_names=[],
            variations_in_name="concept.image_variations",
            balancing_in_name="concept.balancing",
            balancing_strategy_in_name="concept.balancing_strategy",
            variations_group_in_name=["concept.path"],
            group_enabled_in_name="concept.enabled",
            modeltype="testmodel",
            source_path_in_name="image_path",
            sourceless=sourceless,
        )

    def dataset(modules: list) -> MGDS:
        return MGDS(
            device=torch.device("cpu"),
            concepts=[{"name": "concept-a", "path": "concept-a"}],
            settings={},
            definition=[modules, OutputPipelineModule(names=["latent_image"])],
            batch_size=1,
            state=PipelineState(),
            seed=42,
        )

    dataset([_TextSource(paths, "REPEATS", 1.0), text_cache(False)]).start_next_epoch()
    cache = text_cache(True)
    ds = dataset([PlaceholderModule(), cache])
    ds.start_next_epoch()

    assert cache.aggregate_loads == []
    assert len(list(ds)) == 1


# ---------------------------------------------------------------------------
# Cache file existence from one listing
# ---------------------------------------------------------------------------


def _entry_pt_names(cache_dir) -> list[str]:
    with open(os.path.join(cache_dir, "cache.json"), encoding="utf-8") as f:
        entries = json.load(f)["entries"]
    return [f"{next(iter(entry['variants'].values()))['cache_file']}_1.pt" for entry in entries.values()]


def _record_isfile(monkeypatch) -> list[str]:
    asked: list[str] = []
    real_isfile = os.path.isfile

    def isfile(path):
        asked.append(os.path.basename(str(path)))
        return real_isfile(path)

    monkeypatch.setattr(os.path, "isfile", isfile)
    return asked


def test_listed_cache_files_are_not_asked_of_the_filesystem(tmp_path, monkeypatch):
    cache_dir, paths = _build_cache(tmp_path)
    pt_names = _entry_pt_names(cache_dir)
    assert len(pt_names) == len(paths)
    ds, _ = _sourceless(cache_dir)
    asked = _record_isfile(monkeypatch)

    ds.start_next_epoch()

    assert not set(asked) & set(pt_names)
    assert len(_rows(ds)) == len(paths)


def test_a_missing_cache_file_is_refused_with_the_same_error(tmp_path, monkeypatch):
    cache_dir, _ = _build_cache(tmp_path)
    missing = _entry_pt_names(cache_dir)[1]
    os.remove(os.path.join(cache_dir, missing))
    ds, cache = _sourceless(cache_dir)
    asked = _record_isfile(monkeypatch)

    with pytest.raises(RuntimeError) as error:
        ds.start_next_epoch()

    expected_path = os.path.join(cache._real_cache_dir, missing)
    assert str(error.value) == f"Sourceless training: cache file '{expected_path}' is missing. Rebuild your cache."
    # Only the name the listing lacks falls back to the filesystem.
    assert missing in asked


def test_a_symlinked_cache_file_is_still_accepted(tmp_path):
    cache_dir, paths = _build_cache(tmp_path)
    name = _entry_pt_names(cache_dir)[2]
    link = os.path.join(cache_dir, name)
    target = link + ".real"
    os.replace(link, target)
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available here")

    ds, cache = _sourceless(cache_dir)
    # The listing holds regular files only, so this one is asked of the
    # filesystem, which follows the link exactly as before.
    assert name not in cache._scan_existing_pt_files()
    ds.start_next_epoch()

    assert len(_rows(ds)) == len(paths)


def test_an_unlistable_cache_dir_falls_back_to_the_filesystem(tmp_path, monkeypatch):
    cache_dir, paths = _build_cache(tmp_path)
    pt_names = _entry_pt_names(cache_dir)
    ds, cache = _sourceless(cache_dir)
    monkeypatch.setattr(cache, "_scan_existing_pt_files", set)
    asked = _record_isfile(monkeypatch)

    ds.start_next_epoch()

    assert set(pt_names) <= set(asked)
    assert len(_rows(ds)) == len(paths)
