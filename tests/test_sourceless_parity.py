import json
import os
import shutil

from mgds.MGDS import MGDS
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModules.SmartDiskCache import CACHE_VERSION, NO_RESOLUTION_KEY, SmartDiskCache
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch


class PairedSource(PipelineModule, RandomAccessPipelineModule):
    def __init__(self, image_paths: list[str], text_paths: list[str]):
        super().__init__()
        self.image_paths = image_paths
        self.text_paths = text_paths

    def length(self) -> int:
        return len(self.image_paths)

    def get_inputs(self) -> list[str]:
        return []

    def get_outputs(self) -> list[str]:
        return ["image_path", "sample_prompt_path", "latent_image", "text_embedding", "concept"]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return {
            "image_path": self.image_paths[index],
            "sample_prompt_path": self.text_paths[index],
            "latent_image": torch.tensor([index, variation], dtype=torch.float32),
            "text_embedding": torch.tensor([index + 100, variation], dtype=torch.float32),
            "concept": {
                "name": "concept-a",
                "path": "concept-a",
                "seed": 123,
                "enabled": True,
                "image_variations": 2,
                "text_variations": 2,
                "balancing": 1.0,
                "balancing_strategy": "REPEATS",
                "image": {"resolution": "test"},
                "text": {"source": "sample"},
            },
        }


def _write_file(path, data: bytes = b"x") -> str:
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return str(path)


def _cache_modules(cache_root, *, sourceless: bool):
    image_cache = SmartDiskCache(
        cache_dir=str(cache_root / "image"),
        split_names=["latent_image"],
        aggregate_names=["image_path"],
        variations_in_name="concept.image_variations",
        balancing_in_name="concept.balancing",
        balancing_strategy_in_name="concept.balancing_strategy",
        variations_group_in_name=["concept.path", "concept.seed", "concept.image"],
        group_enabled_in_name="concept.enabled",
        modeltype="testmodel",
        source_path_in_name="image_path",
        sourceless=sourceless,
    )
    text_cache = SmartDiskCache(
        cache_dir=str(cache_root / "text"),
        split_names=["text_embedding"],
        aggregate_names=[],
        variations_in_name="concept.text_variations",
        balancing_in_name="concept.balancing",
        balancing_strategy_in_name="concept.balancing_strategy",
        variations_group_in_name=["concept.path", "concept.seed", "concept.text"],
        group_enabled_in_name="concept.enabled",
        modeltype="testmodel",
        source_path_in_name="sample_prompt_path",
        sourceless=sourceless,
    )
    return image_cache, text_cache


def _build_dataset(cache_root, image_paths=None, text_paths=None, *, sourceless=False):
    modules = []
    if not sourceless:
        modules.append(PairedSource(image_paths, text_paths))
    modules.extend(_cache_modules(cache_root, sourceless=sourceless))
    output = OutputPipelineModule(names=["image_path", "latent_image", "text_embedding"])
    return MGDS(
        device=torch.device("cpu"),
        concepts=[{"name": "concept-a", "path": "concept-a"}],
        settings={},
        definition=[modules, output],
        batch_size=1,
        state=PipelineState(),
        seed=42,
    )


def _drain_epoch(ds):
    ds.start_next_epoch()
    rows = []
    for batch in ds:
        image_path = batch["image_path"]
        rows.append(
            (
                image_path,
                tuple(batch["latent_image"].squeeze(0).tolist()),
                tuple(batch["text_embedding"].squeeze(0).tolist()),
            )
        )
    return rows


def _add_stale_text_cache_entry(cache_root, stale_path: str):
    text_cache_dir = cache_root / "text"
    with open(text_cache_dir / "cache.json", "r", encoding="utf-8") as f:
        index = json.load(f)

    first_entry = next(iter(index["entries"].values()))
    source_cache_file = first_entry["variants"][NO_RESOLUTION_KEY]["cache_file"]
    stale_cache_file = "000_stale_caption"
    shutil.copy2(text_cache_dir / f"{source_cache_file}_1.pt", text_cache_dir / f"{stale_cache_file}_1.pt")
    shutil.copy2(text_cache_dir / f"{source_cache_file}_2.pt", text_cache_dir / f"{stale_cache_file}_2.pt")
    index["entries"][os.path.normpath(stale_path)] = {
        "filename": os.path.basename(stale_path),
        "hash": "stale",
        "mtime": 1.0,
        "modeltype": "testmodel",
        "variants": {NO_RESOLUTION_KEY: {"cache_file": stale_cache_file, "schema_keys": ["text_embedding"]}},
        "cache_version": CACHE_VERSION,
        "sidecar_mtimes": {},
        "sidecar_hashes": {},
    }
    with open(text_cache_dir / "cache.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def test_sourceless_matches_sourced_pairing_and_variations_with_stale_text_entries(tmp_path):
    image_paths = [
        _write_file(tmp_path / "src" / "b_image.png", b"image b"),
        _write_file(tmp_path / "src" / "a_image.png", b"image a"),
    ]
    text_paths = [
        _write_file(tmp_path / "src" / "b_image.txt", b"text b"),
        _write_file(tmp_path / "src" / "a_image.txt", b"text a"),
    ]
    cache_root = tmp_path / "cache"

    sourced_builder = _build_dataset(cache_root, image_paths, text_paths, sourceless=False)
    sourced_epoch0 = _drain_epoch(sourced_builder)
    sourced_epoch1 = _drain_epoch(sourced_builder)
    assert sourced_epoch0 != sourced_epoch1

    _add_stale_text_cache_entry(cache_root, str(tmp_path / "src" / "000_stale.txt"))

    sourced_restart = _build_dataset(cache_root, image_paths, text_paths, sourceless=False)
    expected_epoch0 = _drain_epoch(sourced_restart)
    expected_epoch1 = _drain_epoch(sourced_restart)

    sourceless_restart = _build_dataset(cache_root, sourceless=True)
    assert _drain_epoch(sourceless_restart) == expected_epoch0
    assert _drain_epoch(sourceless_restart) == expected_epoch1
