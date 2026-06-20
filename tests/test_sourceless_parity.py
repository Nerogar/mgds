import json
import os
import shutil

from mgds.MGDS import MGDS
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModules.AspectBatchSorting import AspectBatchSorting
from mgds.pipelineModules.PlaceholderModule import PlaceholderModule
from mgds.pipelineModules.SmartDiskCache import CACHE_VERSION, NO_RESOLUTION_KEY, SmartDiskCache
from mgds.pipelineModules.VariationSorting import VariationSorting
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
        return [
            "image_path",
            "sample_prompt_path",
            "latent_image",
            "latent_mask",
            "latent_conditioning_image",
            "latent_image_rejected",
            "original_resolution",
            "crop_resolution",
            "crop_offset",
            "text_embedding",
            "prompt",
            "prompt_1",
            "prompt_2",
            "concept",
        ]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        crop_resolutions = [(512, 768), (768, 512)]
        return {
            "image_path": self.image_paths[index],
            "sample_prompt_path": self.text_paths[index],
            "latent_image": torch.tensor([index, variation], dtype=torch.float32),
            "latent_mask": torch.tensor([index + 10, variation], dtype=torch.float32),
            "latent_conditioning_image": torch.tensor([index + 20, variation], dtype=torch.float32),
            "latent_image_rejected": torch.tensor([index + 30, variation], dtype=torch.float32),
            "original_resolution": (600 + index, 800 - index),
            "crop_resolution": crop_resolutions[(index + variation) % len(crop_resolutions)],
            "crop_offset": (variation, index),
            "text_embedding": torch.tensor([index + 100, variation], dtype=torch.float32),
            "prompt": f"prompt-{index}-variation-{variation}",
            "prompt_1": f"clip-l-{index}-variation-{variation}",
            "prompt_2": f"clip-g-{index}-variation-{variation}",
            "concept": {
                "name": f"concept-{index}",
                "path": f"concept-{index}",
                "seed": 123 + index,
                "enabled": True,
                "image_variations": 2,
                "text_variations": 2,
                "balancing": 1.0,
                "balancing_strategy": "REPEATS",
                "loss_weight": 0.25 + index,
                "include_subdirectories": False,
                "dpo_chosen_pattern": "chosen/{}",
                "dpo_rejected_pattern": "rejected/{}",
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
        variations_group_in_name=[
            "concept.path",
            "concept.seed",
            "concept.include_subdirectories",
            "concept.image",
            "concept.dpo_chosen_pattern",
            "concept.dpo_rejected_pattern",
        ],
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
        variations_group_in_name=[
            "concept.path",
            "concept.seed",
            "concept.include_subdirectories",
            "concept.text",
            "concept.dpo_chosen_pattern",
            "concept.dpo_rejected_pattern",
        ],
        group_enabled_in_name="concept.enabled",
        modeltype="testmodel",
        source_path_in_name="sample_prompt_path",
        sourceless=sourceless,
    )
    return image_cache, text_cache


def _realistic_cache_modules(cache_root, *, sourceless: bool):
    image_cache = SmartDiskCache(
        cache_dir=str(cache_root / "image"),
        split_names=[
            "latent_image",
            "latent_mask",
            "latent_conditioning_image",
            "latent_image_rejected",
            "original_resolution",
            "crop_offset",
        ],
        aggregate_names=["crop_resolution", "image_path"],
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
        content_key_in_name="prompt",
    )
    variation_sorting = VariationSorting(
        names=["prompt", "prompt_1", "prompt_2", "concept"],
        balancing_in_name="concept.balancing",
        balancing_strategy_in_name="concept.balancing_strategy",
        variations_group_in_name=[
            "concept.path",
            "concept.seed",
            "concept.include_subdirectories",
            "concept.text",
            "concept.dpo_chosen_pattern",
            "concept.dpo_rejected_pattern",
        ],
        group_enabled_in_name="concept.enabled",
    )
    return image_cache, text_cache, variation_sorting


def _build_dataset(cache_root, image_paths=None, text_paths=None, *, sourceless=False, realistic=False):
    modules = []
    if not sourceless:
        modules.append(PairedSource(image_paths, text_paths))
    elif realistic:
        modules.append(PlaceholderModule())
    modules.extend(
        _realistic_cache_modules(cache_root, sourceless=sourceless)
        if realistic
        else _cache_modules(cache_root, sourceless=sourceless)
    )
    output_names = ["image_path", "latent_image", "text_embedding"]
    if realistic:
        output_names.extend(
            [
                "original_resolution",
                "crop_resolution",
                "crop_offset",
                "latent_mask",
                "latent_conditioning_image",
                "latent_image_rejected",
                "prompt",
                "prompt_1",
                "prompt_2",
                "concept",
            ]
        )
        modules.append(
            AspectBatchSorting(
                resolution_in_name="crop_resolution",
                names=output_names + ["concept"],
                batch_size=1,
            )
        )
    output = OutputPipelineModule(names=output_names)
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


def _freeze(value):
    if torch.is_tensor(value):
        return tuple(value.squeeze(0).tolist())
    if isinstance(value, dict):
        return tuple((key, _freeze(value[key])) for key in sorted(value))
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _drain_full_epoch(ds):
    ds.start_next_epoch()
    return [
        tuple((key, _freeze(value)) for key, value in sorted(batch.items()))
        for batch in ds
    ]


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


def _strip_sourceless_runtime_values(cache_root):
    for pt_path in cache_root.glob("*/*.pt"):
        cached = torch.load(pt_path, weights_only=False, map_location="cpu")
        if not isinstance(cached, dict):
            continue
        cached.pop("__sourceless_values", None)
        torch.save(cached, pt_path)
    for cache_index_path in cache_root.glob("*/cache.json"):
        with open(cache_index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        for entry in index.get("entries", {}).values():
            entry.pop("sourceless_runtime_values", None)
        with open(cache_index_path, "w", encoding="utf-8") as f:
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


def test_sourceless_matches_sourced_runtime_metadata_and_linked_latents(tmp_path):
    image_paths = [
        _write_file(tmp_path / "src" / "b_image.png", b"image b"),
        _write_file(tmp_path / "src" / "a_image.png", b"image a"),
    ]
    text_paths = [
        _write_file(tmp_path / "src" / "b_image.txt", b"text b"),
        _write_file(tmp_path / "src" / "a_image.txt", b"text a"),
    ]
    cache_root = tmp_path / "cache"

    sourced_builder = _build_dataset(cache_root, image_paths, text_paths, realistic=True)
    _drain_full_epoch(sourced_builder)
    _drain_full_epoch(sourced_builder)
    _strip_sourceless_runtime_values(cache_root)

    sourced_restart = _build_dataset(cache_root, image_paths, text_paths, realistic=True)
    expected_epoch0 = _drain_full_epoch(sourced_restart)
    expected_epoch1 = _drain_full_epoch(sourced_restart)

    sourceless_restart = _build_dataset(cache_root, sourceless=True, realistic=True)
    assert _drain_full_epoch(sourceless_restart) == expected_epoch0
    assert _drain_full_epoch(sourceless_restart) == expected_epoch1


def test_sourceless_preserves_row_metadata_when_text_payloads_dedup(tmp_path):
    image_paths = [
        _write_file(tmp_path / "src" / "left.png", b"image left"),
        _write_file(tmp_path / "src" / "right.png", b"image right"),
    ]
    text_paths = [
        _write_file(tmp_path / "src" / "left.txt", b"same caption"),
        _write_file(tmp_path / "src" / "right.txt", b"same caption"),
    ]
    cache_root = tmp_path / "cache"

    sourced_builder = _build_dataset(cache_root, image_paths, text_paths, realistic=True)
    _drain_full_epoch(sourced_builder)

    sourced_restart = _build_dataset(cache_root, image_paths, text_paths, realistic=True)
    expected_epoch0 = _drain_full_epoch(sourced_restart)
    assert len({row[0][1] for row in expected_epoch0}) == 2

    sourceless_restart = _build_dataset(cache_root, sourceless=True, realistic=True)
    assert _drain_full_epoch(sourceless_restart) == expected_epoch0
