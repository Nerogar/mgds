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


def _training_features(batch) -> torch.Tensor:
    parts = [
        batch["latent_image"].to(torch.float32),
        batch["latent_mask"].to(torch.float32),
        batch["latent_conditioning_image"].to(torch.float32),
        batch["latent_image_rejected"].to(torch.float32),
        batch["text_embedding"].to(torch.float32),
        torch.tensor(batch["original_resolution"], dtype=torch.float32) / 1000.0,
        torch.tensor(batch["crop_resolution"], dtype=torch.float32) / 1000.0,
        torch.tensor(batch["crop_offset"], dtype=torch.float32),
        torch.tensor([batch["concept"]["loss_weight"]], dtype=torch.float32),
    ]
    return torch.cat([part.flatten() for part in parts]).unsqueeze(0)


def _new_cpu_dummy_model() -> torch.nn.Module:
    torch.manual_seed(20240620)
    return torch.nn.Sequential(
        torch.nn.Linear(17, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 1),
    )


def _run_cpu_dummy_training(cache_root, image_paths=None, text_paths=None, *, sourceless=False):
    model = _new_cpu_dummy_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-4)
    ds = _build_dataset(cache_root, image_paths, text_paths, sourceless=sourceless, realistic=True)
    losses = []

    for _ in range(2):
        ds.start_next_epoch()
        for batch in ds:
            features = _training_features(batch)
            target = features.sum(dim=1, keepdim=True) * 0.01
            target = target + torch.tensor([[batch["concept"]["loss_weight"]]], dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(features), target)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().clone())

    final_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    return losses, final_state


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
            entry.pop("sourceless_rows", None)
        with open(cache_index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)


def _strip_embedded_sourceless_runtime_values(cache_root):
    for pt_path in cache_root.glob("*/*.pt"):
        cached = torch.load(pt_path, weights_only=False, map_location="cpu")
        if not isinstance(cached, dict):
            continue
        cached.pop("__sourceless_values", None)
        torch.save(cached, pt_path)


def _delete_source_tree(paths: list[str]) -> None:
    roots = {os.path.dirname(path) for path in paths}
    for root in roots:
        shutil.rmtree(root)
    assert all(not os.path.exists(path) for path in paths)


def _concept_names(rows):
    names = []

    def visit(value):
        if isinstance(value, dict) and "name" in value:
            names.append(value["name"])
        elif isinstance(value, tuple) and len(value) == 2 and value[0] == "name":
            names.append(value[1])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    for row in rows:
        visit(row)
    return names


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


def test_sourceless_uses_index_runtime_metadata_without_pt_rewrite(tmp_path):
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
    _strip_sourceless_runtime_values(cache_root)

    sourced_restart = _build_dataset(cache_root, image_paths, text_paths, realistic=True)
    expected_epoch0 = _drain_full_epoch(sourced_restart)
    expected_epoch1 = _drain_full_epoch(sourced_restart)

    _strip_embedded_sourceless_runtime_values(cache_root)
    _delete_source_tree(image_paths + text_paths)

    sourceless_restart = _build_dataset(cache_root, sourceless=True, realistic=True)
    assert _drain_full_epoch(sourceless_restart) == expected_epoch0
    assert _drain_full_epoch(sourceless_restart) == expected_epoch1


def test_sourceless_preserves_duplicate_source_rows(tmp_path):
    image_path = _write_file(tmp_path / "src" / "duplicate.png", b"same image")
    text_path = _write_file(tmp_path / "src" / "duplicate.txt", b"same text")
    image_paths = [image_path, image_path]
    text_paths = [text_path, text_path]
    cache_root = tmp_path / "cache"

    sourced_builder = _build_dataset(cache_root, image_paths, text_paths, realistic=True)
    _drain_full_epoch(sourced_builder)

    sourced_restart = _build_dataset(cache_root, image_paths, text_paths, realistic=True)
    expected_epoch0 = _drain_full_epoch(sourced_restart)
    expected_epoch1 = _drain_full_epoch(sourced_restart)

    assert _concept_names(expected_epoch0) == ["concept-0", "concept-1"]

    _delete_source_tree(image_paths + text_paths)

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


def test_sourceless_runs_when_original_source_files_are_missing(tmp_path):
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

    sourced_restart = _build_dataset(cache_root, image_paths, text_paths, realistic=True)
    expected_epoch0 = _drain_full_epoch(sourced_restart)
    expected_epoch1 = _drain_full_epoch(sourced_restart)

    _delete_source_tree(image_paths + text_paths)

    sourceless_restart = _build_dataset(cache_root, sourceless=True, realistic=True)
    assert _drain_full_epoch(sourceless_restart) == expected_epoch0
    assert _drain_full_epoch(sourceless_restart) == expected_epoch1


def test_sourceless_cpu_dummy_training_matches_sourced_training(tmp_path):
    image_paths = [
        _write_file(tmp_path / "src" / "b_image.png", b"image b"),
        _write_file(tmp_path / "src" / "a_image.png", b"image a"),
    ]
    text_paths = [
        _write_file(tmp_path / "src" / "b_image.txt", b"text b"),
        _write_file(tmp_path / "src" / "a_image.txt", b"text a"),
    ]
    cache_root = tmp_path / "cache"

    cache_builder = _build_dataset(cache_root, image_paths, text_paths, realistic=True)
    _drain_full_epoch(cache_builder)
    _drain_full_epoch(cache_builder)

    sourced_losses, sourced_state = _run_cpu_dummy_training(
        cache_root,
        image_paths,
        text_paths,
        sourceless=False,
    )

    _delete_source_tree(image_paths + text_paths)

    sourceless_losses, sourceless_state = _run_cpu_dummy_training(
        cache_root,
        sourceless=True,
    )

    assert len(sourceless_losses) == len(sourced_losses)
    for sourceless_loss, sourced_loss in zip(sourceless_losses, sourced_losses, strict=True):
        torch.testing.assert_close(sourceless_loss, sourced_loss, rtol=0.0, atol=0.0)

    assert sourceless_state.keys() == sourced_state.keys()
    for key in sourced_state:
        torch.testing.assert_close(sourceless_state[key], sourced_state[key], rtol=0.0, atol=0.0)


def test_sourceless_raises_when_entry_metadata_missing(tmp_path):
    import json

    import pytest

    image_paths = [
        _write_file(tmp_path / "src" / "a_image.png", b"image a"),
        _write_file(tmp_path / "src" / "b_image.png", b"image b"),
    ]
    text_paths = [
        _write_file(tmp_path / "src" / "a_image.txt", b"text a"),
        _write_file(tmp_path / "src" / "b_image.txt", b"text b"),
    ]
    cache_root = tmp_path / "cache"
    _drain_full_epoch(_build_dataset(cache_root, image_paths, text_paths, realistic=True))

    # Simulate a cache built before the metadata bake (or under trust mode):
    # strip one image entry's sourceless metadata entirely.
    image_index = cache_root / "image" / "cache.json"
    index = json.loads(image_index.read_text(encoding="utf-8"))
    victim = next(iter(index["entries"]))
    index["entries"][victim].pop("sourceless", None)
    index["entries"][victim].pop("sourceless_rows", None)
    image_index.write_text(json.dumps(index), encoding="utf-8")

    _delete_source_tree(image_paths + text_paths)

    with pytest.raises(RuntimeError, match="missing sourceless metadata"):
        _drain_full_epoch(_build_dataset(cache_root, sourceless=True, realistic=True))
