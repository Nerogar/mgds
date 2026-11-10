"""Re-hashing after an mtime change: the files the validation loop would hash
one by one are hashed in a thread pool first, with the same digests and the
same decisions."""

import os
import sys
import threading
from pathlib import Path

from mgds.pipelineModules.SmartDiskCache import SmartDiskCache

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_smartcache import (  # noqa: E402
    _build_smart_pipeline,
    _create_source_file,
    _drain,
    _make_tensors,
    _read_cache_json,
)

LATER = 1_900_000_000


def _bare_cache(watched=("mask_path",)) -> SmartDiskCache:
    sdc = SmartDiskCache.__new__(SmartDiskCache)
    sdc.modeltype = "testmodel"
    sdc.tolerate_missing_source = False
    sdc.extra_watched_paths_in_names = list(watched)
    sdc._extra_paths_by_filepath = {}
    sdc._source_mtimes = {}
    sdc._index_lock = threading.Lock()
    sdc.cache_index = {"entries": {}}
    return sdc


def _entry(sdc: SmartDiskCache, path: str, *, mtime: float, sidecars: dict[str, float] | None = None) -> dict:
    sidecars = sidecars or {}
    return {
        "hash": sdc._hash_file(path),
        "mtime": mtime,
        "modeltype": "testmodel",
        "variants": {},
        "sidecar_mtimes": dict(sidecars),
        "sidecar_hashes": {sidecar: sdc._hash_file(sidecar) for sidecar in sidecars},
    }


def _count_hash_calls(monkeypatch) -> list[str]:
    calls: list[str] = []
    original = SmartDiskCache._hash_file

    def hash_file(self, filepath):
        calls.append(os.path.normpath(filepath))
        return original(self, filepath)

    monkeypatch.setattr(SmartDiskCache, "_hash_file", hash_file)
    return calls


def _record_statuses(monkeypatch) -> dict[str, str]:
    statuses: dict[str, str] = {}
    original = SmartDiskCache._validate_entry

    def validate(self, filepath, entry, requested_key, variations, current_mtime):
        status = original(self, filepath, entry, requested_key, variations, current_mtime)
        statuses[os.path.basename(filepath)] = status
        return status

    monkeypatch.setattr(SmartDiskCache, "_validate_entry", validate)
    return statuses


def _record_prehashed(monkeypatch) -> list[set[str]]:
    seen: list[set[str]] = []
    original = SmartDiskCache._prehash_changed_mtimes

    def prehash(self, filepaths):
        count = original(self, filepaths)
        seen.append({os.path.basename(path) for path in self._prehashed})
        return count

    monkeypatch.setattr(SmartDiskCache, "_prehash_changed_mtimes", prehash)
    return seen


@pytest.mark.parametrize("size", [0, 1, 65535, 65536, 65537, (1 << 20) - 1, 1 << 20, (1 << 20) + 1, 2_500_000])
def test_pool_digest_equals_the_serial_digest(tmp_path, size):
    sdc = _bare_cache(watched=())
    paths = [
        os.path.normpath(_create_source_file(tmp_path, name, (name.encode() * (size // 5 + 1))[:size]))
        for name in ("a.bin", "b.bin")
    ]
    for path in paths:
        sdc.cache_index["entries"][path] = _entry(sdc, path, mtime=1.0)
        sdc._source_mtimes[path] = 2.0

    assert sdc._prehash_changed_mtimes(paths) == 2
    assert sdc._prehashed == {path: sdc._hash_file(path) for path in paths}


def test_only_files_the_loop_would_hash_are_pre_hashed(tmp_path):
    sdc = _bare_cache()

    def source(name: str, *, drift: bool, sidecar_drift: bool | None = None, rewrite: bool = False) -> str:
        path = os.path.normpath(_create_source_file(tmp_path, f"{name}.bin", f"source {name}".encode()))
        sidecars = {}
        if sidecar_drift is not None:
            sidecar = os.path.normpath(_create_source_file(tmp_path, f"{name}-mask.bin", f"mask {name}".encode()))
            sidecars[sidecar] = 5.0
            sdc._source_mtimes[sidecar] = 6.0 if sidecar_drift else 5.0
        sdc.cache_index["entries"][path] = _entry(sdc, path, mtime=1.0, sidecars=sidecars)
        sdc._source_mtimes[path] = 2.0 if drift else 1.0
        if rewrite:
            with open(path, "wb") as f:
                f.write(b"different bytes")
        return path

    paths = [
        source("same", drift=False),
        source("touched", drift=True),
        source("touched_with_touched_mask", drift=True, sidecar_drift=True),
        source("same_with_touched_mask", drift=False, sidecar_drift=True),
        source("same_with_same_mask", drift=False, sidecar_drift=False),
        source("rewritten_with_touched_mask", drift=True, sidecar_drift=True, rewrite=True),
    ]
    not_indexed = os.path.normpath(_create_source_file(tmp_path, "new.bin", b"new"))
    sdc._source_mtimes[not_indexed] = 2.0
    other_model = source("other_model", drift=True)
    sdc.cache_index["entries"][other_model]["modeltype"] = "another"
    gone = source("gone", drift=True)
    os.remove(gone)
    unscanned = source("unscanned", drift=True, sidecar_drift=True)
    del sdc._source_mtimes[unscanned]

    sdc._prehash_changed_mtimes([*paths, paths[1], not_indexed, other_model, gone, unscanned])

    assert {os.path.basename(path) for path in sdc._prehashed} == {
        "touched.bin",
        "touched_with_touched_mask.bin",
        "touched_with_touched_mask-mask.bin",
        "same_with_touched_mask-mask.bin",
        # Its content changed, so the loop never reaches its sidecar.
        "rewritten_with_touched_mask.bin",
    }
    assert all(sdc._prehashed[path] == sdc._hash_file(path) for path in sdc._prehashed)


def test_a_single_changed_file_is_left_to_the_loop(tmp_path):
    sdc = _bare_cache(watched=())
    path = os.path.normpath(_create_source_file(tmp_path, "a.bin", b"a"))
    sdc.cache_index["entries"][path] = _entry(sdc, path, mtime=1.0)
    sdc._source_mtimes[path] = 2.0

    assert sdc._prehash_changed_mtimes([path]) == 0
    assert sdc._prehashed == {}
    assert sdc._validation_hash(path) == sdc._hash_file(path)


def test_validation_hash_prefers_the_pre_hashed_digest(tmp_path):
    sdc = _bare_cache()
    path = _create_source_file(tmp_path, "a.bin", b"a")
    sdc._prehashed = {path: "from-the-pool"}
    assert sdc._validation_hash(path) == "from-the-pool"
    sdc._prehashed = {}
    assert sdc._validation_hash(path) == sdc._hash_file(path)


def _mask_dataset(tmp_path, count: int = 6):
    src_dir = tmp_path / "sources"
    paths = [_create_source_file(src_dir, f"i{i}.bin", f"src_{i}".encode()) for i in range(count)]
    masks = [_create_source_file(src_dir, f"i{i}-masklabel.png", f"mask_{i}".encode()) for i in range(count)]

    def build():
        return _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": _make_tensors(count, seed=7), "image_path": paths, "mask_path": masks},
            dummy_length=count,
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
            extra_watched_paths_in_names=["mask_path"],
        )

    return paths, masks, build


def test_touching_every_file_revalidates_without_a_serial_hash(tmp_path, capsys, monkeypatch):
    paths, masks, build = _mask_dataset(tmp_path)
    ds, cache_dir, _ = build()
    _drain(ds)
    before = _read_cache_json(cache_dir)["entries"]
    for path in paths + masks:
        os.utime(path, (LATER, LATER))
    capsys.readouterr()

    ds, _, _ = build()
    hashed = _count_hash_calls(monkeypatch)
    statuses = _record_statuses(monkeypatch)
    prehashed = _record_prehashed(monkeypatch)
    _drain(ds)

    assert "Cached 0/6 files. 6 reused from cache." in capsys.readouterr().out
    assert set(statuses.values()) == {"valid"} and len(statuses) == 6
    assert prehashed == [{os.path.basename(path) for path in paths + masks}]
    assert hashed == []
    after = _read_cache_json(cache_dir)["entries"]
    for path, mask in zip(paths, masks, strict=True):
        key, mask_key = os.path.normpath(path), os.path.normpath(mask)
        assert after[key]["hash"] == before[key]["hash"]
        assert after[key]["sidecar_hashes"] == before[key]["sidecar_hashes"]
        assert after[key]["variants"] == before[key]["variants"]
        assert after[key]["mtime"] == LATER
        assert after[key]["sidecar_mtimes"] == {mask_key: LATER}


def test_decisions_match_the_serial_loop_on_a_mixed_change(tmp_path, capsys, monkeypatch):
    def run(root, *, pool: bool) -> tuple[dict[str, str], str]:
        paths, masks, build = _mask_dataset(root)
        ds, _, _ = build()
        _drain(ds)
        os.utime(paths[0], (LATER, LATER))  # touched
        with open(paths[1], "wb") as f:  # rewritten, same length
            f.write(b"SRC_1")
        os.utime(paths[1], (LATER, LATER))
        with open(paths[2], "wb") as f:  # rewritten, other length
            f.write(b"a longer source 2")
        os.utime(paths[2], (LATER, LATER))
        os.utime(masks[3], (LATER, LATER))  # mask touched
        with open(masks[4], "wb") as f:  # mask rewritten
            f.write(b"MASK_4")
        os.utime(masks[4], (LATER, LATER))
        os.utime(paths[5], (LATER, LATER))  # both touched
        os.utime(masks[5], (LATER, LATER))
        capsys.readouterr()

        with monkeypatch.context() as patch:
            if not pool:
                patch.setattr(SmartDiskCache, "_prehash_changed_mtimes", lambda self, filepaths: 0)
            statuses = _record_statuses(patch)
            ds, _, _ = build()
            _drain(ds)
        return statuses, capsys.readouterr().out.strip().splitlines()[-1]

    serial = run(tmp_path / "serial", pool=False)
    pooled = run(tmp_path / "pooled", pool=True)

    assert pooled == serial
    assert serial[0] == {
        "i0.bin": "valid",
        "i1.bin": "content_changed",
        "i2.bin": "content_changed",
        "i3.bin": "valid",
        "i4.bin": "content_changed",
        "i5.bin": "valid",
    }


def test_pre_hashed_digests_are_dropped_after_the_group(tmp_path):
    paths, masks, build = _mask_dataset(tmp_path)
    ds, _, _ = build()
    _drain(ds)
    for path in paths:
        os.utime(path, (LATER, LATER))

    ds, _, _ = build()
    ds.start_next_epoch()
    cache = next(module for module in ds.loading_pipeline.modules if isinstance(module, SmartDiskCache))
    assert cache._prehashed == {}
