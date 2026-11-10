"""Re-hashing after an mtime change: the files the validation loop would hash
one by one are hashed in a thread pool first, with the same digests and the
same decisions, and a source whose length differs from the recorded one is
not hashed at all."""

import json
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


# ---------------------------------------------------------------------------
# Recorded source size: a different length settles "changed" without a hash
# ---------------------------------------------------------------------------


def _write_cache_json(cache_dir: str, index: dict) -> None:
    with open(os.path.join(cache_dir, "cache.json"), "w", encoding="utf-8") as f:
        json.dump(index, f)


def _rewrite(path: str, content: bytes, mtime: int = LATER) -> None:
    with open(path, "wb") as f:
        f.write(content)
    os.utime(path, (mtime, mtime))


def test_new_entries_record_the_size_of_what_was_hashed(tmp_path):
    paths, _masks, build = _mask_dataset(tmp_path)
    ds, cache_dir, _ = build()
    _drain(ds)

    entries = _read_cache_json(cache_dir)["entries"]
    for path in paths:
        entry = entries[os.path.normpath(path)]
        assert entry["size"] == os.path.getsize(path)
        assert entry["size_hash"] == entry["hash"]


def test_a_different_length_is_changed_without_a_validation_hash(tmp_path, capsys, monkeypatch):
    paths, _masks, build = _mask_dataset(tmp_path)
    ds, cache_dir, _ = build()
    _drain(ds)
    _rewrite(paths[1], b"a source of another length")
    _rewrite(paths[2], b"SRC_2")  # same length as before
    capsys.readouterr()

    ds, _, _ = build()
    hashed = _count_hash_calls(monkeypatch)
    statuses = _record_statuses(monkeypatch)
    _drain(ds)

    assert "Cached 2/6 files. 4 reused from cache." in capsys.readouterr().out
    assert statuses["i1.bin"] == "content_changed" and statuses["i2.bin"] == "content_changed"
    # i1: hashed once, by its rebuild. i2: by validation (same length) and by its rebuild.
    assert hashed.count(os.path.normpath(paths[1])) == 1
    assert hashed.count(os.path.normpath(paths[2])) == 2
    entry = _read_cache_json(cache_dir)["entries"][os.path.normpath(paths[1])]
    assert entry["size"] == len(b"a source of another length")
    assert entry["size_hash"] == entry["hash"]


def test_an_entry_without_a_size_is_hashed_as_before(tmp_path, capsys, monkeypatch):
    paths, _masks, build = _mask_dataset(tmp_path)
    ds, cache_dir, _ = build()
    _drain(ds)
    index = _read_cache_json(cache_dir)
    for entry in index["entries"].values():
        del entry["size"], entry["size_hash"]
    _write_cache_json(cache_dir, index)
    _rewrite(paths[1], b"a source of another length")
    os.utime(paths[4], (LATER, LATER))
    capsys.readouterr()

    ds, _, _ = build()
    prehashed = _record_prehashed(monkeypatch)
    statuses = _record_statuses(monkeypatch)
    _drain(ds)

    assert statuses["i1.bin"] == "content_changed" and statuses["i4.bin"] == "valid"
    # No recorded size: the rewritten source is read to find out, as before.
    assert "i1.bin" in prehashed[0]
    entries = _read_cache_json(cache_dir)["entries"]
    # Validated by a pool hash: the touched entry now carries its size.
    touched = entries[os.path.normpath(paths[4])]
    assert touched["size"] == os.path.getsize(paths[4]) and touched["size_hash"] == touched["hash"]
    # Untouched legacy entries stay as they were.
    assert "size" not in entries[os.path.normpath(paths[0])]


def test_a_size_left_behind_by_an_older_writer_is_ignored(tmp_path, capsys, monkeypatch):
    paths, _masks, build = _mask_dataset(tmp_path)
    ds, cache_dir, _ = build()
    _drain(ds)
    # An older mgds that rebuilt this entry replaced its hash and knew nothing
    # of the size next to it: the size no longer belongs to the hash.
    index = _read_cache_json(cache_dir)
    key = os.path.normpath(paths[3])
    index["entries"][key]["size"] += 100
    index["entries"][key]["size_hash"] = "0" * 16
    _write_cache_json(cache_dir, index)
    os.utime(paths[3], (LATER, LATER))
    os.utime(paths[0], (LATER, LATER))
    capsys.readouterr()

    ds, _, _ = build()
    statuses = _record_statuses(monkeypatch)
    _drain(ds)

    assert statuses["i3.bin"] == "valid"
    assert "Cached 0/6 files. 6 reused from cache." in capsys.readouterr().out
    entry = _read_cache_json(cache_dir)["entries"][key]
    assert entry["size"] == os.path.getsize(paths[3]) and entry["size_hash"] == entry["hash"]


def test_the_scanned_size_is_only_a_hint(tmp_path, monkeypatch):
    sdc = _bare_cache(watched=())
    path = os.path.normpath(_create_source_file(tmp_path, "a.bin", b"twelve bytes"))
    entry = _entry(sdc, path, mtime=1.0)
    SmartDiskCache._stamp_hashed_size(entry, entry["hash"], 12)

    stats: list[str] = []
    real_stat = os.stat

    def stat(target, *args, **kwargs):
        stats.append(str(target))
        return real_stat(target, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat)

    # No scan, or a scan that agrees: nothing is ruled out and nothing is asked.
    assert sdc._size_rules_out_recorded_content(path, entry) is False
    sdc._source_sizes = {path: 12}
    assert sdc._size_rules_out_recorded_content(path, entry) is False
    assert stats == []

    # A directory entry lagging behind the file (NTFS hard links): the scan
    # disagrees, the file itself does not, so the content is still hashed.
    sdc._source_sizes = {path: 5}
    assert sdc._size_rules_out_recorded_content(path, entry) is False
    assert stats == [path]

    # The file itself has another length: ruled out.
    monkeypatch.undo()
    with open(path, "wb") as f:
        f.write(b"short")
    assert sdc._size_rules_out_recorded_content(path, entry) is True
    # ... unless it cannot be read at all: then the hash fails on it as before.
    os.remove(path)
    assert sdc._size_rules_out_recorded_content(path, entry) is False


def test_recorded_size_needs_a_matching_hash_and_a_plain_int():
    entry = {"hash": "abc", "size": 10, "size_hash": "abc"}
    assert SmartDiskCache._recorded_size(entry) == 10
    assert SmartDiskCache._recorded_size({**entry, "size_hash": "other"}) is None
    assert SmartDiskCache._recorded_size({"hash": "abc", "size": 10}) is None
    assert SmartDiskCache._recorded_size({**entry, "size": "10"}) is None
    assert SmartDiskCache._recorded_size({**entry, "size": True}) is None
    assert SmartDiskCache._recorded_size({**entry, "size": -1}) is None
    assert SmartDiskCache._recorded_size({"size": 10, "size_hash": None}) is None

    SmartDiskCache._stamp_hashed_size(entry, "def", None)
    assert "size" not in entry and "size_hash" not in entry


def test_a_file_that_changed_while_it_was_hashed_gets_no_size(tmp_path):
    path = _create_source_file(tmp_path, "a.bin", b"first")
    before = os.stat(path)
    assert SmartDiskCache._unchanged_size(path, before) == 5
    _rewrite(path, b"second version")
    assert SmartDiskCache._unchanged_size(path, before) is None
    os.remove(path)
    assert SmartDiskCache._unchanged_size(path, before) is None
