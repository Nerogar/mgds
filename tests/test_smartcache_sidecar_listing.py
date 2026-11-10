"""Optional sidecars that are not on disk are ruled out from one directory
listing instead of a failing stat per file, and everything a listing cannot
prove is still asked of the filesystem."""

import os
import sys
import threading
from pathlib import Path

from mgds.PipelineModule import PipelineState
from mgds.pipelineModules.SmartDiskCache import (
    SmartDiskCache,
    _list_dir_keys,
    _listing_key,
    _listing_rules_out,
)

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_smartcache import (  # noqa: E402
    _adaptive_dataset,
    _build_adaptive_pipeline,
    _build_smart_pipeline,
    _create_source_file,
    _drain,
    _make_tensors,
    _read_cache_json,
)


def _bare_cache(tmp_path, watched=("mask_path",)) -> SmartDiskCache:
    sdc = SmartDiskCache.__new__(SmartDiskCache)
    sdc.cache_dir = str(tmp_path / "cache")
    sdc._real_cache_dir = os.path.realpath(sdc.cache_dir)
    sdc.extra_watched_paths_in_names = list(watched)
    sdc._extra_paths_by_filepath = {}
    sdc._source_mtimes = {}
    sdc._index_lock = threading.Lock()
    sdc._state = PipelineState()
    return sdc


def _count_calls(monkeypatch, module, name: str) -> list[str]:
    calls: list[str] = []
    real = getattr(module, name)

    def wrapper(path, *args, **kwargs):
        calls.append(str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(module, name, wrapper)
    return calls


# ---------------------------------------------------------------------------
# What a listing can and cannot prove
# ---------------------------------------------------------------------------


def test_listing_key_folds_case_and_what_filesystems_treat_as_ascii():
    assert _listing_key("Img_01-MaskLabel.PNG") == "img_01-masklabel.png"
    # Names a filesystem may match against an ASCII spelling fold onto it.
    assert _listing_key("ımage.png") == "image.png"  # dotless i upper-cases to I
    assert _listing_key("maſk.png") == "mask.png"  # long s upper-cases to S
    assert _listing_key("K.png") == "k.png"  # Kelvin sign decomposes to K
    # Two spellings of one non-ASCII name share a key (final sigma, case).
    assert _listing_key("ΑΣ.png") == _listing_key("ας.png")


def test_a_plain_name_missing_from_the_listing_is_ruled_out():
    keys = {"a.png", "b-masklabel.png"}
    assert _listing_rules_out("a-masklabel.png", keys) is True
    assert _listing_rules_out("b-masklabel.png", keys) is False
    assert _listing_rules_out("B-MaskLabel.PNG", keys) is False  # case-insensitive filesystems


@pytest.mark.parametrize(
    "name",
    [
        "",
        "café-masklabel.png",  # not ASCII: folding rules differ per filesystem
        "a-masklabel.png.",  # Windows strips a trailing dot
        "a-masklabel.png ",  # ... and a trailing space
        "A-MASK~1.PNG",  # 8.3 short name
        "a.png:mask",  # alternate data stream
        "nul",
        "NUL.png",
        "con.txt",
        "com1",
        "lpt9.png",
    ],
)
def test_names_a_listing_cannot_vouch_for_are_never_ruled_out(name):
    assert _listing_rules_out(name, {"unrelated.png"}) is False


def test_an_unlistable_directory_rules_nothing_out(tmp_path):
    assert _list_dir_keys(str(tmp_path / "missing")) is None
    assert _listing_rules_out("a-masklabel.png", None) is False


def test_a_folded_on_disk_name_keeps_the_ascii_lookup_on_the_filesystem(tmp_path):
    (tmp_path / "ımage-masklabel.png").write_bytes(b"x")
    keys = _list_dir_keys(str(tmp_path))
    assert _listing_rules_out("image-masklabel.png", keys) is False
    assert _listing_rules_out("other-masklabel.png", keys) is True


# ---------------------------------------------------------------------------
# The fast-path guard's lookup
# ---------------------------------------------------------------------------


def test_sidecar_exists_agrees_with_the_filesystem(tmp_path, monkeypatch):
    present = _create_source_file(tmp_path, "a-masklabel.png", b"mask")
    os.mkdir(tmp_path / "dir-masklabel.png")
    paths = [
        present,
        str(tmp_path / "b-masklabel.png"),
        str(tmp_path / "A-MASKLABEL.PNG"),
        str(tmp_path / "dir-masklabel.png"),
        str(tmp_path / "café-masklabel.png"),
        str(tmp_path / "b-masklabel.png."),
        str(tmp_path / "no_such_dir" / "c-masklabel.png"),
        "bare-masklabel.png",
    ]
    expected = [os.path.exists(path) for path in paths]
    scans = _count_calls(monkeypatch, os, "scandir")
    asked = _count_calls(monkeypatch, os.path, "exists")
    listings: dict = {}

    assert [SmartDiskCache._sidecar_exists(path, listings) for path in paths] == expected

    # One listing per directory, however many names are looked up in it.
    assert sorted(scans) == sorted({os.path.dirname(path) for path in paths})
    # The plain absent name cost no stat; the rest were asked as before.
    assert str(tmp_path / "b-masklabel.png") not in asked
    assert set(asked) == set(paths) - {str(tmp_path / "b-masklabel.png")}


def test_a_broken_symlink_is_still_reported_absent(tmp_path):
    link = tmp_path / "a-masklabel.png"
    try:
        os.symlink(tmp_path / "gone.png", link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available here")
    # Listed, but os.path.exists follows the link: the listing never says "present".
    assert SmartDiskCache._sidecar_exists(str(link), {}) is False


def test_adaptive_fast_path_does_not_stat_an_absent_optional_sidecar(tmp_path, capsys, monkeypatch):
    sources = _adaptive_dataset(tmp_path, "dpo")
    absent = os.path.join(str(tmp_path / "src_dpo_rejected"), "never_made.bin")
    sources["rejected_paths"][0] = absent
    ds1, _cache1, _ = _build_adaptive_pipeline(tmp_path, sources, role="dpo")
    _drain(ds1)
    capsys.readouterr()

    ds2, _cache2, _ = _build_adaptive_pipeline(tmp_path, sources, role="dpo")
    asked = _count_calls(monkeypatch, os.path, "exists")
    _drain(ds2)

    assert "Fast validation passed" in capsys.readouterr().out
    assert os.path.normpath(absent) not in {os.path.normpath(path) for path in asked}


# ---------------------------------------------------------------------------
# Full validation's lookup (_check_sidecars after the bulk scan)
# ---------------------------------------------------------------------------


def test_bulk_scan_keeps_listings_only_when_sidecars_are_watched(tmp_path):
    a = os.path.normpath(_create_source_file(tmp_path, "a.png", b"a"))
    _create_source_file(tmp_path, "Unrelated.TXT", b"u")
    missing = os.path.normpath(str(tmp_path / "a-masklabel.png"))

    watching = _bare_cache(tmp_path)
    mtimes = watching._bulk_stat_source_files({a, missing})
    assert mtimes == {a: os.path.getmtime(a)}
    assert watching._scanned_dir_keys == {os.path.dirname(a): {"a.png", "unrelated.txt"}}

    plain = _bare_cache(tmp_path, watched=())
    assert plain._bulk_stat_source_files({a, missing}) == mtimes
    assert plain._scanned_dir_keys == {}


def test_bulk_scan_of_an_unlistable_directory_leaves_no_listing(tmp_path):
    sdc = _bare_cache(tmp_path)
    assert sdc._bulk_stat_source_files({str(tmp_path / "missing_dir" / "a.png")}) == {}
    assert sdc._scanned_dir_keys == {}
    assert sdc._scan_rules_out(str(tmp_path / "missing_dir" / "a-masklabel.png")) is False


def test_check_sidecars_skips_the_stat_of_a_sidecar_the_scan_rules_out(tmp_path, monkeypatch):
    primary = os.path.normpath(_create_source_file(tmp_path, "a.png", b"a"))
    absent = os.path.normpath(str(tmp_path / "a-masklabel.png"))
    sdc = _bare_cache(tmp_path)
    sdc._extra_paths_by_filepath[primary] = {"mask_path": absent}
    sdc._source_mtimes = sdc._bulk_stat_source_files({primary, absent})
    stats = _count_calls(monkeypatch, os.path, "getmtime")

    # Recorded as absent, still absent: valid.
    assert sdc._check_sidecars(primary, {"sidecar_mtimes": {}, "sidecar_hashes": {}}) is True
    # Recorded as present, now gone: rebuild.
    recorded = {"sidecar_mtimes": {absent: 100.0}, "sidecar_hashes": {absent: "deadbeef"}}
    assert sdc._check_sidecars(primary, recorded) is False
    assert stats == []


def test_check_sidecars_asks_the_filesystem_without_a_fresh_listing(tmp_path, monkeypatch):
    primary = os.path.normpath(_create_source_file(tmp_path, "a.png", b"a"))
    mask = os.path.normpath(str(tmp_path / "a-masklabel.png"))
    sdc = _bare_cache(tmp_path)
    sdc._extra_paths_by_filepath[primary] = {"mask_path": mask}
    sdc._source_mtimes = sdc._bulk_stat_source_files({primary, mask})
    entry = {"sidecar_mtimes": {}, "sidecar_hashes": {}}

    # Created after the scan: the listing still rules it out for this pass...
    _create_source_file(tmp_path, "a-masklabel.png", b"late mask")
    assert sdc._check_sidecars(primary, entry) is True
    # ... and once the listing is dropped (a build started, or the refresh
    # ended) the lookup is the per-file stat again, which sees it.
    sdc._scanned_dir_keys = None
    stats = _count_calls(monkeypatch, os.path, "getmtime")
    assert sdc._check_sidecars(primary, entry) is False
    assert stats == [mask]


def test_check_sidecars_stats_a_name_listed_under_another_case(tmp_path, monkeypatch):
    primary = os.path.normpath(_create_source_file(tmp_path, "a.png", b"a"))
    _create_source_file(tmp_path, "A-MaskLabel.PNG", b"mask")
    wanted = os.path.normpath(str(tmp_path / "a-masklabel.png"))
    sdc = _bare_cache(tmp_path)
    sdc._extra_paths_by_filepath[primary] = {"mask_path": wanted}
    sdc._source_mtimes = sdc._bulk_stat_source_files({primary, wanted})
    stats = _count_calls(monkeypatch, os.path, "getmtime")
    expected_valid = not os.path.exists(wanted)  # False on a case-insensitive filesystem

    assert sdc._check_sidecars(primary, {"sidecar_mtimes": {}, "sidecar_hashes": {}}) is expected_valid
    assert stats == [wanted]


def test_full_validation_end_to_end_with_mostly_absent_sidecars(tmp_path, capsys, monkeypatch):
    src_dir = tmp_path / "sources"
    src_dir.mkdir()
    paths = [_create_source_file(src_dir, f"i{i}.bin", f"src_{i}".encode()) for i in range(4)]
    masks = [os.path.join(str(src_dir), f"i{i}-masklabel.png") for i in range(4)]
    _create_source_file(src_dir, "i0-masklabel.png", b"mask_0")

    def build():
        return _build_smart_pipeline(
            tmp_path,
            concepts=[{"name": "A", "path": "dummy"}],
            dummy_data={"latent": _make_tensors(4, seed=5), "image_path": paths, "mask_path": masks},
            dummy_length=len(paths),
            split_names=["latent"],
            aggregate_names=[],
            modeltype="testmodel",
            source_path_in_name="image_path",
            extra_watched_paths_in_names=["mask_path"],
        )

    ds, cache_dir, _ = build()
    _drain(ds)
    capsys.readouterr()

    # Touch one source: the fast path declines and every row is validated.
    os.utime(paths[3], (1_900_000_000, 1_900_000_000))
    ds, _, _ = build()
    stats = _count_calls(monkeypatch, os.path, "getmtime")
    _drain(ds)
    out = capsys.readouterr().out
    assert "Cached 0/4 files. 4 reused from cache." in out
    absent = {os.path.normpath(mask) for mask in masks[1:]}
    assert not absent & {os.path.normpath(path) for path in stats}
    monkeypatch.undo()

    # A mask that appears later is still noticed and rebuilds its entry only.
    _create_source_file(src_dir, "i2-masklabel.png", b"mask_2")
    os.utime(paths[3], (1_900_000_100, 1_900_000_100))
    ds, _, _ = build()
    _drain(ds)
    assert "Cached 1/4 files. 3 reused from cache." in capsys.readouterr().out
    entry = _read_cache_json(cache_dir)["entries"][os.path.normpath(paths[2])]
    assert os.path.normpath(masks[2]) in entry["sidecar_mtimes"]
