"""Content-addressed caption reuse by hard link: a duplicate whose payload is
exactly the donor's shares the donor's file instead of being written again,
and nothing done to one name ever changes what another name holds."""

import errno
import hashlib
import json
import os
import sys
from pathlib import Path

from mgds.MGDS import MGDS
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModules.SmartDiskCache import MAX_CONTENT_LINK_FAILURES, SmartDiskCache
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_smartcache_content_reuse import (  # noqa: E402
    CaptionSource,
    CountingEncoder,
    PickLine,
    _build_text_pipeline,
    _drain,
    _embedding_for,
    _variant_pt_path,
    _write_caption_file,
)

LINE = "shared trigger line"


def _concept(name: str = "C", **extra) -> dict:
    return {"name": name, "text_variations": 1, "balancing": 1.0, "balancing_strategy": "REPEATS", **extra}


def _digest(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _same_file(a: str, b: str) -> bool:
    return os.path.samefile(a, b)


def _duplicates(tmp_path, count: int = 3) -> list[str]:
    """Caption files with different bytes whose picked line is the same."""
    return [_write_caption_file(tmp_path / "src", f"c{i}.txt", [LINE, f"never picked {i}"]) for i in range(count)]


def _build(tmp_path, paths: list[str], concepts: list[dict] | None = None):
    ds, encoder, _ = _build_text_pipeline(tmp_path, paths, text_variations=1, concepts_per_index=concepts)
    _drain(ds)
    return encoder


def _donor_then_duplicates(tmp_path, count: int = 3) -> tuple[list[str], list[str]]:
    """Cache the first file, then the rest in a later pass; returns (sources, .pt paths)."""
    paths = _duplicates(tmp_path, count)
    _build(tmp_path, paths[:1])
    encoder = _build(tmp_path, paths)
    assert encoder.calls == []
    cache_dir = str(tmp_path / "cache")
    return paths, [_variant_pt_path(cache_dir, path, 0) for path in paths]


@pytest.fixture
def links_available(tmp_path):
    probe, link = tmp_path / "probe", tmp_path / "probe.link"
    probe.write_bytes(b"x")
    try:
        os.link(probe, link)
    except OSError:
        pytest.skip("hard links are not available here")
    os.remove(link)
    os.remove(probe)


def test_an_identical_duplicate_is_linked_to_its_donor(tmp_path, links_available):
    _paths, pts = _donor_then_duplicates(tmp_path)

    assert all(_same_file(pts[0], pt) for pt in pts[1:])
    assert os.stat(pts[0]).st_nlink == 3
    donor = torch.load(pts[0], weights_only=False)
    for pt in pts[1:]:
        loaded = torch.load(pt, weights_only=False)
        assert list(loaded) == list(donor)
        assert torch.equal(loaded["embedding"], _embedding_for(LINE))
        assert {k: repr(v) for k, v in loaded.items() if k != "embedding"} == {
            k: repr(v) for k, v in donor.items() if k != "embedding"
        }
    assert not [name for name in os.listdir(tmp_path / "cache") if name.endswith(".tmp")]


def test_a_duplicate_with_other_stamped_metadata_is_copied(tmp_path, links_available):
    paths = _duplicates(tmp_path, 2)
    concepts = [_concept("A", loss_weight=1.0), _concept("B", loss_weight=2.5)]
    _build(tmp_path, paths[:1], concepts[:1])
    encoder = _build(tmp_path, paths, concepts)

    assert encoder.calls == []
    cache_dir = str(tmp_path / "cache")
    donor_pt, copy_pt = (_variant_pt_path(cache_dir, path, 0) for path in paths)
    assert not _same_file(donor_pt, copy_pt)
    assert os.stat(donor_pt).st_nlink == 1
    assert torch.load(copy_pt, weights_only=False)["__concept_name"] == "B"
    assert torch.load(donor_pt, weights_only=False)["__concept_name"] == "A"


def test_rebuilding_a_linked_name_leaves_the_others_untouched(tmp_path, links_available):
    paths, pts = _donor_then_duplicates(tmp_path)
    before = _digest(pts[0])

    # The donor's own caption changes: its entry moves on, the links keep the bytes.
    _write_caption_file(tmp_path / "src", "c0.txt", ["the donor says something else", "never picked 0"])
    encoder = _build(tmp_path, paths)
    assert encoder.calls == ["the donor says something else"]
    cache_dir = str(tmp_path / "cache")
    new_donor_pt = _variant_pt_path(cache_dir, paths[0], 0)
    assert not _same_file(new_donor_pt, pts[1])
    assert _digest(pts[1]) == before and _digest(pts[2]) == before
    assert torch.equal(torch.load(pts[1], weights_only=False)["embedding"], _embedding_for(LINE))
    assert torch.equal(
        torch.load(new_donor_pt, weights_only=False)["embedding"], _embedding_for("the donor says something else")
    )

    # A linked duplicate's caption changes: same again from the other side.
    _write_caption_file(tmp_path / "src", "c1.txt", ["and so does a duplicate", "never picked 1"])
    _build(tmp_path, paths)
    assert _digest(pts[2]) == before
    assert torch.equal(torch.load(pts[2], weights_only=False)["embedding"], _embedding_for(LINE))


def test_a_lost_linked_name_is_restored_without_touching_the_others(tmp_path, links_available):
    paths, pts = _donor_then_duplicates(tmp_path)
    before = _digest(pts[0])
    # One of the names goes missing: removing it only drops that name, and
    # validation rebuilds just that variant, as a link again.
    cache_dir = str(tmp_path / "cache")
    os.remove(pts[1])
    assert _digest(pts[0]) == before and os.stat(pts[0]).st_nlink == 2
    encoder = _build(tmp_path, paths)

    assert encoder.calls == []
    assert os.path.isfile(pts[1])
    assert _digest(pts[0]) == before and _digest(pts[2]) == before
    assert torch.equal(torch.load(pts[1], weights_only=False)["embedding"], _embedding_for(LINE))
    assert _same_file(pts[1], _variant_pt_path(cache_dir, paths[1], 0))


class _IndexedExtra(PipelineModule, RandomAccessPipelineModule):
    """A second cached name whose value differs per row."""

    def get_inputs(self) -> list[str]:
        return ["sample_prompt_path"]

    def get_outputs(self) -> list[str]:
        return ["extra"]

    def length(self) -> int:
        return self._get_previous_length("sample_prompt_path")

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return {"extra": torch.full((2,), float(index))}


def test_augmenting_linked_files_writes_each_name_separately(tmp_path, links_available):
    paths, pts = _donor_then_duplicates(tmp_path)
    assert _same_file(pts[0], pts[1])

    # Same cache, one more required name: the schema-drift pass loads every
    # .pt, adds the missing value and writes it back.
    cache = SmartDiskCache(
        cache_dir=str(tmp_path / "cache"),
        split_names=["embedding", "extra"],
        aggregate_names=[],
        variations_in_name="concept.text_variations",
        balancing_in_name="concept.balancing",
        balancing_strategy_in_name="concept.balancing_strategy",
        variations_group_in_name="concept.name",
        modeltype="testmodel",
        source_path_in_name="sample_prompt_path",
        content_key_in_name="prompt",
    )
    ds = MGDS(
        device=torch.device("cpu"),
        concepts=[{"name": "C", "path": "dummy"}],
        settings={},
        definition=[
            [CaptionSource(paths, [_concept()] * len(paths)), PickLine(), CountingEncoder(), _IndexedExtra()],
            [cache],
            [OutputPipelineModule(names=["embedding", "extra"])],
        ],
        batch_size=1,
        state=PipelineState(),
        seed=42,
    )
    ds.start_next_epoch()

    for index, pt in enumerate(pts):
        loaded = torch.load(pt, weights_only=False)
        # Each name got its own row's value: no write reached a linked sibling.
        assert torch.equal(loaded["extra"], torch.full((2,), float(index)))
        assert torch.equal(loaded["embedding"], _embedding_for(LINE))
    assert len({os.stat(pt).st_ino for pt in pts}) == len(pts)


def test_a_filesystem_without_links_gets_copies(tmp_path, monkeypatch):
    attempts: list[str] = []

    def no_link(src, dst, **kwargs):
        attempts.append(os.path.basename(dst))
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "link", no_link)
    count = MAX_CONTENT_LINK_FAILURES + 40
    paths = _duplicates(tmp_path, count)
    _build(tmp_path, paths[:1])
    encoder = _build(tmp_path, paths)

    assert encoder.calls == []
    # After enough failures in a row the pass stops trying.
    assert MAX_CONTENT_LINK_FAILURES <= len(attempts) < count - 1
    cache_dir = str(tmp_path / "cache")
    for path in paths:
        pt = _variant_pt_path(cache_dir, path, 0)
        assert os.stat(pt).st_nlink == 1
        assert torch.equal(torch.load(pt, weights_only=False)["embedding"], _embedding_for(LINE))
    assert not [name for name in os.listdir(cache_dir) if name.endswith(".tmp")]


@pytest.mark.parametrize("windows_error", [False, True])
def test_a_full_donor_hands_over_to_the_copy(tmp_path, monkeypatch, links_available, windows_error):
    real_link = os.link
    limit = 3  # names per file

    def limited_link(src, dst, **kwargs):
        if os.stat(src).st_nlink >= limit:
            if windows_error:
                error = OSError(errno.EINVAL, "An attempt was made to create more links on a file than supported")
                error.winerror = 1142
                raise error
            raise OSError(errno.EMLINK, "Too many links")
        return real_link(src, dst, **kwargs)

    monkeypatch.setattr(os, "link", limited_link)
    paths = _duplicates(tmp_path, 9)
    _build(tmp_path, paths[:1])
    # One duplicate per pass, so every hand-over is visible to the next one.
    for end in range(2, len(paths) + 1):
        encoder = _build(tmp_path, paths[:end])
        assert encoder.calls == []

    cache_dir = str(tmp_path / "cache")
    pts = [_variant_pt_path(cache_dir, path, 0) for path in paths]
    groups: dict[int, int] = {}
    for pt in pts:
        groups[os.stat(pt).st_ino] = groups.get(os.stat(pt).st_ino, 0) + 1
        assert torch.equal(torch.load(pt, weights_only=False)["embedding"], _embedding_for(LINE))
    assert sorted(groups.values()) == [3, 3, 3]
    with open(os.path.join(cache_dir, "content_index.json"), encoding="utf-8") as f:
        donors = json.load(f)["entries"]
    assert len(donors) == 1
    assert os.path.isfile(os.path.join(cache_dir, next(iter(donors.values()))))


def test_the_environment_switch_turns_links_off(tmp_path, monkeypatch, links_available):
    monkeypatch.setenv("OT_SMARTCACHE_NO_HARDLINKS", "1")
    _paths, pts = _donor_then_duplicates(tmp_path)
    assert [os.stat(pt).st_nlink for pt in pts] == [1, 1, 1]
    assert torch.equal(torch.load(pts[2], weights_only=False)["embedding"], _embedding_for(LINE))


def test_a_second_start_keeps_the_links_and_writes_nothing(tmp_path, links_available):
    paths, pts = _donor_then_duplicates(tmp_path)
    stamps = [os.stat(pt).st_mtime_ns for pt in pts]

    encoder = _build(tmp_path, paths)

    assert encoder.calls == []
    assert [os.stat(pt).st_mtime_ns for pt in pts] == stamps
    assert os.stat(pts[0]).st_nlink == 3


# ---------------------------------------------------------------------------
# The rule that decides link or copy
# ---------------------------------------------------------------------------


def _bare(scrub=()) -> SmartDiskCache:
    sdc = SmartDiskCache.__new__(SmartDiskCache)
    sdc.scrub_split_names = list(scrub)
    sdc._content_donor_meta = {}
    return sdc


def _donor_payload() -> dict:
    return {
        "embedding": torch.ones(4),
        "tokens": torch.zeros(3, dtype=torch.int64),
        "__cache_version": 3,
        "__modeltype": "testmodel",
        "__content_hash": "h",
        "__sourceless_values": {"concept": {"name": "A", "loss_weight": 1.0}, "prompt": "a prompt"},
        "__concept_loss_weight": 1.0,
        "__concept_name": "A",
    }


def _snapshot(sdc: SmartDiskCache, donor: dict, name: str = "donor_1.pt") -> None:
    sdc._content_donor_meta["h"] = (
        name,
        tuple(donor),
        {key: repr(value) for key, value in donor.items() if not torch.is_tensor(value)},
    )


def test_only_an_exact_payload_counts_as_the_donors():
    sdc = _bare()
    donor = _donor_payload()
    _snapshot(sdc, donor)

    def judge(candidate: dict, donor_name: str = "donor_1.pt") -> bool:
        return sdc._is_donor_file_payload("h", donor_name, donor, candidate)

    assert judge(dict(donor)) is True
    # Equal values in fresh objects (what a re-stamp of the same concept produces).
    restamped = dict(donor)
    restamped["__sourceless_values"] = {"concept": {"name": "A", "loss_weight": 1.0}, "prompt": "a prompt"}
    assert judge(restamped) is True

    assert judge(dict(donor, __concept_name="B")) is False
    assert judge(dict(donor, __concept_loss_weight=1)) is False  # int where the file has a float
    assert judge(dict(donor, __cache_version=4)) is False
    assert judge({key: donor[key] for key in reversed(donor)}) is False
    assert judge({key: value for key, value in donor.items() if key != "__concept_name"}) is False
    assert judge(dict(donor, embedding=torch.ones(4))) is False  # not the donor's tensor
    assert judge(dict(donor), donor_name="another_1.pt") is False
    assert sdc._is_donor_file_payload("unknown", "donor_1.pt", donor, dict(donor)) is False


def test_a_scrubbed_tensor_counts_only_if_the_donor_was_already_zero():
    sdc = _bare(scrub=["tokens", "embedding"])
    donor = _donor_payload()
    _snapshot(sdc, donor)

    scrubbed = dict(donor, tokens=torch.zeros_like(donor["tokens"]))
    assert sdc._is_donor_file_payload("h", "donor_1.pt", donor, scrubbed) is True
    scrubbed["embedding"] = torch.zeros_like(donor["embedding"])
    assert sdc._is_donor_file_payload("h", "donor_1.pt", donor, scrubbed) is False


def test_the_donor_is_judged_by_what_its_file_held_when_read():
    sdc = _bare()
    donor = _donor_payload()
    _snapshot(sdc, donor)
    # A refresh that blanks a prompt inside the memoized dict itself must not
    # make the file (which still holds the prompt) look identical.
    donor["__sourceless_values"]["prompt"] = ""
    assert sdc._is_donor_file_payload("h", "donor_1.pt", donor, dict(donor)) is False


# ---------------------------------------------------------------------------
# Size accounting with linked names
# ---------------------------------------------------------------------------


def test_gc_preview_counts_shared_bytes_once_and_only_when_freed(tmp_path, links_available):
    paths, pts = _donor_then_duplicates(tmp_path)
    cache_dir = str(tmp_path / "cache")
    size = os.path.getsize(pts[0])
    index_path = os.path.join(cache_dir, "cache.json")

    def drop_entries(sources: list[str]) -> None:
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        for source in sources:
            del index["entries"][os.path.normpath(source)]
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f)

    assert SmartDiskCache.gc_preview(cache_dir) == {"orphan_count": 0, "orphan_bytes": 0}

    # Two of the three names are orphans: removing them frees nothing.
    drop_entries(paths[1:])
    assert SmartDiskCache.gc_preview(cache_dir) == {"orphan_count": 2, "orphan_bytes": 0}

    # All three are orphans: the file's bytes are freed, once.
    drop_entries(paths[:1])
    assert SmartDiskCache.gc_preview(cache_dir) == {"orphan_count": 3, "orphan_bytes": size}


def test_gc_clean_of_a_linked_orphan_keeps_the_file_for_the_other_names(tmp_path, links_available):
    paths, pts = _donor_then_duplicates(tmp_path)
    cache_dir = str(tmp_path / "cache")
    before = _digest(pts[0])
    index_path = os.path.join(cache_dir, "cache.json")
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    del index["entries"][os.path.normpath(paths[1])]
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f)

    SmartDiskCache.gc_clean(cache_dir)

    assert not os.path.exists(pts[1])
    assert _digest(pts[0]) == before and _digest(pts[2]) == before
    assert os.stat(pts[0]).st_nlink == 2
