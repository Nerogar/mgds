"""Upscale tolerance: a source may fall short of a tier's crop by a fraction
and still train there (DPO loaders), while 0 keeps today's rule and hashes."""

import sys
from pathlib import Path

from mgds.pipelineModules.ResolutionPolicy import (
    all_resolutions_cover_scale,
    scale_resolution_for_crop,
    upscale_tolerance_fraction,
)
from mgds.pipelineModules.SmartDiskCache import SmartDiskCache

sys.path.insert(0, str(Path(__file__).parent))
from test_aspect_bucketing_resolution_eligibility import AspectBucketingHarness  # noqa: E402
from test_sourceless_adaptive_tier import _cache, _entry, _unstarted_bucketing, _variant  # noqa: E402

LADDER = "1536,1280,1024,768,512,256,128"


def test_tolerance_fraction_is_clamped():
    assert upscale_tolerance_fraction(None) == 0.0
    assert upscale_tolerance_fraction("") == 0.0
    assert upscale_tolerance_fraction(-0.1) == 0.0
    assert upscale_tolerance_fraction(float("nan")) == 0.0
    assert upscale_tolerance_fraction(0.03) == 0.03
    assert upscale_tolerance_fraction(5) == 0.99


def test_cover_scale_with_tolerance():
    assert all_resolutions_cover_scale([(1000, 1000)], (1024, 1024)) is False
    assert all_resolutions_cover_scale([(1000, 1000)], (1024, 1024), 0.03) is True
    assert all_resolutions_cover_scale([(1000, 1000)], (1024, 1024), 0.02) is False
    # Every constraining image has to be within the tolerance.
    assert all_resolutions_cover_scale([(1024, 1024), (900, 900)], (1024, 1024), 0.03) is False


def test_off_policy_rejected_pair_moves_up_a_tier_within_tolerance():
    # A square 1024 chosen image pairs with a portrait ~1 MP rejected image.
    # The pair inherits the chosen bucket (1024x1024), which the rejected
    # image's 832 width cannot cover, so the pair trains at 768 today; with a
    # tolerance covering the shortfall the pair is eligible for 1024.
    chosen, rejected = (1024, 1024), (1248, 832)
    strict = AspectBucketingHarness(
        resolution=chosen,
        targets="1024",
        adaptive_resolution=True,
        adaptive_resolution_ladder=LADDER,
        additional_resolutions=[rejected],
    )
    tolerant = AspectBucketingHarness(
        resolution=chosen,
        targets="1024",
        adaptive_resolution=True,
        adaptive_resolution_ladder=LADDER,
        additional_resolutions=[rejected],
    )
    tolerant.upscale_tolerance = 0.2
    tiers = [128, 256, 512, 768, 1024]
    assert max(strict.eligible_target_resolutions(chosen, tiers, [rejected])) == 768
    assert max(tolerant.eligible_target_resolutions(chosen, tiers, [rejected])) == 1024
    assert strict.variant_key_from_resolutions(0, 7, chosen, [rejected]) != tolerant.variant_key_from_resolutions(
        0, 7, chosen, [rejected]
    )


def test_three_percent_tolerance_admits_a_source_three_percent_short():
    bucketing = AspectBucketingHarness(resolution=(1000, 1000), targets="1024", adaptive_resolution=True)
    bucketing.upscale_tolerance = 0.03
    assert max(bucketing.eligible_target_resolutions((1000, 1000), [768, 1024])) == 1024
    bucketing.upscale_tolerance = 0.02
    assert max(bucketing.eligible_target_resolutions((1000, 1000), [768, 1024])) == 768


def test_bucket_method_hash_changes_only_when_tolerance_is_set():
    strict = AspectBucketingHarness(targets="1024", adaptive_resolution=True)
    assert strict.upscale_tolerance == 0.0
    baseline = strict.compute_bucket_method_hash()
    strict.upscale_tolerance = 0.0
    assert strict.compute_bucket_method_hash() == baseline
    strict.upscale_tolerance = 0.03
    assert strict.compute_bucket_method_hash() != baseline


def test_sourceless_selection_prefers_the_tolerated_tier_when_cached():
    source = (1000, 1000)
    unstarted = _unstarted_bucketing()
    strict_cache = _cache(unstarted, maximum=1024)
    tolerant_cache = _cache(unstarted, maximum=1024)
    tolerant_cache.upscale_tolerance = 0.03
    entry = _entry(
        source,
        [((768, 768), _variant("v768", (768, 768), 768)), ((1024, 1024), _variant("v1024", (1024, 1024), 1024))],
        maximum=1024,
    )
    assert strict_cache._sourceless_variant(entry, 0)["cache_file"] == "v768"
    assert tolerant_cache._sourceless_variant(entry, 0)["cache_file"] == "v1024"


def test_sourceless_selection_falls_back_to_the_cached_tier_and_counts_it():
    source = (1000, 1000)
    tolerant_cache = _cache(_unstarted_bucketing(), maximum=1024)
    tolerant_cache.upscale_tolerance = 0.03
    tolerant_cache._tier_fallback_entries = set()
    entry = _entry(source, [((768, 768), _variant("v768", (768, 768), 768))], maximum=1024)
    assert tolerant_cache._sourceless_variant(entry, 0)["cache_file"] == "v768"
    assert len(tolerant_cache._tier_fallback_entries) == 1


def test_prune_plan_keeps_tolerated_variants(tmp_path):
    import json

    cache_dir = tmp_path / "image"
    cache_dir.mkdir()
    entry = _entry(
        (1000, 1000),
        [((768, 768), _variant("v768.pt", (768, 768), 768)), ((1024, 1024), _variant("v1024.pt", (1024, 1024), 1024))],
        maximum=1024,
    )
    (cache_dir / "cache.json").write_text(json.dumps({"entries": {"a.png": entry}}), encoding="utf-8")

    strict = SmartDiskCache.preview_adaptive_prune(str(cache_dir), adaptive_resolution_ladder=LADDER)
    tolerant = SmartDiskCache.preview_adaptive_prune(
        str(cache_dir), adaptive_resolution_ladder=LADDER, upscale_tolerance=0.03
    )
    assert strict["removed_variants"] == 1
    assert tolerant["removed_variants"] == 0


def test_exact_bucket_rejected_keeps_the_generation_tier():
    # A 12 MP portrait photo paired with a rejected image generated at the
    # photo's own 1024 bucket (896x1152). The photo's aspect (0.75) sits 3.7%
    # off that bucket's (0.778), so its cover scale is 896x1195: checking the
    # rejected against THAT size (the old rule) failed it at 3% tolerance and
    # dropped the pair to 768, where the bucket shape differs and the rejected
    # got cover-cropped. The rejected covers its own scale size exactly.
    chosen, rejected = (4032, 3024), (1152, 896)
    bucketing = AspectBucketingHarness(
        resolution=chosen,
        targets="1536",
        adaptive_resolution=True,
        adaptive_resolution_ladder=LADDER,
        additional_resolutions=[rejected],
    )
    bucketing.upscale_tolerance = 0.03
    tiers = [128, 256, 512, 768, 1024, 1280, 1536]
    assert bucketing.bucket_for_aspect(4032 / 3024, 1024) == rejected
    assert scale_resolution_for_crop(chosen, rejected) == (1195, 896)
    # the pair lands at the tier the rejected was generated at, no higher
    assert max(bucketing.eligible_target_resolutions(chosen, tiers, [rejected])) == 1024
    # the photo alone could go higher; the rejected is the only cap
    assert max(bucketing.eligible_target_resolutions(chosen, tiers, [])) == 1536
    # a rejected that is genuinely too small for the bucket still drops
    assert max(bucketing.eligible_target_resolutions(chosen, tiers, [(1100, 850)])) == 768
    assert max(bucketing.eligible_target_resolutions(chosen, tiers, [(1120, 870)])) == 1024  # within 3%


def test_pair_rule_marker_leaves_single_image_hashes_alone():
    single = AspectBucketingHarness(targets="1024", adaptive_resolution=True)
    paired = AspectBucketingHarness(targets="1024", adaptive_resolution=True, additional_resolutions=[(1024, 1024)])
    assert single.additional_resolution_in_names == []
    assert paired.additional_resolution_in_names == ["rejected_original_resolution"]
    assert single.compute_bucket_method_hash() != paired.compute_bucket_method_hash()
