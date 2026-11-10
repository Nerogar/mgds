import hashlib
import itertools
import json
import math
from random import Random
from typing import Any

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModules.ResolutionPolicy import (
    all_resolutions_cover_scale,
    scale_resolution_for_crop,
    target_resolutions_for_spec,
    upscale_tolerance_fraction,
)
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import numpy as np


class AspectBucketing(
    PipelineModule,
    RandomAccessPipelineModule,
):
    # all possible target aspect ratios
    all_possible_input_aspects = [
        (1.0, 1.0),
        (1.0, 1.25),
        (1.0, 1.5),
        (1.0, 1.75),
        (1.0, 2.0),
        (1.0, 2.5),
        (1.0, 3.0),
        (1.0, 3.5),
        (1.0, 4.0),
    ]

    # Native generator buckets at a 1024 reference target, as (h, w).
    # Unlike ``all_possible_input_aspects`` these are NOT normalized to an
    # equal pixel count before quantization: SDXL-style presets such as
    # 1216x768 or 1536x640 carry fewer pixels than 1024^2, so deriving them
    # from a ratio would round to a bigger bucket the source cannot cover
    # (forcing an upscale or a tier drop). Scaled by target/1024, both
    # orientations, and quantized alongside the ratio-derived buckets so an
    # image emitted at exactly one of these sizes lands in an exact bucket
    # with zero crop.
    reference_target_resolution = 1024
    all_possible_input_resolutions = [
        (1024, 1024),
        (1152, 896),
        (1216, 832),
        (1216, 768),
        (1344, 768),
        (1536, 640),
    ]

    def __init__(
        self,
        quantization: int,
        resolution_in_name: str,
        target_resolution_in_name: str,
        enable_target_resolutions_override_in_name: str,
        target_resolutions_override_in_name: str,
        target_frames_in_name: str,
        frame_dim_enabled: bool,
        scale_resolution_out_name: str,
        crop_resolution_out_name: str,
        possible_resolutions_out_name: str,
        adaptive_resolution: bool = False,
        additional_resolution_in_names: list[str] | None = None,
        adaptive_resolution_ladder: str | list[int] | None = None,
        upscale_tolerance: float = 0.0,
    ):
        super().__init__()

        self.quantization = quantization
        self.resolution_in_name = resolution_in_name
        # Fraction a source (or any additional constraining image) may fall
        # short of a tier's crop by and still train there, slightly enlarged.
        # 0 = the strict no-upscale rule; part of the bucket method hash only
        # when set, so existing caches keep their stamps.
        self.upscale_tolerance = upscale_tolerance_fraction(upscale_tolerance)

        self.target_resolutions_in_name = target_resolution_in_name
        self.enable_target_resolutions_override_in_name = enable_target_resolutions_override_in_name
        self.target_resolutions_override_in_name = target_resolutions_override_in_name
        self.target_frames_in_name = target_frames_in_name
        self.frame_dim_enabled = frame_dim_enabled

        self.scale_resolution_out_name = scale_resolution_out_name
        self.crop_resolution_out_name = crop_resolution_out_name
        self.possible_resolutions_out_name = possible_resolutions_out_name
        self.adaptive_resolution = adaptive_resolution
        self.additional_resolution_in_names = list(additional_resolution_in_names or [])
        self.adaptive_resolution_ladder = adaptive_resolution_ladder

        self.bucket_resolutions = {}
        self.bucket_aspects = {}
        self.flattened_possible_resolutions = []

        # Per-call target_resolution override. SmartDiskCache sets this around
        # multi-variant cache build passes to drive get_item to a specific
        # target instead of rand.choice. Single-threaded by contract — the
        # cache builder groups items by target and runs one parallel pass per
        # target with the override held for that pass.
        self._target_override: int | None = None

    def length(self) -> int:
        return self._get_previous_length(self.resolution_in_name)

    def get_inputs(self) -> list[str]:
        return [
            self.resolution_in_name,
            self.target_resolutions_in_name,
            self.enable_target_resolutions_override_in_name,
            self.target_resolutions_override_in_name,
            self.target_frames_in_name,
        ] + self.additional_resolution_in_names

    def get_outputs(self) -> list[str]:
        return [self.scale_resolution_out_name, self.crop_resolution_out_name, self.possible_resolutions_out_name]

    def __quantize_resolution(self, resolution: tuple[float, float], quantization: int) -> tuple[int, int]:
        return (
            round(resolution[0] / quantization) * quantization,
            round(resolution[1] / quantization) * quantization,
        )

    def __create_automatic_buckets(
        self,
        target_resolutions: list[int],
    ) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[float]]]:

        possible_resolutions = {}
        possible_aspects = {}

        for target_resolution in target_resolutions:
            # normalize to the same pixel count
            new_resolutions = [
                (h / math.sqrt(h * w) * target_resolution, w / math.sqrt(h * w) * target_resolution)
                for (h, w) in self.all_possible_input_aspects
            ]

            # native generator buckets, scaled linearly from the reference target
            scale = target_resolution / self.reference_target_resolution
            new_resolutions = new_resolutions + [
                (h * scale, w * scale) for (h, w) in self.all_possible_input_resolutions
            ]

            # add inverted dimensions
            new_resolutions = new_resolutions + [(w, h) for (h, w) in new_resolutions]

            # quantization
            new_resolutions = [
                self.__quantize_resolution(resolution, self.quantization) for resolution in new_resolutions
            ]

            # Very small adaptive tiers combined with coarse model
            # quantization can round one axis to zero (for example a 128 tier
            # at 1:4 with quantization 128). Such a bucket can never be
            # resized/cropped or batched, so exclude it from the grid.
            new_resolutions = [(h, w) for h, w in new_resolutions if h > 0 and w > 0]

            # remove duplicates
            new_resolutions = list(set(new_resolutions))

            # add to lists
            possible_resolutions[target_resolution] = new_resolutions
            possible_aspects[target_resolution] = [h / w for (h, w) in new_resolutions]

        return possible_resolutions, possible_aspects

    def __get_bucket(self, rand: Random, h: int, w: int, target_resolution: int) -> tuple[int, int]:
        return self.bucket_for_aspect(h / w, target_resolution)

    def eligible_target_resolutions(
        self,
        resolution: tuple[int, ...] | list[int],
        target_resolutions: list[int],
        additional_resolutions: list[tuple[int, ...] | list[int]] | None = None,
    ) -> list[int]:
        """Return target tiers that do not require enlarging the source.

        A target tier is usable when the source already covers the aspect-
        matched crop bucket in both dimensions.  When every configured tier
        is larger than the source, retain only the smallest tier so small
        images remain trainable without also generating every larger upscale.

        The input may include a leading frame dimension; only its final H/W
        dimensions participate in image-resolution eligibility.
        """
        if not target_resolutions:
            return []

        source_h = int(resolution[-2])
        source_w = int(resolution[-1])
        aspect = source_h / source_w
        all_resolutions = [resolution, *(additional_resolutions or [])]

        eligible = []
        for target in target_resolutions:
            bucket = self.bucket_for_aspect(aspect, target)
            # Every image is cover-scaled on its OWN aspect to the source's
            # crop bucket (an additional image, e.g. a DPO rejected half, by
            # the loader's cover-scale-crop), so each must reach its own scale
            # size for that bucket, i.e. cover the bucket without enlarging.
            # Checking the additional images against the source's scale size
            # instead dropped an exact-bucket rejected image a tier whenever
            # the source's aspect slack at that bucket exceeded the tolerance.
            if all(
                all_resolutions_cover_scale([image], scale_resolution_for_crop(image, bucket), self.upscale_tolerance)
                for image in all_resolutions
            ):
                eligible.append(target)

        return eligible or [min(target_resolutions)]

    def __choose_target_resolution(
        self,
        rand: Random,
        resolution: tuple[int, ...] | list[int],
        target_resolutions: list[int],
        additional_resolutions: list[tuple[int, ...] | list[int]] | None = None,
    ) -> int:
        eligible = self.eligible_target_resolutions(resolution, target_resolutions, additional_resolutions)

        if self.adaptive_resolution:
            return max(eligible)

        # SmartDiskCache pins a target while building an already-selected
        # resolution variant.  A stale/foreign override must not bypass the
        # source-size guard and create a higher-resolution upscale.
        if self._target_override is not None and self._target_override in eligible:
            return self._target_override
        return rand.choice(eligible)

    def _additional_resolutions(self, variation: int, index: int) -> list[tuple[int, ...] | list[int]]:
        return [self._get_previous_item(variation, name, index) for name in self.additional_resolution_in_names]

    def _target_resolutions_for_item(self, variation: int, index: int) -> str:
        target_resolutions = self._get_previous_item(variation, self.target_resolutions_in_name, index)
        if self.enable_target_resolutions_override_in_name is not None:
            enable_resolution_override = self._get_previous_item(
                variation, self.enable_target_resolutions_override_in_name, index
            )
            if enable_resolution_override:
                target_resolutions = self._get_previous_item(variation, self.target_resolutions_override_in_name, index)
        return target_resolutions

    def bucket_for_aspect(self, aspect: float, target_resolution: int) -> tuple[int, int]:
        """Pure aspect-to-bucket assignment. No image read, no RNG.

        Used by SmartDiskCache to re-derive bucket assignments from a cached
        resolution string when the user changes target_resolutions, without
        opening any source images.
        """
        bucket_index = np.argmin(abs(self.bucket_aspects[target_resolution] - aspect))
        return self.bucket_resolutions[target_resolution][bucket_index]

    def _reconstructed_target_buckets(self, target_resolution: int) -> tuple[list[tuple[int, int]], np.ndarray]:
        # Kept apart from bucket_resolutions: SmartDiskCache treats a
        # populated bucket_resolutions as a started module, and a sourceless
        # run must stay on its own path.
        reconstructed = self.__dict__.setdefault("_reconstructed_buckets", {})
        target_resolution = int(target_resolution)
        buckets = reconstructed.get(target_resolution)
        if buckets is None:
            resolutions, aspects = self.__create_automatic_buckets([target_resolution])
            buckets = (resolutions[target_resolution], np.array(aspects[target_resolution]))
            reconstructed[target_resolution] = buckets
        return buckets

    def reconstruct_bucket_resolutions(self, target_resolution: int) -> list[tuple[int, int]]:
        """Bucket grid ``start()`` builds for ``target_resolution``, without ``start()``.

        A target's grid depends only on the target and the quantization and is
        built by the same code in the same order, so the result (including the
        order, which breaks ``bucket_for_aspect`` ties) equals a started
        module's ``bucket_resolutions[target_resolution]``. Sourceless
        SmartDiskCache runs never start this module and use it to tell which
        tier a cached bucket belongs to.
        """
        return list(self._reconstructed_target_buckets(target_resolution)[0])

    def reconstruct_bucket_for_aspect(self, aspect: float, target_resolution: int) -> tuple[int, int]:
        """``bucket_for_aspect`` against the reconstructed grid; needs no ``start()``."""
        resolutions, aspects = self._reconstructed_target_buckets(target_resolution)
        bucket_index = np.argmin(abs(aspects - aspect))
        return resolutions[bucket_index]

    def compute_bucket_method_hash(self) -> str:
        """Hash of the post-start() bucketing config.

        Stamped into ``cache_index['bucket_method']`` so a config change
        (different target_resolutions, quantization, etc.) drives drift
        recovery instead of silently serving stale .pt files. Must be called
        after ``start()`` populates ``bucket_resolutions``.
        """
        payload = {
            # v3 filters target tiers against the source resolution.  Bumping
            # the method marker makes existing caches run their normal bucket-
            # drift recovery before the new selector starts serving variants.
            "method": "aspect_v4_adaptive" if self.adaptive_resolution else "aspect_v3_no_upscale",
            "adaptive_resolution": self.adaptive_resolution,
            "additional_resolution_inputs": self.additional_resolution_in_names,
            "quantization": self.quantization,
            "frame_dim_enabled": self.frame_dim_enabled,
            "targets": sorted(self.bucket_resolutions.keys()),
            "flattened": sorted([list(r) for r in self.flattened_possible_resolutions]),
        }
        if self.upscale_tolerance > 0.0:
            # Changing the tolerance moves images between tiers, so the
            # variants have to be re-derived; the key is absent at 0 so the
            # hash of every existing cache is unchanged.
            payload["upscale_tolerance"] = round(self.upscale_tolerance, 6)
        if self.additional_resolution_in_names:
            # Additional images are now checked against their own cover-scale
            # size, which moves pairs between tiers; only pair-bearing
            # instances carry the marker, so single-image caches keep theirs.
            payload["additional_cover_rule"] = "own"
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def get_meta(self, variation: int, name: str) -> Any:
        if name == self.possible_resolutions_out_name:
            return self.flattened_possible_resolutions
        else:
            return None

    def start(self, variation: int):
        possible_target_resolutions = set()
        possible_fixed_resolutions = set()
        possible_frames = {1}

        def register_resolution_spec(resolutions: str | int):
            if isinstance(resolutions, int):
                possible_target_resolutions.add(resolutions)
                return
            if "x" in resolutions and "," not in resolutions:
                res = resolutions.strip().split("x")
                possible_fixed_resolutions.add(
                    self.__quantize_resolution((int(res[1]), int(res[0])), self.quantization)
                )
            else:
                possible_target_resolutions.update(
                    target_resolutions_for_spec(
                        resolutions,
                        self.adaptive_resolution,
                        self.adaptive_resolution_ladder,
                    )
                )

        for index in range(self._get_previous_length(self.target_resolutions_in_name)):
            resolutions = self._get_previous_item(variation, self.target_resolutions_in_name, index)
            register_resolution_spec(resolutions)

        if self.target_resolutions_override_in_name is not None:
            for index in range(self._get_previous_length(self.target_resolutions_override_in_name)):
                resolutions = self._get_previous_item(variation, self.target_resolutions_override_in_name, index)
                register_resolution_spec(resolutions)

        for index in range(self._get_previous_length(self.target_frames_in_name)):
            frames = self._get_previous_item(variation, self.target_frames_in_name, index)
            possible_frames.add(int(frames))

        self.bucket_resolutions, self.bucket_aspects = self.__create_automatic_buckets(
            list(possible_target_resolutions)
        )

        self.flattened_possible_resolutions = list(
            set(sum(self.bucket_resolutions.values(), [])) | possible_fixed_resolutions
        )
        if self.frame_dim_enabled:
            self.flattened_possible_resolutions = [
                (f, *r) for f, r in itertools.product(possible_frames, self.flattened_possible_resolutions)
            ]

        self.bucket_aspects = {k: np.array(v) for (k, v) in self.bucket_aspects.items()}

    def variant_key_from_aspect(self, variation: int, index: int, aspect: float) -> str | None:
        """Compute the 'HxW' variant key for (variation, index) without
        triggering the upstream image decode.

        Replicates the get_item RNG path exactly — same _get_rand seed,
        same target_resolutions parse, same rand.choice — but accepts the
        aspect ratio as an argument instead of pulling it through
        CalcAspect → LoadImage. SmartDiskCache recovers aspect from an
        existing variant key (deterministic since buckets quantize from
        aspect) and uses this to fast-validate per-epoch variant
        rotation without re-decoding every image on disk.

        Returns None when target_resolutions is the fixed-WxH form
        (no aspect bucketing applies) or any upstream walk fails.
        """
        # Adaptive eligibility depends on every source image's actual H/W, so
        # an aspect alone cannot safely select a cache variant.
        if self.adaptive_resolution:
            return None

        rand = self._get_rand(variation, index)
        try:
            target_resolutions = self._get_previous_item(variation, self.target_resolutions_in_name, index)
        except Exception:
            return None

        if self.enable_target_resolutions_override_in_name is not None:
            try:
                enable_resolution_override = self._get_previous_item(
                    variation, self.enable_target_resolutions_override_in_name, index
                )
            except Exception:
                # Can't replicate get_item's read — signal "fall back to the
                # slow path" like the sibling reads above/below do. Silently
                # assuming the override is disabled would compute a key from
                # the wrong resolution set and let fast-validation serve a
                # wrong-bucket variant where get_item would have raised.
                return None
            if enable_resolution_override:
                try:
                    target_resolutions = self._get_previous_item(
                        variation, self.target_resolutions_override_in_name, index
                    )
                except Exception:
                    return None

        if "x" in target_resolutions and "," not in target_resolutions:
            # Fixed WxH: the variant key is the configured dims directly,
            # no aspect math involved. Fall back to the slow path.
            return None

        target_resolutions_list = [int(res.strip()) for res in target_resolutions.split(",")]
        if not target_resolutions_list:
            return None

        if self._target_override is not None and self._target_override in target_resolutions_list:
            target_resolution = self._target_override
        else:
            target_resolution = rand.choice(target_resolutions_list)

        if target_resolution not in self.bucket_aspects:
            return None
        h, w = self.bucket_for_aspect(aspect, target_resolution)
        return f"{h}x{w}"

    def variant_key_from_resolution(
        self,
        variation: int,
        index: int,
        resolution: tuple[int, ...] | list[int],
    ) -> str | None:
        """Compute the selected variant key with source-size eligibility.

        SmartDiskCache calls this with ``original_resolution`` from its index,
        avoiding an image decode while reproducing :meth:`get_item` exactly.
        """
        return self.variant_key_from_resolutions(variation, index, resolution, [])

    def variant_key_from_resolutions(
        self,
        variation: int,
        index: int,
        resolution: tuple[int, ...] | list[int],
        additional_resolutions: list[tuple[int, ...] | list[int]],
    ) -> str | None:
        """Compute the selected key using every image that constrains a crop."""
        rand = self._get_rand(variation, index)
        try:
            target_resolutions = self._target_resolutions_for_item(variation, index)
        except Exception:
            return None
        target_resolutions = str(target_resolutions)

        if "x" in target_resolutions and "," not in target_resolutions:
            return None

        target_resolutions_list = target_resolutions_for_spec(
            target_resolutions,
            self.adaptive_resolution,
            self.adaptive_resolution_ladder,
        )
        if not target_resolutions_list:
            return None

        target_resolution = self.__choose_target_resolution(
            rand, resolution, target_resolutions_list, additional_resolutions
        )
        if target_resolution not in self.bucket_aspects:
            return None
        h, w = self.bucket_for_aspect(resolution[-2] / resolution[-1], target_resolution)
        return f"{h}x{w}"

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        rand = self._get_rand(variation, index)
        resolution = self._get_previous_item(variation, self.resolution_in_name, index)
        additional_resolutions = self._additional_resolutions(variation, index)
        target_resolutions = str(self._target_resolutions_for_item(variation, index))

        if "x" in target_resolutions and "," not in target_resolutions:
            res = target_resolutions.strip().split("x")
            target_resolution = self.__quantize_resolution((int(res[1]), int(res[0])), self.quantization)
        else:
            target_resolutions = target_resolutions_for_spec(
                target_resolutions,
                self.adaptive_resolution,
                self.adaptive_resolution_ladder,
            )
            target_resolution = self.__choose_target_resolution(
                rand, resolution, target_resolutions, additional_resolutions
            )
            target_resolution = self.__get_bucket(rand, resolution[-2], resolution[-1], target_resolution)

        aspect = resolution[-2] / resolution[-1]
        target_aspect = target_resolution[-2] / target_resolution[-1]

        if aspect > target_aspect:
            scale = target_resolution[-1] / resolution[-1]
            scale_resolution = (*resolution[:-2], round(resolution[-2] * scale), target_resolution[-1])
        else:
            scale = target_resolution[-2] / resolution[-2]
            scale_resolution = (*resolution[:-2], target_resolution[-2], round(resolution[-1] * scale))

        target_resolution = (*resolution[:-2], *target_resolution)

        return {
            self.scale_resolution_out_name: scale_resolution,
            self.crop_resolution_out_name: target_resolution,
        }
