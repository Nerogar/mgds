import hashlib
import json
from typing import Any

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModules.ResolutionPolicy import (
    all_resolutions_cover_scale,
    scale_resolution_for_crop,
    target_resolutions_for_spec,
    upscale_tolerance_fraction,
)
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class SingleAspectCalculation(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
        self,
        resolution_in_name: str,
        target_resolution_in_name: str,
        enable_target_resolutions_override_in_name: str,
        target_resolutions_override_in_name: str,
        scale_resolution_out_name: str,
        crop_resolution_out_name: str,
        possible_resolutions_out_name: str,
        adaptive_resolution: bool = False,
        additional_resolution_in_names: list[str] | None = None,
        adaptive_resolution_ladder: str | list[int] | None = None,
        upscale_tolerance: float = 0.0,
    ):
        super().__init__()

        self.resolution_in_name = resolution_in_name
        self.upscale_tolerance = upscale_tolerance_fraction(upscale_tolerance)

        self.target_resolutions_in_name = target_resolution_in_name
        self.enable_target_resolutions_override_in_name = enable_target_resolutions_override_in_name
        self.target_resolutions_override_in_name = target_resolutions_override_in_name

        self.scale_resolution_out_name = scale_resolution_out_name
        self.crop_resolution_out_name = crop_resolution_out_name
        self.possible_resolutions_out_name = possible_resolutions_out_name
        self.adaptive_resolution = adaptive_resolution
        self.additional_resolution_in_names = additional_resolution_in_names or []
        self.adaptive_resolution_ladder = adaptive_resolution_ladder

        self.possible_target_resolutions = []
        self.fixed_target_resolutions = []
        self.frame_dim_enabled = False
        self._target_override = None
        self.bucket_resolutions = {}
        self.bucket_aspects = {}

    def length(self) -> int:
        return self._get_previous_length(self.resolution_in_name)

    def get_inputs(self) -> list[str]:
        return [self.resolution_in_name, *self.additional_resolution_in_names]

    def get_outputs(self) -> list[str]:
        return [self.scale_resolution_out_name, self.crop_resolution_out_name, self.possible_resolutions_out_name]

    def get_meta(self, variation: int, name: str) -> Any:
        if name == self.possible_resolutions_out_name:
            return [
                *((x, x) for x in self.possible_target_resolutions),
                *self.fixed_target_resolutions,
            ]
        else:
            return None

    def start(self, variation: int):
        possible_target_resolutions = set()
        fixed_target_resolutions = set()

        def add_resolution_spec(resolutions):
            if isinstance(resolutions, int):
                possible_target_resolutions.add(resolutions)
            elif isinstance(resolutions, str):
                if "x" in resolutions and "," not in resolutions:
                    width, height = (int(part.strip()) for part in resolutions.split("x"))
                    fixed_target_resolutions.add((height, width))
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
            add_resolution_spec(resolutions)

        if self.target_resolutions_override_in_name is not None:
            for index in range(self._get_previous_length(self.target_resolutions_override_in_name)):
                resolutions = self._get_previous_item(variation, self.target_resolutions_override_in_name, index)
                add_resolution_spec(resolutions)

        self.possible_target_resolutions = sorted(possible_target_resolutions)
        self.fixed_target_resolutions = sorted(fixed_target_resolutions)
        self.bucket_resolutions = {target: [(target, target)] for target in self.possible_target_resolutions}
        self.bucket_aspects = {target: [1.0] for target in self.possible_target_resolutions}

    def eligible_target_resolutions(
        self,
        resolution: tuple[int, ...] | list[int],
        target_resolutions: list[int],
        additional_resolutions: list[tuple[int, ...] | list[int]] | None = None,
    ) -> list[int]:
        """Square target tiers that do not enlarge either source dimension.

        If the image cannot cover even the smallest square crop, keep only the
        smallest configured tier so it remains trainable at one resolution.
        """
        if not target_resolutions:
            return []
        all_resolutions = [resolution, *(additional_resolutions or [])]
        # Each image covers the square crop on its own aspect (see
        # AspectBucketing.eligible_target_resolutions), so each is checked
        # against its own scale size for the crop.
        eligible = [
            target
            for target in target_resolutions
            if all(
                all_resolutions_cover_scale(
                    [image],
                    scale_resolution_for_crop(image, (target, target)),
                    self.upscale_tolerance,
                )
                for image in all_resolutions
            )
        ]
        return eligible or [min(target_resolutions)]

    def _target_resolutions_for_item(self, variation: int, index: int) -> str:
        target_resolutions = self._get_previous_item(variation, self.target_resolutions_in_name, index)
        if self.enable_target_resolutions_override_in_name is not None:
            enable_override = self._get_previous_item(variation, self.enable_target_resolutions_override_in_name, index)
            if enable_override:
                target_resolutions = self._get_previous_item(variation, self.target_resolutions_override_in_name, index)
        return target_resolutions

    def _additional_resolutions(self, variation: int, index: int) -> list:
        return [self._get_previous_item(variation, name, index) for name in self.additional_resolution_in_names]

    def _choose_target_resolution(self, rand, resolution, targets, additional_resolutions):
        eligible = self.eligible_target_resolutions(resolution, targets, additional_resolutions)
        if self.adaptive_resolution:
            return max(eligible)
        if self._target_override is not None and self._target_override in eligible:
            return self._target_override
        return rand.choice(eligible)

    def bucket_for_aspect(self, aspect: float, target_resolution: int) -> tuple[int, int]:
        return target_resolution, target_resolution

    def reconstruct_bucket_resolutions(self, target_resolution: int) -> list[tuple[int, int]]:
        """Bucket grid for ``target_resolution`` without ``start()`` (see AspectBucketing)."""
        return [(int(target_resolution), int(target_resolution))]

    def reconstruct_bucket_for_aspect(self, aspect: float, target_resolution: int) -> tuple[int, int]:
        return int(target_resolution), int(target_resolution)

    def compute_bucket_method_hash(self) -> str:
        payload = {
            "method": "single_aspect_v2_adaptive" if self.adaptive_resolution else "single_aspect_v1_no_upscale",
            "adaptive_resolution": self.adaptive_resolution,
            "additional_resolution_inputs": self.additional_resolution_in_names,
            "resolutions": self.possible_target_resolutions,
            "fixed_resolutions": self.fixed_target_resolutions,
        }
        if self.upscale_tolerance > 0.0:
            payload["upscale_tolerance"] = round(self.upscale_tolerance, 6)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def variant_key_from_aspect(self, variation: int, index: int, aspect: float) -> str | None:
        # Source dimensions are required for the no-upscale policy.
        return None

    def variant_key_from_resolution(self, variation, index, resolution) -> str | None:
        return self.variant_key_from_resolutions(variation, index, resolution, [])

    def variant_key_from_resolutions(self, variation, index, resolution, additional_resolutions) -> str | None:
        try:
            spec = self._target_resolutions_for_item(variation, index)
        except Exception:
            return None
        if isinstance(spec, str) and "x" in spec and "," not in spec:
            return None
        targets = target_resolutions_for_spec(
            str(spec),
            self.adaptive_resolution,
            self.adaptive_resolution_ladder,
        )
        if not targets:
            return None
        target = self._choose_target_resolution(
            self._get_rand(variation, index), resolution, targets, additional_resolutions
        )
        return f"{target}x{target}"

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        rand = self._get_rand(variation, index)
        resolution = self._get_previous_item(variation, self.resolution_in_name, index)
        additional_resolutions = self._additional_resolutions(variation, index)
        target_resolutions = self._target_resolutions_for_item(variation, index)

        if isinstance(target_resolutions, str) and "x" in target_resolutions and "," not in target_resolutions:
            width, height = (int(part.strip()) for part in target_resolutions.split("x"))
            target_resolution = (height, width)
        else:
            target_resolutions = target_resolutions_for_spec(
                str(target_resolutions),
                self.adaptive_resolution,
                self.adaptive_resolution_ladder,
            )
            target = self._choose_target_resolution(rand, resolution, target_resolutions, additional_resolutions)
            target_resolution = (target, target)

        aspect = resolution[-2] / resolution[-1]
        target_aspect = target_resolution[0] / target_resolution[1]

        if aspect > target_aspect:
            scale = target_resolution[1] / resolution[-1]
            scale_resolution = (round(resolution[-2] * scale), target_resolution[1])
        else:
            scale = target_resolution[0] / resolution[-2]
            scale_resolution = (target_resolution[0], round(resolution[-1] * scale))

        return {
            self.scale_resolution_out_name: scale_resolution,
            self.crop_resolution_out_name: target_resolution,
        }
