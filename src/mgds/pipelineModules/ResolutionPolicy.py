import math
from collections.abc import Iterable

ADAPTIVE_MIN_RESOLUTION = 128
DEFAULT_ADAPTIVE_RESOLUTION_LADDER = (1024, 768, 512, 256, 128)


def parse_scalar_resolutions(spec: str) -> list[int]:
    """Parse a scalar/comma resolution specification.

    Exact ``WxH`` specifications deliberately return an empty list: adaptive
    resolution does not reinterpret fixed rectangles.
    """
    if not isinstance(spec, str):
        return []
    spec = spec.strip()
    if not spec or ("x" in spec.lower() and "," not in spec):
        return []
    return [int(value.strip()) for value in spec.split(",") if value.strip()]


def parse_adaptive_resolution_ladder(ladder: str | Iterable[int] | None = None) -> list[int]:
    """Return a validated, de-duplicated adaptive snap ladder.

    Values are returned in ascending order because resolution selectors walk
    them from the smallest fallback to the largest supported tier. ``None``
    preserves the original built-in ladder for backwards compatibility.
    """
    if ladder is None:
        values = list(DEFAULT_ADAPTIVE_RESOLUTION_LADDER)
    elif isinstance(ladder, str):
        if not ladder.strip():
            raise ValueError("adaptive resolution ladder must contain at least one resolution")
        try:
            values = [int(value.strip()) for value in ladder.split(",") if value.strip()]
        except ValueError as exc:
            raise ValueError("adaptive resolution ladder must be a comma-separated list of integers") from exc
    else:
        try:
            values = [int(value) for value in ladder]
        except (TypeError, ValueError) as exc:
            raise ValueError("adaptive resolution ladder must contain integers") from exc

    if not values:
        raise ValueError("adaptive resolution ladder must contain at least one resolution")
    if any(value <= 0 for value in values):
        raise ValueError("adaptive resolution ladder values must be positive integers")
    return sorted(set(values))


def adaptive_resolution_tiers(spec: str, ladder: str | Iterable[int] | None = None) -> list[int]:
    """Return ``ladder`` capped by ``spec``'s maximum, including that maximum."""
    configured = parse_scalar_resolutions(spec)
    if not configured:
        return []

    maximum = max(configured)
    tiers = {maximum}
    tiers.update(value for value in parse_adaptive_resolution_ladder(ladder) if value <= maximum)
    return sorted(tiers)


def target_resolutions_for_spec(
    spec: str,
    adaptive: bool,
    adaptive_resolution_ladder: str | Iterable[int] | None = None,
) -> list[int]:
    return adaptive_resolution_tiers(spec, adaptive_resolution_ladder) if adaptive else parse_scalar_resolutions(spec)


def scale_resolution_for_crop(
    source_resolution: tuple[int, ...] | list[int],
    crop_resolution: tuple[int, int] | list[int],
) -> tuple[int, int]:
    """Mirror the scale-before-crop calculation used by resolution modules."""
    source_h = int(source_resolution[-2])
    source_w = int(source_resolution[-1])
    crop_h = int(crop_resolution[-2])
    crop_w = int(crop_resolution[-1])

    source_aspect = source_h / source_w
    crop_aspect = crop_h / crop_w
    if source_aspect > crop_aspect:
        scale = crop_w / source_w
        return round(source_h * scale), crop_w
    scale = crop_h / source_h
    return crop_h, round(source_w * scale)


def upscale_tolerance_fraction(value) -> float:
    """``value`` as a clean fraction in ``[0, 1)``; anything unset or invalid is 0 (no upscale)."""
    try:
        fraction = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if fraction != fraction or fraction <= 0.0:  # NaN or non-positive
        return 0.0
    return min(fraction, 0.99)


def all_resolutions_cover_scale(
    resolutions: Iterable[tuple[int, ...] | list[int]],
    scale_resolution: tuple[int, int],
    upscale_tolerance: float = 0.0,
) -> bool:
    """Every resolution reaches ``scale_resolution`` in both dimensions.

    ``upscale_tolerance`` (a fraction, e.g. 0.03) lets a source fall short of
    the scale size by up to that fraction of it -- the image is then enlarged
    by at most that much to fill the bucket. 0 keeps the exact no-upscale rule.
    """
    scale_h, scale_w = scale_resolution
    tolerance = upscale_tolerance_fraction(upscale_tolerance)
    if tolerance > 0.0:
        scale_h = scale_h * (1.0 - tolerance)
        scale_w = scale_w * (1.0 - tolerance)
    return all(int(resolution[-2]) >= scale_h and int(resolution[-1]) >= scale_w for resolution in resolutions)


def nominal_resolution_for_crop(crop_resolution: tuple[int, int] | list[int]) -> float:
    return math.sqrt(int(crop_resolution[-2]) * int(crop_resolution[-1]))
