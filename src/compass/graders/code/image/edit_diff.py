"""Image difference utilities for edit evaluation graders.

Shared tools for computing pixel-level differences between original
and edited images, used by EditPreservationGrader and EditLocalityGrader.
"""

from __future__ import annotations

import math

from PIL import Image, ImageChops, ImageStat


def ensure_comparable(
    original: Image.Image, edited: Image.Image
) -> tuple[Image.Image, Image.Image]:
    """Ensure two images can be compared: unify mode and size.

    - Converts both to RGB if they aren't already.
    - Resizes ``edited`` to match ``original`` if sizes differ.
    """
    if original.mode != "RGB":
        original = original.convert("RGB")
    if edited.mode != "RGB":
        edited = edited.convert("RGB")
    if edited.size != original.size:
        edited = edited.resize(original.size, Image.LANCZOS)
    return original, edited


def compute_diff_image(
    original: Image.Image, edited: Image.Image
) -> Image.Image:
    """Compute per-pixel absolute difference (RGB)."""
    original, edited = ensure_comparable(original, edited)
    return ImageChops.difference(original, edited)


def compute_change_mask(
    original: Image.Image,
    edited: Image.Image,
    pixel_threshold: int = 10,
) -> Image.Image:
    """Binary change mask: pixels differing above *pixel_threshold* are white.

    Returns an ``"L"`` mode image (single channel, 0 or 255).
    """
    diff = compute_diff_image(original, edited).convert("L")
    return diff.point(lambda v: 255 if v > pixel_threshold else 0, mode="L")


def compute_change_ratio(change_mask: Image.Image) -> float:
    """Fraction of white (changed) pixels in a binary mask."""
    stat = ImageStat.Stat(change_mask)
    # mean of L channel: 0..255  →  fraction of 255-pixels
    return stat.mean[0] / 255.0


def compute_mse_in_region(
    original: Image.Image,
    edited: Image.Image,
    region_mask: Image.Image,
) -> float:
    """Mean Squared Error computed only inside *region_mask* (white pixels).

    Both images must already be comparable (same mode/size).
    ``region_mask`` must be mode ``"L"`` with same size.
    """
    original, edited = ensure_comparable(original, edited)
    if region_mask.mode != "L":
        region_mask = region_mask.convert("L")
    if region_mask.size != original.size:
        region_mask = region_mask.resize(original.size, Image.LANCZOS)

    orig_pixels = list(original.get_flattened_data())
    edit_pixels = list(edited.get_flattened_data())
    mask_pixels = list(region_mask.get_flattened_data())

    total_err = 0.0
    count = 0
    for o, e, m in zip(orig_pixels, edit_pixels, mask_pixels, strict=True):
        if m > 127:  # inside mask
            total_err += sum((oc - ec) ** 2 for oc, ec in zip(o, e, strict=True)) / 3.0
            count += 1

    if count == 0:
        return 0.0
    return total_err / count


def mse_to_similarity(mse: float, max_mse: float = 255.0 * 255.0) -> float:
    """Convert MSE to a similarity score in [0, 1] (1 = identical)."""
    return max(0.0, 1.0 - mse / max_mse)


def compute_histogram_similarity(
    original: Image.Image,
    edited: Image.Image,
    mask: Image.Image | None = None,
) -> float:
    """Histogram correlation between two images, optionally restricted to *mask*.

    Returns a similarity in [0, 1].
    """
    original, edited = ensure_comparable(original, edited)

    pil_mask: Image.Image | None = None
    if mask is not None:
        pil_mask = mask.convert("L") if mask.mode != "L" else mask
        if pil_mask.size != original.size:
            pil_mask = pil_mask.resize(original.size, Image.LANCZOS)

    # Collect per-channel histograms (each 256 bins)
    hist_orig = original.histogram(mask=pil_mask)
    hist_edit = edited.histogram(mask=pil_mask)

    # Normalised correlation across all channels
    return _histogram_correlation(hist_orig, hist_edit)


def _histogram_correlation(h1: list[int], h2: list[int]) -> float:
    """Pearson-like correlation between two histograms, mapped to [0, 1]."""
    n = len(h1)
    if n == 0:
        return 1.0

    mean1 = sum(h1) / n
    mean2 = sum(h2) / n

    num = sum((a - mean1) * (b - mean2) for a, b in zip(h1, h2, strict=True))
    den1 = math.sqrt(sum((a - mean1) ** 2 for a in h1))
    den2 = math.sqrt(sum((b - mean2) ** 2 for b in h2))

    if den1 == 0 or den2 == 0:
        return 1.0  # both constant → identical distribution

    corr = num / (den1 * den2)  # in [-1, 1]
    return (corr + 1.0) / 2.0   # map to [0, 1]


def load_edit_mask(
    context_metadata: dict,
    image_size: tuple[int, int],
) -> tuple[Image.Image | None, str]:
    """Load an edit region mask from context metadata.

    Priority:
      1. ``metadata["edit_mask"]`` — PIL Image or file path
      2. ``metadata["edit_region"]`` — ``[x1, y1, x2, y2]`` bbox → rectangle mask
      3. Nothing available → ``(None, "none")``

    Returns:
        ``(mask_image_or_None, source_label)``
    """
    # 1. Explicit mask
    raw_mask = context_metadata.get("edit_mask")
    if raw_mask is not None:
        if isinstance(raw_mask, Image.Image):
            mask = raw_mask.convert("L") if raw_mask.mode != "L" else raw_mask
            if mask.size != image_size:
                mask = mask.resize(image_size, Image.LANCZOS)
            return mask, "provided"
        if isinstance(raw_mask, str):
            mask = Image.open(raw_mask).convert("L")
            if mask.size != image_size:
                mask = mask.resize(image_size, Image.LANCZOS)
            return mask, "provided"

    # 2. Bounding box
    bbox = context_metadata.get("edit_region")
    if bbox is not None and len(bbox) == 4:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        mask = Image.new("L", image_size, 0)
        for x in range(max(0, x1), min(image_size[0], x2)):
            for y in range(max(0, y1), min(image_size[1], y2)):
                mask.putpixel((x, y), 255)
        return mask, "bbox"

    # 3. Nothing
    return None, "none"


def invert_mask(mask: Image.Image) -> Image.Image:
    """Invert a binary mask (swap black ↔ white)."""
    return ImageChops.invert(mask.convert("L"))
