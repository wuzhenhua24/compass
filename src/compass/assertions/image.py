"""Image assertion functions."""

from typing import Literal

from PIL import Image


class ImageAssertionError(AssertionError):
    """Custom exception for image assertion failures."""

    pass


def assert_image_size(
    image: Image.Image,
    width: int | None = None,
    height: int | None = None,
    min_width: int | None = None,
    min_height: int | None = None,
    max_width: int | None = None,
    max_height: int | None = None,
) -> None:
    """Assert image dimensions.

    Args:
        image: Image to check.
        width: Exact width required.
        height: Exact height required.
        min_width: Minimum width.
        min_height: Minimum height.
        max_width: Maximum width.
        max_height: Maximum height.

    Raises:
        ImageAssertionError: If assertion fails.
    """
    img_width, img_height = image.size

    if width is not None and img_width != width:
        raise ImageAssertionError(
            f"Image width {img_width} does not match expected {width}"
        )

    if height is not None and img_height != height:
        raise ImageAssertionError(
            f"Image height {img_height} does not match expected {height}"
        )

    if min_width is not None and img_width < min_width:
        raise ImageAssertionError(
            f"Image width {img_width} is less than minimum {min_width}"
        )

    if min_height is not None and img_height < min_height:
        raise ImageAssertionError(
            f"Image height {img_height} is less than minimum {min_height}"
        )

    if max_width is not None and img_width > max_width:
        raise ImageAssertionError(
            f"Image width {img_width} exceeds maximum {max_width}"
        )

    if max_height is not None and img_height > max_height:
        raise ImageAssertionError(
            f"Image height {img_height} exceeds maximum {max_height}"
        )


def assert_image_format(
    image: Image.Image,
    format: str | list[str],
) -> None:
    """Assert image format.

    Args:
        image: Image to check.
        format: Expected format(s) (e.g., "PNG", "JPEG", ["PNG", "WEBP"]).

    Raises:
        ImageAssertionError: If assertion fails.
    """
    if isinstance(format, str):
        format = [format]

    format = [f.upper() for f in format]
    img_format = (image.format or "").upper()

    if img_format not in format:
        raise ImageAssertionError(
            f"Image format '{img_format}' not in expected formats: {format}"
        )


def assert_image_mode(
    image: Image.Image,
    mode: str | list[str],
) -> None:
    """Assert image color mode.

    Args:
        image: Image to check.
        mode: Expected mode(s) (e.g., "RGB", "RGBA", ["RGB", "L"]).

    Raises:
        ImageAssertionError: If assertion fails.
    """
    if isinstance(mode, str):
        mode = [mode]

    if image.mode not in mode:
        raise ImageAssertionError(
            f"Image mode '{image.mode}' not in expected modes: {mode}"
        )


def assert_image_aspect_ratio(
    image: Image.Image,
    ratio: float,
    tolerance: float = 0.01,
) -> None:
    """Assert image aspect ratio.

    Args:
        image: Image to check.
        ratio: Expected aspect ratio (width/height).
        tolerance: Allowed deviation from ratio.

    Raises:
        ImageAssertionError: If assertion fails.
    """
    width, height = image.size
    actual_ratio = width / height

    if abs(actual_ratio - ratio) > tolerance:
        raise ImageAssertionError(
            f"Image aspect ratio {actual_ratio:.3f} differs from expected {ratio:.3f} "
            f"(tolerance: {tolerance})"
        )


def assert_image_not_blank(
    image: Image.Image,
    threshold: float = 0.01,
) -> None:
    """Assert image is not blank (all same color).

    Args:
        image: Image to check.
        threshold: Minimum variance threshold.

    Raises:
        ImageAssertionError: If image appears blank.
    """
    import statistics

    # Convert to grayscale for simplicity
    gray = image.convert("L")
    pixels = list(gray.getdata())

    if len(set(pixels)) == 1:
        raise ImageAssertionError("Image is blank (all pixels same color)")

    # Check variance
    try:
        variance = statistics.variance(pixels)
        if variance < threshold:
            raise ImageAssertionError(
                f"Image variance {variance} below threshold {threshold}, appears nearly blank"
            )
    except statistics.StatisticsError:
        pass  # Not enough data points


def assert_image_has_transparency(image: Image.Image) -> None:
    """Assert image has alpha channel with some transparency.

    Args:
        image: Image to check.

    Raises:
        ImageAssertionError: If image has no transparency.
    """
    if image.mode not in ("RGBA", "LA", "PA"):
        raise ImageAssertionError(
            f"Image mode '{image.mode}' does not support transparency"
        )

    # Check if any pixels are actually transparent
    if image.mode == "RGBA":
        alpha = image.split()[3]
        if alpha.getextrema() == (255, 255):
            raise ImageAssertionError("Image has alpha channel but no transparent pixels")
