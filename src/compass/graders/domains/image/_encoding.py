"""Getting an image onto the wire.

Every VLM in this domain takes its image the same way — PNG bytes, base64, in
a JSON body — so the conversion is one function rather than a private method on
each grader that happens to call one.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


def image_to_base64(image: Image.Image) -> str:
    """A PIL image as a base64-encoded PNG."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
