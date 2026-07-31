"""
Compress and downscale images before they are sent to a VLM.

Full-resolution PNG screenshots dominate multimodal token usage. Keep OCR /
click coordinates on the original capture; only the model-facing payload is
resized and JPEG-encoded here.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MAX_SIDE: int = 1280
DEFAULT_JPEG_QUALITY: int = 72
_MIN_BYTES_TO_PREP: int = 180_000


def prepare_image_bytes_for_vlm(
    data: bytes,
    *,
    max_side: int = DEFAULT_MAX_SIDE,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    mime_type: str | None = None,
    info: dict[str, Any] | None = None,
) -> tuple[bytes, str]:
    """Return ``(bytes, mime_type)`` suitable for a cheaper VLM image part.

    No-ops when the payload is already small or OpenCV is unavailable.

    When ``info`` is provided and the prepared image is used, it is populated
    with ``vlm_scale`` (prepared / original size ratio, 1.0 when not resized)
    and ``original_size`` ([width, height] of the decoded input).
    """

    if not data or len(data) < _MIN_BYTES_TO_PREP:
        return data, mime_type or "image/png"

    try:
        import cv2
        import numpy as np
    except Exception:
        return data, mime_type or "image/png"

    try:
        array = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None or getattr(image, "size", 0) == 0:
            return data, mime_type or "image/png"

        height, width = image.shape[:2]
        longest = max(height, width)
        scale = 1.0
        if longest > max_side:
            scale = max_side / float(longest)
            image = cv2.resize(
                image,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )

        quality = max(40, min(95, int(jpeg_quality)))
        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if not ok:
            return data, mime_type or "image/png"
        prepared = encoded.tobytes()
        if len(prepared) >= len(data):
            return data, mime_type or "image/png"
        if info is not None:
            info["vlm_scale"] = scale
            info["original_size"] = [int(width), int(height)]
        logger.debug(
            "Prepared VLM image: %d -> %d bytes (max_side=%d, q=%d)",
            len(data),
            len(prepared),
            max_side,
            quality,
        )
        return prepared, "image/jpeg"
    except Exception as error:
        logger.debug("VLM image prep skipped: %s", error)
        return data, mime_type or "image/png"


def prepare_numpy_image_for_vlm(
    image: Any,
    *,
    max_side: int = DEFAULT_MAX_SIDE,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    info: dict[str, Any] | None = None,
) -> tuple[bytes, str] | None:
    """Encode a BGR/RGB numpy screenshot for VLM use.

    When ``info`` is provided and encoding succeeds, it is populated with
    ``vlm_scale`` (prepared / original size ratio, 1.0 when not resized) and
    ``original_size`` ([width, height] of the input image).
    """

    try:
        import cv2
        import numpy as np
    except Exception:
        return None

    if not isinstance(image, np.ndarray) or image.size == 0:
        return None

    prepared = image
    if prepared.ndim == 2:
        prepared = cv2.cvtColor(prepared, cv2.COLOR_GRAY2BGR)
    elif prepared.ndim == 3 and prepared.shape[2] == 4:
        prepared = cv2.cvtColor(prepared, cv2.COLOR_BGRA2BGR)

    height, width = prepared.shape[:2]
    longest = max(height, width)
    scale = 1.0
    if longest > max_side:
        scale = max_side / float(longest)
        prepared = cv2.resize(
            prepared,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    quality = max(40, min(95, int(jpeg_quality)))
    ok, encoded = cv2.imencode(
        ".jpg",
        prepared,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
        return None
    if info is not None:
        info["vlm_scale"] = scale
        info["original_size"] = [int(width), int(height)]
    return encoded.tobytes(), "image/jpeg"


__all__ = [
    "DEFAULT_JPEG_QUALITY",
    "DEFAULT_MAX_SIDE",
    "prepare_image_bytes_for_vlm",
    "prepare_numpy_image_for_vlm",
]
