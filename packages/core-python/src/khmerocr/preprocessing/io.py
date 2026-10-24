"""Safe image loading.

Images are decoded in memory. The container format is identified from magic
bytes, never from a file extension or a declared content type, and the pixel
count is checked before the pixel data is decoded.
"""

from __future__ import annotations

import io
import os
import warnings
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

from ..errors import InvalidImageError

ImageArray = NDArray[np.uint8]
ImageInput = Union[str, "os.PathLike[str]", bytes, bytearray, memoryview, ImageArray, Image.Image]

SUPPORTED_FORMATS: dict[str, str] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


def sniff_format(data: bytes) -> str | None:
    """Return the PIL format name for supported magic bytes, else ``None``."""
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    return None


def load_image(source: ImageInput, *, max_bytes: int, max_pixels: int) -> ImageArray:
    """Decode ``source`` into an RGB ``uint8`` array of shape ``(H, W, 3)``."""
    if isinstance(source, np.ndarray):
        return _from_array(source, max_pixels)
    if isinstance(source, Image.Image):
        _check_pixels(source.width, source.height, max_pixels)
        return np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
    if isinstance(source, (bytes, bytearray, memoryview)):
        data = bytes(source)
    elif isinstance(source, (str, os.PathLike)):
        data = _read_file(Path(source), max_bytes)
    else:
        raise InvalidImageError(f"Unsupported image input type: {type(source).__name__}")
    return decode_image_bytes(data, max_bytes=max_bytes, max_pixels=max_pixels)


def decode_image_bytes(data: bytes, *, max_bytes: int, max_pixels: int) -> ImageArray:
    if not data:
        raise InvalidImageError("The image is empty.")
    if len(data) > max_bytes:
        raise InvalidImageError(
            "The image exceeds the maximum allowed size.",
            details={"max_bytes": max_bytes},
        )
    fmt = sniff_format(data)
    if fmt is None:
        raise InvalidImageError(
            "Unsupported or unrecognized image format. Supported formats: JPEG, PNG, WebP."
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data), formats=[fmt]) as img:
                _check_pixels(img.width, img.height, max_pixels)
                img.load()
                img = ImageOps.exif_transpose(img) or img
                rgb = img.convert("RGB")
    except InvalidImageError:
        raise
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise InvalidImageError("The image could not be decoded.") from exc
    return np.asarray(rgb, dtype=np.uint8).copy()


def _read_file(path: Path, max_bytes: int) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise InvalidImageError("The image file could not be read.") from exc
    if size > max_bytes:
        raise InvalidImageError(
            "The image exceeds the maximum allowed size.",
            details={"max_bytes": max_bytes},
        )
    try:
        return path.read_bytes()
    except OSError as exc:
        raise InvalidImageError("The image file could not be read.") from exc


def _from_array(array: NDArray[np.generic], max_pixels: int) -> ImageArray:
    if array.dtype != np.uint8:
        raise InvalidImageError("Image arrays must have dtype uint8.")
    if array.ndim == 2:
        _check_pixels(array.shape[1], array.shape[0], max_pixels)
        return np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim == 3 and array.shape[2] in (3, 4):
        _check_pixels(array.shape[1], array.shape[0], max_pixels)
        return np.ascontiguousarray(array[:, :, :3])
    raise InvalidImageError("Image arrays must have shape (H, W), (H, W, 3) or (H, W, 4).")


def _check_pixels(width: int, height: int, max_pixels: int) -> None:
    if width <= 0 or height <= 0:
        raise InvalidImageError("The image has no pixels.")
    if width * height > max_pixels:
        raise InvalidImageError(
            "The image dimensions exceed the maximum allowed pixel count.",
            details={"max_pixels": max_pixels},
        )
