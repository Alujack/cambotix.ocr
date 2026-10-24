from __future__ import annotations

import io

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from khmerocr.errors import InvalidImageError
from khmerocr.preprocessing import load_image, sniff_format
from PIL import Image

LIMITS = {"max_bytes": 10 * 1024 * 1024, "max_pixels": 40_000_000}


def _encode(fmt: str, size: tuple[int, int] = (64, 48)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (200, 10, 10)).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_supported_formats_decode(fmt: str) -> None:
    data = _encode(fmt)
    assert sniff_format(data) == fmt
    image = load_image(data, **LIMITS)
    assert image.shape == (48, 64, 3) and image.dtype == np.uint8


def test_loads_from_path(tmp_path) -> None:
    path = tmp_path / "card.png"
    path.write_bytes(_encode("PNG"))
    assert load_image(path, **LIMITS).shape == (48, 64, 3)


def test_rejects_unsupported_format_even_with_image_extension(tmp_path) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buffer, format="GIF")
    path = tmp_path / "card.png"
    path.write_bytes(buffer.getvalue())
    with pytest.raises(InvalidImageError):
        load_image(path, **LIMITS)


def test_rejects_empty_and_oversized() -> None:
    with pytest.raises(InvalidImageError):
        load_image(b"", **LIMITS)
    with pytest.raises(InvalidImageError):
        load_image(_encode("PNG"), max_bytes=10, max_pixels=LIMITS["max_pixels"])


def test_rejects_pixel_bomb_before_decoding() -> None:
    data = _encode("PNG", size=(2000, 2000))
    with pytest.raises(InvalidImageError) as excinfo:
        load_image(data, max_bytes=LIMITS["max_bytes"], max_pixels=1_000_000)
    assert excinfo.value.details["max_pixels"] == 1_000_000


def test_rejects_truncated_image() -> None:
    data = _encode("JPEG", size=(400, 300))
    with pytest.raises(InvalidImageError):
        load_image(data[: len(data) // 3], **LIMITS)


def test_array_inputs() -> None:
    gray = np.zeros((20, 30), dtype=np.uint8)
    assert load_image(gray, **LIMITS).shape == (20, 30, 3)
    rgba = np.zeros((20, 30, 4), dtype=np.uint8)
    assert load_image(rgba, **LIMITS).shape == (20, 30, 3)
    with pytest.raises(InvalidImageError):
        load_image(np.zeros((20, 30), dtype=np.float32), **LIMITS)


@settings(max_examples=200)
@given(st.binary(max_size=2048))
def test_arbitrary_bytes_only_raise_invalid_image(data: bytes) -> None:
    try:
        load_image(data, **LIMITS)
    except InvalidImageError:
        pass


@settings(max_examples=100)
@given(st.binary(min_size=1, max_size=512))
def test_corrupted_png_body_only_raises_invalid_image(noise: bytes) -> None:
    data = _encode("PNG")
    corrupted = data[:16] + noise + data[16 + len(noise):]
    try:
        load_image(corrupted, **LIMITS)
    except InvalidImageError:
        pass
