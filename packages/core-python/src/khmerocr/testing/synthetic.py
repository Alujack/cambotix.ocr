"""Synthetic ID-card images with clearly fake values.

The generated cards follow the engine's *provisional* layout template, not a
real card design. They are marked SPECIMEN and must only be used to test the
software pipeline (geometry, quality checks, field mapping); accuracy on these
images says nothing about accuracy on real cards.

Correct Khmer shaping requires Pillow built with libraqm and a Khmer font
(set ``KHMEROCR_FONT`` or install Noto Sans Khmer).
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..documents.layout import DocumentTemplate, load_builtin_template
from ..documents.national_id import DEFAULT_TEMPLATE_ID
from ..preprocessing.io import ImageArray

_KHMER_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/noto/NotoSansKhmer-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansKhmer-Regular.ttf",
    "/Library/Fonts/NotoSansKhmer-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Khmer Sangam MN.ttf",
    "C:/Windows/Fonts/KhmerUI.ttf",
)
_LATIN_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
)

_FAKE_KHMER_NAMES = ["គំរូ សាកល្បង", "សុខ គំរូ", "ចាន់ សាកល្បង", "កែវ គំរូ"]
_FAKE_LATIN_NAMES = ["KUMRU SAKALBANG", "SOK KUMRU", "CHAN SAKALBANG", "KEO KUMRU"]


@dataclass(frozen=True)
class SyntheticIdentity:
    id_number: str = "000000001"
    full_name_km: str = "គំរូ សាកល្បង"
    full_name_latin: str = "KUMRU SAKALBANG"
    date_of_birth: str = "01.01.1990"
    gender: str = "M"

    def expected_fields(self) -> dict[str, str]:
        day, month, year = self.date_of_birth.split(".")
        expected = asdict(self)
        expected["date_of_birth"] = f"{year}-{month}-{day}"
        return expected


def random_identity(rng: random.Random) -> SyntheticIdentity:
    index = rng.randrange(len(_FAKE_KHMER_NAMES))
    return SyntheticIdentity(
        # The 00 prefix keeps synthetic numbers visibly fake.
        id_number="00" + "".join(str(rng.randrange(10)) for _ in range(7)),
        full_name_km=_FAKE_KHMER_NAMES[index],
        full_name_latin=_FAKE_LATIN_NAMES[index],
        date_of_birth=(f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}."
                       f"{rng.randint(1950, 2005)}"),
        gender=rng.choice("MF"),
    )


def find_font(candidates: tuple[str, ...], env_var: str | None = None) -> str | None:
    if env_var and os.environ.get(env_var):
        return os.environ[env_var]
    return next((path for path in candidates if Path(path).is_file()), None)


def render_card(identity: SyntheticIdentity | None = None,
                template: DocumentTemplate | None = None) -> ImageArray:
    identity = identity or SyntheticIdentity()
    template = template or load_builtin_template(DEFAULT_TEMPLATE_ID)
    width, height = template.canonical_size
    card = Image.new("RGB", (width, height), (226, 233, 240))
    draw = ImageDraw.Draw(card)

    # Light guilloche-like background lines so the card is not flat white.
    for line_y in range(0, height, 14):
        draw.line([(0, line_y), (width, line_y + 40)], fill=(214, 224, 234), width=1)
    draw.rectangle([0, 0, width - 1, int(height * 0.16)], fill=(186, 205, 226))
    draw.rectangle([int(width * 0.03), int(height * 0.22), int(width * 0.29),
                    int(height * 0.86)], fill=(160, 168, 178))

    khmer_font = _font(find_font(_KHMER_FONT_CANDIDATES, "KHMEROCR_FONT"), int(height * 0.052))
    latin_font = _font(find_font(_LATIN_FONT_CANDIDATES), int(height * 0.055))
    title_font = _font(find_font(_LATIN_FONT_CANDIDATES), int(height * 0.06))
    draw.text((int(width * 0.32), int(height * 0.04)), "SPECIMEN - SYNTHETIC TEST CARD",
              fill=(40, 40, 60), font=title_font)

    gender_km = {"M": "ប្រុស", "F": "ស្រី"}[identity.gender]
    lines = {
        "id_number": (identity.id_number, latin_font),
        "full_name_km": (f"គោត្តនាមនិងនាម: {identity.full_name_km}", khmer_font),
        "full_name_latin": (identity.full_name_latin, latin_font),
        "date_of_birth": (f"ថ្ងៃខែឆ្នាំកំណើត: {identity.date_of_birth}", khmer_font),
        "gender": (f"ភេទ: {gender_km}", khmer_font),
    }
    for spec in template.fields:
        if spec.name not in lines:
            continue
        text, font = lines[spec.name]
        x, y, _, h = spec.box
        draw.text((int((x + 0.01) * width), int((y + h / 2) * height)), text,
                  fill=(20, 20, 30), font=font, anchor="lm")
    return np.asarray(card, dtype=np.uint8).copy()


def place_on_background(card: ImageArray, *, canvas: tuple[int, int] = (1600, 1200),
                        angle: float = 0.0, scale: float = 0.75, perspective: float = 0.0,
                        background: tuple[int, int, int] = (60, 52, 45),
                        seed: int = 0) -> ImageArray:
    """Photograph-like placement: rotate, scale, and skew the card on a backdrop."""
    rng = np.random.default_rng(seed)
    cw, ch = canvas
    h, w = card.shape[:2]
    target_w = cw * scale
    target_h = target_w * h / w
    center = np.array([cw / 2, ch / 2])
    corners = np.array([[-target_w / 2, -target_h / 2], [target_w / 2, -target_h / 2],
                        [target_w / 2, target_h / 2], [-target_w / 2, target_h / 2]])
    theta = np.deg2rad(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    jitter = rng.uniform(-perspective, perspective, size=(4, 2)) * target_w
    dst = (corners @ rotation.T + center + jitter).astype(np.float32)
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    noise = rng.normal(0, 6, size=(ch, cw, 3))
    backdrop = np.clip(np.array(background, dtype=np.float64) + noise, 0, 255).astype(np.uint8)
    warped = cv2.warpPerspective(card, matrix, (cw, ch), borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), matrix, (cw, ch))
    return np.where(mask[:, :, None] > 0, warped, backdrop).astype(np.uint8)


def add_blur(image: ImageArray, sigma: float) -> ImageArray:
    return np.asarray(cv2.GaussianBlur(image, (0, 0), sigma), dtype=np.uint8)


def add_glare(image: ImageArray, *, center: tuple[float, float] = (0.6, 0.45),
              radius: float = 0.1, intensity: float = 220.0) -> ImageArray:
    """Add a specular hotspot. Raise ``intensity``/``radius`` for glare that
    destroys the text underneath."""
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.hypot(xx - center[0] * w, yy - center[1] * h) / (radius * max(h, w))
    strength = np.clip(1.4 - dist, 0, 1)[:, :, None]
    glared = image.astype(np.float64) + strength * intensity
    return np.clip(glared, 0, 255).astype(np.uint8)


def to_png_bytes(image: ImageArray) -> bytes:
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return encoded.tobytes()


def _font(path: str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)
