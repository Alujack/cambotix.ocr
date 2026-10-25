"""Render synthetic text-line images that look like crops from ID cards.

Each sample is rendered with one Khmer font and one Latin font (lines mix
scripts, as on bilingual cards), placed on a card-like background, degraded
with photo artifacts, and then passed through the *same* ``trim_to_ink`` crop
the SDK applies at inference, so training and inference inputs match.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fontTools.ttLib import TTFont
from khmerocr.recognition.onnx_ctc import trim_to_ink
from PIL import Image, ImageDraw, ImageFont

INPUT_HEIGHT = 32
MAX_WIDTH = 512
FONT_SIZES = list(range(22, 66, 4))
# Share of lines drawn with a thickened stroke. Weight axes only reach the
# variable fonts; a stroke makes any font as heavy as card print.
BOLD_RATE = 0.25
# Muol-style fonts are already heavy; card names are often printed in a
# lighter cut (Muol Light), so heading lines are thickened less often and
# sometimes thinned instead.
HEADING_BOLD_RATE = 0.1
HEADING_THIN_RATE = 0.4


@dataclass(frozen=True)
class FontSpec:
    path: str
    script: str  # "khmer" | "latin"
    variable: bool
    holdout: bool
    heading: bool  # Muol style, used for names on ID cards
    cmap: frozenset[int]


def load_fonts(training_dir: Path, *, holdout: bool) -> tuple[list[FontSpec], list[FontSpec]]:
    """Return (khmer, latin) fonts for training (``holdout=False``) or evaluation."""
    spec = json.loads((training_dir / "fonts.json").read_text(encoding="utf-8"))
    root = training_dir / "data" / "fonts"
    khmer: list[FontSpec] = []
    latin: list[FontSpec] = []
    for entry in spec["fonts"]:
        is_holdout = bool(entry.get("holdout", False))
        path = root / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing; run training/fetch_fonts.py")
        cmap = frozenset(TTFont(str(path), lazy=True).getBestCmap() or {})
        font = FontSpec(str(path), entry["script"], bool(entry.get("variable")), is_holdout,
                        bool(entry.get("heading")), cmap)
        # Holdout evaluation still needs Latin fonts for mixed lines; use any.
        if is_holdout == holdout or (holdout and entry["script"] == "latin"):
            (khmer if entry["script"] == "khmer" else latin).append(font)
    return khmer, latin


@lru_cache(maxsize=512)
def _pil_font(path: str, size: int, weight: int | None) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)
    if weight is not None:
        try:
            axes = font.get_variation_axes()
            values = []
            for axis in axes:
                if axis.get("name", b"") in (b"Weight", "Weight"):
                    values.append(max(axis["minimum"], min(axis["maximum"], weight)))
                else:
                    values.append(axis.get("default", axis["minimum"]))
            font.set_variation_by_axes(values)
        except (OSError, KeyError):
            pass
    return font


def _is_khmer(ch: str) -> bool:
    return 0x1780 <= ord(ch) <= 0x17FF


def _runs(text: str, khmer_cmap: frozenset[int]) -> list[tuple[str, bool]]:
    """Split into (segment, is_khmer) runs. Khmer characters use the Khmer
    font, Latin letters and digits the Latin font; spaces and punctuation stay
    in the current run when its font has them, so separators between Khmer
    digits are drawn at the same size as the digits."""
    runs: list[tuple[str, bool]] = []
    for ch in text:
        if _is_khmer(ch):
            khmer = True
        elif ch.isalnum():
            khmer = False
        elif runs and (not runs[-1][1] or ord(ch) in khmer_cmap):
            khmer = runs[-1][1]
        else:
            khmer = False
        if runs and runs[-1][1] == khmer:
            runs[-1] = (runs[-1][0] + ch, khmer)
        else:
            runs.append((ch, khmer))
    return runs


class LineRenderer:
    def __init__(self, khmer_fonts: list[FontSpec], latin_fonts: list[FontSpec],
                 rng: random.Random) -> None:
        self.khmer_fonts = khmer_fonts
        self.heading_fonts = [f for f in khmer_fonts if f.heading]
        # Latin text uses dedicated Latin fonts only: several Khmer display
        # fonts draw lowercase Latin as capitals, which would mislabel samples.
        self.latin_fonts = latin_fonts
        self.rng = rng
        self.np_rng = np.random.default_rng(rng.randrange(1 << 31))

    def render(self, text: str, *, heading: bool = False,
               bold: bool | None = None) -> np.ndarray | None:
        """Return a float32 ``[32, W]`` line in [0, 1], or None if the text
        cannot be rendered faithfully with the chosen fonts.

        ``heading`` draws Khmer in a Muol-style font when the font set has
        one; ``bold`` forces (or rules out) a thickened stroke, which is
        otherwise drawn at random."""
        pool = self.heading_fonts if heading and self.heading_fonts else self.khmer_fonts
        khmer_font = self._pick(pool, "".join(ch for ch in text if _is_khmer(ch)))
        if khmer_font is None:
            return None
        runs = _runs(text, khmer_font.cmap)
        latin_font = self._pick(self.latin_fonts, "".join(r for r, k in runs if not k))
        if latin_font is None:
            return None
        size = self.rng.choice(FONT_SIZES)
        weight = self.rng.choice([300, 400, 400, 500, 700, 800, 900])
        latin_size = max(12, int(size * self.rng.uniform(0.65, 0.9)))
        fonts = [_pil_font(khmer_font.path, size, weight if khmer_font.variable else None)
                 if is_khmer else
                 _pil_font(latin_font.path, latin_size, weight if latin_font.variable else None)
                 for _, is_khmer in runs]

        ascent = max(f.getmetrics()[0] for f in fonts)
        descent = max(f.getmetrics()[1] for f in fonts)
        widths = [f.getlength(segment) for (segment, _), f in zip(runs, fonts, strict=True)]
        if bold is None:
            bold = self.rng.random() < (HEADING_BOLD_RATE if heading else BOLD_RATE)
        thin = heading and not bold and self.rng.random() < HEADING_THIN_RATE
        stroke = max(1, round(size * self.rng.uniform(0.015, 0.045))) if bold else 0
        pad_x, pad_y = self.rng.randint(4, 30) + stroke, self.rng.randint(3, 16) + stroke
        width = int(sum(widths)) + 2 * pad_x
        height = ascent + descent + 2 * pad_y
        if width > 6000:
            return None

        background, ink = self._colors()
        canvas = self._background(width, height, background)
        draw = ImageDraw.Draw(canvas)
        x = float(pad_x)
        baseline = pad_y + ascent
        for (segment, _), font, advance in zip(runs, fonts, widths, strict=True):
            draw.text((x, baseline), segment, font=font, fill=ink, anchor="ls",
                      stroke_width=stroke, stroke_fill=ink)
            x += advance
        image = np.asarray(canvas, dtype=np.uint8)
        if thin:  # a lighter-on-dark max filter erodes the dark strokes
            k = max(2, round(size * self.rng.uniform(0.025, 0.04)))
            image = cv2.dilate(image, np.ones((k, k), np.uint8))
        image = self._degrade(image, text_height=size)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        trimmed = trim_to_ink(gray)
        if trimmed is None:
            return None
        out_width = max(8, round(trimmed.shape[1] * INPUT_HEIGHT / trimmed.shape[0]))
        if out_width > MAX_WIDTH:
            return None
        line = cv2.resize(trimmed, (out_width, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
        return line.astype(np.float32) / 255.0

    def _pick(self, fonts: list[FontSpec], text: str) -> FontSpec | None:
        needed = {ord(c) for c in text if c != " "}
        candidates = [f for f in fonts if needed <= f.cmap]
        return self.rng.choice(candidates) if candidates else None

    def _colors(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        rng = self.rng
        base = rng.randint(185, 255)
        tint = [rng.randint(-25, 10) for _ in range(3)]
        background = tuple(int(np.clip(base + t, 0, 255)) for t in tint)
        dark = rng.randint(0, 90)
        ink_tint = [rng.randint(0, 40) for _ in range(3)]
        ink = tuple(int(np.clip(dark + t, 0, 255)) for t in ink_tint)
        return background, ink  # type: ignore[return-value]

    def _background(self, width: int, height: int,
                    color: tuple[int, int, int]) -> Image.Image:
        rng = self.rng
        img = np.empty((height, width, 3), dtype=np.float32)
        img[:] = color
        style = rng.random()
        if style < 0.3:  # gradient
            ramp = np.linspace(0, rng.uniform(-35, 35), width, dtype=np.float32)
            img += ramp[None, :, None]
        elif style < 0.55:  # guilloche-like waves, as on security printing
            yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
            freq, amp = rng.uniform(0.02, 0.12), rng.uniform(2, 10)
            waves = np.sin(xx * freq + np.sin(yy * freq * 2) * amp)
            strength = rng.uniform(6, 30)
            channel = rng.randrange(3)
            img[:, :, channel] -= (waves > 0.8) * strength
            img -= ((waves > 0.9) * strength * 0.5)[:, :, None]
        elif style < 0.7:  # blotchy texture
            noise = self.np_rng.normal(0, 1, (max(1, height // 8), max(1, width // 8)))
            noise = cv2.resize(noise.astype(np.float32), (width, height))
            img += (noise * rng.uniform(4, 14))[:, :, None]
        return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

    def _degrade(self, img: np.ndarray, *, text_height: int) -> np.ndarray:
        """Photo artifacts, bounded so the text stays legible to a person:
        a label nobody could read from the image only teaches noise."""
        rng = self.rng
        h, w = img.shape[:2]
        border = tuple(int(v) for v in img[0, 0])
        if rng.random() < 0.5:
            angle = rng.uniform(-1.5, 1.5)
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, matrix, (w, h), borderValue=border)
        if rng.random() < 0.3:
            shear = rng.uniform(-0.15, 0.15)
            matrix = np.array([[1, shear, -shear * h / 2], [0, 1, 0]], dtype=np.float32)
            img = cv2.warpAffine(img, matrix, (w, h), borderValue=border)
        if rng.random() < 0.08:  # soft glare spot
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            yy, xx = np.mgrid[0:h, 0:w]
            spot = np.clip(1 - np.hypot(xx - cx, yy - cy) / (rng.uniform(0.3, 0.8) * w), 0, 1)
            img = np.clip(img + spot[:, :, None] * rng.uniform(40, 120), 0, 255).astype(np.uint8)
        degraded = False
        if rng.random() < 0.4:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, max(0.4, text_height / 35)))
            degraded = True
        if rng.random() < (0.2 if degraded else 0.35):  # low resolution capture
            # Keep at least ~14 px of text height after downscaling.
            factor = rng.uniform(min(0.8, max(0.35, 14 / text_height)), 0.85)
            small = cv2.resize(img, (max(1, int(w * factor)), max(1, int(h * factor))),
                               interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        if rng.random() < 0.4:
            noise = self.np_rng.normal(0, rng.uniform(2, 10), img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
        if rng.random() < 0.5:
            ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(30, 95)])
            if ok:
                img = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        return img
