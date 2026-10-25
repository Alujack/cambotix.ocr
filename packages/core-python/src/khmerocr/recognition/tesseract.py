"""Tesseract recognizer backend.

Requires the ``tesseract`` binary (4.1+) and the ``pytesseract`` package
(``pip install "cambotix-ocr[tesseract]"``). Language data (``khm.traineddata``,
``eng.traineddata``) is read only from the model bundle's ``tessdata``
directory, never from a system-wide location, so results are reproducible and
pinned by the bundle checksum.

Set ``KHMEROCR_TESSERACT_CMD`` to use a binary that is not on ``PATH``.

pytesseract passes images to the tesseract process through short-lived files
in the system temporary directory and deletes them after each call.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ..errors import InferenceFailedError, ModelNotAvailableError
from ..preprocessing.io import ImageArray
from ..schemas import BoundingBox, TextRegion
from .base import RecognizedText, Recognizer, RegionHint

_SCRIPT_LANGUAGES = {"khmer": "khm", "latin": "eng", "digits": "eng", "mixed": "khm+eng"}
_MIN_LINE_HEIGHT = 48
# Retry ladder for a line crop: (preprocessing, page segmentation mode).
# psm 7 (single line) fails on short, widely spaced text such as numbers
# under a barcode, where psm 8 (single word) succeeds; Otsu binarization
# recovers text printed over patterned backgrounds.
_LINE_ATTEMPTS = (("gray", 7), ("otsu", 7))
_DIGIT_ATTEMPTS = (("gray", 7), ("otsu", 7), ("gray", 8))
# Stop retrying once a read reaches this confidence.
_ACCEPT_CONFIDENCE = 0.85


class TesseractRecognizer(Recognizer):
    name = "tesseract"

    def __init__(self, *, tessdata_dir: Path, default_languages: str = "khm+eng") -> None:
        try:
            import pytesseract
        except ImportError as exc:
            raise ModelNotAvailableError(
                "The tesseract backend requires the 'pytesseract' package: "
                "pip install 'cambotix-ocr[tesseract]'."
            ) from exc
        self._pytesseract: Any = pytesseract
        cmd = os.environ.get("KHMEROCR_TESSERACT_CMD")
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
        try:
            self.version = str(pytesseract.get_tesseract_version())
        except Exception as exc:  # TesseractNotFoundError, OSError, ...
            raise ModelNotAvailableError("The tesseract binary was not found.") from exc

        if os.name == "nt" and any(ch.isspace() for ch in str(tessdata_dir)):
            raise ModelNotAvailableError(
                "On Windows, the model bundle path must not contain spaces.",
                details={"tessdata_dir": str(tessdata_dir)},
            )
        self.tessdata_dir = tessdata_dir
        self.default_languages = default_languages
        languages = set(default_languages.split("+")) | {"khm", "eng"}
        missing = sorted(lang for lang in languages
                         if not (tessdata_dir / f"{lang}.traineddata").is_file())
        if missing:
            raise ModelNotAvailableError(
                "Tesseract language data is missing from the model bundle.",
                details={"missing_languages": missing},
            )

    def describe(self) -> dict[str, str]:
        return {"backend": self.name, "engine_version": self.version,
                "languages": self.default_languages}

    def recognize_line(self, image: ImageArray, hint: RegionHint | None = None) -> RecognizedText:
        hint = hint or RegionHint()
        languages = hint.languages or _SCRIPT_LANGUAGES.get(hint.script, self.default_languages)
        attempts = _DIGIT_ATTEMPTS if hint.script == "digits" else _LINE_ATTEMPTS
        best = RecognizedText(text="", confidence=None)
        for mode, psm in attempts:
            data = self._image_to_data(_prepare_line(image, mode), languages, psm=psm,
                                       whitelist=hint.char_whitelist)
            result = _join_words(data)
            if result.text and (best.confidence is None
                                or (result.confidence or 0.0) > best.confidence):
                best = result
            if best.confidence is not None and best.confidence >= _ACCEPT_CONFIDENCE:
                break
        return best

    def recognize_page(self, image: ImageArray) -> list[TextRegion]:
        data = self._image_to_data(image, self.default_languages, psm=3, whitelist=None)
        lines: dict[tuple[int, int, int], list[int]] = {}
        for i, text in enumerate(data["text"]):
            if str(text).strip() and float(data["conf"][i]) >= 0:
                key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                lines.setdefault(key, []).append(i)
        regions: list[TextRegion] = []
        for indices in lines.values():
            x0 = min(data["left"][i] for i in indices)
            y0 = min(data["top"][i] for i in indices)
            x1 = max(data["left"][i] + data["width"][i] for i in indices)
            y1 = max(data["top"][i] + data["height"][i] for i in indices)
            regions.append(TextRegion(
                text=" ".join(str(data["text"][i]).strip() for i in indices),
                confidence=float(np.mean([float(data["conf"][i]) for i in indices])) / 100.0,
                bbox=BoundingBox(int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
            ))
        return regions

    def _image_to_data(self, image: ImageArray, languages: str, *, psm: int,
                       whitelist: str | None) -> dict[str, list[Any]]:
        # pytesseract splits config with shlex in non-POSIX mode on Windows, which
        # keeps quotes as literal characters, so the path is only quoted on POSIX.
        tessdata = self.tessdata_dir if os.name == "nt" else f'"{self.tessdata_dir}"'
        config = f"--oem 1 --psm {psm} --tessdata-dir {tessdata}"
        if whitelist:
            config += f" -c tessedit_char_whitelist={whitelist}"
        try:
            result: dict[str, list[Any]] = self._pytesseract.image_to_data(
                Image.fromarray(image), lang=languages, config=config,
                output_type=self._pytesseract.Output.DICT,
            )
        except Exception as exc:
            raise InferenceFailedError("Text recognition failed.") from exc
        return result


def _join_words(data: dict[str, list[Any]]) -> RecognizedText:
    words, confidences = [], []
    for text, conf in zip(data["text"], data["conf"], strict=True):
        text = str(text).strip()
        if text and float(conf) >= 0:
            words.append(text)
            confidences.append(float(conf) / 100.0)
    if not words:
        return RecognizedText(text="", confidence=None)
    return RecognizedText(text=" ".join(words), confidence=float(np.mean(confidences)))


def _prepare_line(image: ImageArray, mode: str = "gray") -> ImageArray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    height = max(1, gray.shape[0])
    scale = max(3.0, _MIN_LINE_HEIGHT / height) if mode == "otsu" else _MIN_LINE_HEIGHT / height
    if scale > 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if mode == "otsu":
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # A white border helps Tesseract's line finder on tight crops.
    gray = cv2.copyMakeBorder(gray, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    return np.asarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), dtype=np.uint8)
