"""ONNX CTC text-line recognizer backend.

Runs CRNN-style line recognizers exported to ONNX (for example the Khmer
recognizer from github.com/seanghay/KhmerOCR) with ONNX Runtime
(``pip install "cambotix-ocr[onnx]"``). Expected model contract:

* input ``[batch, 1, height, width]`` float32 grayscale in ``[0, 1]``, where
  ``height`` is fixed (32 for seanghay/KhmerOCR) and ``width`` is dynamic;
* first output: logits ``[time, batch, classes]`` with the CTC blank at
  index 0 and token ``i`` at class ``i + token_offset``. Other outputs (such
  as a font-style head) are ignored.

The vocabulary comes from a tokens file in the model bundle. Classes that do
not map to a token are dropped instead of being decoded as wrong characters.

Line recognizers like this are trained on tight text-line crops, so each crop
is trimmed to its ink before inference. Vocabularies without a space
character cannot separate words; when ``restore_spaces`` is set, spaces are
re-inserted where the image has a wide blank gap between two decoded
characters.

Decoded lines are checked against Khmer orthography. Edge words that are only
a stray Khmer mark and punctuation (such as ``.្`` picked up from the next
line at a crop edge) are dropped, but the read's confidence is capped, since
the same pattern also appears when the letter of a real word was misread
(``(ៅ`` for ``ពៅ``). Any other impossible sequence, such as a coeng (្) that
does not join two consonants, cannot be repaired without guessing, so it is
kept and the read gets confidence 0.

The line confidence is the mean probability of the decoded characters. That
mean ranks reads well but sits high even for wrong lines, so a bundle can
ship ``confidence_calibration``: ``[raw, calibrated]`` breakpoints, fitted on
held-out lines, that map the mean to the share of lines read exactly right.
"""

from __future__ import annotations

import os
import unicodedata
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from ..errors import InferenceFailedError, ModelNotAvailableError
from ..preprocessing.io import ImageArray
from .base import RecognizedText, Recognizer, RegionHint

# onnxruntime 1.29.0 added 1DS telemetry that resolves and contacts
# mobile.events.data.microsoft.com; neither ORT_DISABLE_TELEMETRY nor
# disable_telemetry_events() stopped the lookups in our tests (1.29-1.31).
FIRST_TELEMETRY_ORT_VERSION = (1, 29)

# Blank column runs at least this many line-heights wide count as word gaps.
_SPACE_GAP_RATIO = 0.22
_CROP_MARGIN_Y = 4
_CROP_MARGIN_X = 6

_COENG = "្"
_KHMER_CONSONANTS = range(0x1780, 0x17A3)  # ក .. អ
_KHMER_BASES = range(0x1780, 0x17B4)  # consonants and independent vowels
# Upper bound on the confidence of a read that had edge noise removed, low
# enough that the engine flags it for review.
_CLEANED_CONFIDENCE = 0.5
# Dependent vowels and signs that must follow a Khmer letter. U+17C8 (ៈ) is
# left out: recognizers read a printed ":" as ៈ, often after a space.
_KHMER_DEPENDENT = frozenset([*range(0x17B6, 0x17C8), *range(0x17C9, 0x17D2), 0x17D3, 0x17DD])


class OnnxCtcRecognizer(Recognizer):
    name = "onnx-ctc"

    def __init__(self, *, model_path: Path, tokens: str, token_offset: int = 3,
                 input_height: int = 32, restore_spaces: bool = True,
                 confidence_calibration: Sequence[Sequence[float]] | None = None,
                 session: Any = None) -> None:
        self.calibration = (_check_calibration(confidence_calibration)
                            if confidence_calibration is not None else None)
        if session is None:
            os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ModelNotAvailableError(
                    "The onnx-ctc backend requires the 'onnxruntime' package: "
                    "pip install 'cambotix-ocr[onnx]'."
                ) from exc
            check_onnxruntime_version(ort.__version__)
            ort.disable_telemetry_events()
            if not model_path.is_file():
                raise ModelNotAvailableError("The ONNX model file is missing from the bundle.")
            try:
                session = ort.InferenceSession(str(model_path),
                                               providers=["CPUExecutionProvider"])
            except Exception as exc:
                raise ModelNotAvailableError("The ONNX model could not be loaded.") from exc
        if not tokens:
            raise ModelNotAvailableError("The recognizer vocabulary is empty.")
        self._session = session
        self._input_name: str = session.get_inputs()[0].name
        self.model_file = model_path.name
        self.tokens = tokens
        self.token_offset = token_offset
        self.input_height = input_height
        self.restore_spaces = restore_spaces and " " not in tokens

    def describe(self) -> dict[str, str]:
        return {"backend": self.name, "model": self.model_file,
                "vocabulary_size": str(len(self.tokens))}

    def recognize_line(self, image: ImageArray, hint: RegionHint | None = None) -> RecognizedText:
        gray = trim_to_ink(np.asarray(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), dtype=np.uint8))
        if gray is None:
            return RecognizedText(text="", confidence=None)
        width = max(1, round(gray.shape[1] * self.input_height / gray.shape[0]))
        line = np.asarray(cv2.resize(gray, (width, self.input_height),
                                     interpolation=cv2.INTER_AREA), dtype=np.uint8)
        tensor = (line.astype(np.float32) / 255.0)[None, None]
        try:
            logits = self._session.run(None, {self._input_name: tensor})[0]
        except Exception as exc:
            raise InferenceFailedError("Text recognition failed.") from exc
        chars, steps, confidences = self._decode(np.asarray(logits)[:, 0, :])
        if not chars:
            return RecognizedText(text="", confidence=None)
        scores: list[float | None] = list(confidences)
        if self.restore_spaces:
            # The vocabulary has no space, so every space here was inserted.
            chars = insert_word_spaces(chars, steps, line, len(logits))
            remaining = iter(confidences)
            scores = [None if ch == " " else next(remaining) for ch in chars]
        cleaned, scores = drop_edge_noise(chars, scores)
        kept = [score for score in scores if score is not None]
        if not kept:
            return RecognizedText(text="", confidence=None)
        text = "".join(cleaned)
        confidence = float(np.mean(kept))
        if self.calibration is not None:
            confidence = float(np.interp(confidence, *self.calibration))
        if len(cleaned) < len(chars):
            confidence = min(confidence, _CLEANED_CONFIDENCE)
        if not khmer_structure_valid(text):
            confidence = 0.0
        return RecognizedText(text=text, confidence=confidence)

    def close(self) -> None:
        self._session = None

    def _decode(self, logits: NDArray[Any]) -> tuple[list[str], list[int], list[float]]:
        """Greedy CTC decoding: collapse repeats, drop blanks and unmapped classes."""
        shifted = logits - logits.max(axis=-1, keepdims=True)
        probs = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
        best = probs.argmax(axis=-1)
        chars: list[str] = []
        steps: list[int] = []
        confidences: list[float] = []
        previous = -1
        for step, index in enumerate(best.tolist()):
            token = index - self.token_offset
            if index != previous and index != 0 and 0 <= token < len(self.tokens):
                chars.append(self.tokens[token])
                steps.append(step)
                confidences.append(float(probs[step, index]))
            previous = index
        return chars, steps, confidences


def check_onnxruntime_version(version: str) -> None:
    """Refuse onnxruntime builds that make outbound telemetry connections."""
    try:
        major, minor = (int(part) for part in version.split(".")[:2])
    except ValueError as exc:
        raise ModelNotAvailableError("Unrecognized onnxruntime version.") from exc
    if (major, minor) >= FIRST_TELEMETRY_ORT_VERSION:
        raise ModelNotAvailableError(
            "This onnxruntime version sends telemetry to Microsoft, which breaks KhmerOCR's "
            "offline guarantee. Install 'onnxruntime<1.29'.",
            details={"onnxruntime_version": version},
        )


def _check_calibration(points: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    try:
        raw = [float(p[0]) for p in points]
        calibrated = [float(p[1]) for p in points]
        valid = (len(points) >= 2 and all(len(p) == 2 for p in points)
                 and all(0.0 <= v <= 1.0 for v in raw + calibrated)
                 and all(a < b for a, b in pairwise(raw))
                 and all(a <= b for a, b in pairwise(calibrated)))
    except (TypeError, ValueError, IndexError):
        valid = False
    if not valid:
        raise ModelNotAvailableError(
            "confidence_calibration must be at least two [raw, calibrated] pairs in [0, 1], "
            "with raw strictly increasing and calibrated non-decreasing.")
    return raw, calibrated


def _is_noise_word(chars: Sequence[str]) -> bool:
    # Punctuation-only words such as "។", ")" or "-" are real text; a Khmer
    # mark with no letter to attach to is not.
    return (not any(unicodedata.category(ch)[0] in "LN" for ch in chars)
            and any(ch == _COENG or ord(ch) in _KHMER_DEPENDENT for ch in chars))


def drop_edge_noise(chars: list[str],
                    scores: list[float | None]) -> tuple[list[str], list[float | None]]:
    """Drop leading and trailing words made of a Khmer mark and punctuation
    with no letter or digit, such as ``.្`` read from a neighbouring line at
    the crop edge. A line that is a single word is left alone."""
    start, end = 0, len(chars)
    while True:
        while start < end and chars[start] == " ":
            start += 1
        while end > start and chars[end - 1] == " ":
            end -= 1
        inner = chars[start:end]
        if " " not in inner:
            break
        first_space = inner.index(" ")
        last_space = len(inner) - 1 - inner[::-1].index(" ")
        if _is_noise_word(inner[last_space + 1:]):
            end = start + last_space
        elif _is_noise_word(inner[:first_space]):
            start += first_space
        else:
            break
    return chars[start:end], scores[start:end]


def khmer_structure_valid(text: str) -> bool:
    """False when the text has a Khmer sequence no correctly read word can
    contain: a coeng (្) not between a base letter and a consonant, or a
    dependent vowel or sign with no Khmer character before it."""
    for i, ch in enumerate(text):
        prev = ord(text[i - 1]) if i else None
        if ch == _COENG:
            nxt = ord(text[i + 1]) if i + 1 < len(text) else None
            if prev not in _KHMER_BASES or nxt not in _KHMER_CONSONANTS:
                return False
        elif ord(ch) in _KHMER_DEPENDENT and (prev is None or not 0x1780 <= prev <= 0x17FF):
            return False
    return True


def trim_to_ink(gray: NDArray[np.uint8]) -> NDArray[np.uint8] | None:
    """Crop a field region to its text, assuming dark text on a light background.

    Returns ``None`` when the region contains no ink at all.
    """
    if gray.size == 0 or gray.std() < 8:
        return None
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rows = np.flatnonzero(ink.sum(axis=1) >= max(1, gray.shape[1] // 200))
    cols = np.flatnonzero(ink.sum(axis=0) >= 1)
    if rows.size == 0 or cols.size == 0:
        return None
    y0, y1 = max(0, rows[0] - _CROP_MARGIN_Y), min(gray.shape[0], rows[-1] + _CROP_MARGIN_Y + 1)
    x0, x1 = max(0, cols[0] - _CROP_MARGIN_X), min(gray.shape[1], cols[-1] + _CROP_MARGIN_X + 1)
    return np.ascontiguousarray(gray[y0:y1, x0:x1])


def insert_word_spaces(chars: list[str], steps: list[int], line: NDArray[np.uint8],
                       time_steps: int) -> list[str]:
    """Insert a space between two decoded characters when a wide blank column
    run in the line image lies between their positions."""
    _, ink = cv2.threshold(line, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    empty = ink.sum(axis=0) == 0
    min_gap = max(2, round(_SPACE_GAP_RATIO * line.shape[0]))
    gap_centers: list[float] = []
    run_start: int | None = None
    for x, is_empty in enumerate([*empty.tolist(), False]):
        if is_empty and run_start is None:
            run_start = x
        elif not is_empty and run_start is not None:
            if x - run_start >= min_gap and run_start > 0 and x < len(empty):
                gap_centers.append((run_start + x) / 2)
            run_start = None
    if not gap_centers:
        return chars
    column_per_step = line.shape[1] / max(1, time_steps)
    positions = [(step + 0.5) * column_per_step for step in steps]
    spaced = [chars[0]]
    for i in range(1, len(chars)):
        if any(positions[i - 1] < gap < positions[i] for gap in gap_centers):
            spaced.append(" ")
        spaced.append(chars[i])
    return spaced
