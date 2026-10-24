"""ONNX CTC text-line recognizer backend.

Runs CRNN-style line recognizers exported to ONNX (for example the Khmer
recognizer from github.com/seanghay/KhmerOCR) with ONNX Runtime
(``pip install "khmerocr[onnx]"``). Expected model contract:

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
"""

from __future__ import annotations

import os
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


class OnnxCtcRecognizer(Recognizer):
    name = "onnx-ctc"

    def __init__(self, *, model_path: Path, tokens: str, token_offset: int = 3,
                 input_height: int = 32, restore_spaces: bool = True,
                 session: Any = None) -> None:
        if session is None:
            os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ModelNotAvailableError(
                    "The onnx-ctc backend requires the 'onnxruntime' package: "
                    "pip install 'khmerocr[onnx]'."
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
        if self.restore_spaces:
            chars = insert_word_spaces(chars, steps, line, len(logits))
        return RecognizedText(text="".join(chars), confidence=float(np.mean(confidences)))

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
