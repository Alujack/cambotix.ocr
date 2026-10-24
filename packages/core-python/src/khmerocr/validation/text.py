"""Unicode and script normalization for Khmer and Latin text."""

from __future__ import annotations

import re
import unicodedata

KHMER_DIGITS = "០១២៣៤៥៦៧៨៩"
_DIGIT_TABLE = str.maketrans(KHMER_DIGITS, "0123456789")

# Khmer (U+1780-U+17FF) and Khmer Symbols (U+19E0-U+19FF).
_KHMER_CHAR = re.compile(r"[ក-៿᧠-᧿]")
_ZERO_WIDTH = re.compile(r"[​‌‍⁠﻿]")
_WHITESPACE = re.compile(r"\s+")


def to_ascii_digits(text: str) -> str:
    """Convert Khmer digits to ASCII digits."""
    return text.translate(_DIGIT_TABLE)


def normalize_text(text: str) -> str:
    """NFC-normalize, drop zero-width characters, and collapse whitespace.

    Zero-width spaces are commonly inserted between Khmer words by input
    methods and OCR engines; they carry no meaning for field values.
    """
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH.sub("", text)
    return _WHITESPACE.sub(" ", text).strip()


def contains_khmer(text: str) -> bool:
    return bool(_KHMER_CHAR.search(text))


def strip_labels(text: str, labels: list[str]) -> str:
    """Remove a leading printed label (and a following separator) from ``text``.

    Labels are compared after removing spaces, since recognizers frequently
    insert or drop spaces inside Khmer label text.
    """
    compact = text.replace(" ", "")
    for label in sorted(labels, key=len, reverse=True):
        label_compact = normalize_text(label).replace(" ", "")
        if label_compact and compact.startswith(label_compact):
            # Walk the original string until the label's characters are consumed.
            consumed, i = 0, 0
            while i < len(text) and consumed < len(label_compact):
                if text[i] != " ":
                    consumed += 1
                i += 1
            return text[i:].lstrip(" :：-").strip()
    return text
