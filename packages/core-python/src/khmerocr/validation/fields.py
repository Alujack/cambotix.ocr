"""Field-type normalizers.

Each normalizer turns cleaned recognizer output into a canonical value and
reports whether it passed format validation. Normalizers never invent data: if
the text cannot be interpreted, the value is ``None`` or the result is marked
invalid so the field is sent to human review.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .dates import is_plausible_birth_date, parse_day_first_date
from .text import contains_khmer, normalize_text, to_ascii_digits

# Characters commonly confused with digits by OCR engines.
_DIGIT_CONFUSIONS = str.maketrans({"O": "0", "o": "0", "D": "0", "I": "1", "l": "1",
                                   "|": "1", "S": "5", "B": "8", "Z": "2"})
_LATIN_NAME = re.compile(r"^[A-Z][A-Z' -]*[A-Z]$")
_NON_KHMER_NAME = re.compile(r"[^ក-៿᧠-᧿ ]")


@dataclass
class Normalized:
    value: str | None
    valid: bool
    # Warning codes (without the field suffix), e.g. "ocr_corrected".
    warnings: list[str] = field(default_factory=list)


def normalize_id_number(text: str, spec: dict[str, Any]) -> Normalized:
    digits_only = re.sub(r"[\s\-./]", "", to_ascii_digits(text))
    warnings: list[str] = []
    corrected = digits_only.translate(_DIGIT_CONFUSIONS)
    if corrected != digits_only:
        warnings.append("ocr_corrected")
    if not corrected:
        return Normalized(None, False, warnings)
    pattern = spec.get("pattern", r"^[0-9]+$")
    valid = re.fullmatch(pattern, corrected) is not None
    if not valid:
        warnings.append("invalid_format")
    return Normalized(corrected, valid, warnings)


def normalize_khmer_name(text: str, spec: dict[str, Any]) -> Normalized:
    if not contains_khmer(text):
        return Normalized(None, False, ["invalid_script"] if text else [])
    cleaned = normalize_text(_NON_KHMER_NAME.sub(" ", text))
    warnings = [] if cleaned == text else ["non_khmer_characters_removed"]
    return Normalized(cleaned or None, bool(cleaned), warnings)


def normalize_latin_name(text: str, spec: dict[str, Any]) -> Normalized:
    cleaned = normalize_text(text.upper())
    if not cleaned:
        return Normalized(None, False)
    valid = _LATIN_NAME.fullmatch(cleaned) is not None
    return Normalized(cleaned, valid, [] if valid else ["invalid_format"])


def normalize_birth_date(text: str, spec: dict[str, Any]) -> Normalized:
    parsed = parse_day_first_date(text)
    if parsed is None:
        return Normalized(None, False, ["invalid_format"] if text else [])
    if not is_plausible_birth_date(parsed):
        return Normalized(parsed.isoformat(), False, ["implausible_date"])
    return Normalized(parsed.isoformat(), True)


def normalize_gender(text: str, spec: dict[str, Any]) -> Normalized:
    compact = text.replace(" ", "")
    has_male = "ប្រុស" in compact
    has_female = "ស្រី" in compact
    if not (has_male or has_female):
        tokens = set(re.findall(r"[A-Za-z]+", text.upper()))
        has_male = bool(tokens & {"M", "MALE"})
        has_female = bool(tokens & {"F", "FEMALE"})
    if has_male == has_female:
        return Normalized(None, False, ["ambiguous_value"] if text else [])
    return Normalized("M" if has_male else "F", True)


def normalize_text_field(text: str, spec: dict[str, Any]) -> Normalized:
    return Normalized(text or None, bool(text))


NORMALIZERS: dict[str, Callable[[str, dict[str, Any]], Normalized]] = {
    "id_number": normalize_id_number,
    "khmer_name": normalize_khmer_name,
    "latin_name": normalize_latin_name,
    "birth_date": normalize_birth_date,
    "gender": normalize_gender,
    "text": normalize_text_field,
}
