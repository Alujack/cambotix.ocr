from __future__ import annotations

from datetime import date

import pytest
from hypothesis import given
from hypothesis import strategies as st
from khmerocr.validation import (
    NORMALIZERS,
    normalize_text,
    parse_day_first_date,
    strip_labels,
    to_ascii_digits,
)
from khmerocr.validation.dates import is_plausible_birth_date
from khmerocr.validation.fields import (
    normalize_birth_date,
    normalize_gender,
    normalize_id_number,
    normalize_khmer_name,
    normalize_latin_name,
)

ID_SPEC = {"pattern": r"^[0-9]{9}$"}


def test_khmer_digits_are_converted() -> None:
    assert to_ascii_digits("០១២៣៤៥៦៧៨៩") == "0123456789"


def test_normalize_text_removes_zero_width_and_collapses_spaces() -> None:
    assert normalize_text("  គំរូ\u200b  សាកល្បង \n") == "គំរូ សាកល្បង"


def test_strip_labels_handles_spacing_inside_label() -> None:
    assert strip_labels("គោត្តនាម និងនាម : គំរូ", ["គោត្តនាមនិងនាម"]) == "គំរូ"
    assert strip_labels("NAME: SOK", ["NAME"]) == "SOK"
    assert strip_labels("SOK", ["NAME"]) == "SOK"


def test_day_first_dates() -> None:
    assert parse_day_first_date("01.02.1990") == date(1990, 2, 1)
    assert parse_day_first_date("៣១/១២/២០០០") == date(2000, 12, 31)
    assert parse_day_first_date("31.02.1990") is None
    assert parse_day_first_date("no date") is None


def test_birth_date_plausibility() -> None:
    today = date(2026, 1, 1)
    assert is_plausible_birth_date(date(1990, 1, 1), today=today)
    assert not is_plausible_birth_date(date(2027, 1, 1), today=today)
    assert not is_plausible_birth_date(date(1800, 1, 1), today=today)


def test_id_number_valid_and_invalid() -> None:
    ok = normalize_id_number("០០០ ០០០ ០០១", ID_SPEC)
    assert (ok.value, ok.valid, ok.warnings) == ("000000001", True, [])
    short = normalize_id_number("12345", ID_SPEC)
    assert not short.valid and "invalid_format" in short.warnings


def test_id_number_confusable_characters_are_flagged() -> None:
    result = normalize_id_number("O0000000I", ID_SPEC)
    assert result.value == "000000001"
    assert "ocr_corrected" in result.warnings


def test_khmer_name_requires_khmer_script() -> None:
    assert normalize_khmer_name("គំរូ សាកល្បង", {}).valid
    latin = normalize_khmer_name("SOK", {})
    assert latin.value is None and "invalid_script" in latin.warnings


@pytest.mark.parametrize("text", ["សៀុខ ៥ៅក", "សុខ ១២", "សុខ។", "ាសុខ"])
def test_khmer_name_with_impossible_characters_is_invalid(text: str) -> None:
    # Shaped like a real misread of a bold printed name: a digit and two
    # vowel signs in a row.
    result = normalize_khmer_name(text, {})
    assert result.value == text and not result.valid
    assert "invalid_format" in result.warnings


def test_khmer_name_with_register_shifter_is_valid() -> None:
    assert normalize_khmer_name("សុខ ម៉ៅ", {}).valid


def test_latin_name_is_uppercased_and_validated() -> None:
    assert normalize_latin_name("kumru sakalbang", {}).value == "KUMRU SAKALBANG"
    assert not normalize_latin_name("SOK 123", {}).valid


def test_birth_date_normalizes_to_iso() -> None:
    result = normalize_birth_date("01.01.1990", {})
    assert (result.value, result.valid) == ("1990-01-01", True)
    future = normalize_birth_date("01.01.2999", {})
    assert not future.valid and "implausible_date" in future.warnings


def test_gender() -> None:
    assert normalize_gender("ប្រុស", {}).value == "M"
    assert normalize_gender("ស្រី", {}).value == "F"
    assert normalize_gender("Sex: F", {}).value == "F"
    ambiguous = normalize_gender("ប្រុស ស្រី", {})
    assert ambiguous.value is None and not ambiguous.valid


@given(st.text(max_size=200))
def test_normalizers_never_crash(text: str) -> None:
    for name, normalizer in NORMALIZERS.items():
        result = normalizer(text, ID_SPEC if name == "id_number" else {})
        assert result.value is None or isinstance(result.value, str)
        if result.valid:
            assert result.value
