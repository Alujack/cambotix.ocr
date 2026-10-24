"""Date parsing for document fields.

Only day-first numeric dates are accepted (``DD.MM.YYYY`` with ``.``, ``/``,
``-`` or space separators, in ASCII or Khmer digits). The date convention
printed on the card must be confirmed against authorized reference material;
until then, parsed dates are only as trustworthy as that assumption.
"""

from __future__ import annotations

import re
from datetime import date

from .text import to_ascii_digits

_DATE = re.compile(r"(\d{1,2})\s*[./\- ]\s*(\d{1,2})\s*[./\- ]\s*(\d{4})")
MAX_AGE_YEARS = 130
EARLIEST_ISSUE_YEAR = 1950

# Gregorian month names as printed on Cambodian documents. Lunar-calendar
# months (e.g. ខែផល្គុន) are deliberately not mapped.
KHMER_MONTHS = {
    "មករា": 1, "កុម្ភៈ": 2, "កុម្ភ": 2, "មីនា": 3, "មេសា": 4, "ឧសភា": 5, "មិថុនា": 6,
    "កក្កដា": 7, "សីហា": 8, "កញ្ញា": 9, "តុលា": 10, "វិច្ឆិកា": 11, "ធ្នូ": 12,
}
_KHMER_MONTH_DATE = re.compile(
    r"(\d{1,2})ខែ(" + "|".join(sorted(KHMER_MONTHS, key=len, reverse=True)) + r")ឆ្នាំ(\d{4})"
)


def parse_day_first_date(text: str) -> date | None:
    match = _DATE.search(to_ascii_digits(text))
    if not match:
        return None
    day, month, year = (int(group) for group in match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_khmer_month_date(text: str) -> date | None:
    """Parse dates such as ``ថ្ងៃទី០១ ខែមីនា ឆ្នាំ២០២៣`` (1 March 2023)."""
    compact = re.sub(r"[\s\u200b]", "", to_ascii_digits(text)).replace("ថ្ងៃទី", "")
    match = _KHMER_MONTH_DATE.search(compact)
    if not match:
        return None
    try:
        return date(int(match.group(3)), KHMER_MONTHS[match.group(2)], int(match.group(1)))
    except ValueError:
        return None


def is_plausible_issue_date(value: date, *, today: date | None = None) -> bool:
    today = today or date.today()
    return EARLIEST_ISSUE_YEAR <= value.year and value <= today


def is_plausible_birth_date(value: date, *, today: date | None = None) -> bool:
    today = today or date.today()
    if value > today:
        return False
    return (today.year - value.year) <= MAX_AGE_YEARS
