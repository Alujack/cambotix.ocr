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


def parse_day_first_date(text: str) -> date | None:
    match = _DATE.search(to_ascii_digits(text))
    if not match:
        return None
    day, month, year = (int(group) for group in match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None


def is_plausible_birth_date(value: date, *, today: date | None = None) -> bool:
    today = today or date.today()
    if value > today:
        return False
    return (today.year - value.year) <= MAX_AGE_YEARS
