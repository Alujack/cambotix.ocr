"""Recognizer vocabulary: Khmer, Latin, digits, space, and common punctuation.

Class 0 is the CTC blank; token ``i`` is class ``i + 1`` (``token_offset=1``
in the model bundle). Unlike Khmer-only vocabularies, this one covers every
character printed on bilingual Cambodian cards, so one model can read every
field.
"""

from __future__ import annotations

# Khmer block characters that are visible and in current use. Excluded:
# U+17B4/U+17B5 (invisible inherent vowels), U+17D3 and U+17D8 (deprecated),
# U+17F0-U+17F9 (lunar-date digits, rare).
_EXCLUDED = {0x17B4, 0x17B5, 0x17D3, 0x17D8}
KHMER = "".join(chr(c) for c in list(range(0x1780, 0x17DE)) + list(range(0x17E0, 0x17EA))
                if c not in _EXCLUDED and c not in range(0x17DE, 0x17E0))
LATIN = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"
PUNCTUATION = " .,:;/-()'\"!?%&+#@*_=«»"

TOKENS = KHMER + LATIN + DIGITS + PUNCTUATION
TOKEN_OFFSET = 1
NUM_CLASSES = len(TOKENS) + TOKEN_OFFSET

_INDEX = {ch: i + TOKEN_OFFSET for i, ch in enumerate(TOKENS)}
assert len(_INDEX) == len(TOKENS), "duplicate tokens"


def encode(text: str) -> list[int]:
    return [_INDEX[ch] for ch in text]


def decode(indices: list[int]) -> str:
    return "".join(TOKENS[i - TOKEN_OFFSET] for i in indices
                   if TOKEN_OFFSET <= i < NUM_CLASSES)


def is_supported(text: str) -> bool:
    return all(ch in _INDEX for ch in text)


def greedy_decode(best_path: list[int]) -> str:
    """Collapse repeats and drop blanks from per-time-step argmax classes."""
    out: list[int] = []
    previous = -1
    for index in best_path:
        if index != previous and index != 0:
            out.append(index)
        previous = index
    return decode(out)
