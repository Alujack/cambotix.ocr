"""Text sampling for synthetic training lines.

Mixes general Khmer text (Wikipedia sample), ID-card style content (labels,
names, dates, numbers, addresses), Latin text, and random characters so every
vocabulary entry is seen. All names and numbers are synthetic.
"""

from __future__ import annotations

import random
import re
import unicodedata
from pathlib import Path

from .vocab import DIGITS, KHMER, LATIN, is_supported

KHMER_DIGITS = "០១២៣៤៥៦៧៨៩"
_COMBINING = {chr(c) for c in range(0x17B4, 0x17D4)} | {chr(0x17DD)}
_COENG = chr(0x17D2)  # KHMER SIGN COENG (subscript marker)
_SPACES = re.compile(r"\s+")

SURNAMES = [("សុខ", "SOK"), ("ចាន់", "CHAN"), ("កែវ", "KEO"), ("លី", "LY"), ("ហេង", "HENG"),
            ("ស៊ឹម", "SIM"), ("ឈឹម", "CHHIM"), ("ម៉ៅ", "MAO"), ("ញឹម", "NHIM"), ("ប៉ែន", "PEN"),
            ("សួន", "SUON"), ("ទេព", "TEP"), ("យឹម", "YIM"), ("គឹម", "KIM"), ("ជា", "CHEA"),
            ("ឡុង", "LONG"), ("ម៉ម", "MOM"), ("សេង", "SENG"), ("ខៀវ", "KHIEV"), ("ឯម", "EM"),
            ("ពៅ", "POV"), ("អ៊ុក", "OUK"), ("វង្ស", "VONG"), ("ស្រី", "SREY"), ("នូ", "NOU"),
            ("ឃុន", "KHUN"), ("ប៊ុន", "BUN"), ("ថាច់", "THACH"), ("អ៊ឹម", "IM"), ("ហ៊ុយ", "HUY")]
GIVEN = [("សុភា", "SOPHEA"), ("ដារ៉ា", "DARA"), ("វិចិត្រ", "VICHET"), ("ស្រីនាង", "SREYNEANG"),
         ("សុខា", "SOKHA"), ("ពិសិដ្ឋ", "PISETH"), ("រតនា", "RATANA"), ("ចន្ថា", "CHANTHA"),
         ("បុប្ផា", "BOPHA"), ("វណ្ណា", "VANNA"), ("សំណាង", "SAMNANG"), ("ធារី", "THEARY"),
         ("មករា", "MAKARA"), ("សុវណ្ណ", "SOVANN"), ("ពន្លក", "PONLOK"), ("រស្មី", "RASMEY"),
         ("ចិន្តា", "CHENDA"), ("ស៊ីណា", "SINA"), ("វាសនា", "VEASNA"), ("គន្ធា", "KUNTHEA"),
         ("ណារិទ្ធ", "NARITH"), ("ភក្តី", "PHEAKDEY"), ("សុជាតា", "SOCHEATA"), ("ធីតា", "THIDA")]
PROVINCES = ["ភ្នំពេញ", "សៀមរាប", "បាត់ដំបង", "កំពង់ចាម", "កណ្ដាល", "តាកែវ", "ព្រៃវែង",
             "ស្វាយរៀង", "កំពត", "កែប", "ព្រះសីហនុ", "កោះកុង", "ពោធិ៍សាត់", "បន្ទាយមានជ័យ",
             "ឧត្ដរមានជ័យ", "ព្រះវិហារ", "ស្ទឹងត្រែង", "រតនគិរី", "មណ្ឌលគិរី", "ក្រចេះ",
             "កំពង់ធំ", "កំពង់ស្ពឺ", "កំពង់ឆ្នាំង", "ប៉ៃលិន", "ត្បូងឃ្មុំ"]
LABELS_KM = ["គោត្តនាមនិងនាម", "ថ្ងៃខែឆ្នាំកំណើត", "ភេទ", "ទីកន្លែងកំណើត", "អាសយដ្ឋាន",
             "សញ្ជាតិ", "កម្ពស់", "សុពលភាព", "ថ្ងៃផុតកំណត់", "លេខ", "អត្តសញ្ញាណប័ណ្ណ",
             "ប័ណ្ណសម្គាល់ខ្លួននិស្សិត", "ព្រះរាជាណាចក្រកម្ពុជា", "ជាតិ សាសនា ព្រះមហាក្សត្រ",
             "សាកលវិទ្យាល័យ", "មហាវិទ្យាល័យ", "ដេប៉ាតឺម៉ង់", "បរិញ្ញាបត្រ", "អនុបណ្ឌិត",
             "ភូមិ", "ឃុំ", "សង្កាត់", "ស្រុក", "ខណ្ឌ", "ខេត្ត", "រាជធានី", "ខ្មែរ"]
MONTHS_KM = ["មករា", "កុម្ភៈ", "មីនា", "មេសា", "ឧសភា", "មិថុនា", "កក្កដា", "សីហា",
             "កញ្ញា", "តុលា", "វិច្ឆិកា", "ធ្នូ"]
LATIN_WORDS = ["NAME", "SEX", "MALE", "FEMALE", "Male", "Female", "DATE OF BIRTH", "NATIONALITY",
               "CAMBODIAN", "KINGDOM OF CAMBODIA", "ADDRESS", "PLACE OF BIRTH", "EXPIRY",
               "IDENTITY CARD", "STUDENT ID CARD", "Department", "of", "Information", "Technology",
               "Engineering", "Bachelor", "Master", "University", "Faculty", "Phnom Penh",
               "Siem Reap", "Battambang", "Student", "Card", "No.", "ID", "Valid", "until",
               "Business", "Management", "Accounting", "Law", "Medicine", "Science", "and", "the"]


def normalize_label(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace(chr(0x200B), "")
    return _SPACES.sub(" ", text).strip()


def khmer_clusters(text: str) -> list[str]:
    """Split into orthographic clusters so samples never cut a syllable."""
    clusters: list[str] = []
    for ch in text:
        if clusters and (ch in _COMBINING or clusters[-1].endswith(_COENG)):
            clusters[-1] += ch
        else:
            clusters.append(ch)
    return clusters


def _date(rng: random.Random, khmer_digits: bool) -> str:
    d, m, y = rng.randint(1, 28), rng.randint(1, 12), rng.randint(1940, 2030)
    sep = rng.choice("./-")
    text = f"{d:02d}{sep}{m:02d}{sep}{y}"
    return to_khmer_digits(text) if khmer_digits else text


def to_khmer_digits(text: str) -> str:
    return text.translate(str.maketrans(DIGITS, KHMER_DIGITS))


class TextSampler:
    def __init__(self, corpus_path: Path | None, rng: random.Random) -> None:
        self.rng = rng
        self.paragraphs: list[list[str]] = []
        if corpus_path and corpus_path.is_file():
            for line in corpus_path.read_text(encoding="utf-8").splitlines():
                line = normalize_label(line)
                clusters = khmer_clusters("".join(ch for ch in line if is_supported(ch)))
                if len(clusters) >= 8:
                    self.paragraphs.append(clusters)

    def sample(self) -> str:
        r = self.rng.random()
        if self.paragraphs and r < 0.40:
            text = self._corpus_span()
        elif r < 0.65:
            text = self._id_line()
        elif r < 0.75:
            text = self._latin_line()
        elif r < 0.85:
            text = self._numbers()
        elif r < 0.93:
            text = self._name()
        else:
            text = self._random_chars()
        return normalize_label(text)

    def _corpus_span(self) -> str:
        clusters = self.rng.choice(self.paragraphs)
        length = self.rng.randint(1, min(20, len(clusters)))
        start = self.rng.randint(0, len(clusters) - length)
        return "".join(clusters[start:start + length])

    def _name(self) -> str:
        surname, given = self.rng.choice(SURNAMES), self.rng.choice(GIVEN)
        mode = self.rng.random()
        if mode < 0.4:
            return f"{surname[0]} {given[0]}"
        if mode < 0.7:
            return f"{surname[1]} {given[1]}"
        return f"{surname[0]} {given[0]} / {surname[1]} {given[1]}"

    def card_name(self) -> str:
        """A Khmer-only name as printed on the card: two or three words, half
        from the composite name lists and half made of syllables in the
        patterns Khmer names use, so unlisted names are seen too."""
        rng = self.rng
        if rng.random() < 0.5:
            return f"{rng.choice(SURNAMES)[0]} {rng.choice(GIVEN)[0]}"
        words = 2 if rng.random() < 0.85 else 3
        return " ".join("".join(_name_syllable(rng) for _ in range(rng.choice([1, 1, 1, 2])))
                        for _ in range(words))

    def _id_line(self) -> str:
        rng = self.rng
        kind = rng.randrange(8)
        if kind == 0:
            return f"{rng.choice(LABELS_KM[:2])}{rng.choice([': ', ':', ' ', '៖ '])}{self._name()}"
        if kind == 1:
            return f"ថ្ងៃខែឆ្នាំកំណើត{rng.choice([': ', ' '])}{_date(rng, rng.random() < 0.4)}"
        if kind == 2:
            gender = rng.choice([("ប្រុស", "Male"), ("ស្រី", "Female")])
            return rng.choice([f"ភេទ: {gender[0]}", f"{gender[0]} / {gender[1]}", gender[0]])
        if kind == 3:
            day = to_khmer_digits(f"{rng.randint(1, 28):02d}")
            year = to_khmer_digits(str(rng.randint(1990, 2030)))
            return f"ថ្ងៃទី{day} ខែ{rng.choice(MONTHS_KM)} ឆ្នាំ{year}"
        if kind == 4:
            parts = [f"ភូមិ{rng.choice(GIVEN)[0]}", f"ឃុំ{rng.choice(GIVEN)[0]}",
                     f"ស្រុក{rng.choice(PROVINCES)}", f"ខេត្ត{rng.choice(PROVINCES)}"]
            return rng.choice([",", ", ", " "]).join(parts[rng.randint(0, 2):])
        if kind == 5:
            number = "".join(rng.choice(DIGITS) for _ in range(rng.choice([9, 10, 12])))
            return rng.choice([number, f"ID {number}", f"No. {number}", to_khmer_digits(number)])
        if kind == 6:
            return " ".join(rng.sample(LABELS_KM, rng.randint(1, 3)))
        return f"{rng.choice(LABELS_KM)} / {rng.choice(LATIN_WORDS)}"

    def _latin_line(self) -> str:
        words = [self.rng.choice(LATIN_WORDS) for _ in range(self.rng.randint(1, 5))]
        text = " ".join(words)
        return text.upper() if self.rng.random() < 0.3 else text

    def _numbers(self) -> str:
        rng = self.rng
        kind = rng.randrange(4)
        if kind == 0:
            return _date(rng, rng.random() < 0.3)
        if kind == 1:
            return "".join(rng.choice(DIGITS) for _ in range(rng.randint(3, 12)))
        if kind == 2:
            return to_khmer_digits("".join(rng.choice(DIGITS) for _ in range(rng.randint(2, 10))))
        return f"{rng.randint(100, 220)} cm"

    def _random_chars(self) -> str:
        """Random but well-formed clusters, so rare characters are still seen.
        Combining marks always follow a base, so nothing renders as a dotted
        circle that the label would not contain."""
        rng = self.rng
        parts = []
        for _ in range(rng.randint(2, 12)):
            if rng.random() < 0.25:
                parts.append(rng.choice(LATIN + DIGITS + " .,:/-()"))
                continue
            if rng.random() < 0.15:
                parts.append(rng.choice(_STANDALONE))  # digits and punctuation take no marks
                continue
            cluster = rng.choice(_BASES)
            if rng.random() < 0.3:
                cluster += _COENG + rng.choice(_CONSONANTS)
            if rng.random() < 0.6:
                cluster += rng.choice(_VOWELS)
            if rng.random() < 0.2:
                cluster += rng.choice(_SIGNS)
            parts.append(cluster)
        return "".join(parts)


def _name_syllable(rng: random.Random) -> str:
    """Onset, optional subscript, optional register shifter, vowel, and an
    optional final consonant or sign, in Unicode storage order (យ៉ាន, សុខ)."""
    syllable = rng.choice(_CONSONANTS)
    if rng.random() < 0.2:
        syllable += _COENG + rng.choice(_NAME_SUBSCRIPTS)
    if rng.random() < 0.15:
        syllable += rng.choice("៉៊")
    if rng.random() < 0.85:
        syllable += rng.choice(_NAME_VOWELS)
    r = rng.random()
    if r < 0.55:
        syllable += rng.choice(_NAME_FINALS) + ("់" if rng.random() < 0.15 else "")
    elif r < 0.65:
        syllable += rng.choice("ំះ")
    return syllable


_NAME_SUBSCRIPTS = "កខងចញតទនបពមយរលវស"
_NAME_VOWELS = "ាិីឹឺុូួើឿៀេែៃោៅ"
_NAME_FINALS = "កងចញណតនបមយលវស"
_CONSONANTS = "".join(chr(c) for c in range(0x1780, 0x17A3))
_BASES = _CONSONANTS + "".join(chr(c) for c in range(0x17A3, 0x17B4) if chr(c) in KHMER)
_STANDALONE = "០១២៣៤៥៦៧៨៩។៕៖ៗ៛"
_VOWELS = "".join(chr(c) for c in range(0x17B6, 0x17C6))
_SIGNS = "".join(chr(c) for c in range(0x17C6, 0x17D2) if chr(c) in KHMER) + "\u17DD"
