"""Fast checks of the training pipeline (no real training)."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fontTools")

from training.khmer_rec.data import bucketed, collate, ctc_feasible  # noqa: E402
from training.khmer_rec.model import CRNN  # noqa: E402
from training.khmer_rec.textgen import TextSampler, khmer_clusters, normalize_label  # noqa: E402
from training.khmer_rec.vocab import (  # noqa: E402
    NUM_CLASSES,
    TOKEN_OFFSET,
    TOKENS,
    decode,
    encode,
    greedy_decode,
    is_supported,
)

TRAINING_DIR = Path(__file__).resolve().parents[1]
FONTS_PRESENT = (TRAINING_DIR / "data" / "fonts" / "fonts.lock.json").is_file()


def test_vocabulary_covers_bilingual_cards() -> None:
    for text in ("គំរូ សាកល្បង / KUMRU SAKALBANG", "ថ្ងៃខែឆ្នាំកំណើត: ០១.០២.២០០៣",
                 "ID 000000001", "ប្រុស / Male", "ភូមិ, ឃុំ (ខេត្ត)"):
        assert is_supported(text)
        assert decode(encode(text)) == text
    assert len(set(TOKENS)) == len(TOKENS) and NUM_CLASSES == len(TOKENS) + TOKEN_OFFSET
    # Invisible inherent vowels must never be targets.
    assert "឴" not in TOKENS and "឵" not in TOKENS


def test_greedy_decode_collapses_repeats_and_blanks() -> None:
    a, b = encode("ក")[0], encode("A")[0]
    assert greedy_decode([0, a, a, 0, a, b, b, 0]) == "កកA"


def test_clusters_never_split_subscripts_or_vowels() -> None:
    assert khmer_clusters("ស្រុកខ្មែរ") == ["ស្រុ", "ក", "ខ្មែ", "រ"]
    assert normalize_label("  ក​ខ   គ ") == "កខ គ"


def test_text_sampler_yields_supported_text() -> None:
    sampler = TextSampler(None, random.Random(0))
    for _ in range(500):
        text = sampler.sample()
        assert text and is_supported(text) and text == normalize_label(text)


def test_card_names_are_well_formed_khmer() -> None:
    from khmerocr.recognition.onnx_ctc import khmer_structure_valid

    sampler = TextSampler(None, random.Random(0))
    for _ in range(500):
        name = sampler.card_name()
        assert is_supported(name) and name == normalize_label(name)
        assert khmer_structure_valid(name) and " " in name


def test_ctc_feasibility() -> None:
    assert ctc_feasible(40, "abcd")          # 10 steps >= 4
    assert not ctc_feasible(12, "aab")       # 3 steps < 3 + 1 repeat


def test_collate_and_bucketing() -> None:
    samples = [(np.full((32, w), 0.5, np.float32), "ក" * (w // 16)) for w in range(16, 400, 8)]
    batches = list(bucketed(iter(samples), batch_size=4, pool_batches=2))
    assert batches and all(len(b) == 4 for b in batches)
    for b in batches:  # widths inside a bucket are close
        widths = [s[0].shape[1] for s in b]
        assert max(widths) - min(widths) <= 32
    data = collate(samples[:3])
    assert data["images"].shape == (3, 1, 32, samples[2][0].shape[1])
    assert data["targets"].numel() == sum(len(t) for _, t in samples[:3])
    assert data["input_lengths"].tolist() == [s[0].shape[1] // 4 for s in samples[:3]]


def test_model_output_contract() -> None:
    model = CRNN(NUM_CLASSES).eval()
    with torch.no_grad():
        logits = model(torch.rand(2, 1, 32, 100))
    assert logits.shape == (25, 2, NUM_CLASSES)


@pytest.mark.skipif(not FONTS_PRESENT, reason="run training/fetch_fonts.py first")
def test_renderer_produces_model_ready_lines() -> None:
    from training.khmer_rec.render import INPUT_HEIGHT, LineRenderer, load_fonts

    khmer, latin = load_fonts(TRAINING_DIR, holdout=False)
    holdout_khmer, _ = load_fonts(TRAINING_DIR, holdout=True)
    assert not {f.path for f in khmer} & {f.path for f in holdout_khmer}
    renderer = LineRenderer(khmer, latin, random.Random(3))
    for text in ("120 cm", "ប្រុស / Male", "ថ្ងៃខែឆ្នាំកំណើត: ២៦.០៤.១៩៩១"):
        line = renderer.render(text)
        assert line is not None and line.shape[0] == INPUT_HEIGHT
        assert line.dtype == np.float32 and 0.0 <= line.min() and line.max() <= 1.0
