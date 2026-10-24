from __future__ import annotations

import numpy as np
import pytest
from khmerocr.config import QualityThresholds
from khmerocr.preprocessing import assess_quality, detect_card_quad, normalize_card
from khmerocr.testing import add_blur, add_glare, place_on_background, render_card

THRESHOLDS = QualityThresholds()


@pytest.fixture(scope="module")
def card() -> np.ndarray:
    return render_card()


def test_clean_photo_is_usable(card) -> None:
    report = assess_quality(place_on_background(card), THRESHOLDS)
    assert report.usable and report.issues == []


def test_blur_is_detected(card) -> None:
    photo = place_on_background(card)
    assert "blurry" in assess_quality(add_blur(photo, 2), THRESHOLDS).issues
    severe = assess_quality(add_blur(photo, 12), THRESHOLDS)
    assert "too_blurry" in severe.blocking_issues and not severe.usable


def test_glare_is_detected(card) -> None:
    photo = place_on_background(card)
    report = assess_quality(add_glare(photo), THRESHOLDS)
    assert "glare_detected" in report.issues and report.usable
    severe = assess_quality(add_glare(photo, radius=0.18, intensity=400), THRESHOLDS)
    assert "severe_glare" in severe.blocking_issues and not severe.usable


def test_small_and_blank_images_are_blocked() -> None:
    small = np.full((100, 160, 3), 128, dtype=np.uint8)
    report = assess_quality(small, THRESHOLDS)
    assert {"image_too_small", "low_contrast"} <= set(report.blocking_issues)


@pytest.mark.parametrize("angle,perspective", [(0, 0), (12, 0.02), (-20, 0.04), (35, 0.0)])
def test_card_is_detected_and_rectified(card, angle: float, perspective: float) -> None:
    photo = place_on_background(card, angle=angle, perspective=perspective, seed=5)
    assert detect_card_quad(photo) is not None
    normalized = normalize_card(photo, (1012, 638))
    assert normalized.detected and normalized.image.shape == (638, 1012, 3)
    # The rectified card should closely match the rendered original.
    diff = np.abs(normalized.image.astype(int) - card.astype(int)).mean()
    assert diff < 25


def test_portrait_card_is_turned_landscape(card) -> None:
    photo = place_on_background(card, angle=90, canvas=(1200, 1600), scale=0.7)
    normalized = normalize_card(photo, (1012, 638))
    assert normalized.detected and normalized.quarter_turns == 1


def test_tight_crop_falls_back_to_whole_image(card) -> None:
    normalized = normalize_card(card, (1012, 638))
    assert not normalized.detected
