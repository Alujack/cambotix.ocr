from __future__ import annotations

import json
from pathlib import Path

import pytest
from khmerocr import KhmerOCR
from khmerocr.models import build_manifest
from khmerocr.recognition import register_recognizer
from khmerocr.testing import ScriptedRecognizer, place_on_background, render_card, to_png_bytes
from khmerocr.testing.recognizers import scripted_factory

register_recognizer("scripted", scripted_factory)

GOOD_RESPONSES: dict[str, tuple[str, float | None]] = {
    "id_number": ("000000001", 0.97),
    "full_name_km": ("គោត្តនាមនិងនាម: គំរូ សាកល្បង", 0.93),
    "full_name_latin": ("KUMRU SAKALBANG", 0.95),
    "date_of_birth": ("ថ្ងៃខែឆ្នាំកំណើត: ០១.០១.១៩៩០", 0.91),
    "gender": ("ភេទ: ប្រុស", 0.9),
}


@pytest.fixture
def card_photo_png() -> bytes:
    return to_png_bytes(place_on_background(render_card(), angle=7, perspective=0.03, seed=1))


@pytest.fixture
def scripted_engine() -> KhmerOCR:
    return KhmerOCR(ScriptedRecognizer(GOOD_RESPONSES))


def make_bundle(root: Path, model_id: str = "test-model",
                responses: dict[str, tuple[str, float | None]] | None = None) -> Path:
    bundle = root / model_id
    (bundle / "weights").mkdir(parents=True)
    (bundle / "weights" / "model.bin").write_bytes(b"fake weights for tests")
    manifest = build_manifest(
        bundle,
        file_paths=["weights/model.bin"],
        model_id=model_id,
        task="khmer_id_extraction",
        runtime="local",
        recognition_model="scripted-test",
        model_version="0.0.1",
        schema_version="1.0",
        license="Apache-2.0",
        recognizer={"backend": "scripted",
                    "options": {"responses": {k: list(v) for k, v in
                                              (responses or GOOD_RESPONSES).items()}}},
    )
    (bundle / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False),
                                          encoding="utf-8")
    return bundle
