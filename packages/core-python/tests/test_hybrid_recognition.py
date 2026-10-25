"""ONNX CTC backend (with a fake session) and script routing."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from khmerocr import ModelNotAvailableError
from khmerocr.recognition import RegionHint, create_recognizer
from khmerocr.recognition.onnx_ctc import OnnxCtcRecognizer, insert_word_spaces, trim_to_ink
from khmerocr.recognition.routing import ScriptRouter
from khmerocr.testing import ScriptedRecognizer
from khmerocr.validation import strip_labels

TOKENS = "កខគ"  # class i -> TOKENS[i - 3]; class 0 = blank


class _Input:
    name = "input"


class FakeSession:
    """Emits a fixed class sequence, one class per time step."""

    def __init__(self, classes: list[int], n_classes: int = 8) -> None:
        self.classes = classes
        self.n_classes = n_classes
        self.last_input: np.ndarray | None = None

    def get_inputs(self):
        return [_Input()]

    def run(self, _outputs, feeds):
        self.last_input = feeds["input"]
        logits = np.full((len(self.classes), 1, self.n_classes), -10.0, dtype=np.float32)
        for t, c in enumerate(self.classes):
            logits[t, 0, c] = 10.0
        return [logits, np.zeros((1, 6), np.float32)]


def _line(width: int = 200, gaps: tuple[tuple[int, int], ...] = ()) -> np.ndarray:
    img = np.full((40, width, 3), 240, dtype=np.uint8)
    img[10:30, 10:width - 10] = 20
    for start, end in gaps:
        img[:, start:end] = 240
    return img


def _recognizer(classes: list[int], **kwargs) -> tuple[OnnxCtcRecognizer, FakeSession]:
    session = FakeSession(classes)
    return OnnxCtcRecognizer(model_path=Path("rec.onnx"), tokens=TOKENS, session=session,
                             **kwargs), session


def test_ctc_decoding_collapses_repeats_and_blanks() -> None:
    rec, session = _recognizer([3, 3, 0, 3, 4, 0, 0, 5], restore_spaces=False)
    result = rec.recognize_line(_line())
    assert result.text == "កកខគ"
    assert result.confidence == pytest.approx(1.0, abs=1e-3)
    assert session.last_input.shape[2] == 32  # fixed input height


def test_unmapped_classes_are_dropped_not_misdecoded() -> None:
    # Classes 1, 2 and 7 have no token; a naive decoder wraps 1 and 2 to the
    # end of the vocabulary via negative indexing.
    rec, _ = _recognizer([1, 3, 2, 7, 4], restore_spaces=False)
    assert rec.recognize_line(_line()).text == "កខ"


def test_blank_region_returns_nothing() -> None:
    rec, session = _recognizer([3])
    result = rec.recognize_line(np.full((40, 200, 3), 240, dtype=np.uint8))
    assert result.text == "" and result.confidence is None
    assert session.last_input is None


def test_trim_to_ink_crops_tightly() -> None:
    gray = np.full((100, 300), 250, dtype=np.uint8)
    gray[40:60, 100:200] = 10
    trimmed = trim_to_ink(gray)
    assert trimmed is not None and trimmed.shape == (20 + 8, 100 + 12)


def test_word_spaces_are_restored_at_wide_gaps() -> None:
    line = np.full((32, 100), 250, dtype=np.uint8)
    line[:, 5:40] = 10
    line[:, 60:95] = 10  # 20px blank gap between the two words
    spaced = insert_word_spaces(["ក", "ខ", "គ"], [2, 8, 18], line, time_steps=25)
    assert "".join(spaced) == "កខ គ"
    narrow = line.copy()
    narrow[:, 40:57] = 10  # gap of 3px is not a word break
    assert "".join(insert_word_spaces(["ក", "ខ", "គ"], [2, 8, 18], narrow, 25)) == "កខគ"


CARD_TOKENS = "ក្រ .ិB"  # class i -> CARD_TOKENS[i - 1]: ក ្ រ space . ិ B


def _card_recognizer(text: str, **kwargs) -> OnnxCtcRecognizer:
    # A blank between classes keeps repeated characters from collapsing.
    classes = [c for ch in text for c in (CARD_TOKENS.index(ch) + 1, 0)]
    return OnnxCtcRecognizer(model_path=Path("rec.onnx"), tokens=CARD_TOKENS, token_offset=1,
                             session=FakeSession(classes), **kwargs)


@pytest.mark.parametrize("text", ["ក្រ B .្", ".្ ក្រ B", "ិ ក្រ B ្"])
def test_edge_noise_is_dropped_and_the_read_flagged(text: str) -> None:
    # "Bachelor .្" from a real card: the last word is a dot and a stray coeng.
    result = _card_recognizer(text).recognize_line(_line())
    assert result.text == "ក្រ B"
    assert result.confidence == pytest.approx(0.5)  # below the 0.85 review threshold


def test_clean_read_keeps_its_confidence() -> None:
    assert _card_recognizer("ក្រ B").recognize_line(_line()).confidence == pytest.approx(
        1.0, abs=1e-3)


def test_punctuation_words_at_the_edges_are_kept() -> None:
    assert _card_recognizer(". ក្រ .").recognize_line(_line()).text == ". ក្រ ."


def test_single_word_line_is_not_dropped_as_noise() -> None:
    assert _card_recognizer(".្").recognize_line(_line()).text == ".្"


@pytest.mark.parametrize("text", ["ក្ ក", "ក ិក", "ក្.ក"])
def test_impossible_khmer_sequence_gets_zero_confidence(text: str) -> None:
    result = _card_recognizer(text).recognize_line(_line())
    assert result.text == text  # kept as read, but never trusted
    assert result.confidence == 0.0


def test_confidence_calibration_maps_the_mean() -> None:
    rec = _card_recognizer("ក្រ", confidence_calibration=[[0.0, 0.0], [0.9, 0.5], [1.0, 0.8]])
    assert rec.recognize_line(_line()).confidence == pytest.approx(0.8, abs=1e-3)


@pytest.mark.parametrize("points", [[[0.5, 0.5]], [[0.2, 0.1], [0.1, 0.2]],
                                    [[0.0, 0.9], [1.0, 0.1]], [[0.0, 0.0], [1.0, 1.5]],
                                    "bad"])
def test_invalid_calibration_is_rejected(points) -> None:
    with pytest.raises(ModelNotAvailableError):
        _card_recognizer("ក", confidence_calibration=points)


def test_router_sends_khmer_regions_to_khmer_backend() -> None:
    khmer = ScriptedRecognizer({"name_km": ("គំរូ", 0.99)})
    latin = ScriptedRecognizer({"name_latin": ("KUMRU", 0.98)})
    router = ScriptRouter({"khmer": khmer, "latin": latin}, {"khmer": "khmer"}, default="latin")
    image = _line()
    assert router.recognize_line(image, RegionHint(name="name_km", script="khmer")).text == "គំរូ"
    for script in ("latin", "digits", "mixed"):
        router.recognize_line(image, RegionHint(name="name_latin", script=script))
    assert khmer.calls == ["name_km"] and latin.calls == ["name_latin"] * 3
    router.close()
    assert khmer.closed and latin.closed


def test_router_rejects_unknown_backends() -> None:
    with pytest.raises(ValueError):
        ScriptRouter({"a": ScriptedRecognizer()}, {"khmer": "missing"}, default="a")


def test_router_factory_validates_configuration(tmp_path) -> None:
    with pytest.raises(ModelNotAvailableError, match="routing"):
        create_recognizer("router", tmp_path, {"backends": {}})
    with pytest.raises(ModelNotAvailableError, match="nested"):
        create_recognizer("router", tmp_path, {
            "backends": {"r": {"backend": "router"}}, "default": "r"})


def test_onnx_factory_refuses_paths_outside_bundle(tmp_path) -> None:
    with pytest.raises(ModelNotAvailableError, match="inside the model bundle"):
        create_recognizer("onnx-ctc", tmp_path, {"model": "../rec.onnx",
                                                 "tokens_file": "../tokens.txt"})


def test_truncated_label_is_not_returned_as_value() -> None:
    labels = ["គោត្តនាមនិងនាម"]
    assert strip_labels("គោត្តនាមនិងន", labels) == ""
    assert strip_labels("គោត្តនាមនិងនាមៈ", labels) == ""
    assert strip_labels("គោត្តនាមនិងនាមៈគំរូ សាកល្បង", labels) == "គំរូ សាកល្បង"


REC = os.environ.get("KHMEROCR_KHMER_ONNX")


@pytest.mark.skipif(not (REC and Path(REC).is_file()),
                    reason="set KHMEROCR_KHMER_ONNX to a seanghay/KhmerOCR rec.onnx")
def test_real_khmer_model_reads_synthetic_name() -> None:
    import importlib.util

    from khmerocr.testing import render_card

    script = Path(__file__).resolve().parents[3] / "scripts" / "build_bundle.py"
    spec = importlib.util.spec_from_file_location("build_bundle", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rec = OnnxCtcRecognizer(model_path=Path(REC), tokens=module.SEANGHAY_TOKENS)
    from khmerocr.documents import template_for_document

    card = render_card()
    box = next(f.box for f in template_for_document("khmer_national_id").fields
               if f.name == "full_name_km")
    crop = card[int(box[1] * 638):int((box[1] + box[3]) * 638), int(box[0] * 1012):]
    text = strip_labels(rec.recognize_line(crop).text, ["គោត្តនាមនិងនាម"])
    assert text == "គំរូ សាកល្បង", json.dumps(text, ensure_ascii=False)


@pytest.mark.parametrize("version,allowed", [("1.20.1", True), ("1.28.0", True),
                                             ("1.29.0", False), ("1.31.0", False),
                                             ("2.0.0", False)])
def test_telemetry_enabled_onnxruntime_is_refused(version: str, allowed: bool) -> None:
    from khmerocr.recognition.onnx_ctc import check_onnxruntime_version

    if allowed:
        check_onnxruntime_version(version)
    else:
        with pytest.raises(ModelNotAvailableError, match="telemetry"):
            check_onnxruntime_version(version)


def _hybrid_engine(khmer_responses, fallback_responses):
    from khmerocr import KhmerOCR
    from khmerocr.config import EngineConfig

    khmer = ScriptedRecognizer(khmer_responses)
    fallback = ScriptedRecognizer(fallback_responses)
    router = ScriptRouter({"khmer": khmer, "tesseract": fallback},
                          {"khmer": ["khmer", "tesseract"]}, default="tesseract")
    engine = KhmerOCR(router, config=EngineConfig(try_alternate_orientations=False))
    return engine, khmer, fallback


def test_fallback_is_not_called_when_primary_read_validates() -> None:
    from khmerocr.testing import place_on_background, render_card

    engine, khmer, fallback = _hybrid_engine(
        {"full_name_km": ("គោត្តនាមនិងនាមៈគំរូ សាកល្បង", 0.99)},
        {"full_name_km": ("wrong", 0.99), "id_number": ("000000001", 0.97)})
    result = engine.extract_id(place_on_background(render_card()))
    assert result.fields["full_name_km"].value == "គំរូ សាកល្បង"
    assert "full_name_km" not in fallback.calls  # lazily skipped
    from khmerocr.documents import template_for_document

    khmer_fields = [f.name for f in template_for_document("khmer_national_id").fields
                    if f.script == "khmer"]
    assert khmer.calls == khmer_fields  # only Khmer-script regions go to the Khmer model
    assert "fallback_recognizer:full_name_km" not in result.warnings


def test_fallback_is_used_when_primary_read_fails_validation() -> None:
    from khmerocr.testing import place_on_background, render_card

    engine, _, _ = _hybrid_engine(
        {"full_name_km": ("", None)},
        {"full_name_km": ("គំរូ សាកល្បង", 0.9)})
    result = engine.extract_id(place_on_background(render_card()))
    assert result.fields["full_name_km"].value == "គំរូ សាកល្បង"
    assert "fallback_recognizer:full_name_km" in result.warnings
    region = next(r for r in result.regions if r.label == "full_name_km")
    assert region.text == "គំរូ សាកល្បង"  # the region reports the read actually used


def test_failed_reads_report_primary_problem() -> None:
    from khmerocr.testing import place_on_background, render_card

    engine, _, _ = _hybrid_engine({"full_name_km": ("SOK", 0.99)},
                                  {"full_name_km": ("123", 0.99)})
    result = engine.extract_id(place_on_background(render_card()))
    assert result.fields["full_name_km"].value is None
    assert "invalid_script:full_name_km" in result.warnings
    assert result.review_required
