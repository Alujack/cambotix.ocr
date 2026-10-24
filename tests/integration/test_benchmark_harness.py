from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from khmerocr import KhmerOCR
from khmerocr.testing import ScriptedRecognizer

from ..conftest import SYNTHETIC_RESPONSES

ROOT = Path(__file__).resolve().parents[2]


def _load_evaluate():
    spec = importlib.util.spec_from_file_location("evaluate", ROOT / "benchmarks" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metrics() -> None:
    ev = _load_evaluate()
    assert ev.char_error_rate("abc", "abc") == 0
    assert ev.char_error_rate("abd", "abc") == 1 / 3
    assert ev.char_error_rate(None, "ab") == 1
    low, high = ev.wilson_interval(98, 100)
    assert 0.92 < low < 0.98 < high <= 1


def test_harness_flags_silent_errors(tmp_path) -> None:
    subprocess.run([sys.executable, str(ROOT / "benchmarks" / "generate_synthetic.py"),
                    "--out", str(tmp_path), "--count", "6"], check=True, capture_output=True)
    ev = _load_evaluate()
    import dataclasses

    from khmerocr.documents import load_builtin_template

    # A verified template + a recognizer that always reads the same fixed
    # identity: confident, valid, and wrong for most images.
    template = dataclasses.replace(load_builtin_template("khmer_national_id_front_v1"),
                                   verified=True)
    engine = KhmerOCR(ScriptedRecognizer(SYNTHETIC_RESPONSES), template=template)
    report = ev.evaluate(engine, tmp_path)
    assert report["samples"] == 6
    assert report["fields"]["id_number"]["silent_errors"] > 0
    problems = ev.gate(report, min_samples=1)
    assert any("not flagged for review" in p for p in problems)
    # Reports carry no values unless explicitly requested.
    assert "000000001" not in json.dumps(report["failures"])
