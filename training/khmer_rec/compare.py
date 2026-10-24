"""Compare ONNX line recognizers on synthetic lines rendered with holdout fonts.

Both models run through the SDK's ``OnnxCtcRecognizer`` (same cropping and
decoding as in production). Khmer-only models are scored on the Khmer-only
subset; models with a bilingual vocabulary are also scored on all lines.

    python -m training.khmer_rec.compare --ours training/data/runs/poc \\
        --seanghay path/to/seanghay/rec.onnx
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from khmerocr.recognition.onnx_ctc import OnnxCtcRecognizer

from .data import fixed_set
from .train import TRAINING_DIR, edit_distance

SEANGHAY_TOKEN_OFFSET = 3


def _khmer_only(text: str) -> bool:
    return all(0x1780 <= ord(c) <= 0x17FF or c == " " for c in text)


def score(recognizer: OnnxCtcRecognizer, samples: list) -> dict[str, float]:
    errors = chars = exact = 0
    started = time.perf_counter()
    for image, label in samples:
        rgb = np.repeat((image * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        prediction = recognizer.recognize_line(rgb).text
        errors += edit_distance(prediction, label)
        chars += len(label)
        exact += prediction == label
    return {"n": len(samples), "cer": round(errors / max(1, chars), 4),
            "line_accuracy": round(exact / max(1, len(samples)), 4),
            "ms_per_line": round((time.perf_counter() - started) * 1000 / max(1, len(samples)), 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ours", type=Path, required=True, help="Run dir with rec.onnx.")
    parser.add_argument("--seanghay", type=Path, help="seanghay/KhmerOCR rec.onnx")
    parser.add_argument("--seanghay-tokens", type=Path,
                        help="Vocabulary file for the seanghay model (one line).")
    parser.add_argument("--size", type=int, default=1500)
    args = parser.parse_args()

    samples = fixed_set(TRAINING_DIR, size=args.size, seed=30_003, holdout=True)
    khmer = [s for s in samples if _khmer_only(s[1])]
    ours = OnnxCtcRecognizer(model_path=args.ours / "rec.onnx",
                             tokens=(args.ours / "tokens.txt").read_text(encoding="utf-8")
                             .rstrip("\n"), token_offset=1, restore_spaces=False)
    report = {"holdout_fonts": True,
              "ours": {"all_lines": score(ours, samples), "khmer_only": score(ours, khmer)}}
    if args.seanghay and args.seanghay_tokens:
        theirs = OnnxCtcRecognizer(
            model_path=args.seanghay,
            tokens=args.seanghay_tokens.read_text(encoding="utf-8").rstrip("\n"),
            token_offset=SEANGHAY_TOKEN_OFFSET, restore_spaces=True)
        report["seanghay"] = {"khmer_only": score(theirs, khmer)}
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
