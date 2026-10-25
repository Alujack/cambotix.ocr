"""Fit the confidence calibration for an exported recognizer.

    python -m training.khmer_rec.calibrate --run-name poc

The SDK's line confidence is the mean probability of the decoded characters.
It ranks reads well, but wrong lines still average around 0.9, so the
engine's review threshold (0.85) lets most of them through. This script reads
lines rendered with the holdout fonts through the SDK's own recognizer, fits
an isotonic map from that mean to the share of lines read exactly right, checks
it on a second set rendered with another seed, and writes ``calibration.json``
(``[raw, calibrated]`` breakpoints) next to ``rec.onnx``. Pass it to
``scripts/build_bundle.py --khmer-onnx-calibration``.

Synthetic lines are cleaner than real cards, so the map is optimistic there.
Refit it on authorized real crops when they are available.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from khmerocr.recognition.onnx_ctc import OnnxCtcRecognizer

from .data import fixed_set
from .train import TRAINING_DIR

# Raw means below 0.5 are rare and almost always wrong; detail matters near 1.
GRID = [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, *(k / 100 for k in range(82, 101))]
REVIEW_THRESHOLD = 0.85  # EngineConfig.review_confidence_threshold


def read_lines(recognizer: OnnxCtcRecognizer, *, size: int,
               seed: int) -> tuple[np.ndarray, np.ndarray]:
    scores, correct = [], []
    for image, label in fixed_set(TRAINING_DIR, size=size, seed=seed, holdout=True):
        rgb = np.repeat((image * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        result = recognizer.recognize_line(rgb)
        scores.append(result.confidence or 0.0)
        correct.append(result.text == label)
    return np.array(scores), np.array(correct)


def isotonic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators fit of a non-decreasing step function."""
    order = np.argsort(x, kind="stable")
    blocks: list[list[float]] = []  # [mean, weight, x_min, x_max]
    for xi, yi in zip(x[order], y[order].astype(float), strict=True):
        blocks.append([yi, 1.0, xi, xi])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            mean_b, weight_b, _, hi = blocks.pop()
            mean_a, weight_a, lo, _ = blocks.pop()
            weight = weight_a + weight_b
            blocks.append([(mean_a * weight_a + mean_b * weight_b) / weight, weight, lo, hi])
    centers = np.array([(lo + hi) / 2 for _, _, lo, hi in blocks])
    return centers, np.array([mean for mean, _, _, _ in blocks])


def flagged(scores: np.ndarray, correct: np.ndarray) -> str:
    low = scores < REVIEW_THRESHOLD
    return (f"wrong lines flagged {low[~correct].mean():.1%}, "
            f"correct lines flagged {low[correct].mean():.1%}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--size", type=int, default=4000, help="Lines to fit on.")
    parser.add_argument("--check-size", type=int, default=3000, help="Lines to check on.")
    args = parser.parse_args()

    run_dir = TRAINING_DIR / "data" / "runs" / args.run_name
    recognizer = OnnxCtcRecognizer(
        model_path=run_dir / "rec.onnx",
        tokens=(run_dir / "tokens.txt").read_text(encoding="utf-8").rstrip("\n"),
        token_offset=1, restore_spaces=False)
    fit_scores, fit_correct = read_lines(recognizer, size=args.size, seed=50_005)
    centers, means = isotonic(fit_scores, fit_correct)
    calibrated = np.interp(GRID, centers, means)
    points = [[float(raw), round(float(cal), 3)] for raw, cal in zip(GRID, calibrated,
                                                                     strict=True)]

    check_scores, check_correct = read_lines(recognizer, size=args.check_size, seed=60_006)
    mapped = np.interp(check_scores, [p[0] for p in points], [p[1] for p in points])
    report = {
        "fit_lines": int(fit_scores.size),
        "check_lines": int(check_scores.size),
        "check_wrong_lines": int((~check_correct).sum()),
        "raw": flagged(check_scores, check_correct),
        "calibrated": flagged(mapped, check_correct),
    }
    out = run_dir / "calibration.json"
    out.write_text(json.dumps({"confidence_calibration": points, "report": report},
                              indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
