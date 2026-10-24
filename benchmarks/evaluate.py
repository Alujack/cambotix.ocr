"""Field-level benchmark for National ID extraction.

Dataset format: a directory with ``labels.jsonl``, one JSON object per line::

    {"image": "images/0001.jpg", "category": "glare",
     "fields": {"id_number": "...", "full_name_km": "...", "date_of_birth": "YYYY-MM-DD", ...}}

Expected values use the engine's normalized forms (ISO dates, uppercase Latin
names, ``M``/``F``). A field that is absent from ``fields`` is not scored.

Reports (JSON + Markdown) are written to ``benchmarks/reports/``. By default
they contain no field values: failure cases list the image, field, and kind of
failure only. Never pass ``--include-values`` for real ID images.

    python benchmarks/evaluate.py --dataset path/to/set --model khmerocr-id-v1 --gate
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "core-python" / "src"))

from khmerocr import KhmerOCR, KhmerOCRError, __version__  # noqa: E402

CRITICAL_GATE_FIELDS = ("id_number", "date_of_birth")
GATE_ACCURACY = 0.98
CER_FIELDS = ("full_name_km", "full_name_latin")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def char_error_rate(predicted: str | None, expected: str) -> float:
    predicted = predicted or ""
    if not expected:
        return 0.0 if not predicted else 1.0
    prev = list(range(len(predicted) + 1))
    for i, ec in enumerate(expected, 1):
        cur = [i]
        for j, pc in enumerate(predicted, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ec != pc)))
        prev = cur
    return prev[-1] / len(expected)


def evaluate(engine: KhmerOCR, dataset: Path, *, include_values: bool = False,
             document_type: str = "khmer_national_id") -> dict[str, Any]:
    samples = [json.loads(line) for line in
               (dataset / "labels.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    field_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "exact": 0, "missing": 0, "silent_errors": 0, "cer": []})
    category_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "all_fields_exact": 0, "review_required": 0, "errors": 0})
    error_codes: dict[str, int] = defaultdict(int)
    latencies: list[float] = []
    failures: list[dict[str, Any]] = []
    review_count = 0

    for sample in samples:
        category = sample.get("category", "uncategorized")
        cat = category_stats[category]
        cat["n"] += 1
        started = time.perf_counter()
        try:
            result = engine.extract_id(dataset / sample["image"], document_type=document_type)
        except KhmerOCRError as exc:
            latencies.append(time.perf_counter() - started)
            error_codes[exc.code.value] += 1
            cat["errors"] += 1
            failures.append({"image": sample["image"], "category": category,
                             "error": exc.code.value})
            for name in sample["fields"]:
                field_stats[name]["n"] += 1
                field_stats[name]["missing"] += 1
            continue
        latencies.append(time.perf_counter() - started)
        review_count += result.review_required
        cat["review_required"] += result.review_required

        all_exact = True
        for name, expected in sample["fields"].items():
            stats = field_stats[name]
            stats["n"] += 1
            field = result.fields.get(name)
            predicted = field.value if field else None
            if name in CER_FIELDS:
                stats["cer"].append(char_error_rate(predicted, expected))
            if predicted == expected:
                stats["exact"] += 1
                continue
            all_exact = False
            kind = "missing" if predicted is None else "mismatch"
            if predicted is None:
                stats["missing"] += 1
            elif not result.review_required:
                # A wrong value presented as not needing review: the most
                # dangerous failure mode.
                stats["silent_errors"] += 1
                kind = "silent_mismatch"
            failure = {"image": sample["image"], "category": category, "field": name,
                       "kind": kind, "confidence": field.confidence if field else None}
            if include_values:
                failure.update({"expected": expected, "predicted": predicted})
            failures.append(failure)
        cat["all_fields_exact"] += all_exact

    fields_report = {}
    for name, stats in sorted(field_stats.items()):
        n = stats["n"]
        low, high = wilson_interval(stats["exact"], n)
        fields_report[name] = {
            "n": n,
            "exact_match": round(stats["exact"] / n, 4) if n else None,
            "exact_match_ci95": [round(low, 4), round(high, 4)],
            "missing": stats["missing"],
            "silent_errors": stats["silent_errors"],
            "mean_cer": round(statistics.fmean(stats["cer"]), 4) if stats["cer"] else None,
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset.name,
        "document_type": document_type,
        "sdk_version": __version__,
        "engine": engine.info(),
        "platform": platform.platform(),
        "samples": len(samples),
        "review_required_rate": round(review_count / len(samples), 4) if samples else None,
        "fields": fields_report,
        "categories": dict(category_stats),
        "error_codes": dict(error_codes),
        "latency_ms": {
            "p50": round(statistics.median(latencies) * 1000, 1) if latencies else None,
            "p95": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))] * 1000, 1)
            if latencies else None,
        },
        "failures": failures,
    }


def gate(report: dict[str, Any], min_samples: int) -> list[str]:
    problems = []
    if report["samples"] < min_samples:
        problems.append(f"only {report['samples']} samples (< {min_samples})")
    for name in CRITICAL_GATE_FIELDS:
        stats = report["fields"].get(name)
        if not stats or stats["exact_match"] is None:
            problems.append(f"{name}: not evaluated")
        elif stats["exact_match"] < GATE_ACCURACY:
            problems.append(f"{name}: exact match {stats['exact_match']:.2%} < {GATE_ACCURACY:.0%}")
    for name, stats in report["fields"].items():
        if stats["silent_errors"]:
            problems.append(f"{name}: {stats['silent_errors']} wrong values not flagged for review")
    return problems


def to_markdown(report: dict[str, Any], problems: list[str] | None) -> str:
    lines = [
        f"# Benchmark: {report['dataset']}",
        "",
        f"- Generated: {report['generated_at']}",
        f"- SDK: {report['sdk_version']}; model: "
        f"{(report['engine'].get('model') or {}).get('model_id', 'n/a')}",
        f"- Samples: {report['samples']}; review required: {report['review_required_rate']}",
        f"- Latency p50/p95 (ms): {report['latency_ms']['p50']} / {report['latency_ms']['p95']}",
        "",
        "| Field | n | Exact match | 95% CI | Missing | Silent errors | Mean CER |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, s in report["fields"].items():
        lines.append(f"| {name} | {s['n']} | {s['exact_match']} | {s['exact_match_ci95']} | "
                     f"{s['missing']} | {s['silent_errors']} | {s['mean_cer']} |")
    lines += ["", "| Category | n | All fields exact | Review required | Errors |",
              "| --- | --- | --- | --- | --- |"]
    for name, s in report["categories"].items():
        lines.append(f"| {name} | {s['n']} | {s['all_fields_exact']} | "
                     f"{s['review_required']} | {s['errors']} |")
    if problems is not None:
        lines += ["", "## Release gate", ""]
        lines += [f"- FAIL: {p}" for p in problems] or ["- PASS"]
    lines += ["", f"Failure cases: {len(report['failures'])} (see JSON report)."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Model ID or bundle directory.")
    parser.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "reports")
    parser.add_argument("--document-type", default="khmer_national_id")
    parser.add_argument("--gate", action="store_true", help="Exit 1 if release gates fail.")
    parser.add_argument("--min-samples", type=int, default=300)
    parser.add_argument("--include-values", action="store_true",
                        help="Include expected/predicted values (synthetic data only).")
    args = parser.parse_args()

    with KhmerOCR.from_pretrained(args.model) as engine:
        report = evaluate(engine, args.dataset, include_values=args.include_values,
                          document_type=args.document_type)
    problems = gate(report, args.min_samples) if args.gate else None
    report["gate"] = None if problems is None else {"passed": not problems,
                                                     "problems": problems}

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{args.dataset.name}"
    (args.out / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (args.out / f"{stem}.md").write_text(to_markdown(report, problems), encoding="utf-8")
    print(to_markdown(report, problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
