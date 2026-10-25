"""Build the Python-vs-TypeScript comparison set that test/parity.test.ts checks.

Renders synthetic SPECIMEN cards (benchmarks/generate_synthetic.py: clean,
rotated, perspective, blur, glare, upside-down), runs the Python engine on
each, and stores its results next to the images:

    .venv/bin/python packages/sdk-local/test/parity/make_parity_set.py \\
        --model ~/.khmerocr/models/khmerocr-id-v1 --out /tmp/parity --count 120
    cd packages/sdk-local
    KHMEROCR_PARITY_DIR=/tmp/parity KHMEROCR_TEST_BUNDLE=~/.khmerocr/models/khmerocr-id-v1 npm test

The images contain only fake values. Put the `tesseract` binary on PATH so
the Python engine can use the bundle's fallback recognizer.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "packages" / "core-python" / "src"))

from khmerocr import KhmerOCR, KhmerOCRError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Model bundle folder or installed model ID.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=120)
    args = parser.parse_args()

    # Runs this repository's own generator with the current interpreter.
    subprocess.run([sys.executable, str(ROOT / "benchmarks" / "generate_synthetic.py"),  # noqa: S603
                    "--out", str(args.out), "--count", str(args.count)], check=True)
    engine = KhmerOCR.from_pretrained(args.model)
    results = {}
    for line in (args.out / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        image = json.loads(line)["image"]
        try:
            result = engine.extract_id(str(args.out / image))
            results[image] = result.to_dict(include_text_regions=True)
        except KhmerOCRError as exc:
            results[image] = {"error": exc.code.value}
    (args.out / "python.json").write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(results)} Python results to {args.out / 'python.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
