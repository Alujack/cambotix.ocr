"""Build a Tesseract model bundle from local traineddata files.

The bundle is written with ``license_approved: false``. Approval is a manual
step recorded in ``models/manifest.json`` after the licensing audit in
docs/model-licensing.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "core-python" / "src"))

from khmerocr.models import build_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-version", default="0.1.0")
    parser.add_argument("--traineddata", type=Path, nargs="+", required=True,
                        help="Paths to khm.traineddata, eng.traineddata, ...")
    parser.add_argument("--source-url", required=True,
                        help="Where the traineddata files were obtained.")
    parser.add_argument("--license", default="Apache-2.0")
    parser.add_argument("--recognition-model", default="tesseract-tessdata_fast")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    names = {p.name for p in args.traineddata}
    missing = {"khm.traineddata", "eng.traineddata"} - names
    if missing:
        parser.error(f"missing required language data: {', '.join(sorted(missing))}")
    if args.out.exists():
        parser.error(f"{args.out} already exists")

    tessdata = args.out / "tessdata"
    tessdata.mkdir(parents=True)
    for path in args.traineddata:
        shutil.copyfile(path, tessdata / path.name)

    manifest = build_manifest(
        args.out,
        file_paths=[f"tessdata/{p.name}" for p in args.traineddata],
        model_id=args.model_id,
        task="khmer_id_extraction",
        runtime="local",
        recognition_model=args.recognition_model,
        model_version=args.model_version,
        schema_version="1.0",
        license=args.license,
        license_approved=False,
        license_components=[{"name": "tesseract traineddata", "license": args.license,
                             "url": args.source_url}],
        recognizer={"backend": "tesseract", "options": {"languages": "khm+eng"}},
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"bundle": str(args.out), "weights_sha256": manifest["weights_sha256"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
