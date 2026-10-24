"""Build a model bundle from local model files.

Tesseract only (development baseline)::

    python scripts/build_bundle.py --model-id khmerocr-id-dev \\
      --traineddata khm.traineddata eng.traineddata \\
      --source-url https://github.com/tesseract-ocr/tessdata_fast/tree/4.1.0 --out dist/dev

Hybrid: Khmer regions read by an ONNX CTC recognizer (e.g. seanghay/KhmerOCR
``rec.onnx``), Latin and digit regions by Tesseract::

    python scripts/build_bundle.py --model-id khmerocr-id-hybrid \\
      --traineddata khm.traineddata eng.traineddata \\
      --source-url https://github.com/tesseract-ocr/tessdata_fast/tree/4.1.0 \\
      --khmer-onnx path/to/rec.onnx \\
      --khmer-onnx-source-url https://github.com/seanghay/KhmerOCR/tree/68c2c72 \\
      --out dist/hybrid

Bundles are written with ``license_approved: false``. Approval is a manual
step recorded in ``models/manifest.json`` after the licensing audit in
docs/model-licensing.md. Do not bundle seanghay/KhmerOCR ``det.onnx``: its
embedded metadata declares Ultralytics AGPL-3.0.
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

# Vocabulary of seanghay/KhmerOCR rec.onnx (MIT, Copyright (c) 2026 Seanghay
# Yath), copied from khmerocr/__init__.py at commit 68c2c72. Class i of the
# model maps to SEANGHAY_TOKENS[i - 3]; class 0 is the CTC blank.
SEANGHAY_TOKENS = (
    "កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវឝឞសហឡអឣឤឥឦឧឩឪឫឬឭឮឯឰឱឲឳាិីឹឺុូួើឿៀេែៃោៅំះៈ៉៊់៌៍៎៏័៑្"
    "។៕៖ៗ៘៛៝០១២៣៤៥៦៧៨៩៳"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-version", default="0.1.0")
    parser.add_argument("--traineddata", type=Path, nargs="+", required=True,
                        help="Paths to khm.traineddata, eng.traineddata, ...")
    parser.add_argument("--source-url", required=True,
                        help="Where the traineddata files were obtained.")
    parser.add_argument("--khmer-onnx", type=Path,
                        help="ONNX CTC recognizer used for Khmer-script regions.")
    parser.add_argument("--khmer-onnx-source-url",
                        help="Where the ONNX recognizer was obtained (required with --khmer-onnx).")
    parser.add_argument("--khmer-onnx-license", default="MIT")
    parser.add_argument("--khmer-onnx-tokens", type=Path,
                        help="Vocabulary file; defaults to the seanghay/KhmerOCR vocabulary.")
    parser.add_argument("--khmer-onnx-token-offset", type=int, default=3,
                        help="Class index of the first token (3 for seanghay/KhmerOCR, "
                             "1 for models from training/).")
    parser.add_argument("--onnx-scripts", default="khmer",
                        help="Comma-separated region scripts routed to the ONNX model "
                             "(khmer,latin,digits,mixed). Only use more than 'khmer' for "
                             "models whose vocabulary covers Latin letters and digits.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    names = {p.name for p in args.traineddata}
    missing = {"khm.traineddata", "eng.traineddata"} - names
    if missing:
        parser.error(f"missing required language data: {', '.join(sorted(missing))}")
    if args.khmer_onnx and not args.khmer_onnx_source_url:
        parser.error("--khmer-onnx-source-url is required with --khmer-onnx")
    if args.out.exists():
        parser.error(f"{args.out} already exists")

    tessdata = args.out / "tessdata"
    tessdata.mkdir(parents=True)
    files = []
    for path in args.traineddata:
        shutil.copyfile(path, tessdata / path.name)
        files.append(f"tessdata/{path.name}")
    components = [{"name": "tesseract traineddata", "license": "Apache-2.0",
                   "url": args.source_url}]
    tesseract = {"backend": "tesseract", "options": {"languages": "khm+eng"}}

    if args.khmer_onnx:
        khmer_dir = args.out / "khmer"
        khmer_dir.mkdir()
        shutil.copyfile(args.khmer_onnx, khmer_dir / "rec.onnx")
        tokens = (args.khmer_onnx_tokens.read_text(encoding="utf-8").rstrip("\n")
                  if args.khmer_onnx_tokens else SEANGHAY_TOKENS)
        (khmer_dir / "tokens.txt").write_text(tokens + "\n", encoding="utf-8")
        files += ["khmer/rec.onnx", "khmer/tokens.txt"]
        components.append({"name": "Khmer ONNX CTC recognizer (rec.onnx)",
                           "license": args.khmer_onnx_license,
                           "url": args.khmer_onnx_source_url,
                           "notes": "Confirm weights license and training data with the author."})
        recognizer = {"backend": "router", "options": {
            "backends": {
                "khmer": {"backend": "onnx-ctc", "options": {
                    "model": "khmer/rec.onnx", "tokens_file": "khmer/tokens.txt",
                    "token_offset": args.khmer_onnx_token_offset, "input_height": 32,
                    "restore_spaces": True}},
                "tesseract": tesseract,
            },
            # ONNX model first, Tesseract when its read fails validation.
            "routes": {script: ["khmer", "tesseract"]
                       for script in args.onnx_scripts.split(",")},
            "default": "tesseract",
            "page": "tesseract",
        }}
        recognition_model = (f"hybrid: onnx-ctc ({args.onnx_scripts}) "
                             "+ tesseract-tessdata_fast")
        license_expr = f"Apache-2.0 AND {args.khmer_onnx_license}"
    else:
        recognizer = tesseract
        recognition_model = "tesseract-tessdata_fast"
        license_expr = "Apache-2.0"

    manifest = build_manifest(
        args.out,
        file_paths=files,
        model_id=args.model_id,
        task="khmer_id_extraction",
        runtime="local",
        recognition_model=recognition_model,
        model_version=args.model_version,
        schema_version="1.0",
        license=license_expr,
        license_approved=False,
        license_components=components,
        recognizer=recognizer,
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)
                                            + "\n", encoding="utf-8")
    print(json.dumps({"bundle": str(args.out), "weights_sha256": manifest["weights_sha256"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
