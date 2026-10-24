"""Extract National ID fields locally.

    python examples/python-local/extract_id.py national-id.jpg --model khmerocr-id-v1

Without a model installed, run with --demo to see the result format using a
scripted recognizer and a synthetic SPECIMEN card.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from khmerocr import KhmerOCR, KhmerOCRError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, nargs="?")
    parser.add_argument("--model", default="khmerocr-id-v1")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        from khmerocr.testing import ScriptedRecognizer, place_on_background, render_card

        ocr = KhmerOCR(ScriptedRecognizer({
            "id_number": ("000000001", 0.97),
            "full_name_km": ("គោត្តនាមនិងនាម: គំរូ សាកល្បង", 0.93),
            "full_name_latin": ("KUMRU SAKALBANG", 0.95),
            "date_of_birth": ("ថ្ងៃខែឆ្នាំកំណើត: 01.01.1990", 0.91),
            "gender": ("ភេទ: ប្រុស", 0.90),
        }))
        image = place_on_background(render_card(), angle=8)
    else:
        if args.image is None:
            parser.error("an image path is required unless --demo is set")
        ocr = KhmerOCR.from_pretrained(args.model)
        image = args.image

    with ocr:
        quality = ocr.check_image(image)
        if not quality.usable:
            print("Please retake the photo:", ", ".join(quality.blocking_issues))
            return 1
        try:
            result = ocr.extract_id(image)
        except KhmerOCRError as exc:
            print(json.dumps(exc.to_dict(), indent=2))
            return 1

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    if result.review_required:
        print("\nA person must confirm these fields before they are used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
