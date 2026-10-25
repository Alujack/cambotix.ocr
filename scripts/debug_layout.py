"""Check a card layout template against real photos without printing their content.

For each photo, writes to --out (default benchmarks/data/private/layout-debug/,
which Git ignores):

* ``<name>.quad.jpg``  the photo with the detected card outline;
* ``<name>.card.png``  the rectified card at the template's canonical size;
* ``<name>.boxes.png`` the card with every template field box drawn on it;
* ``<name>.lines.png`` the card with detected text lines, numbered;
* ``<name>.json``      card detection, line boxes in template coordinates
                       (x, y, width, height as fractions of the card), and
                       per-field results: valid, and correct when labels are given.

Stdout lists field names with valid/correct flags only, never values. Real ID
photos and everything written here are personal data: keep them out of Git.

    python scripts/debug_layout.py photo1.jpg photo2.jpg --model khmerocr-id-v1 \\
        --labels expected.json [--max-side 1600]

``expected.json`` maps each photo's file name to its true field values in the
engine's normalized form, e.g. ``{"card.jpg": {"id_number": "...",
"date_of_birth": "YYYY-MM-DD", ...}}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "core-python" / "src"))

from khmerocr import KhmerOCR, KhmerOCRError  # noqa: E402
from khmerocr.detection import detect_text_lines  # noqa: E402
from khmerocr.documents import template_for_document  # noqa: E402
from khmerocr.preprocessing import load_image, normalize_card  # noqa: E402

DEFAULT_OUT = ROOT / "benchmarks" / "data" / "private" / "layout-debug"


def bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def edit_distance(a: str, b: str) -> int:
    """Characters to change to turn one value into the other; 0 means exactly right."""
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def shrink(image: np.ndarray, max_side: int | None) -> np.ndarray:
    h, w = image.shape[:2]
    if not max_side or max(h, w) <= max_side:
        return image
    scale = max_side / max(h, w)
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("photos", type=Path, nargs="+")
    parser.add_argument("--document-type", default="khmer_national_id")
    parser.add_argument("--model",
                        help="Model bundle folder or ID; without it only geometry is checked.")
    parser.add_argument("--labels", type=Path, help="JSON of true values per photo file name.")
    parser.add_argument("--max-side", type=int, help="Shrink photos first, as the member app does.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    template = template_for_document(args.document_type)
    width, height = template.canonical_size
    labels = json.loads(args.labels.read_text(encoding="utf-8")) if args.labels else {}
    engine = KhmerOCR.from_pretrained(args.model) if args.model else None
    args.out.mkdir(parents=True, exist_ok=True)

    for photo in args.photos:
        image = shrink(load_image(photo, max_bytes=50 << 20, max_pixels=100_000_000), args.max_side)
        card = normalize_card(image, template.canonical_size)
        stem = args.out / (photo.stem + (f".{args.max_side}" if args.max_side else ""))

        outline = bgr(image).copy()
        if card.quad is not None:
            cv2.polylines(outline, [card.quad.astype(np.int32)], True, (0, 255, 0),
                          max(2, max(image.shape[:2]) // 300))
        cv2.imwrite(f"{stem}.quad.jpg", outline)
        cv2.imwrite(f"{stem}.card.png", bgr(card.image))

        boxes = bgr(card.image).copy()
        for spec in template.fields:
            x, y, w, h = spec.box
            p0 = (int(x * width), int(y * height))
            p1 = (int((x + w) * width), int((y + h) * height))
            cv2.rectangle(boxes, p0, p1, (0, 0, 255), 2)
            cv2.putText(boxes, spec.name, (p0[0] + 3, p0[1] + 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1)
        cv2.imwrite(f"{stem}.boxes.png", boxes)

        lines = detect_text_lines(card.image)
        drawn = bgr(card.image).copy()
        for i, box in enumerate(lines):
            cv2.rectangle(drawn, (box.x, box.y), (box.x + box.width, box.y + box.height),
                          (255, 0, 0), 2)
            cv2.putText(drawn, str(i), (box.x, max(12, box.y - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1)
        cv2.imwrite(f"{stem}.lines.png", drawn)

        report: dict[str, object] = {
            "photo_size": [image.shape[1], image.shape[0]],
            "card_detected": card.detected,
            "quarter_turns": card.quarter_turns,
            "lines": [[round(b.x / width, 4), round(b.y / height, 4),
                       round(b.width / width, 4), round(b.height / height, 4)] for b in lines],
        }
        print(f"{photo.name}{' @' + str(args.max_side) if args.max_side else ''}: "
              f"card_detected={card.detected} lines={len(lines)}")
        if engine is not None:
            expected = labels.get(photo.name, {})
            try:
                result = engine.extract_id(image, document_type=args.document_type)
                raw = {r.label: r.text for r in result.regions}
            except KhmerOCRError as exc:
                report["error"] = exc.code.value
                print(f"  error {exc.code.value}")
            else:
                fields = {}
                for name, field in result.fields.items():
                    correct = (field.value == expected[name]) if name in expected else None
                    distance = (edit_distance(field.value or "", expected[name])
                                if name in expected else None)
                    confidence = None if field.confidence is None else round(field.confidence, 3)
                    fields[name] = {"valid": field.valid, "correct": correct, "distance": distance,
                                    "confidence": confidence,
                                    "read_chars": len(raw.get(name, "") or "")}
                    print(f"  {name:16s} valid={field.valid!s:5s} correct={correct!s:5s} "
                          f"distance={distance} confidence={confidence} "
                          f"read_chars={fields[name]['read_chars']}")
                report["fields"] = fields
                report["quarter_turns_used"] = result.metadata.get("quarter_turns")
        Path(f"{stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"debug output: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
