"""Generate a synthetic benchmark set with clearly fake identities.

Synthetic results validate the *pipeline* (geometry, quality gating, mapping,
review flags). They are not evidence of accuracy on real cards; the release
gate must be evaluated on an authorized, held-out set of real images.

    python benchmarks/generate_synthetic.py --out benchmarks/data/synthetic --count 60
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "core-python" / "src"))

from khmerocr.testing.synthetic import (  # noqa: E402
    add_blur,
    add_glare,
    place_on_background,
    random_identity,
    render_card,
)

CATEGORIES = {
    "clean": {},
    "rotated": {"angle": (-30, 30)},
    "perspective": {"angle": (-10, 10), "perspective": 0.05},
    "blur": {"blur": (1.2, 2.0)},
    "glare": {"glare": True},
    "upside_down": {"angle": (170, 190)},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=60, help="Total images.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    images = args.out / "images"
    images.mkdir(parents=True, exist_ok=True)
    names = list(CATEGORIES)
    with (args.out / "labels.jsonl").open("w", encoding="utf-8") as labels:
        for i in range(args.count):
            category = names[i % len(names)]
            spec = CATEGORIES[category]
            identity = random_identity(rng)
            angle = rng.uniform(*spec["angle"]) if "angle" in spec else 0.0
            photo = place_on_background(render_card(identity), angle=angle,
                                        perspective=float(spec.get("perspective", 0.0)),
                                        seed=rng.randrange(1 << 30))
            if "blur" in spec:
                photo = add_blur(photo, rng.uniform(*spec["blur"]))
            if spec.get("glare"):
                photo = add_glare(photo, center=(rng.uniform(0.4, 0.7), rng.uniform(0.3, 0.6)))
            name = f"synthetic_{i:04d}.jpg"
            Image.fromarray(photo).save(images / name, quality=92)
            labels.write(json.dumps({
                "image": f"images/{name}",
                "category": category,
                "fields": identity.expected_fields(),
            }, ensure_ascii=False) + "\n")
    print(f"wrote {args.count} images to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
