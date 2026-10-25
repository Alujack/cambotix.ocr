"""Record OpenCV outputs that test/opencv.test.ts compares the TypeScript image code against.

    .venv/bin/python packages/sdk-local/test/fixtures/make_opencv_fixtures.py

Inputs are small crops of the engine's synthetic SPECIMEN card, so the
fixtures contain no real document data.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "packages" / "core-python" / "src"))

from khmerocr.preprocessing.geometry import detect_card_quad, normalize_card  # noqa: E402
from khmerocr.preprocessing.quality import assess_quality  # noqa: E402
from khmerocr.config import QualityThresholds  # noqa: E402
from khmerocr.recognition.onnx_ctc import trim_to_ink  # noqa: E402
from khmerocr.testing.synthetic import (  # noqa: E402
    SyntheticIdentity, add_glare, render_card)

OUT = Path(__file__).with_name("opencv.json")


def png(a: np.ndarray) -> dict:
    """Large images are stored as PNG to keep the fixture small."""
    bgr = cv2.cvtColor(a, cv2.COLOR_RGB2BGR) if a.ndim == 3 else a
    ok, data = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    assert ok
    return {"shape": list(a.shape), "png": base64.b64encode(data.tobytes()).decode()}


def photograph(card: np.ndarray) -> np.ndarray:
    """A tilted card on a smooth backdrop (no sensor noise, so the PNG stays small)."""
    h, w = card.shape[:2]
    yy, xx = np.mgrid[0:600, 0:800]
    backdrop = np.dstack([60 + xx // 40, 52 + yy // 40, 45 + (xx + yy) // 80]).astype(np.uint8)
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    dst = np.array([[112, 128], [690, 92], [716, 452], [126, 488]], np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(card, matrix, (800, 600))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), matrix, (800, 600))
    return np.where(mask[:, :, None] > 0, warped, backdrop).astype(np.uint8)


def enc(a: np.ndarray) -> dict:
    a = np.ascontiguousarray(a)
    return {"shape": list(a.shape), "dtype": str(a.dtype), "b64": base64.b64encode(a.tobytes()).decode()}


def main() -> None:
    card = render_card(SyntheticIdentity())
    rgb = np.ascontiguousarray(card[180:300, 300:480])  # 180x120 crop with text
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    otsu_t, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dil = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    closed = cv2.morphologyEx(dil, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_areas = sorted((float(cv2.contourArea(c)) for c in contours), reverse=True)
    inv_contours, _ = cv2.findContours(cv2.bitwise_not(otsu), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    inv_areas = sorted((float(cv2.contourArea(c)) for c in inv_contours), reverse=True)
    sat = (gray >= 200).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(sat, connectivity=8)
    comp = sorted([[int(v) for v in s] for s in stats[1:]])

    src = np.array([[0, 0], [179, 0], [179, 119], [0, 119]], np.float32)
    dst = np.array([[8, 5], [170, 12], [175, 110], [3, 116]], np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(rgb, matrix, (180, 120), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)

    # Whole-pipeline checks on a photographed card.
    photo = photograph(render_card(SyntheticIdentity()))
    glare = add_glare(photo)
    quad = detect_card_quad(photo)
    normalized = normalize_card(photo, (1012, 638))
    quality = {name: assess_quality(img, QualityThresholds()).__dict__ for name, img in
               (("photo", photo), ("glare", glare))}
    trimmed = trim_to_ink(cv2.cvtColor(normalized.image[200:290, 330:980], cv2.COLOR_RGB2GRAY))

    data = {
        "rgb": enc(rgb),
        "gray": enc(gray),
        "resize": {
            "area_down": enc(cv2.resize(rgb, (71, 47), interpolation=cv2.INTER_AREA)),
            "area_int": enc(cv2.resize(gray, (60, 40), interpolation=cv2.INTER_AREA)),
            "area_up": enc(cv2.resize(gray, (250, 170), interpolation=cv2.INTER_AREA)),
            "area_mixed": enc(cv2.resize(gray, (90, 150), interpolation=cv2.INTER_AREA)),
            "cubic_up": enc(cv2.resize(rgb, (257, 171), interpolation=cv2.INTER_CUBIC)),
            "cubic_down": enc(cv2.resize(gray, (101, 67), interpolation=cv2.INTER_CUBIC)),
        },
        "blurred": enc(blurred),
        "laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "edges": enc(edges),
        "otsu_threshold": float(otsu_t),
        "otsu": enc(otsu),
        "closed": enc(closed),
        "contour_areas": contour_areas,
        "inverted_contour_areas": inv_areas,
        "components": comp,
        "matrix": matrix.ravel().tolist(),
        "warped": enc(warped),
        "photo": png(photo),
        "quad": None if quad is None else quad.tolist(),
        "normalized": png(normalized.image),
        "quality": quality,
        "glare": png(glare),
        "trimmed_shape": None if trimmed is None else list(trimmed.shape),
    }
    OUT.write_text(json.dumps(data), encoding="utf-8")
    print(OUT, OUT.stat().st_size)


if __name__ == "__main__":
    main()
