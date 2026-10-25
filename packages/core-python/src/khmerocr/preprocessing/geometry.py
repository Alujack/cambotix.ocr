"""Card boundary detection, perspective correction, and rotation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .io import ImageArray

# ISO/IEC 7810 ID-1 card: 85.60 mm x 53.98 mm.
ID1_ASPECT = 85.60 / 53.98
_DETECTION_LONG_SIDE = 800
# The card must fill at least this share of the frame. Lower values let
# inner rectangles (such as the portrait photo on a tightly cropped card) be
# mistaken for the card itself.
_MIN_CARD_AREA_RATIO = 0.2
_ASPECT_TOLERANCE = 0.25


@dataclass
class CardNormalization:
    image: ImageArray
    detected: bool
    # Card corners in source-image coordinates (tl, tr, br, bl), if detected.
    quad: NDArray[np.float32] | None
    # Quarter turns (counter-clockwise) applied after warping.
    quarter_turns: int


def normalize_card(image: ImageArray, size: tuple[int, int]) -> CardNormalization:
    """Locate the card, rectify it, and resize it to ``size`` (width, height).

    If no card boundary is found the whole image is used, on the assumption
    that it is already a tight crop of the card; callers must surface that as
    a warning because the field layout may then be misaligned.
    """
    quad = detect_card_quad(image)
    if quad is not None:
        card = warp_quad(image, quad)
    else:
        card = image
    turns = 0
    if card.shape[0] > card.shape[1]:
        card = rotate_quarter_turns(card, 1)
        turns = 1
    interpolation = cv2.INTER_AREA if card.shape[1] > size[0] else cv2.INTER_CUBIC
    card = np.asarray(cv2.resize(card, size, interpolation=interpolation), dtype=np.uint8)
    return CardNormalization(image=card, detected=quad is not None, quad=quad, quarter_turns=turns)


def detect_card_quad(image: ImageArray) -> NDArray[np.float32] | None:
    h, w = image.shape[:2]
    scale = _DETECTION_LONG_SIDE / max(h, w)
    small = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    area = gray.shape[0] * gray.shape[1]

    for mask in _candidate_masks(gray):
        quad = _largest_card_quad(mask, area)
        if quad is not None:
            # Reject quads that are simply the image border.
            if _covers_whole_image(quad, gray.shape[1], gray.shape[0]):
                return None
            return (quad / scale).astype(np.float32)
    return None


def warp_quad(image: ImageArray, quad: NDArray[np.float32]) -> ImageArray:
    tl, tr, br, bl = quad
    width = round(float(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = round(float(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                   dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
    return np.asarray(warped, dtype=np.uint8)


def rotate_quarter_turns(image: ImageArray, turns: int) -> ImageArray:
    return np.ascontiguousarray(np.rot90(image, k=turns % 4))


def order_quad(points: NDArray[Any]) -> NDArray[np.float32]:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    pts = points.reshape(4, 2).astype(np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(sums)], pts[np.argmin(diffs)],
                     pts[np.argmax(sums)], pts[np.argmax(diffs)]], dtype=np.float32)


def _candidate_masks(gray: NDArray[Any]) -> list[NDArray[Any]]:
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [edges, otsu, cv2.bitwise_not(otsu)]


def _largest_card_quad(mask: NDArray[Any], image_area: int) -> NDArray[np.float32] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        contour_area = cv2.contourArea(contour)
        if contour_area < _MIN_CARD_AREA_RATIO * image_area:
            break
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
        if len(approx) == 4:
            quad = order_quad(approx)
        else:
            rect = cv2.minAreaRect(hull)
            if contour_area / max(rect[1][0] * rect[1][1], 1.0) < 0.85:
                continue
            quad = order_quad(cv2.boxPoints(rect))
        if _plausible_card(quad):
            return quad
    return None


def _plausible_card(quad: NDArray[np.float32]) -> bool:
    tl, tr, br, bl = quad
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if min(width, height) < 1:
        return False
    aspect = max(width, height) / min(width, height)
    return bool(abs(aspect - ID1_ASPECT) <= _ASPECT_TOLERANCE)


def _covers_whole_image(quad: NDArray[np.float32], width: int, height: int) -> bool:
    margin = 3
    xs, ys = quad[:, 0], quad[:, 1]
    return bool(np.min(xs) <= margin and np.min(ys) <= margin
                and np.max(xs) >= width - 1 - margin and np.max(ys) >= height - 1 - margin)
