/** Image usability checks: size, blur, glare, exposure, and contrast. */
import type { QualityThresholds } from "../config.js";
import { connectedComponents } from "../image/components.js";
import { dilateRect, laplacianVariance } from "../image/filters.js";
import { meanStd, roundHalfEven, toGray, type Raster } from "../image/raster.js";
import { resize } from "../image/resize.js";
import type { QualityMeasurements } from "../schemas.js";

const ANALYSIS_LONG_SIDE = 1000;
const SATURATED = 250;
// Glare hotspots fade out through a soft bright halo; printed white card
// backgrounds end at sharp edges. A saturated region counts as glare when
// this share of its surrounding ring is bright-but-not-clipped.
const HALO_RING_PX = 10;
const HALO_MIN_FRACTION = 0.6;
const MIN_GLARE_COMPONENT_PX = 50;

export function assessQuality(image: Raster, thresholds: QualityThresholds): QualityMeasurements {
  const { width, height } = image;
  const gray = resizeLongSide(toGray(image), ANALYSIS_LONG_SIDE);

  const blurScore = laplacianVariance(gray);
  const glareRatio = glareRatioOf(gray);
  const { mean: brightness, std: contrast } = meanStd(gray.data);

  const issues: string[] = [];
  const blocking: string[] = [];

  if (Math.min(width, height) < thresholds.minShortSide) blocking.push("image_too_small");
  if (contrast < thresholds.minContrast) blocking.push("low_contrast");
  if (brightness < thresholds.minBrightness) blocking.push("image_too_dark");
  else if (brightness > thresholds.maxBrightness) blocking.push("image_too_bright");

  if (blurScore < thresholds.blurReject) blocking.push("too_blurry");
  else if (blurScore < thresholds.blurWarn) issues.push("blurry");
  if (glareRatio > thresholds.glareRejectRatio) blocking.push("severe_glare");
  else if (glareRatio > thresholds.glareWarnRatio) issues.push("glare_detected");
  if (height > width) issues.push("portrait_orientation");

  return {
    width, height, blurScore, glareRatio, brightness, contrast,
    issues: [...blocking, ...issues],
    blockingIssues: blocking,
  };
}

/** Share of the image covered by saturated regions that look like glare. */
function glareRatioOf(gray: Raster): number {
  const { width: w, height: h, data } = gray;
  const saturated = new Uint8Array(w * h);
  for (let i = 0; i < data.length; i++) saturated[i] = data[i]! >= SATURATED ? 1 : 0;
  const { labels, stats } = connectedComponents(saturated, w, h);
  const ring = 2 * HALO_RING_PX + 1;
  let glarePixels = 0;
  stats.forEach((s, index) => {
    const label = index + 1;
    if (s.area < MIN_GLARE_COMPONENT_PX) return;
    const x0 = Math.max(0, s.x - HALO_RING_PX);
    const y0 = Math.max(0, s.y - HALO_RING_PX);
    const x1 = Math.min(w, s.x + s.width + HALO_RING_PX);
    const y1 = Math.min(h, s.y + s.height + HALO_RING_PX);
    const cw = x1 - x0;
    const ch = y1 - y0;
    const component = new Uint8Array(cw * ch);
    for (let y = 0; y < ch; y++) {
      for (let x = 0; x < cw; x++) {
        component[y * cw + x] = labels[(y + y0) * w + x + x0] === label ? 1 : 0;
      }
    }
    const dilated = dilateRect({ width: cw, height: ch, channels: 1, data: component }, ring);
    let ringCount = 0;
    let halo = 0;
    for (let y = 0; y < ch; y++) {
      for (let x = 0; x < cw; x++) {
        const i = y * cw + x;
        if (dilated.data[i]! > component[i]!) {
          ringCount++;
          const v = data[(y + y0) * w + x + x0]!;
          if (v >= 200 && v < SATURATED) halo++;
        }
      }
    }
    if (ringCount === 0) return;
    if (halo / ringCount >= HALO_MIN_FRACTION) glarePixels += s.area;
  });
  return glarePixels / (w * h);
}

function resizeLongSide(gray: Raster, longSide: number): Raster {
  const { width: w, height: h } = gray;
  const scale = longSide / Math.max(h, w);
  if (Math.abs(scale - 1) < 1e-3) return gray;
  return resize(gray, Math.max(1, roundHalfEven(w * scale)), Math.max(1, roundHalfEven(h * scale)),
    scale < 1 ? "area" : "cubic");
}
