import { createRaster, type Raster } from "./raster.js";

/**
 * OpenCV `Canny(gray, low, high)` with a 3x3 Sobel aperture and the L1
 * gradient norm: Sobel with a replicated border, OpenCV's integer
 * non-maximum suppression, and 8-connected hysteresis.
 */
export function canny(gray: Raster, lowThreshold: number, highThreshold: number): Raster {
  const { width: w, height: h, data } = gray;
  const low = Math.floor(Math.min(lowThreshold, highThreshold));
  const high = Math.floor(Math.max(lowThreshold, highThreshold));
  // The image with a one-pixel replicated border, for the Sobel filter.
  const pw = w + 2;
  const src = new Int32Array(pw * (h + 2));
  for (let y = -1; y <= h; y++) {
    const sy = y < 0 ? 0 : y >= h ? h - 1 : y;
    for (let x = -1; x <= w; x++) {
      src[(y + 1) * pw + x + 1] = data[sy * w + (x < 0 ? 0 : x >= w ? w - 1 : x)]!;
    }
  }
  const dx = new Int32Array(w * h);
  const dy = new Int32Array(w * h);
  // Magnitudes padded by one pixel of zeros on every side.
  const mag = new Int32Array(pw * (h + 2));
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const c = (y + 1) * pw + x + 1;
      const nw = src[c - pw - 1]!, n = src[c - pw]!, ne = src[c - pw + 1]!;
      const west = src[c - 1]!, east = src[c + 1]!;
      const sw = src[c + pw - 1]!, s = src[c + pw]!, se = src[c + pw + 1]!;
      const gx = ne + 2 * east + se - nw - 2 * west - sw;
      const gy = sw + 2 * s + se - nw - 2 * n - ne;
      dx[y * w + x] = gx;
      dy[y * w + x] = gy;
      mag[c] = Math.abs(gx) + Math.abs(gy);
    }
  }

  // 0: weak candidate, 1: not an edge, 2: strong edge.
  const map = new Uint8Array(w * h).fill(1);
  const stack: number[] = [];
  const TG22 = Math.floor(0.4142135623730950488016887242097 * (1 << 15) + 0.5);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = (y + 1) * pw + x + 1;
      const m = mag[p]!;
      if (m <= low) continue;
      const xs = dx[y * w + x]!;
      const ys = dy[y * w + x]!;
      const ax = Math.abs(xs);
      const ay = Math.abs(ys) * 32768;
      const tg22x = ax * TG22;
      let isMax: boolean;
      if (ay < tg22x) {
        isMax = m > mag[p - 1]! && m >= mag[p + 1]!;
      } else {
        const tg67x = tg22x + ax * 65536;
        if (ay > tg67x) {
          isMax = m > mag[p - pw]! && m >= mag[p + pw]!;
        } else {
          const s = (xs ^ ys) < 0 ? -1 : 1;
          isMax = m > mag[p - pw - s]! && m > mag[p + pw + s]!;
        }
      }
      if (!isMax) continue;
      if (m > high) {
        map[y * w + x] = 2;
        stack.push(y * w + x);
      } else {
        map[y * w + x] = 0;
      }
    }
  }

  while (stack.length > 0) {
    const i = stack.pop()!;
    const x = i % w;
    const y = (i - x) / w;
    for (let oy = -1; oy <= 1; oy++) {
      const yy = y + oy;
      if (yy < 0 || yy >= h) continue;
      for (let ox = -1; ox <= 1; ox++) {
        const xx = x + ox;
        if (xx < 0 || xx >= w) continue;
        const j = yy * w + xx;
        if (map[j] === 0) {
          map[j] = 2;
          stack.push(j);
        }
      }
    }
  }

  const out = createRaster(w, h, 1);
  for (let i = 0; i < map.length; i++) out.data[i] = map[i] === 2 ? 255 : 0;
  return out;
}
