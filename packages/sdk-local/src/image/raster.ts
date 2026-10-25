/**
 * Pixel buffers and the basic operations on them.
 *
 * The image code in this folder reimplements the OpenCV functions the Python
 * engine uses, following OpenCV's own algorithms (fixed-point arithmetic,
 * border modes, rounding) so that both engines see nearly identical pixels.
 * Results can still differ by a grey level here and there; the parity tests
 * in `test/parity.test.ts` measure how much that changes the extracted fields.
 */

/** Row-major 8-bit pixels with 1 (grey) or 3 (RGB) interleaved channels. */
export interface Raster {
  width: number;
  height: number;
  channels: 1 | 3;
  data: Uint8Array;
}

export function createRaster(width: number, height: number, channels: 1 | 3): Raster {
  return { width, height, channels, data: new Uint8Array(width * height * channels) };
}

/** Python's `round()` and OpenCV's `cvRound()`: halves go to the even neighbour. */
export function roundHalfEven(x: number): number {
  const floor = Math.floor(x);
  const diff = x - floor;
  if (diff < 0.5) return floor;
  if (diff > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

export function clampByte(x: number): number {
  return x < 0 ? 0 : x > 255 ? 255 : x;
}

/** OpenCV `saturate_cast<uchar>` from a floating-point value. */
export function saturateByte(x: number): number {
  return clampByte(roundHalfEven(x));
}

/** OpenCV `cvtColor(..., COLOR_RGB2GRAY)` for 8-bit images (fixed point, 14 bits). */
export function toGray(image: Raster): Raster {
  if (image.channels === 1) return image;
  const { width, height, data } = image;
  const out = new Uint8Array(width * height);
  for (let i = 0, j = 0; i < out.length; i++, j += 3) {
    out[i] = (data[j]! * 4899 + data[j + 1]! * 9617 + data[j + 2]! * 1868 + 8192) >> 14;
  }
  return { width, height, channels: 1, data: out };
}

/** Grey to RGB by repeating the channel. */
export function toRgb(image: Raster): Raster {
  if (image.channels === 3) return image;
  const { width, height, data } = image;
  const out = new Uint8Array(width * height * 3);
  for (let i = 0, j = 0; i < data.length; i++, j += 3) {
    out[j] = out[j + 1] = out[j + 2] = data[i]!;
  }
  return { width, height, channels: 3, data: out };
}

/** `image[y:y+height, x:x+width]`, clipped to the image like a NumPy slice. */
export function crop(image: Raster, x: number, y: number, width: number, height: number): Raster {
  const x0 = Math.max(0, Math.min(image.width, x));
  const y0 = Math.max(0, Math.min(image.height, y));
  const x1 = Math.max(x0, Math.min(image.width, x + width));
  const y1 = Math.max(y0, Math.min(image.height, y + height));
  const cn = image.channels;
  const w = x1 - x0;
  const h = y1 - y0;
  const out = new Uint8Array(w * h * cn);
  for (let row = 0; row < h; row++) {
    const from = ((y0 + row) * image.width + x0) * cn;
    out.set(image.data.subarray(from, from + w * cn), row * w * cn);
  }
  return { width: w, height: h, channels: cn, data: out };
}

/** NumPy `rot90(image, k=turns)`: counter-clockwise quarter turns. */
export function rotateQuarterTurns(image: Raster, turns: number): Raster {
  const k = ((turns % 4) + 4) % 4;
  if (k === 0) return image;
  const { width: w, height: h, channels: cn, data } = image;
  const outW = k === 2 ? w : h;
  const outH = k === 2 ? h : w;
  const out = new Uint8Array(data.length);
  for (let i = 0; i < outH; i++) {
    for (let j = 0; j < outW; j++) {
      let sy: number;
      let sx: number;
      if (k === 1) {
        sy = j;
        sx = w - 1 - i;
      } else if (k === 2) {
        sy = h - 1 - i;
        sx = w - 1 - j;
      } else {
        sy = h - 1 - j;
        sx = i;
      }
      const src = (sy * w + sx) * cn;
      const dst = (i * outW + j) * cn;
      for (let c = 0; c < cn; c++) out[dst + c] = data[src + c]!;
    }
  }
  return { width: outW, height: outH, channels: cn, data: out };
}

/** Population mean and standard deviation, as NumPy's `mean()` and `std()`. */
export function meanStd(data: Uint8Array): { mean: number; std: number } {
  if (data.length === 0) return { mean: 0, std: 0 };
  let sum = 0;
  for (let i = 0; i < data.length; i++) sum += data[i]!;
  const mean = sum / data.length;
  let sq = 0;
  for (let i = 0; i < data.length; i++) {
    const d = data[i]! - mean;
    sq += d * d;
  }
  return { mean, std: Math.sqrt(sq / data.length) };
}

/** Add a constant-colour border (OpenCV `copyMakeBorder` with `BORDER_CONSTANT`). */
export function padConstant(image: Raster, pad: number, value: number): Raster {
  const cn = image.channels;
  const out = createRaster(image.width + 2 * pad, image.height + 2 * pad, cn);
  out.data.fill(value);
  for (let y = 0; y < image.height; y++) {
    const from = y * image.width * cn;
    out.data.set(image.data.subarray(from, from + image.width * cn),
      ((y + pad) * out.width + pad) * cn);
  }
  return out;
}
