/**
 * Safe image loading. The format is identified from magic bytes, never from
 * a file name or declared type, and the pixel count is read from the file
 * header and checked before any pixel data is decoded.
 */
import { ocrError } from "../errors.js";
import type { ImageFormat, Platform } from "../platform.js";
import type { Raster } from "./raster.js";

/** Pixels you already have, e.g. `ImageData` from a canvas (RGBA). */
export interface RawImage {
  data: Uint8Array | Uint8ClampedArray;
  width: number;
  height: number;
  /** 1 (grey), 3 (RGB) or 4 (RGBA). Inferred from the data length when omitted. */
  channels?: 1 | 3 | 4;
}

/** A JPEG, PNG or WebP file (`Blob`/`File`, `ArrayBuffer`, bytes), or raw pixels. */
export type LocalImageInput = Blob | ArrayBuffer | ArrayBufferView | RawImage;

export interface ImageLimits {
  maxBytes: number;
  maxPixels: number;
}

export function sniffFormat(head: Uint8Array): ImageFormat | null {
  if (head[0] === 0xff && head[1] === 0xd8 && head[2] === 0xff) return "image/jpeg";
  const png = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  if (png.every((byte, i) => head[i] === byte)) return "image/png";
  const ascii = (from: number, to: number) => String.fromCharCode(...head.subarray(from, to));
  if (head.length >= 12 && ascii(0, 4) === "RIFF" && ascii(8, 12) === "WEBP") return "image/webp";
  return null;
}

/** Width and height from the file header, or null when the header is not understood. */
export function readImageSize(data: Uint8Array, format: ImageFormat): { width: number; height: number } | null {
  const u16be = (i: number) => (data[i]! << 8) | data[i + 1]!;
  const u32be = (i: number) => ((data[i]! << 24) >>> 0) + (data[i + 1]! << 16) + (data[i + 2]! << 8) + data[i + 3]!;
  const u24le = (i: number) => data[i]! | (data[i + 1]! << 8) | (data[i + 2]! << 16);
  if (format === "image/png") {
    if (data.length < 24) return null;
    return { width: u32be(16), height: u32be(20) };
  }
  if (format === "image/webp") {
    if (data.length < 30) return null;
    const chunk = String.fromCharCode(...data.subarray(12, 16));
    if (chunk === "VP8 ") return { width: (data[26]! | (data[27]! << 8)) & 0x3fff, height: (data[28]! | (data[29]! << 8)) & 0x3fff };
    if (chunk === "VP8L") {
      const [b0, b1, b2, b3] = [data[21]!, data[22]!, data[23]!, data[24]!];
      return { width: 1 + (((b1 & 0x3f) << 8) | b0), height: 1 + (((b3 & 0x0f) << 10) | (b2 << 2) | ((b1 & 0xc0) >> 6)) };
    }
    if (chunk === "VP8X") return { width: 1 + u24le(24), height: 1 + u24le(27) };
    return null;
  }
  // JPEG: walk the marker segments to the first start-of-frame.
  let i = 2;
  while (i + 9 < data.length) {
    if (data[i] !== 0xff) return null;
    let marker = data[i + 1]!;
    while (marker === 0xff && i + 2 < data.length) {
      i++;
      marker = data[i + 1]!;
    }
    if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) {
      i += 2;
      continue;
    }
    const isFrame = marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc;
    if (isFrame) return { width: u16be(i + 7), height: u16be(i + 5) };
    i += 2 + u16be(i + 2);
  }
  return null;
}

function checkPixels(width: number, height: number, maxPixels: number): void {
  if (width <= 0 || height <= 0) throw ocrError("INVALID_IMAGE", "The image has no pixels.");
  if (width * height > maxPixels) {
    throw ocrError("INVALID_IMAGE", "The image dimensions exceed the maximum allowed pixel count.",
      { max_pixels: maxPixels });
  }
}

function isRawImage(input: unknown): input is RawImage {
  return typeof input === "object" && input !== null && "data" in input && "width" in input && "height" in input;
}

/** Decode any supported input into RGB pixels. */
export async function loadImage(input: LocalImageInput, limits: ImageLimits, platform: Platform): Promise<Raster> {
  if (isRawImage(input)) return fromRaw(input, limits.maxPixels);
  let bytes: Uint8Array;
  if (typeof Blob !== "undefined" && input instanceof Blob) {
    if (input.size > limits.maxBytes) throw tooLarge(limits.maxBytes);
    bytes = new Uint8Array(await input.arrayBuffer());
  } else if (input instanceof ArrayBuffer) {
    bytes = new Uint8Array(input);
  } else if (ArrayBuffer.isView(input)) {
    bytes = new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  } else {
    throw ocrError("INVALID_IMAGE", "Unsupported image input type.");
  }
  if (bytes.byteLength === 0) throw ocrError("INVALID_IMAGE", "The image is empty.");
  if (bytes.byteLength > limits.maxBytes) throw tooLarge(limits.maxBytes);
  const format = sniffFormat(bytes.subarray(0, 12));
  if (format === null) {
    throw ocrError("UNSUPPORTED_MEDIA_TYPE",
      "Unsupported or unrecognized image format. Supported formats: JPEG, PNG, WebP.");
  }
  const size = readImageSize(bytes, format);
  if (size === null) throw ocrError("INVALID_IMAGE", "The image could not be decoded.");
  checkPixels(size.width, size.height, limits.maxPixels);
  let image: Raster;
  try {
    image = await platform.decodeImage(bytes, format);
  } catch (cause) {
    if (cause instanceof Error && cause.name === "KhmerOCRError") throw cause;
    throw ocrError("INVALID_IMAGE", "The image could not be decoded.", undefined, cause);
  }
  checkPixels(image.width, image.height, limits.maxPixels);
  return image;
}

function tooLarge(maxBytes: number) {
  return ocrError("PAYLOAD_TOO_LARGE", "The image exceeds the maximum allowed size.", { max_bytes: maxBytes });
}

function fromRaw(input: RawImage, maxPixels: number): Raster {
  const { width, height, data } = input;
  if (!Number.isInteger(width) || !Number.isInteger(height)) {
    throw ocrError("INVALID_IMAGE", "Image width and height must be integers.");
  }
  checkPixels(width, height, maxPixels);
  const channels = input.channels ?? data.length / (width * height);
  if (![1, 3, 4].includes(channels) || data.length !== width * height * channels) {
    throw ocrError("INVALID_IMAGE", "Raw pixels must have 1, 3 or 4 channels and match width x height.");
  }
  const out = new Uint8Array(width * height * 3);
  for (let i = 0, j = 0; i < width * height; i++, j += channels) {
    if (channels === 1) {
      out[3 * i] = out[3 * i + 1] = out[3 * i + 2] = data[j]!;
    } else {
      out[3 * i] = data[j]!;
      out[3 * i + 1] = data[j + 1]!;
      out[3 * i + 2] = data[j + 2]!;
    }
  }
  return { width, height, channels: 3, data: out };
}
