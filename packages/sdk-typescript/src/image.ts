import { KhmerOCRError } from "./errors.js";

/** A card photo: a `Blob`/`File`, an `ArrayBuffer`, or bytes (`Uint8Array`, Node `Buffer`). */
export type ImageInput = Blob | ArrayBuffer | ArrayBufferView;

export type ImageContentType = "image/jpeg" | "image/png" | "image/webp";

const ACCEPTED: ReadonlySet<string> = new Set(["image/jpeg", "image/png", "image/webp"]);
const EXTENSIONS: Record<ImageContentType, string> = {
  "image/jpeg": "jpg",
  "image/png": "png",
  "image/webp": "webp",
};

export interface PreparedImage {
  blob: Blob;
  filename: string;
}

/** Identify JPEG, PNG or WebP from the file's first bytes. */
export function sniffContentType(head: Uint8Array): ImageContentType | null {
  const at = (i: number) => head[i];
  if (at(0) === 0xff && at(1) === 0xd8 && at(2) === 0xff) return "image/jpeg";
  const png = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  if (png.every((byte, i) => at(i) === byte)) return "image/png";
  const ascii = (from: number, to: number) => String.fromCharCode(...head.subarray(from, to));
  if (head.length >= 12 && ascii(0, 4) === "RIFF" && ascii(8, 12) === "WEBP") return "image/webp";
  return null;
}

/**
 * Validate an image and wrap it as a typed `Blob` before any request is made,
 * so unusable input fails fast and never leaves the device.
 */
export async function prepareImage(
  image: ImageInput,
  options: { filename?: string | undefined; contentType?: string | undefined; maxUploadBytes: number },
): Promise<PreparedImage> {
  let bytes: Uint8Array | null = null;
  let size: number;
  let declared: string | null = null;
  if (image instanceof Blob) {
    size = image.size;
    declared = image.type ? image.type.split(";")[0]!.trim().toLowerCase() : null;
  } else {
    bytes = image instanceof ArrayBuffer
      ? new Uint8Array(image)
      : new Uint8Array(image.buffer, image.byteOffset, image.byteLength);
    size = bytes.byteLength;
  }
  if (size === 0) {
    throw new KhmerOCRError("INVALID_IMAGE", "The image is empty.");
  }
  if (size > options.maxUploadBytes) {
    throw new KhmerOCRError("PAYLOAD_TOO_LARGE",
      `The image is ${size} bytes; the limit is ${options.maxUploadBytes}.`,
      { details: { size, max_upload_bytes: options.maxUploadBytes } });
  }

  let contentType: string | null = null;
  if (options.contentType !== undefined) {
    contentType = options.contentType.split(";")[0]!.trim().toLowerCase();
  } else if (declared !== null && ACCEPTED.has(declared)) {
    contentType = declared;
  } else {
    const head = bytes ?? new Uint8Array(await (image as Blob).slice(0, 12).arrayBuffer());
    contentType = sniffContentType(head.subarray(0, 12));
  }
  if (contentType === null || !ACCEPTED.has(contentType)) {
    throw new KhmerOCRError("UNSUPPORTED_MEDIA_TYPE",
      "Unsupported image type. Use JPEG, PNG or WebP.");
  }

  const type = contentType as ImageContentType;
  const blob = image instanceof Blob
    ? (image.type === type ? image : image.slice(0, image.size, type))
    : new Blob([bytes as Uint8Array<ArrayBuffer>], { type });
  const filename = options.filename
    ?? (typeof File !== "undefined" && image instanceof File && image.name ? image.name : undefined)
    ?? `image.${EXTENSIONS[type]}`;
  return { blob, filename };
}
