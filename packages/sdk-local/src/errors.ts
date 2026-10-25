import { KhmerOCRError } from "@khmerocr/client";

export { KhmerOCRError };

/**
 * Error codes raised by the local engine. They are the codes the REST API
 * returns for the same problems, so code written against `@khmerocr/client`
 * handles both. Messages never contain document content.
 */
export type LocalErrorCode =
  | "INVALID_IMAGE"
  | "PAYLOAD_TOO_LARGE"
  | "UNSUPPORTED_MEDIA_TYPE"
  | "IMAGE_QUALITY_TOO_LOW"
  | "UNSUPPORTED_DOCUMENT"
  | "MODEL_NOT_AVAILABLE"
  | "INFERENCE_FAILED"
  | "ABORTED";

export function ocrError(code: LocalErrorCode, message: string,
                         details?: Record<string, unknown>, cause?: unknown): KhmerOCRError {
  return new KhmerOCRError(code, message, {
    ...(details === undefined ? {} : { details }),
    ...(cause === undefined ? {} : { cause }),
  });
}
