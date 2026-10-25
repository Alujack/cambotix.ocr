/** Error codes returned by the server (see docs/api-reference.md). */
export type ServerErrorCode =
  | "INVALID_IMAGE"
  | "PAYLOAD_TOO_LARGE"
  | "UNSUPPORTED_MEDIA_TYPE"
  | "INVALID_REQUEST"
  | "IMAGE_QUALITY_TOO_LOW"
  | "UNSUPPORTED_DOCUMENT"
  | "UNAUTHORIZED"
  | "NOT_FOUND"
  | "METHOD_NOT_ALLOWED"
  | "MODEL_NOT_AVAILABLE"
  | "INFERENCE_FAILED"
  | "INTERNAL_ERROR";

/** Error codes raised by the client itself; these have `status: null`. */
export type ClientErrorCode = "NETWORK_ERROR" | "TIMEOUT" | "ABORTED" | "UNEXPECTED_RESPONSE";

export type ErrorCode = ServerErrorCode | ClientErrorCode;

export interface KhmerOCRErrorInit {
  /** HTTP status, or null when no response was received or the error is client-side. */
  status?: number | null;
  requestId?: string | null;
  details?: Record<string, unknown> | null;
  cause?: unknown;
}

export class KhmerOCRError extends Error {
  override readonly name = "KhmerOCRError";
  /** A known `ErrorCode`, or a newer server code this client does not list yet. */
  readonly code: ErrorCode | (string & {});
  readonly status: number | null;
  /** Server request ID: quote it when reporting a problem. */
  readonly requestId: string | null;
  readonly details: Record<string, unknown> | null;

  constructor(code: ErrorCode | (string & {}), message: string, init: KhmerOCRErrorInit = {}) {
    super(message, init.cause === undefined ? undefined : { cause: init.cause });
    this.code = code;
    this.status = init.status ?? null;
    this.requestId = init.requestId ?? null;
    this.details = init.details ?? null;
  }
}
