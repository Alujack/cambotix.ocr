/// Error codes returned by the server or raised by the client itself.
abstract final class KhmerOcrErrorCode {
  // Server codes (see docs/api-reference.md).
  static const invalidImage = 'INVALID_IMAGE';
  static const payloadTooLarge = 'PAYLOAD_TOO_LARGE';
  static const unsupportedMediaType = 'UNSUPPORTED_MEDIA_TYPE';
  static const invalidRequest = 'INVALID_REQUEST';
  static const imageQualityTooLow = 'IMAGE_QUALITY_TOO_LOW';
  static const unsupportedDocument = 'UNSUPPORTED_DOCUMENT';
  static const unauthorized = 'UNAUTHORIZED';
  static const notFound = 'NOT_FOUND';
  static const modelNotAvailable = 'MODEL_NOT_AVAILABLE';
  static const inferenceFailed = 'INFERENCE_FAILED';
  static const internalError = 'INTERNAL_ERROR';

  // Client codes: the request failed before or without a server error body.
  static const networkError = 'NETWORK_ERROR';
  static const timeout = 'TIMEOUT';
  static const unexpectedResponse = 'UNEXPECTED_RESPONSE';
}

/// A failed KhmerOCR request.
///
/// [statusCode] is the HTTP status, or null when the request never got a
/// server response (validation before upload, network failure, timeout).
/// Messages never contain document content.
class KhmerOcrException implements Exception {
  const KhmerOcrException(
    this.code,
    this.message, {
    this.statusCode,
    this.requestId,
    this.details,
  });

  /// One of [KhmerOcrErrorCode], or a newer server code.
  final String code;
  final String message;
  final int? statusCode;

  /// The server's request ID; quote it when reporting a problem.
  final String? requestId;
  final Map<String, dynamic>? details;

  @override
  String toString() {
    final status = statusCode == null ? '' : ' (HTTP $statusCode)';
    final id = requestId == null ? '' : ' [request $requestId]';
    return 'KhmerOcrException: $code$status: $message$id';
  }
}
