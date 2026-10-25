import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';

import 'errors.dart';
import 'image_type.dart';
import 'models.dart';

/// Client for a KhmerOCR server.
///
/// Recognition runs on the server: this client uploads the image and parses
/// the result. All methods throw [KhmerOcrException] on failure.
///
/// ```dart
/// final client = KhmerOcrClient(baseUrl: Uri.parse('https://ocr.example.com'),
///     apiKey: 'your-api-key');
/// final result = await client.extractNationalId(await photo.readAsBytes());
/// ```
class KhmerOcrClient {
  /// [baseUrl] is the server root, such as `http://192.168.1.10:8000`; `/v1`
  /// is appended unless the path already ends with it. [apiKey] is sent as
  /// `Authorization: Bearer <key>`. [headers] are added to every request, for
  /// example your own auth token when calling KhmerOCR through your backend.
  KhmerOcrClient({
    required Uri baseUrl,
    String? apiKey,
    this.timeout = const Duration(seconds: 60),
    this.maxUploadBytes = 10 * 1024 * 1024,
    Map<String, String> headers = const {},
    http.Client? httpClient,
  })  : baseUrl = _apiRoot(baseUrl),
        _headers = {
          ...headers,
          if (apiKey != null) 'Authorization': 'Bearer $apiKey',
        },
        _http = httpClient ?? http.Client(),
        _ownsHttpClient = httpClient == null;

  /// The API root, ending in `/v1`.
  final Uri baseUrl;
  final Duration timeout;

  /// Images larger than this are rejected before upload. Match the server's
  /// `KHMEROCR_MAX_UPLOAD_BYTES` (10 MiB by default).
  final int maxUploadBytes;

  final Map<String, String> _headers;
  final http.Client _http;
  final bool _ownsHttpClient;

  /// Service liveness and whether a model is loaded. Needs no API key.
  Future<HealthStatus> health({String? requestId}) async =>
      HealthStatus.fromJson(await _get('health', requestId));

  /// The active model and the installed bundles.
  Future<ModelsInfo> models({String? requestId}) async =>
      ModelsInfo.fromJson(await _get('models', requestId));

  /// Extracts Cambodian National ID fields from a photo of the card front.
  ///
  /// Check [DocumentResult.reviewRequired] before using the result.
  Future<DocumentResult> extractNationalId(
    Uint8List image, {
    bool includeTextRegions = false,
    String? filename,
    String? contentType,
    String? requestId,
  }) async =>
      DocumentResult.fromJson(await _upload(
        'documents/national-id/extract',
        image,
        filename: filename,
        contentType: contentType,
        fields: {'include_text_regions': '$includeTextRegions'},
        requestId: requestId,
      ));

  /// Extracts university student ID card fields (provisional layout; results
  /// always require review).
  Future<DocumentResult> extractStudentId(
    Uint8List image, {
    bool includeTextRegions = false,
    String? filename,
    String? contentType,
    String? requestId,
  }) async =>
      DocumentResult.fromJson(await _upload(
        'documents/student-id/extract',
        image,
        filename: filename,
        contentType: contentType,
        fields: {'include_text_regions': '$includeTextRegions'},
        requestId: requestId,
      ));

  /// Checks whether an image is usable before extraction.
  Future<QualityReport> checkQuality(
    Uint8List image, {
    String? filename,
    String? contentType,
    String? requestId,
  }) async =>
      QualityReport.fromJson(await _upload('documents/quality-check', image,
          filename: filename, contentType: contentType, requestId: requestId));

  /// Recognizes general Khmer and English text.
  Future<OcrResult> recognizeText(
    Uint8List image, {
    String? filename,
    String? contentType,
    String? requestId,
  }) async =>
      OcrResult.fromJson(await _upload('ocr', image,
          filename: filename, contentType: contentType, requestId: requestId));

  /// Closes the underlying HTTP client if this client created it.
  void close() {
    if (_ownsHttpClient) _http.close();
  }

  Future<Map<String, dynamic>> _get(String path, String? requestId) {
    final request = http.Request('GET', _url(path));
    return _send(request, requestId);
  }

  Future<Map<String, dynamic>> _upload(
    String path,
    Uint8List image, {
    required String? filename,
    required String? contentType,
    Map<String, String> fields = const {},
    required String? requestId,
  }) {
    if (image.isEmpty) {
      throw const KhmerOcrException(
          KhmerOcrErrorCode.invalidImage, 'The image is empty.');
    }
    if (image.length > maxUploadBytes) {
      throw KhmerOcrException(KhmerOcrErrorCode.payloadTooLarge,
          'The image is larger than $maxUploadBytes bytes.');
    }
    final type = contentType != null
        ? ImageType.fromMimeType(contentType)
        : ImageType.sniff(image);
    if (type == null) {
      throw const KhmerOcrException(KhmerOcrErrorCode.unsupportedMediaType,
          'Unsupported image type. Use JPEG, PNG or WebP.');
    }
    final request = http.MultipartRequest('POST', _url(path))
      ..fields.addAll(fields)
      ..files.add(http.MultipartFile.fromBytes(
        'file',
        image,
        filename: filename ?? 'image.${type.extension}',
        contentType: MediaType.parse(type.mimeType),
      ));
    return _send(request, requestId);
  }

  Uri _url(String path) => baseUrl.replace(path: '${baseUrl.path}/$path');

  Future<Map<String, dynamic>> _send(
      http.BaseRequest request, String? requestId) async {
    request.headers.addAll(_headers);
    request.headers['Accept'] = 'application/json';
    if (requestId != null) request.headers['X-Request-ID'] = requestId;

    final http.Response response;
    try {
      response = await () async {
        final streamed = await _http.send(request);
        return http.Response.fromStream(streamed);
      }()
          .timeout(timeout);
    } on TimeoutException {
      throw KhmerOcrException(KhmerOcrErrorCode.timeout,
          'The server did not respond within ${timeout.inSeconds} s.',
          requestId: requestId);
    } on http.ClientException catch (error) {
      throw KhmerOcrException(KhmerOcrErrorCode.networkError,
          'Could not reach the KhmerOCR server: ${error.message}',
          requestId: requestId);
    } on Exception catch (error) {
      // For example a TLS handshake failure, which is not a ClientException.
      throw KhmerOcrException(KhmerOcrErrorCode.networkError,
          'Could not reach the KhmerOCR server (${error.runtimeType}).',
          requestId: requestId);
    }

    final headerRequestId = response.headers['x-request-id'] ?? requestId;
    Object? body;
    try {
      body = jsonDecode(utf8.decode(response.bodyBytes));
    } on FormatException {
      body = null;
    }

    if (response.statusCode >= 200 && response.statusCode < 300) {
      if (body is Map<String, dynamic>) return body;
      throw KhmerOcrException(KhmerOcrErrorCode.unexpectedResponse,
          'The server returned a response that is not a JSON object.',
          statusCode: response.statusCode, requestId: headerRequestId);
    }

    final error = body is Map ? body['error'] : null;
    if (error is Map && error['code'] is String && error['message'] is String) {
      final details = error['details'];
      throw KhmerOcrException(
        error['code'] as String,
        error['message'] as String,
        statusCode: response.statusCode,
        requestId: error['request_id'] as String? ?? headerRequestId,
        details: details is Map ? details.cast<String, dynamic>() : null,
      );
    }
    throw KhmerOcrException(KhmerOcrErrorCode.unexpectedResponse,
        'The server returned HTTP ${response.statusCode} without an error body.',
        statusCode: response.statusCode, requestId: headerRequestId);
  }

  static Uri _apiRoot(Uri baseUrl) {
    var path = baseUrl.path;
    while (path.endsWith('/')) {
      path = path.substring(0, path.length - 1);
    }
    if (!path.endsWith('/v1')) path = '$path/v1';
    return Uri(
      scheme: baseUrl.scheme,
      userInfo: baseUrl.userInfo,
      host: baseUrl.host,
      port: baseUrl.hasPort ? baseUrl.port : null,
      path: path,
    );
  }
}
