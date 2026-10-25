import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:khmerocr/khmerocr.dart';
import 'package:test/test.dart';

final jpeg = Uint8List.fromList([0xFF, 0xD8, 0xFF, 0xE0, 1, 2, 3]);
final png =
    Uint8List.fromList([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0]);
final webp =
    Uint8List.fromList([...'RIFF'.codeUnits, 0, 0, 0, 0, ...'WEBP'.codeUnits]);

final nationalIdJson = {
  'schema_version': '1.0',
  'document_type': 'khmer_national_id',
  'status': 'review_required',
  'fields': {
    'id_number': {
      'value': '000000001',
      'confidence': 0.96,
      'source': 'front_region',
      'valid': true
    },
    'full_name_km': {
      'value': 'គំរូ សាកល្បង',
      'confidence': 0.92,
      'source': 'front_region',
      'valid': true
    },
    'full_name_latin': {
      'value': 'KUMRU SAKALBANG',
      'confidence': 0.92,
      'source': 'front_region',
      'valid': true
    },
    'date_of_birth': {
      'value': null,
      'confidence': null,
      'source': 'front_region',
      'valid': false
    },
    'gender': {
      'value': 'M',
      'confidence': 1,
      'source': 'front_region',
      'valid': true
    },
  },
  'warnings': [
    'layout_template_unverified',
    'low_confidence:full_name_km',
    'field_not_detected:date_of_birth'
  ],
  'review_required': true,
  'metadata': {
    'template_id': 'khmer_national_id_front_v1',
    'card_detected': true
  },
  'text_regions': [
    {
      'text': 'គោត្តនាមនិងនាម: គំរូ សាកល្បង',
      'confidence': 0.92,
      'bbox': {'x': 10, 'y': 20, 'width': 300, 'height': 40},
      'label': 'full_name_km',
    },
  ],
};

/// A client whose HTTP layer records the request and answers with [handler].
(KhmerOcrClient, List<http.Request>) clientWith(
  FutureOr<http.Response> Function(http.Request request) handler, {
  String baseUrl = 'http://ocr.test:8000',
  String? apiKey,
  Map<String, String> headers = const {},
  Duration timeout = const Duration(seconds: 5),
  int maxUploadBytes = 10 * 1024 * 1024,
}) {
  final requests = <http.Request>[];
  final mock = MockClient((request) async {
    requests.add(request);
    return handler(request);
  });
  final client = KhmerOcrClient(
    baseUrl: Uri.parse(baseUrl),
    apiKey: apiKey,
    headers: headers,
    timeout: timeout,
    maxUploadBytes: maxUploadBytes,
    httpClient: mock,
  );
  return (client, requests);
}

http.Response jsonResponse(Object body,
        {int status = 200, Map<String, String> headers = const {}}) =>
    http.Response.bytes(utf8.encode(jsonEncode(body)), status,
        headers: {'content-type': 'application/json', ...headers});

/// The raw multipart body. Latin-1 maps every byte, so the binary image part
/// does not break decoding.
String body(http.Request request) => latin1.decode(request.bodyBytes);

Future<KhmerOcrException> thrown(Future<Object?> future) async {
  try {
    await future;
  } on KhmerOcrException catch (error) {
    return error;
  }
  fail('expected a KhmerOcrException');
}

void main() {
  group('URL building', () {
    for (final (base, expected) in [
      ('http://ocr.test:8000', 'http://ocr.test:8000/v1/health'),
      ('http://ocr.test:8000/', 'http://ocr.test:8000/v1/health'),
      ('http://ocr.test:8000/v1', 'http://ocr.test:8000/v1/health'),
      ('http://ocr.test:8000/v1/', 'http://ocr.test:8000/v1/health'),
      (
        'https://example.com/khmerocr',
        'https://example.com/khmerocr/v1/health'
      ),
      (
        'https://example.com/khmerocr/v1?x=1#f',
        'https://example.com/khmerocr/v1/health'
      ),
    ]) {
      test('$base -> $expected', () async {
        final (client, requests) =
            clientWith((_) => jsonResponse({}), baseUrl: base);
        await client.health();
        expect(requests.single.url.toString(), expected);
      });
    }
  });

  group('headers', () {
    test('sends the API key, extra headers and request ID', () async {
      final (client, requests) = clientWith((_) => jsonResponse({}),
          apiKey: 'secret-key-0123456789',
          headers: {'X-App-Token': 'user-token'});
      await client.models(requestId: 'req-42');
      final headers = requests.single.headers;
      expect(headers['Authorization'], 'Bearer secret-key-0123456789');
      expect(headers['X-App-Token'], 'user-token');
      expect(headers['X-Request-ID'], 'req-42');
      expect(headers['Accept'], 'application/json');
    });

    test('sends no Authorization header without an API key', () async {
      final (client, requests) = clientWith((_) => jsonResponse({}));
      await client.health();
      expect(requests.single.headers.containsKey('Authorization'), isFalse);
      expect(requests.single.headers.containsKey('X-Request-ID'), isFalse);
    });
  });

  group('uploads', () {
    test('extract posts the image as the file part with its type', () async {
      final (client, requests) =
          clientWith((_) => jsonResponse(nationalIdJson));
      await client.extractNationalId(jpeg, includeTextRegions: true);
      final request = requests.single;
      expect(request.method, 'POST');
      expect(request.url.path, '/v1/documents/national-id/extract');
      expect(
          request.headers['content-type'], startsWith('multipart/form-data'));
      final text = body(request);
      expect(text, contains('name="file"; filename="image.jpg"'));
      expect(text, contains('content-type: image/jpeg'));
      expect(text, contains('name="include_text_regions"\r\n\r\ntrue'));
    });

    test(
        'student ID uses its endpoint and defaults include_text_regions to false',
        () async {
      final (client, requests) =
          clientWith((_) => jsonResponse(nationalIdJson));
      await client.extractStudentId(png, filename: 'card.png');
      expect(requests.single.url.path, '/v1/documents/student-id/extract');
      final text = body(requests.single);
      expect(text, contains('filename="card.png"'));
      expect(text, contains('name="include_text_regions"\r\n\r\nfalse'));
    });

    test('quality check and OCR send no include_text_regions field', () async {
      final (client, requests) = clientWith((request) => jsonResponse(
          request.url.path.endsWith('ocr')
              ? {
                  'text': '',
                  'regions': [],
                  'warnings': [],
                  'review_required': false
                }
              : {'usable': true, 'issues': [], 'blocking_issues': []}));
      await client.checkQuality(webp);
      await client.recognizeText(webp);
      expect(requests.map((r) => r.url.path),
          ['/v1/documents/quality-check', '/v1/ocr']);
      for (final request in requests) {
        expect(body(request), isNot(contains('include_text_regions')));
        expect(body(request), contains('content-type: image/webp'));
        expect(body(request), contains('filename="image.webp"'));
      }
    });

    test('an explicit content type wins over sniffing', () async {
      final (client, requests) =
          clientWith((_) => jsonResponse(nationalIdJson));
      await client.extractNationalId(Uint8List.fromList([1, 2, 3]),
          contentType: 'image/png');
      expect(body(requests.single), contains('content-type: image/png'));
    });

    test('sniffs all accepted formats', () {
      expect(ImageType.sniff(jpeg), ImageType.jpeg);
      expect(ImageType.sniff(png), ImageType.png);
      expect(ImageType.sniff(webp), ImageType.webp);
      expect(ImageType.sniff(Uint8List.fromList('%PDF-1.7'.codeUnits)), isNull);
      expect(
          ImageType.sniff(Uint8List.fromList(
              [...'RIFF'.codeUnits, 0, 0, 0, 0, ...'WAVE'.codeUnits])),
          isNull);
    });

    for (final (name, image, contentType, code) in [
      (
        'unknown format',
        Uint8List.fromList('%PDF-1.7'.codeUnits),
        null,
        KhmerOcrErrorCode.unsupportedMediaType
      ),
      (
        'unsupported declared type',
        jpeg,
        'application/pdf',
        KhmerOcrErrorCode.unsupportedMediaType
      ),
      ('empty image', Uint8List(0), null, KhmerOcrErrorCode.invalidImage),
      (
        'oversized image',
        Uint8List.fromList([...jpeg, ...List.filled(20, 0)]),
        null,
        KhmerOcrErrorCode.payloadTooLarge
      ),
    ]) {
      test('rejects $name before any request', () async {
        final (client, requests) =
            clientWith((_) => jsonResponse(nationalIdJson), maxUploadBytes: 16);
        final error = await thrown(
            client.extractNationalId(image, contentType: contentType));
        expect(error.code, code);
        expect(error.statusCode, isNull);
        expect(requests, isEmpty);
      });
    }
  });

  group('errors', () {
    test('maps the server error envelope', () async {
      final (client, _) = clientWith((_) => jsonResponse({
            'error': {
              'code': 'IMAGE_QUALITY_TOO_LOW',
              'message':
                  'The document image is not usable for reliable recognition.',
              'request_id': 'req_abc',
              'details': {
                'issues': ['too_blurry']
              },
            },
          }, status: 422));
      final error = await thrown(client.extractNationalId(jpeg));
      expect(error.code, KhmerOcrErrorCode.imageQualityTooLow);
      expect(error.statusCode, 422);
      expect(error.requestId, 'req_abc');
      expect(error.details, {
        'issues': ['too_blurry']
      });
      expect(error.toString(), contains('IMAGE_QUALITY_TOO_LOW (HTTP 422)'));
    });

    test('401 UNAUTHORIZED falls back to the X-Request-ID header', () async {
      final (client, _) = clientWith((_) => jsonResponse(
          {
            'error': {
              'code': 'UNAUTHORIZED',
              'message': 'A valid API key is required.'
            }
          },
          status: 401,
          headers: {'x-request-id': 'req_hdr'}));
      final error = await thrown(client.models());
      expect(error.code, KhmerOcrErrorCode.unauthorized);
      expect(error.statusCode, 401);
      expect(error.requestId, 'req_hdr');
    });

    test('an error without an envelope is UNEXPECTED_RESPONSE', () async {
      final (client, _) =
          clientWith((_) => http.Response('<html>Bad gateway</html>', 502));
      final error = await thrown(client.health());
      expect(error.code, KhmerOcrErrorCode.unexpectedResponse);
      expect(error.statusCode, 502);
    });

    test('a 2xx body that is not a JSON object is UNEXPECTED_RESPONSE',
        () async {
      for (final text in ['not json', '[1, 2]']) {
        final (client, _) = clientWith((_) => http.Response(text, 200));
        expect((await thrown(client.health())).code,
            KhmerOcrErrorCode.unexpectedResponse);
      }
    });

    test('a connection failure is NETWORK_ERROR', () async {
      final (client, _) =
          clientWith((_) => throw http.ClientException('Connection refused'));
      final error = await thrown(client.health(requestId: 'req-1'));
      expect(error.code, KhmerOcrErrorCode.networkError);
      expect(error.statusCode, isNull);
      expect(error.requestId, 'req-1');
    });

    test('a slow server is TIMEOUT', () async {
      final (client, _) = clientWith((_) async {
        await Future<void>.delayed(const Duration(milliseconds: 300));
        return jsonResponse({});
      }, timeout: const Duration(milliseconds: 50));
      expect((await thrown(client.health())).code, KhmerOcrErrorCode.timeout);
    });
  });

  group('models', () {
    test('parses a national ID result', () async {
      final (client, _) = clientWith((_) => jsonResponse(nationalIdJson));
      final result = await client.extractNationalId(jpeg);
      expect(result.schemaVersion, '1.0');
      expect(result.documentType, 'khmer_national_id');
      expect(result.status, ResultStatus.reviewRequired);
      expect(result.reviewRequired, isTrue);
      expect(result.fields.keys, NationalIdField.all);
      expect(result.field(NationalIdField.fullNameKm)!.value, 'គំរូ សាកល្បង');
      expect(result.field(NationalIdField.gender)!.confidence, 1.0);
      expect(result.field(NationalIdField.dateOfBirth)!.value, isNull);
      expect(result.field('missing'), isNull);
      expect(result.metadata!['card_detected'], isTrue);
      final region = result.textRegions!.single;
      expect(region.label, 'full_name_km');
      expect(region.bbox.width, 300);
      expect(region.text, startsWith('គោត្តនាម'));
    });

    test('splits warnings and lists fields needing review', () {
      final result = DocumentResult.fromJson(nationalIdJson);
      expect(result.parsedWarnings.map((w) => (w.code, w.field)), [
        ('layout_template_unverified', null),
        ('low_confidence', 'full_name_km'),
        ('field_not_detected', 'date_of_birth'),
      ]);
      expect(result.fieldsNeedingReview,
          unorderedEquals(['full_name_km', 'date_of_birth']));
    });

    test('tolerates unknown statuses and missing text regions', () {
      final result = DocumentResult.fromJson({
        ...nationalIdJson,
        'status': 'something_new'
      }..remove('text_regions'));
      expect(result.status, ResultStatus.unknown);
      expect(result.textRegions, isNull);
    });

    test('parses quality, OCR, health and models responses', () async {
      final quality = QualityReport.fromJson({
        'schema_version': '1.0',
        'usable': false,
        'width': 1012,
        'height': 638,
        'blur_score': 12.5,
        'glare_ratio': 0.01,
        'brightness': 180,
        'contrast': 52.3,
        'issues': ['too_blurry'],
        'blocking_issues': ['too_blurry'],
      });
      expect(quality.usable, isFalse);
      expect(quality.brightness, 180.0);
      expect(quality.blockingIssues, ['too_blurry']);

      final ocr = OcrResult.fromJson({
        'schema_version': '1.0',
        'document_type': 'general_text',
        'status': 'success',
        'text': 'ព្រះរាជាណាចក្រកម្ពុជា',
        'regions': [
          {
            'text': 'ព្រះរាជាណាចក្រកម្ពុជា',
            'confidence': 0.9,
            'bbox': {'x': 0, 'y': 0, 'width': 10, 'height': 5},
            'label': null
          }
        ],
        'warnings': [],
        'review_required': false,
      });
      expect(ocr.status, ResultStatus.success);
      expect(ocr.regions.single.label, isNull);

      final health = HealthStatus.fromJson({
        'status': 'ok',
        'model_loaded': false,
        'api_version': '0.1.0',
        'sdk_version': '0.1.0',
        'model_error': 'MODEL_NOT_AVAILABLE'
      });
      expect(health.modelLoaded, isFalse);
      expect(health.modelError, 'MODEL_NOT_AVAILABLE');

      final models = ModelsInfo.fromJson({
        'active': {
          'sdk_version': '0.1.0',
          'model': {
            'model_id': 'khmerocr-id-own',
            'model_version': '0.1.0',
            'license_approved': false,
            'recognizer_backend': 'router'
          },
          'recognizer': {'backend': 'router', 'default': 'tesseract'},
          'templates': {
            'khmer_national_id': {
              'template_id': 'khmer_national_id_front_v1',
              'template_version': '1',
              'verified': false
            }
          },
        },
        'installed': [
          {'model_id': 'khmerocr-id-own', 'model_version': '0.1.0'}
        ],
      });
      expect(models.active!.model!.modelId, 'khmerocr-id-own');
      expect(models.active!.model!.licenseApproved, isFalse);
      expect(models.active!.templates.keys, ['khmer_national_id']);
      expect(models.installed.single.modelVersion, '0.1.0');
      expect(ModelsInfo.fromJson({'active': null, 'installed': []}).active,
          isNull);
    });
  });

  test('close() leaves a caller-supplied HTTP client open', () async {
    var closed = false;
    final mock = _ClosingClient(() => closed = true);
    KhmerOcrClient(baseUrl: Uri.parse('http://ocr.test'), httpClient: mock)
        .close();
    expect(closed, isFalse);
  });
}

class _ClosingClient extends http.BaseClient {
  _ClosingClient(this.onClose);
  final void Function() onClose;

  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) =>
      throw UnimplementedError();

  @override
  void close() => onClose();
}
