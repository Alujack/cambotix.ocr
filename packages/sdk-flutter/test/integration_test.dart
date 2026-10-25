/// Runs against a live KhmerOCR server. Skipped unless KHMEROCR_TEST_URL is set.
///
///     KHMEROCR_TEST_URL=http://127.0.0.1:8000 KHMEROCR_TEST_API_KEY=... \
///     KHMEROCR_TEST_IMAGE=card.png KHMEROCR_TEST_DOCUMENT=national dart test
@TestOn('vm')
library;

import 'dart:io';

import 'package:khmerocr/khmerocr.dart';
import 'package:test/test.dart';

void main() {
  final env = Platform.environment;
  final url = env['KHMEROCR_TEST_URL'];
  final skip =
      url == null ? 'set KHMEROCR_TEST_URL to run against a live server' : null;

  group('live server', skip: skip, () {
    late KhmerOcrClient client;
    setUp(() => client = KhmerOcrClient(
        baseUrl: Uri.parse(url!), apiKey: env['KHMEROCR_TEST_API_KEY']));
    tearDown(() => client.close());

    test('health reports a loaded model', () async {
      final health = await client.health();
      expect(health.status, 'ok');
      expect(health.modelLoaded, isTrue,
          reason: 'model_error: ${health.modelError}');
    });

    test('models lists the active model', () async {
      expect((await client.models()).active, isNotNull);
    });

    test('extracts a card',
        skip: env['KHMEROCR_TEST_IMAGE'] == null
            ? 'set KHMEROCR_TEST_IMAGE to a card photo'
            : null, () async {
      final image = await File(env['KHMEROCR_TEST_IMAGE']!).readAsBytes();
      final student = env['KHMEROCR_TEST_DOCUMENT'] == 'student';
      final result = student
          ? await client.extractStudentId(image, includeTextRegions: true)
          : await client.extractNationalId(image, includeTextRegions: true);
      expect(result.documentType,
          student ? 'student_id_card' : 'khmer_national_id');
      expect(result.fields.keys,
          containsAll(student ? StudentIdField.all : NationalIdField.all));
      expect(result.textRegions, isNotEmpty);
      final quality = await client.checkQuality(image);
      expect(quality.width, greaterThan(0));
      // Printed for the person running the test; never logged by the SDK.
      for (final name in result.fields.keys) {
        final field = result.fields[name]!;
        final flag =
            result.fieldsNeedingReview.contains(name) ? '  <- check' : '';
        print('$name: ${field.value} (${field.confidence})$flag');
      }
      print(
          'status=${result.status.name} review_required=${result.reviewRequired}');
    });

    test('a wrong API key is rejected',
        skip: env['KHMEROCR_TEST_API_KEY'] == null
            ? 'server has no API key configured'
            : null, () async {
      final wrong = KhmerOcrClient(
          baseUrl: Uri.parse(url!), apiKey: 'wrong-key-0123456789');
      try {
        await wrong.models();
        fail('expected UNAUTHORIZED');
      } on KhmerOcrException catch (error) {
        expect(error.code, KhmerOcrErrorCode.unauthorized);
        expect(error.statusCode, 401);
        expect(error.requestId, isNotNull);
      } finally {
        wrong.close();
      }
    });
  });
}
