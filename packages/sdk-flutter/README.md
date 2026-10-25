# khmerocr (Dart / Flutter)

Dart client for the KhmerOCR REST API: Cambodian National ID and student ID
card extraction, image quality checks, and general Khmer/English text
recognition. It works in Flutter (Android, iOS, web, desktop) and in plain
Dart.

Recognition runs on a KhmerOCR server, not on the phone. The app uploads the
photo and gets typed results back. On-device inference is a later milestone.

Not published to pub.dev yet. Apache-2.0.

## Install

```yaml
dependencies:
  khmerocr:
    git:
      url: https://github.com/Alujack/cambotix.ocr.git
      path: packages/sdk-flutter
  # or, inside this repository:
  # khmerocr:
  #   path: ../packages/sdk-flutter
```

Start a server (see [docs/installation.md](../../docs/installation.md)):

```bash
export KHMEROCR_API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
khmerocr serve --host 0.0.0.0 --port 8000 --model khmerocr-id-v1
```

## Quick start

```dart
import 'package:image_picker/image_picker.dart';
import 'package:khmerocr/khmerocr.dart';

final client = KhmerOcrClient(
  baseUrl: Uri.parse('https://ocr.example.com'), // server root; /v1 is added
  apiKey: 'your-api-key',
);

final photo = await ImagePicker()
    .pickImage(source: ImageSource.camera, maxWidth: 2000, imageQuality: 90);
if (photo != null) {
  final result = await client.extractNationalId(await photo.readAsBytes());
  print(result.field(NationalIdField.idNumber)?.value);
  if (result.reviewRequired) {
    // Show the fields to a person and let them confirm or correct them.
  }
}
```

`maxWidth: 2000, imageQuality: 90` keeps phone photos well under the 10 MiB
upload limit while leaving enough resolution for the card text.

A complete camera app is in [example/flutter_scanner](example/flutter_scanner).

## API

| Method | Endpoint | Returns |
| --- | --- | --- |
| `extractNationalId(bytes, {includeTextRegions})` | `POST /v1/documents/national-id/extract` | `DocumentResult` |
| `extractStudentId(bytes, {includeTextRegions})` | `POST /v1/documents/student-id/extract` | `DocumentResult` |
| `checkQuality(bytes)` | `POST /v1/documents/quality-check` | `QualityReport` |
| `recognizeText(bytes)` | `POST /v1/ocr` | `OcrResult` |
| `health()` | `GET /v1/health` | `HealthStatus` |
| `models()` | `GET /v1/models` | `ModelsInfo` |

Uploads take JPEG, PNG or WebP bytes. The type is detected from the bytes;
pass `contentType` to override it. Every method accepts `requestId`, sent as
`X-Request-ID`. Field names are in `NationalIdField` and `StudentIdField`.
Each model keeps the decoded response in `json`.

## Handling review

`reviewRequired` is the signal that matters: when it is true, a person must
confirm the result before it is used. `fieldsNeedingReview` lists the fields to
highlight: those named in a warning (`low_confidence:full_name_km`), plus those
that failed validation or were not read. `parsedWarnings` splits each warning
into `code` and `field`. The warning codes are listed in
[docs/api-reference.md](../../docs/api-reference.md#warning-codes).

`confidence` is the recognizer's confidence in the text it read; it does not
say whether the text belongs to that field.

## Errors

Every failure throws `KhmerOcrException` with `code`, `message`, `statusCode`
(null when no server response arrived), `requestId` and `details`.

| Code | Meaning |
| --- | --- |
| `IMAGE_QUALITY_TOO_LOW` | Blurry, glare, too dark or too small; `details['issues']` says which. Ask for a retake. |
| `UNAUTHORIZED` | Missing or wrong API key. |
| `INVALID_IMAGE`, `UNSUPPORTED_MEDIA_TYPE`, `PAYLOAD_TOO_LARGE` | Not a usable JPEG/PNG/WebP, or larger than `maxUploadBytes`. Checked before upload too. |
| `MODEL_NOT_AVAILABLE` | The server has no model loaded. |
| `INFERENCE_FAILED`, `INTERNAL_ERROR` | Server-side failure; report the `requestId`. |
| `NETWORK_ERROR`, `TIMEOUT` | No response (client-side codes). |
| `UNEXPECTED_RESPONSE` | The response was not a KhmerOCR response, e.g. from a proxy. |

## Security

- An API key compiled into a distributed app can be extracted from it. Only
  embed one in internal or staff apps on a private network. For public apps,
  call your own backend, which authenticates the user and calls KhmerOCR; pass
  your token with the `headers` option and point `baseUrl` at your backend.
- Use HTTPS whenever the server is not on localhost. The KhmerOCR server does
  not terminate TLS itself; put it behind a reverse proxy.
- ID card images are personal data. Do not log, cache or keep them longer
  than the request needs.

Server settings: `KHMEROCR_API_KEY` requires a key on every endpoint except
`/v1/health`; `KHMEROCR_CORS_ORIGINS` (comma-separated) is only needed for
Flutter web, where the browser enforces CORS.

## Development

```bash
dart pub get
dart analyze
dart test
# against a running server:
KHMEROCR_TEST_URL=http://127.0.0.1:8000 KHMEROCR_TEST_API_KEY=... \
  KHMEROCR_TEST_IMAGE=card.png dart test test/integration_test.dart
```
