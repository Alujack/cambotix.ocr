# REST API reference (v1)

Base URL for the local service: `http://127.0.0.1:8000/v1`

The machine-readable contract is served at `/v1/openapi.json` (interactive
docs at `/v1/docs`). Response bodies are defined by the JSON Schemas in
[`schemas/`](../schemas), and the service validates every response against
them before sending it. A future hosted API will implement the same contract.

| Method | Endpoint | Purpose | Response schema |
| --- | --- | --- | --- |
| `GET` | `/health` | Service liveness and model status | — |
| `GET` | `/models` | Active model and installed bundles | — |
| `POST` | `/ocr` | General Khmer/English text recognition | `ocr-result.schema.json` |
| `POST` | `/documents/national-id/extract` | National ID field extraction | `national-id.schema.json` |
| `POST` | `/documents/student-id/extract` | Student ID card extraction (provisional layout) | `document-result.schema.json` |
| `POST` | `/documents/quality-check` | Image usability check | `quality-report.schema.json` |
| `GET` | `/openapi.json` | OpenAPI specification | — |

## Access from apps and browsers

By default the service listens on 127.0.0.1 with no authentication, which is
only safe on a single machine. Before a phone app, another server, or a
browser can reach it:

| Variable | Effect |
| --- | --- |
| `KHMEROCR_API_KEY` | Every endpoint except `/health` requires `Authorization: Bearer <key>` (or `X-API-Key: <key>`). At least 16 characters; generate one with `python -c "import secrets; print(secrets.token_urlsafe(32))"`. The key is checked before the upload is read. |
| `KHMEROCR_CORS_ORIGINS` | Comma-separated browser origins allowed to call the API, e.g. `https://app.example.com,http://localhost:5173`. Only needed for browser apps (including Flutter web). |

Serve over HTTPS (for example behind a reverse proxy) whenever clients are
not on the same machine: requests carry ID images and the key. An API key
compiled into a mobile app or shipped in browser code can be extracted, so
public apps should call their own backend, which authenticates the user and
then calls KhmerOCR. The [TypeScript](../packages/sdk-typescript) and
[Dart/Flutter](../packages/sdk-flutter) clients wrap this API.

## Uploads

`POST` endpoints take `multipart/form-data` with a `file` part.

- Accepted types: `image/jpeg`, `image/png`, `image/webp`. The declared type
  is checked first, then the real format is identified from the file's magic
  bytes. A mismatch is rejected.
- Maximum size: 10 MiB (`KHMEROCR_MAX_UPLOAD_BYTES`). The limit is enforced
  while the body streams in, before it is buffered.
- Decoded images are limited to 40 megapixels.

The `/extract` endpoints also accept `include_text_regions`
(`true`/`false`, default `false`). When it is `true`, the response includes
the raw recognized text of every template region.

```bash
curl -X POST http://127.0.0.1:8000/v1/documents/national-id/extract \
  -F "file=@national-id.jpg" \
  -F "include_text_regions=true"
```

## National ID response

```json
{
  "schema_version": "1.0",
  "document_type": "khmer_national_id",
  "status": "review_required",
  "fields": {
    "id_number":       {"value": "000000001", "confidence": 0.96, "source": "front_region", "valid": true},
    "full_name_km":    {"value": "គំរូ សាកល្បង", "confidence": 0.92, "source": "front_region", "valid": true},
    "full_name_latin": {"value": "KUMRU SAKALBANG", "confidence": 0.92, "source": "front_region", "valid": true},
    "date_of_birth":   {"value": "1990-01-01", "confidence": 0.95, "source": "front_region", "valid": true},
    "gender":          {"value": "M", "confidence": 0.95, "source": "front_region", "valid": true}
  },
  "warnings": ["layout_template_unverified"],
  "review_required": true,
  "metadata": {"template_id": "khmer_national_id_front_v1", "card_detected": true, "...": "..."}
}
```

Field semantics:

- `value`: normalized value (ISO-8601 dates, uppercase Latin names,
  `M`/`F`), or `null` when nothing usable was read.
- `confidence`: the recognizer's confidence in the text it read. It does
  **not** measure whether the text belongs to this field.
- `valid`: whether the value passed format validation for its type.
- `status`: `success` (no review needed), `review_required`, or `failed`
  (no field could be read).

### Student ID cards

`/documents/student-id/extract` returns the same structure with
`document_type: "student_id_card"` and the fields `full_name_km`,
`full_name_latin`, `gender`, `date_of_birth`, `department`, `degree`,
`address_km`, `student_id`, and `issue_date`. The layout was measured from one
real university card design and is unverified, so results always require
review. Bilingual lines (`Khmer / English`) are split on `/`; if the separator
is not recognized, both sides are returned as not detected rather than
guessed.

### Warning codes

Warnings are either document-level codes or `<code>:<field>`.

| Code | Meaning |
| --- | --- |
| `layout_template_unverified` | The layout has not been confirmed against authorized reference cards. Always forces review. |
| `card_boundary_not_detected` | No card edge was found; the whole image was used as the card. |
| `orientation_corrected` | The card was read after rotating it. |
| `blurry`, `glare_detected` | Degraded capture; forces review. |
| `field_not_detected:<f>` | No text in the field's region. |
| `field_unreadable:<f>` | Text was found but could not be interpreted. |
| `low_confidence:<f>` | Confidence is below the review threshold or missing. |
| `invalid_format:<f>`, `invalid_script:<f>`, `implausible_date:<f>`, `ambiguous_value:<f>` | Validation failures. |
| `ocr_corrected:<f>` | Confusable characters such as `O` and `0` were corrected; forces review. |

## Errors

Every error uses one envelope:

```json
{"error": {"code": "IMAGE_QUALITY_TOO_LOW", "message": "...", "request_id": "req_...", "details": {"issues": ["too_blurry"]}}}
```

| HTTP | Code | When |
| --- | --- | --- |
| 400 | `INVALID_IMAGE` | Empty, corrupt, unsupported, or oversized image |
| 413 | `PAYLOAD_TOO_LARGE` | Upload exceeds the limit |
| 415 | `UNSUPPORTED_MEDIA_TYPE` | Declared type not accepted |
| 401 | `UNAUTHORIZED` | `KHMEROCR_API_KEY` is set and the request has no valid key |
| 422 | `INVALID_REQUEST` | Missing or invalid parameters |
| 422 | `IMAGE_QUALITY_TOO_LOW` | Image unusable (too small, too blurry, severe glare, blank, too dark or too bright) |
| 422 | `UNSUPPORTED_DOCUMENT` | Unsupported document type or template |
| 404 | `NOT_FOUND` | Unknown endpoint |
| 503 | `MODEL_NOT_AVAILABLE` | No model loaded, checksum failure, missing backend |
| 500 | `INFERENCE_FAILED` | The recognizer failed |
| 500 | `INTERNAL_ERROR` | Unexpected error |

Every response carries an `X-Request-ID` header. A client-supplied
`X-Request-ID` is echoed when it matches `[A-Za-z0-9._-]{1,64}`. Error
messages never contain document content.
