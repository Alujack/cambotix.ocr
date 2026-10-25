# @khmerocr/client

TypeScript client for the KhmerOCR REST API (`/v1`): Cambodian National ID
and student ID extraction, image quality checks, and Khmer/English OCR.

Recognition runs on the KhmerOCR server, not in the browser or the Node
process. This package uploads the photo and returns typed results. It has no
runtime dependencies and works in Node 18+ and modern browsers.

## Install

```bash
npm install @khmerocr/client
```

The package ships both ESM and CommonJS builds, with type definitions for
each.

Start a server first (see [docs/installation.md](../../docs/installation.md)):

```bash
KHMEROCR_API_KEY=$(openssl rand -hex 24) khmerocr serve --model khmerocr-id-v1
```

## Quick start (Node)

```ts
import { readFile } from "node:fs/promises";
import { KhmerOCRClient, fieldsNeedingReview } from "@khmerocr/client";

const ocr = new KhmerOCRClient({
  baseUrl: "http://127.0.0.1:8000",
  apiKey: process.env.KHMEROCR_API_KEY,
});

const result = await ocr.extractNationalId(await readFile("national-id.jpg"));

console.log(result.fields.id_number?.value);      // "000000001"
console.log(result.fields.date_of_birth?.value);  // "1990-01-01"
if (result.review_required) {
  console.log("Confirm before use:", fieldsNeedingReview(result));
}
```

## Browser

Phones open the rear camera for this input:

```html
<input id="card" type="file" accept="image/*" capture="environment" />
```

```ts
import { KhmerOCRClient, KhmerOCRError } from "@khmerocr/client";

// Calls your own backend route, which forwards to KhmerOCR (see Security).
const ocr = new KhmerOCRClient({ baseUrl: "/api/khmerocr" });

document.querySelector<HTMLInputElement>("#card")!.addEventListener("change", async (event) => {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (!file) return;
  try {
    const result = await ocr.extractStudentId(file);
    console.log(result.fields.full_name_latin?.value, result.fields.student_id?.value);
  } catch (error) {
    if (error instanceof KhmerOCRError && error.code === "IMAGE_QUALITY_TOO_LOW") {
      alert("Retake the photo: hold the card flat, in good light, without glare.");
    } else throw error;
  }
});
```

## API

| Method | Endpoint | Returns |
| --- | --- | --- |
| `health()` | `GET /health` | `HealthStatus` |
| `models()` | `GET /models` | `ModelsInfo` |
| `extractNationalId(image, opts?)` | `POST /documents/national-id/extract` | `DocumentResult<NationalIdField>` |
| `extractStudentId(image, opts?)` | `POST /documents/student-id/extract` | `DocumentResult<StudentIdField>` |
| `checkQuality(image, opts?)` | `POST /documents/quality-check` | `QualityReport` |
| `recognizeText(image, opts?)` | `POST /ocr` | `OcrResult` |

`image` is a `Blob`/`File`, an `ArrayBuffer`, or bytes (`Uint8Array`, Node
`Buffer`). JPEG, PNG and WebP are accepted; the type is detected from the
bytes unless you pass `contentType`. Empty, oversized (`maxUploadBytes`,
default 10 MiB) and unsupported images are rejected before anything is sent.

Options on every call: `signal` (an `AbortSignal`) and `requestId` (sent as
`X-Request-ID`). Extract calls also take `includeTextRegions` to return the raw
text read in each template region.

Client options: `baseUrl` (server root; `/v1` is added), `apiKey`,
`timeoutMs` (default 60 000), `maxUploadBytes`, `headers` (added to every
request), and `fetch` (a custom implementation).

Response types keep the JSON field names of the
[schemas](../../schemas) (`review_required`, `full_name_km`, ...).

## Reviewing results

`review_required: true` means a person must confirm the result before it is
used. It is always true while a card layout is unverified. `confidence`
measures how sure the recognizer is of the text it read, not whether that text
belongs to the field.

`fieldsNeedingReview(result)` lists the fields to show the reviewer first:
those named in a warning, those that failed validation, and those with no
value. `parseWarnings(result)` splits warnings such as
`low_confidence:address_km` into `{ code, field }`.

## Errors

Every failure is a `KhmerOCRError` with `code`, `message`, `status` (HTTP
status, or `null` when no response arrived), `requestId` (quote it when
reporting a problem) and `details`.

| Code | When |
| --- | --- |
| `INVALID_IMAGE` | Empty or corrupt image |
| `UNSUPPORTED_MEDIA_TYPE` | Not JPEG, PNG or WebP |
| `PAYLOAD_TOO_LARGE` | Over the upload limit |
| `IMAGE_QUALITY_TOO_LOW` | Too small, blurry, dark or glary; `details.issues` says why. Ask for a new photo. |
| `UNSUPPORTED_DOCUMENT` | Unsupported document type |
| `INVALID_REQUEST` | Invalid parameters |
| `UNAUTHORIZED` | Missing or wrong API key |
| `MODEL_NOT_AVAILABLE` | The server has no usable model |
| `INFERENCE_FAILED`, `INTERNAL_ERROR` | Server-side failure |
| `NETWORK_ERROR` | The server could not be reached |
| `TIMEOUT` | No response within `timeoutMs` |
| `ABORTED` | Your `signal` aborted the request |
| `UNEXPECTED_RESPONSE` | The response was not a KhmerOCR response (e.g. a proxy error page) |

## Security

ID card photos are personal data.

- **Never put the API key in browser code served to end users.** Anyone can
  read it from the page. Call KhmerOCR from your backend (a Next.js route
  handler, an Express route) with `apiKey`, or route browser calls through
  your own authenticated proxy and pass the user's session token with
  `headers`.
- Use HTTPS whenever the server is not on `localhost`.
- On the server, `KHMEROCR_API_KEY` turns on key checks for every endpoint
  except `/v1/health`. `KHMEROCR_CORS_ORIGINS` (comma-separated) allows
  browser apps on those origins to call the server directly.

## Development

```bash
npm install
npm run build      # dist/esm and dist/cjs
npm test           # unit tests; integration tests need a live server:
KHMEROCR_TEST_URL=http://127.0.0.1:8000 KHMEROCR_TEST_IMAGE=card.jpg npm test
```

Integration tests also read `KHMEROCR_TEST_API_KEY` and
`KHMEROCR_TEST_DOCUMENT` (`national` or `student`).
