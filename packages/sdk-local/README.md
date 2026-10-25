# @khmerocr/local

KhmerOCR running on the device: Cambodian National ID and student ID
extraction, image quality checks, and Khmer/English OCR in the browser or in
Node, with no OCR server. In a browser the photo never leaves the phone.

It is a TypeScript port of the Python engine (`cambotix-ocr`). It loads the
same model bundle and returns the same JSON as the REST API, so results,
types and errors match [`@khmerocr/client`](../sdk-typescript): code written
for the client works with either.

## Install

```bash
npm install @khmerocr/local
```

For Node, also install ONNX Runtime and an image decoder:

```bash
npm install onnxruntime-node@1.27.0 sharp
```

ONNX Runtime 1.29 and later send telemetry to Microsoft even when told not
to, so the engine refuses them (the Python engine does the same).

## Get the model

Download `khmerocr-id-v1-0.1.0.tar.gz` from the
[releases page](https://github.com/Alujack/cambotix.ocr/releases), check it
against its `.sha256` file, and extract it. You get a folder:

```
khmerocr-id-v1/
  manifest.json
  khmer/rec.onnx, khmer/tokens.txt      15.4 MB, reads every field
  tessdata/khm.traineddata, eng.traineddata   5.5 MB, fallback reads
```

When the engine loads, it checks every file against the SHA-256 in
`manifest.json` and refuses a bundle that does not match.

## Browser

1. Serve the model folder as static files, e.g. `public/models/khmerocr-id-v1/`.
2. Copy the ONNX Runtime and Tesseract WebAssembly files into your site
   (run it again after upgrading this package, e.g. from `postinstall`):

   ```bash
   npx khmerocr-local-assets public/khmerocr
   ```

3. Load the engine once and reuse it:

```ts
import { loadKhmerOCR, KhmerOCRError, fieldsNeedingReview } from "@khmerocr/local/browser";

const ocr = await loadKhmerOCR({
  model: "/models/khmerocr-id-v1/",
  onnx: { wasmPaths: "/khmerocr/ort/" },
  tesseractWorker: {
    workerPath: "/khmerocr/tesseract/worker.min.js",
    corePath: "/khmerocr/tesseract/",
  },
});

// `file` from <input type="file" accept="image/*" capture="environment">
try {
  const result = await ocr.extractNationalId(file);
  console.log(result.fields.full_name_km?.value, result.fields.id_number?.value);
  if (result.review_required) console.log("Check:", fieldsNeedingReview(result));
} catch (error) {
  if (error instanceof KhmerOCRError && error.code === "IMAGE_QUALITY_TOO_LOW") {
    alert("Retake the photo: hold the card flat, in good light, without glare.");
  } else throw error;
}
```

Without `wasmPaths` and `tesseractWorker`, ONNX Runtime and tesseract.js load
their code from the jsDelivr CDN instead of your site.

What the browser downloads, once (it is cached afterwards):

| File | Size | When |
| --- | --- | --- |
| ONNX Runtime (WebAssembly) | 13.5 MB | First load |
| Recognizer `rec.onnx` | 15.4 MB | First load |
| Tesseract core and worker | 4 MB | First field that needs a second reading |
| Tesseract language data | 5.5 MB | First field that needs a second reading |

`loadKhmerOCR({ ..., tesseract: false })` skips Tesseract entirely. Fields the
recognizer misreads then get no second reading, so results can differ from
the Python engine.

Recognition takes 0.5 to 1 second per card in desktop Chrome (2 seconds when
Tesseract first starts) and blocks the
thread it runs on. `loadKhmerOCR` works inside a Web Worker; run it there to
keep the page responsive. Speed on phones has not been measured yet.

### Next.js

Run the engine only in the browser: import it inside a client component's
event handler or effect, not at module level, so server rendering never
loads it.

```tsx
"use client";
import type { KhmerOCRLocal } from "@khmerocr/local/browser";

let engine: Promise<KhmerOCRLocal> | undefined;
function getEngine() {
  engine ??= import("@khmerocr/local/browser").then(({ loadKhmerOCR }) =>
    loadKhmerOCR({
      model: "/models/khmerocr-id-v1/",
      onnx: { wasmPaths: "/khmerocr/ort/" },
      tesseractWorker: { workerPath: "/khmerocr/tesseract/worker.min.js", corePath: "/khmerocr/tesseract/" },
    }));
  return engine;
}

export async function scanCard(file: File) {
  return (await getEngine()).extractNationalId(file);
}
```

## Node

```ts
import { readFile } from "node:fs/promises";
import { loadKhmerOCR } from "@khmerocr/local/node";

const ocr = await loadKhmerOCR({ model: "/srv/models/khmerocr-id-v1" });
const result = await ocr.extractNationalId(await readFile("national-id.jpg"));
console.log(result.fields.date_of_birth?.value); // "1990-01-01"
await ocr.close();
```

## API

| Method | Returns |
| --- | --- |
| `extractNationalId(image, opts?)` | `DocumentResult<NationalIdField>` |
| `extractStudentId(image, opts?)` | `DocumentResult<StudentIdField>` |
| `extractDocument(image, documentType, opts?)` | `DocumentResult` |
| `checkQuality(image, opts?)` | `QualityReport` |
| `recognizeText(image, opts?)` | `OcrResult` |
| `info()` | The loaded model and templates (`ActiveModel`) |
| `close()` | Releases ONNX Runtime and Tesseract |

`image` is a JPEG, PNG or WebP file (`Blob`/`File`, `ArrayBuffer`, `Uint8Array`,
Node `Buffer`) or raw pixels such as canvas `ImageData`
(`{ data, width, height }`). The format is detected from the bytes and the
pixel count is checked from the file header before decoding.

Options on every call: `signal` (an `AbortSignal`). Extract calls also take
`includeTextRegions`. Calls on one engine run one at a time.

Load options: `model` (required), `verify` (check checksums, default true),
`tesseract` (default true), and `config`, which overrides the engine's
thresholds (`reviewConfidenceThreshold`, `quality.blurWarn`, ...). The
defaults are the Python engine's.

Results, `review_required` and warnings work exactly as described in the
[`@khmerocr/client` README](../sdk-typescript#reviewing-results);
`fieldsNeedingReview` and `parseWarnings` are re-exported here.

## Errors

Every failure is a `KhmerOCRError` (the same class `@khmerocr/client` exports)
with a `code`, a `message` that never contains document content, and
`details`.

| Code | When |
| --- | --- |
| `INVALID_IMAGE` | Empty, corrupt, or over the pixel limit |
| `UNSUPPORTED_MEDIA_TYPE` | Not JPEG, PNG or WebP |
| `PAYLOAD_TOO_LARGE` | Over `config.maxImageBytes` (default 10 MiB) |
| `IMAGE_QUALITY_TOO_LOW` | Too small, blurry, dark or glary; `details.issues` says why |
| `UNSUPPORTED_DOCUMENT` | Unknown document type |
| `MODEL_NOT_AVAILABLE` | Bundle missing or failing its checksums, runtime missing, or engine closed |
| `INFERENCE_FAILED` | The model or Tesseract failed |
| `ABORTED` | Your `signal` aborted the call |

## How close it is to the Python engine

On 120 synthetic cards (clean, rotated, perspective, blurred, glare,
upside-down) with the `khmerocr-id-v1` bundle, the two engines returned the
same value and validity for 600 of 600 fields, the same status and review
flag for 120 of 120 cards, and identical warnings for 117. The three
differences are `low_confidence` flags on reads whose confidence is right at
the 0.85 threshold.

The image steps follow OpenCV's algorithms and are tested against OpenCV's
output (`test/opencv.test.ts`). One-grey-level differences remain in
resampling; OpenCV itself differs that much between x86 and Apple Silicon
builds. tesseract.js is a different Tesseract build from the `tesseract`
binary, so fallback reads can differ in confidence.

This measures the port, not the model. The model is experimental and the
card layouts are unverified: every ID result has `review_required: true`, and
a person must check it against the card (see the
[model card](../../docs/model-card.md)).

## Development

```bash
npm install
npm run build
npm test
```

`npm test` compares the image code with recorded OpenCV output and checks
that the card templates match the Python engine's. To compare whole results
with the Python engine using the real model:

```bash
.venv/bin/python packages/sdk-local/test/parity/make_parity_set.py \
  --model path/to/khmerocr-id-v1 --out /tmp/parity
KHMEROCR_PARITY_DIR=/tmp/parity KHMEROCR_TEST_BUNDLE=path/to/khmerocr-id-v1 npm test
```

After changing a template in `packages/core-python`, run
`node scripts/sync-templates.mjs`.
