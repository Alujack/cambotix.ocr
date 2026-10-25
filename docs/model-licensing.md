# Model selection and licensing

A repository license does not automatically cover externally hosted model
weights, tokenizers, vocabularies, or the datasets used to train them. Every
component that ships in a model bundle is audited separately.

## Candidates

| Candidate | Status | Notes |
| --- | --- | --- |
| Tesseract + `tessdata_fast` (`khm`, `eng`) | Implemented (`tesseract` backend). Development baseline only. | Engine and tessdata are published under Apache-2.0. Confirm for the exact release you bundle. |
| seanghay/KhmerOCR `rec.onnx` | **Implemented** (`onnx-ctc` backend, routed to Khmer-script regions). Pending written license confirmation. | CTC line recognizer, Khmer-only vocabulary (no Latin, ASCII digits, space, or comma). Repository is MIT (© 2026 Seanghay Yath); confirm the weights license and the training-font provenance with the author. |
| seanghay/KhmerOCR `det.onnx` | **Rejected** | Ultralytics YOLO11n. Its embedded ONNX metadata declares `AGPL-3.0`. Not needed: ID templates already locate fields. |
| Netra-OCR | To benchmark | Khmer-English recognizer designed for local inference. The repository lists MIT, but the full application identifies AGPL-licensed components, and the published V2 checkpoint is described by its authors as an unfinished training snapshot. Audit weights, vocabulary, and each dependency separately. |
| PaddleOCR | To evaluate | Toolkit code is Apache-2.0. Khmer recognition coverage and model-weight licenses must be checked for the exact release. |

New backends plug in through the `khmerocr.recognizers` entry-point group
(see `khmerocr/recognition/registry.py`), so a candidate can be benchmarked
without changing the engine. Keep AGPL-licensed code out of the core package;
if a candidate requires it, isolate it in a separately distributed adapter
and get legal review first.

## Decision process

1. Benchmark candidates on authorized Cambodian ID images and text crops
   (`benchmarks/evaluate.py`).
2. Evaluate detectors and recognizers separately.
3. Compare Khmer names, Latin names, dates, numbers, and difficult capture
   conditions, including the per-category and silent-error columns of the
   report.
4. Pin model and dependency versions.
5. Record each component's license and checksum in the bundle manifest
   (`license_components`, `files[].sha256`).
6. Reject any component whose obligations conflict with the distribution
   model unless permission is obtained.

## Release gate

`models/manifest.json` is the release registry. `scripts/check_release.py`
fails the build while any entry:

- still holds a placeholder (`selected-after-benchmark`,
  `record-during-release`, `verified-license-required`),
- has no SHA-256 bundle digest,
- has `license_approved` other than `true`, or no `license_approved_by`.

With `--bundle DIR`, it also verifies the bundle's checksums and confirms the
bundle digest matches the registry.

Approval is a human decision: set `license_approved: true` and
`license_approved_by` only after reviewing every entry in
`license_components`.

## Hybrid recognizer (current best)

Bundle `recognizer.backend = "router"`:

- **Khmer-script regions:** seanghay `rec.onnx` first. If its read fails field
  validation, Tesseract reads the region instead (`fallback_recognizer:<field>`
  warning).
- **Latin, digit, and mixed regions:** Tesseract only. The Khmer model must
  never see these: on a real card it read a Latin-digit date such as
  `DD/MM/YYYY` as a string of Khmer digits, with 0.94 confidence.

The Khmer model has no space token. Word spaces are restored from wide blank
gaps in the line image.

Synthetic set (60 cards, same images and crops, Docker, networking disabled):

| Field | Tesseract only | Hybrid |
| --- | --- | --- |
| full_name_km (exact, including spaces) | 58.3% | **93.3%** |
| Cards with all fields exact, non-glare categories (50 cards) | 33/50 | **50/50** |
| Glare category, all fields exact | 0/10 | 1/10 |

Recognizer-only comparison on identical tight Khmer-name crops: seanghay 93.3%
vs Tesseract 51.7% (ignoring spaces); 17 ms vs 60 ms per line.

### onnxruntime telemetry

onnxruntime **1.29.0 and later** (tested 1.29, 1.30, 1.31 on Linux) resolve
`mobile.events.data.microsoft.com` for 1DS telemetry, even with
`ORT_DISABLE_TELEMETRY=1` and `onnxruntime.disable_telemetry_events()`.
Versions 1.20–1.28 made no network calls in the same test. The `onnx` extra is
pinned to `onnxruntime<1.29`, and the backend refuses to load newer versions.
`scripts/check_egress.sh` verifies this at the system-call level in CI.
Python socket mocks cannot see native code, so they would not have caught it.

## Baseline result (synthetic data, Tesseract)

Measured on 2026-10-24: 60 synthetic SPECIMEN cards, `tessdata_fast` 4.1.0,
Tesseract 5.5.0 in Docker with networking disabled.

| Field | Exact match | 95% CI |
| --- | --- | --- |
| id_number | 100% | 94.0–100% |
| gender | 100% | 94.0–100% |
| date_of_birth | 91.7% | 81.9–96.4% |
| full_name_latin | 86.7% | 75.8–93.1% |
| full_name_km | 58.3% | 45.7–69.9% (mean CER 0.129) |

These are pipeline sanity numbers on rendered text with the provisional
layout. They are not evidence about real cards. The baseline fails the
release gate, and Khmer name recognition is the clearest gap for the
candidate models to close. (Khmer names were 63.3% before the Otsu retry
was added to the Tesseract backend. That retry fixed real-card failures but
sometimes prefers a more confident wrong read.)

### First real card (2026-10-24)

One real student ID card, a clean digital scan provided with the holder's
consent and kept outside Git. Six scored fields (Khmer name, Latin name,
gender, date of birth, student number, issue date): 6/6 exact after two
fixes this card exposed:

- The glare check flagged the card's printed white background as severe
  glare and rejected the image. Glare now requires a soft bright halo around
  the clipped region.
- Tesseract's single-line mode returned nothing for the widely spaced number
  under the barcode, and nothing for the date printed over a patterned
  background. The backend now retries with Otsu binarization and, for digit
  fields, single-word mode.

Remaining errors on unscored lines: one Khmer consonant substitution in the
department name and stray characters after "Bachelor". Both were flagged as
low confidence. One sample says nothing about accuracy; it only shows that
the pipeline works on a real card.
