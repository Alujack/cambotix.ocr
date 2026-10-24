# Model selection and licensing

A repository license does not automatically cover externally hosted model
weights, tokenizers, vocabularies, or the datasets used to train them. Every
component that ships in a model bundle is audited separately.

## Candidates

| Candidate | Status | Notes |
| --- | --- | --- |
| Tesseract + `tessdata_fast` (`khm`, `eng`) | Implemented (`tesseract` backend). Development baseline only. | Engine and tessdata are published under Apache-2.0. Confirm for the exact release you bundle. |
| Netra-OCR | To benchmark (week 1) | Khmer-English recognizer designed for local inference. The repository lists MIT, but the full application identifies AGPL-licensed components, and the published V2 checkpoint is described by its authors as an unfinished training snapshot. Audit weights, vocabulary, and each dependency separately. |
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
