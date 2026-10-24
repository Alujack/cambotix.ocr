# KhmerOCR

A local-first developer toolkit for Khmer and English text recognition and
Cambodian National ID field extraction. It runs fully offline: no cloud
dependency, no API fees, and no document data leaves the machine.

> **Status: alpha (0.1.0a1).** The software pipeline is built and tested. The
> recognition model has **not** been selected yet (see
> [model licensing](docs/model-licensing.md)), and the ID layout template is
> **provisional**: it has not been confirmed against authorized reference
> cards. Until both are done, every ID result is marked `review_required`.

## What it does

- **`ocr.extract_id(image)`**: finds the card in a photo, corrects rotation
  and perspective, reads each field from a versioned layout template,
  normalizes and validates the values (ISO dates, Khmer digits, names,
  gender), and returns typed results with confidence, warnings, and a
  human-review flag.
- **Other card types:** `ocr.extract_id(image, document_type="student_id_card")`
  uses the same pipeline with a different layout template. Templates are JSON
  files, so adding a card design does not require code changes.
- **`ocr.recognize(image)`**: general Khmer/English text recognition.
- **`ocr.check_image(image)`**: blur, glare, exposure, size, and orientation
  checks.
- **`ocr.extract_batch(images)`**: batch extraction; one bad image does not
  abort the batch.
- **Local REST API** (`khmerocr serve`) implementing a versioned `/v1`
  contract with JSON Schemas, so a future hosted API can serve the same
  clients.

```python
from pathlib import Path
from khmerocr import KhmerOCR

ocr = KhmerOCR.from_pretrained(model_dir="./models/khmerocr-v1")
result = ocr.extract_id(Path("national-id.jpg"))

print(result.fields["id_number"].value)
print(result.review_required)
```

The best current configuration is a **hybrid bundle**. Khmer text is read
by the [seanghay/KhmerOCR](https://github.com/seanghay/KhmerOCR) recognizer
(93% exact on synthetic Khmer names, against 58% for Tesseract); Latin text
and digits are read by Tesseract. See [model licensing](docs/model-licensing.md).

Field mapping is deterministic. No language model guesses or completes
identity fields.

## Non-goals

No facial recognition or biometric matching. No claims that a document is
authentic. No foundation model trained from scratch. No other countries' IDs
until the Cambodian workflow is reliable. No cloud dependency in the local
SDK.

## Repository layout

```
packages/core-python/   khmerocr: engine (no FastAPI/DB/cloud dependencies)
  src/khmerocr/
    engine.py           public KhmerOCR class
    pipeline.py         load -> quality -> extract orchestration
    preprocessing/      safe decoding, quality checks, card detection/rectification
    detection/          model-free text-line detection
    recognition/        Recognizer interface, backend registry, Tesseract backend
    documents/          National ID extractor + versioned layout templates
    validation/         Khmer text, dates, field normalizers
    models.py           bundle manifests, checksums, offline install
    jsonschema/         bundled JSON Schemas (source of truth)
    adapters/cli.py     `khmerocr` command
    testing/            scripted recognizer + synthetic SPECIMEN cards
packages/api/           khmerocr-api: thin FastAPI layer (/v1)
packages/sdk-typescript, packages/sdk-flutter   later milestones (not published)
schemas/                JSON Schemas for non-Python consumers
models/                 release registry (manifest.json); weights are not stored here
benchmarks/             field-level evaluation + synthetic data generator
docs/                   installation, API reference, privacy, model licensing
tests/                  cross-package security and integration tests
docker/                 local API image
scripts/                release gate, bundle builder, schema sync
```

## Development

```bash
uv sync                      # or: pip install -e "packages/core-python[tesseract,validation]" -e packages/api pytest hypothesis
uv run pytest                # unit, API, security, integration tests
uv run ruff check .
uv run mypy
python scripts/check_release.py   # release gate (fails until a model is approved)
```

The Tesseract backend tests run only when `tesseract` is installed and
`KHMEROCR_TESSDATA` points at a directory containing `khm.traineddata` and
`eng.traineddata`.

Benchmark a model:

```bash
python benchmarks/generate_synthetic.py --out benchmarks/data/synthetic --count 60
python benchmarks/evaluate.py --dataset benchmarks/data/synthetic --model khmerocr-id-dev --gate
```

The release gate requires at least 98% exact match on ID number and date of
birth, on a representative held-out set of authorized real images, with zero
wrong values left unflagged for review. Synthetic results do not count toward
it.

## Documentation

- [Installation](docs/installation.md)
- [REST API reference](docs/api-reference.md)
- [Security and privacy](docs/privacy.md)
- [Model selection and licensing](docs/model-licensing.md)

## License

Apache-2.0 (see [LICENSE](LICENSE)). Third-party components are listed in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). Model bundles carry their
own license records.
