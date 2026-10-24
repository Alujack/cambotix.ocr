
# KhmerOCR SDK: Technical Architecture & MVP Plan

Bro, here's the architecture I'd actually build: a local-first, reusable Khmer OCR engine with a Cambodian National ID extraction module, distributed as an open-source SDK, with a paid hosted API added later.

One key research finding changes the implementation plan: Netra-OCR already offers local inference, Python integration, a REST API, and structured document output. Its documentation also identifies AGPL-licensed components in the full application. We should evaluate its recognition model and licensing separately rather than blindly embedding the entire application.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1

Also, Cambodia's public digital services already describe ID OCR with Khmer-English extraction and operator confirmation. Our goal is to create a reusable developer product, not claim that Cambodian ID OCR itself is new.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://digitalsp.gov.kh\&sz=32)

Social Protection

+1

## 1. Product definition

# KhmerOCR

Proposed product name

A developer toolkit for Khmer and English text recognition, document capture, and structured field extraction.

Local SDK

Offline inference, no API fee

ID extraction

National ID fields and validation

Reusable interfaces

Python first, Flutter next

Hosted API

Future paid service

### Non-goals for the first release

- No facial recognition or biometric matching.
- No claim that a document is authentic based on OCR.
- No custom foundation model trained from scratch.
- No multi-country ID support until the Cambodian ID workflow works reliably.
- No cloud dependency for the local SDK.

## 2. Architecture

The central design rule: the OCR engine must not depend on FastAPI, Flutter, a database, or a cloud service. Those are integration layers around the engine.

Consumer application

Banking, HR, fintech, shop management, mobile apps

Public SDK interface

Python API · REST contract · Flutter client

Image pipeline

Crop, rotate, rectify, quality checks

ID layout

Locate card and field regions

OCR recognizer

Khmer + English text recognition

Field extraction

Schema, validation, review flags

Standard result schema

Fields, bounding boxes, confidence, warnings, review status

### Component responsibilities

| Component         | Responsibility                                                      |
| ----------------- | ------------------------------------------------------------------- |
| `capture`       | Camera capture and image selection; mobile-specific                 |
| `preprocessing` | Orientation, perspective correction, blur/glare checks              |
| `detection`     | Locate card boundaries and text/field regions                       |
| `recognition`   | Convert text-line images into Khmer-English text                    |
| `id_card`       | Map recognized regions to ID fields                                 |
| `validation`    | Date parsing, field constraints, checksums where applicable         |
| `schemas`       | Stable public input and output types                                |
| `adapters`      | REST, CLI, mobile clients                                           |
| `telemetry`     | Optional operational metrics, never raw document content by default |

Keep the camera capture feature separate from OCR so developers can use their existing camera or scanner.

## 3. Repository structure

Use a monorepo initially. It keeps the Python engine, API, SDK contracts, tests, and documentation coordinated without creating unnecessary repository-management overhead.

```
khmerocr/
├── packages/
│   ├── core-python/
│   │   ├── src/khmerocr/
│   │   │   ├── engine.py
│   │   │   ├── pipeline.py
│   │   │   ├── config.py
│   │   │   ├── schemas.py
│   │   │   ├── errors.py
│   │   │   ├── preprocessing/
│   │   │   ├── detection/
│   │   │   ├── recognition/
│   │   │   ├── documents/
│   │   │   │   └── national_id.py
│   │   │   ├── validation/
│   │   │   └── adapters/
│   │   └── tests/
│   ├── api/
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── routes/
│   │   │   ├── middleware/
│   │   │   └── dependencies.py
│   │   └── tests/
│   ├── sdk-typescript/
│   └── sdk-flutter/
├── models/
│   ├── manifest.json
│   └── README.md
├── schemas/
│   ├── ocr-result.schema.json
│   └── national-id.schema.json
├── benchmarks/
│   ├── evaluate.py
│   └── reports/
├── examples/
│   ├── python-local/
│   └── flutter-camera/
├── docs/
│   ├── installation.md
│   ├── model-licensing.md
│   ├── privacy.md
│   └── api-reference.md
├── tests/
│   ├── fixtures/
│   ├── integration/
│   └── security/
├── docker/
├── pyproject.toml
├── LICENSE
├── THIRD_PARTY_LICENSES.md
└── README.md
```

Design rule: the `core-python` package must work without the API package installed. The local engine should not import FastAPI or require PostgreSQL.

For the first release, implement `core-python`, `api`, and the schema definitions. Keep TypeScript and Flutter SDK folders for later milestones rather than publishing empty packages.

## 4. Python SDK interface

The public API should remain stable even if we replace the underlying OCR model.

Proposed developer experience:

```
from pathlib import Pathfrom khmerocr import KhmerOCRocr = KhmerOCR.from_pretrained(    model_dir="./models/khmerocr-v1")result = ocr.extract_id(    Path("national-id.jpg"))print(result.fields["id_number"].value)print(result.review_required)
```

This is the interface we intend to implement; it is not an existing package yet.

### Suggested public methods

| Method                        | Purpose                                           |
| ----------------------------- | ------------------------------------------------- |
| `ocr.recognize(image)`      | General text recognition                          |
| `ocr.extract_id(image)`     | Cambodian National ID field extraction            |
| `ocr.check_image(image)`    | Detect blur, glare, rotation, and unusable images |
| `ocr.extract_batch(images)` | Process multiple documents                        |
| `ocr.close()`               | Release model resources                           |

Use a typed result object rather than returning an unstructured dictionary alone.

```
from dataclasses import dataclassfrom typing import Any@dataclassclass ExtractedField:    value: str | None    confidence: float | None    source: str@dataclassclass OCRResult:    document_type: str    fields: dict[str, ExtractedField]    review_required: bool    warnings: list[str]    metadata: dict[str, Any]
```

Keep confidence semantics explicit: recognition confidence is not the same as confidence that a field has been mapped correctly.

## 5. Cambodian National ID extraction

This is the first specialized module, and it deserves its own schema and evaluation suite.

## ID extraction pipeline

1. Check image quality and orientation.
2. Detect the card boundary and correct perspective.
3. Identify the card layout and relevant text regions.
4. Recognize Khmer and Latin-script text in each region.
5. Map text regions to fields using a versioned card template.
6. Normalize dates and validate field formats.
7. Return results with confidence, warnings, and a review requirement.

Do not initially use an LLM to guess missing identity fields. Deterministic field mapping is cheaper, easier to test, and safer for personal data.

Use synthetic fixtures for development. For real ID images, obtain explicit authorization, minimize access, and do not commit them to Git or public benchmarks.

### Proposed ID result schema

```
{
  "schema_version": "1.0",
  "document_type": "khmer_national_id",
  "status": "review_required",
  "fields": {
    "id_number": {
      "value": null,
      "confidence": null,
      "source": "front_region"
    },
    "full_name_km": {
      "value": null,
      "confidence": null,
      "source": "front_region"
    },
    "full_name_latin": {
      "value": null,
      "confidence": null,
      "source": "front_region"
    },
    "date_of_birth": {
      "value": null,
      "confidence": null,
      "source": "front_region"
    },
    "gender": {
      "value": null,
      "confidence": null,
      "source": "front_region"
    }
  },
  "warnings": ["field_not_detected"],
  "review_required": true
}
```

This is a proposed schema, not an official Cambodian ID specification. Confirm the fields, date conventions, card layouts, and any validation rules against authorized reference material before implementation.

### JSON Schema for the field object

Save this as `schemas/national-id.schema.json`:

```
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "KhmerOCR National ID Result",
  "type": "object",
  "required": [
    "schema_version",
    "document_type",
    "status",
    "fields",
    "warnings",
    "review_required"
  ],
  "properties": {
    "schema_version": {
      "type": "string",
      "const": "1.0"
    },
    "document_type": {
      "type": "string",
      "const": "khmer_national_id"
    },
    "status": {
      "type": "string",
      "enum": [
        "success",
        "review_required",
        "failed"
      ]
    },
    "fields": {
      "type": "object",
      "additionalProperties": {
        "$ref": "#/$defs/field"
      }
    },
    "warnings": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "review_required": {
      "type": "boolean"
    }
  },
  "$defs": {
    "field": {
      "type": "object",
      "required": [
        "value",
        "confidence",
        "source"
      ],
      "properties": {
        "value": {
          "type": ["string", "null"]
        },
        "confidence": {
          "type": ["number", "null"],
          "minimum": 0,
          "maximum": 1
        },
        "source": {
          "type": "string"
        }
      },
      "additionalProperties": false
    }
  },
  "additionalProperties": false
}
```

This validates the outer response and field structure. Add stricter rules for actual field formats after verifying the official ID layout.

## 6. REST API specification

Even though the first version is fully local, design the API contract now. Later, the hosted API can implement the same contract without forcing customer applications to change.

Base URL for the local service:

`http://127.0.0.1:8000/v1`

### Endpoints

| Method   | Endpoint                           | Purpose                                 |
| -------- | ---------------------------------- | --------------------------------------- |
| `GET`  | `/health`                        | Check that the local service is running |
| `GET`  | `/models`                        | List installed model versions           |
| `POST` | `/ocr`                           | Recognize general document text         |
| `POST` | `/documents/national-id/extract` | Extract National ID fields              |
| `POST` | `/documents/quality-check`       | Check image usability                   |
| `GET`  | `/openapi.json`                  | Retrieve the API specification          |

### Example request

```
curl -X POST \
  http://127.0.0.1:8000/v1/documents/national-id/extract \
  -F "file=@national-id.jpg" \
  -F "include_text_regions=true"
```

### Example FastAPI implementation skeleton

```
from fastapi import FastAPI, File, UploadFile, HTTPExceptionfrom khmerocr import KhmerOCRapp = FastAPI(title="KhmerOCR API", version="1.0.0")ocr = KhmerOCR.from_pretrained(    model_dir="./models/khmerocr-v1")@app.get("/v1/health")def health():    return {"status": "ok"}@app.post("/v1/documents/national-id/extract")async def extract_id(file: UploadFile = File(...)):    allowed = {"image/jpeg", "image/png", "image/webp"}    if file.content_type not in allowed:        raise HTTPException(            status_code=415,            detail="Unsupported image type"        )    image_bytes = await file.read()    if not image_bytes or len(image_bytes) > 10 * 1024 * 1024:        raise HTTPException(            status_code=413,            detail="Invalid image size"        )    # The production implementation must also validate    # decoded image dimensions and image content.    result = ocr.extract_id(image_bytes)    return result.to_dict()
```

This is a skeleton. Before production, enforce streaming upload limits, verify the actual file format rather than trusting `Content-Type`, handle decoding errors, and return consistent error responses.

### Standard error format

```
{
  "error": {
    "code": "IMAGE_QUALITY_TOO_LOW",
    "message": "The document image is too blurry to process reliably.",
    "request_id": "req_example"
  }
}
```

Use stable error codes such as `INVALID_IMAGE`, `UNSUPPORTED_DOCUMENT`, `MODEL_NOT_AVAILABLE`, `IMAGE_QUALITY_TOO_LOW`, and `INFERENCE_FAILED`.

Offline requirement: model weights must be installed before inference. The application must never silently download a model, call an external AI service, or send a document to the cloud.

## 7. Model selection and licensing audit

This is one of the most important technical decisions.

## Candidate A: Netra-OCR

First recognition benchmark

Netra-OCR documents a 17-million-parameter Khmer-English text recognizer, trained on synthetic text-line images and evaluated on Khmer OCR benchmarks and a smaller collection of real-world legal documents.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

Why test it: it is specifically designed for Khmer recognition and can run locally.

What to verify: the exact license for the selected weights, tokenizer, vocabulary, training datasets, detection models, and each optional dependency. The repository lists an MIT license, but its full application identifies AGPL components. These are separate licensing questions.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1

## Candidate B: PaddleOCR

PaddleOCR is a broader OCR and document-processing toolkit, with its project code released under Apache-2.0 according to its documentation. Its available model language coverage must still be checked for the specific Khmer text we need to recognize.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://paddleocr.dev\&sz=32)

paddleocr.dev

Why test it: useful for evaluating alternative detection and document-processing components.

What to verify: Khmer support, recognition accuracy, model-weight licenses, and dependency licenses for the exact release we package.

### Decision process

1. Benchmark Netra-OCR on real, authorized Cambodian ID images and text crops.
2. Evaluate alternative detectors and recognizers separately.
3. Check recognition quality for Khmer names, Latin names, dates, numbers, and difficult image conditions.
4. Pin model and dependency versions.
5. Record each component's license and model checksum in a machine-readable manifest.
6. Reject any dependency whose licensing obligations are incompatible with the intended distribution model unless permission is obtained.

Do not assume that a repository license automatically covers externally hosted model weights or all datasets used to train them.

### Model manifest

```
{
  "model_id": "khmerocr-id-v1",
  "task": "khmer_id_extraction",
  "runtime": "local",
  "recognition_model": "selected-after-benchmark",
  "weights_sha256": "record-during-release",
  "license": "verified-license-required",
  "model_version": "1.0.0",
  "schema_version": "1.0"
}
```

The placeholder values must be replaced with verified information before release. A release should fail its build if the model license has not been approved.

## 8. Security and privacy

National ID cards contain highly sensitive personal information. Local inference is a strong starting point, but it does not automatically make an application secure.

Process locally by default

No external network calls during inference. Cache model weights locally and provide an explicit offline installation path.

Minimize data retention

Process images in memory where practical. Don't store source images or extracted identity fields unless the integrating application explicitly requires it.

Protect logs and caches

Never log raw ID numbers, names, dates of birth, image contents, or full OCR results by default. Protect temporary files and clear them on completion.

Require human review

Return `review_required=true` whenever a critical field is missing, ambiguous, or below a calibrated confidence threshold.

Separate local and cloud deployments

The future paid API should use authentication, rate limits, encryption in transit, strict tenant isolation, retention controls, and a documented deletion policy.

For the hosted service, obtain legal review of Cambodian privacy, cybersecurity, and identity-document obligations before accepting production ID data. The local SDK should not collect telemetry by default.

## 9. Testing and benchmarking strategy

Do not release this SDK based on a few successful screenshots. A system that reads a clean sample but fails on glare, small print, and rotated cards is not ready for integration.

### Build a controlled test dataset

Use separate datasets for training, validation, and final testing. If you fine-tune the model, split by document/person rather than by individual crops so nearly identical images cannot leak into the test set.

| Test category                           | What to measure                      |
| --------------------------------------- | ------------------------------------ |
| Clean ID images                         | Field-level exact match              |
| Rotated and perspective-distorted cards | Extraction success rate              |
| Glare, shadows, blur                    | Quality detection and recovery       |
| Khmer names                             | Character error rate and exact match |
| Latin names and ID numbers              | Exact field match                    |
| Dates                                   | Exact match after normalization      |
| Missing or obscured fields              | Correctly flag uncertainty           |
| Repeated processing                     | Stability and memory use             |
| Offline execution                       | Zero unexpected network requests     |

### Initial release gates

These are proposed engineering targets, not claims about current model performance.

Critical field exact-match accuracy

ID number and date fields, on a representative held-out set

# ≥ 98%

Review behavior

Ambiguous fields must not be silently treated as verified

## Required

Offline operation

Inference succeeds without internet access after installation

## Required

Privacy tests

No document contents in logs, no unexpected outbound traffic

## Required

A 98% result on a small or unrepresentative dataset is not enough. Report sample counts, confidence intervals, per-field metrics, and failure cases. Critical fields need their own evaluation; high average OCR accuracy can hide unacceptable errors in ID numbers.

### Automated test tools

- `pytest` — unit and integration tests.
- `jsonschema` — validate output contracts.
- `hypothesis` — test malformed input and edge cases.
- `ruff` and `mypy` — Python quality checks.
- `pytest-benchmark` — performance regression tests.
- CI — test installation, offline mode, model loading, schema compatibility, and license manifests.

Use synthetic IDs with clearly fake values for public fixtures. Keep genuine ID images in a controlled, access-restricted evaluation environment.

## 10. Deployment and distribution

### Local installation

For the first version, distribute the core Python package through a package registry and offer a Docker image for users who want a local HTTP API.

```
python -m venv .venv
source .venv/bin/activate

pip install khmerocr
khmerocr models install khmerocr-id-v1
khmerocr serve --host 127.0.0.1 --port 8000
```

These commands define the intended user experience; the CLI and installer need to be implemented. The installation process should let users obtain model weights in advance, verify their checksums, and run without internet access afterward.

### Distribution channels

| Deliverable                | Target audience            |
| -------------------------- | -------------------------- |
| Python wheel               | Backend developers         |
| Local Docker image         | Self-hosting teams         |
| REST/OpenAPI specification | Any language or framework  |
| Flutter package            | Mobile developers          |
| TypeScript SDK             | Web and Node.js developers |
| Model bundle               | Offline deployments        |

For Flutter, I would initially offer a Dart client for the local REST service. A genuinely embedded on-device OCR engine requires a separate compatible runtime, model conversion, mobile performance testing, and possibly a native implementation. Don't promise that feature until it works.

### Future paid hosted API

The open-source core should remain useful on its own. The paid service can add managed hosting, usage quotas, team management, monitoring, and support.

Keep billing and authentication out of the OCR engine itself. This makes the core easier to reuse and allows both deployment modes to share the same result schemas.

## 11. Practical MVP roadmap

Assuming one developer working around 2–3 hours per day:

### Week 1

Research

Benchmark existing models

Install candidate OCR engines, inspect their licenses, create a small authorized test set, and measure field-level accuracy. Deliverable: model-selection report.

### Week 2

Core

Build the Python package

Implement image preprocessing, recognition adapters, typed results, error handling, and the first ID field mapper. Deliverable: working local Python example.

### Week 3

API

Add local REST integration

Implement FastAPI endpoints, input validation, JSON Schema validation, and an OpenAPI contract. Deliverable: working local API.

### Week 4

Quality

Improve reliability

Add quality checks, confidence calibration, human review flags, and regression tests. Deliverable: benchmark report and documented limitations.

### Weeks 5–6

Release

Package and publish

Build the Docker image, installation documentation, offline model bundle, license manifest, and sample integration. Deliverable: a versioned alpha release for external developers to test.

The six-week timeline is a realistic planning estimate for a limited prototype, not a guarantee of production-grade ID accuracy. If you need to collect and annotate data or fine-tune the recognition model, allow additional time.

## 12. Final decisions I would lock in

- Language: Python for the core engine.
- Architecture: modular engine, independently versioned schemas, thin API layer.
- Initial model: benchmark Netra-OCR first; retain alternatives until measured.
- First document: Cambodian National ID, with a single verified layout to start.
- Deployment: fully local and offline.
- Distribution: open-source core plus a future paid hosted API.
- Mobile: Flutter client first, native embedded inference later.
- Database: none required for the local OCR core.
- Privacy: no default storage, telemetry, or external inference.
- Release criterion: measured field-level performance and explicit human review for uncertain results.

The first task is not writing the API. It's proving that the selected OCR model can accurately read authorized Cambodian ID samples. If that fails, the architecture won't save the product. If it succeeds, the rest is a manageable software engineering project.
