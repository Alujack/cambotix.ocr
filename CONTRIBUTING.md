# Contributing

Thanks for helping. Bug reports, fixes, new card layouts, training
improvements, and documentation are all welcome.

## The one rule: no real personal data

Never commit, attach, or paste real ID card images, real names, ID numbers,
or birth dates: not in code, tests, issues, or pull requests. Use the
synthetic SPECIMEN cards (`khmerocr.testing.render_card`) and the fake names
in `training/khmer_rec/textgen.py`. If you describe a real card layout,
describe it in words or with a redrawn mock-up.

## Repository layout

| Path | What | Toolchain |
| --- | --- | --- |
| `packages/core-python` | Engine, `cambotix-ocr` (`import khmerocr`) | Python 3.10+ |
| `packages/api` | REST API, `cambotix-ocr-api` | Python 3.10+ |
| `packages/sdk-typescript` | `@khmerocr/client` | Node 22 for development (package supports 18+) |
| `packages/sdk-flutter` | `khmerocr` Dart/Flutter client + example app | Dart 3 / Flutter stable |
| `training/` | Recognizer training | Python + PyTorch |
| `schemas/` | JSON Schemas, copied from the package by `scripts/sync_schemas.py` | |

## Development

Python (engine and API):

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy
```

Tesseract tests run when `tesseract` is on `PATH` and `KHMEROCR_TESSDATA`
points at a folder with `khm.traineddata` and `eng.traineddata`.

TypeScript client:

```bash
cd packages/sdk-typescript
npm ci && npm run typecheck && npm run build && npm test
```

Dart/Flutter client:

```bash
cd packages/sdk-flutter
dart pub get && dart format . && dart analyze && dart test
cd example/flutter_scanner && flutter pub get && flutter analyze
```

The clients' integration tests run against a live server when
`KHMEROCR_TEST_URL` is set; see each package README.

## Changes to the API contract

The JSON Schemas in `packages/core-python/src/khmerocr/jsonschema/` are the
contract shared by the API and both clients. If you change one, run
`python scripts/sync_schemas.py`, update the TypeScript types and Dart
models, and describe the change in `CHANGELOG.md`. Prefer additive changes;
removing or renaming a field needs a new schema version.

## Pull requests

- Keep each PR to one change, with tests.
- CI must pass: Python lint/types/tests, both clients, the Flutter example.
- Explain *why* in the description, and how you tested it.
- Contributions are licensed under Apache-2.0, the project license.
