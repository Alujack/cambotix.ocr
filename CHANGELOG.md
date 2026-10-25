# Changelog

All notable changes to the Python packages (`cambotix-ocr`,
`cambotix-ocr-api`) and model bundles. The TypeScript and Dart clients keep
their own changelogs in `packages/sdk-typescript` and `packages/sdk-flutter`.

## Unreleased

### Added

- First model bundle, `khmerocr-id-v1` 0.1.0 (experimental): a recognizer
  trained in `training/` for all card fields, confidence calibration, and
  Tesseract as the fallback. See `docs/model-card.md`.
- `KHMEROCR_API_KEY` and `KHMEROCR_CORS_ORIGINS` so apps and browsers on
  other machines can use the REST API safely; new `UNAUTHORIZED` error code.
- TypeScript (`@khmerocr/client`) and Dart/Flutter (`khmerocr`) clients,
  with a Flutter camera scanner example.
- Student ID card extraction (`document_type="student_id_card"`,
  provisional layout).
- Recognizer output checks: impossible Khmer sequences get confidence 0, and
  stray marks at line edges are removed and flagged.

### Changed

- Python distributions renamed to `cambotix-ocr` and `cambotix-ocr-api`
  (PyPI's `khmerocr` belongs to an unrelated project). The import name is
  still `khmerocr`.

### Fixed

- Windows: Tesseract could not load bundle language data, and font license
  downloads used backslash URLs.
