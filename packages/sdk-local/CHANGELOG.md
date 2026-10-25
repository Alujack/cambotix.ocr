# Changelog

## 0.1.2

- Khmer names that no correct read can contain (Khmer digits, punctuation or
  symbols, two vowel signs in a row, a word starting with a vowel sign) are
  now `valid: false`, so a misread name is no longer offered as a value.
- Recognizer: a Khmer vowel sign after a digit or punctuation, or two vowel
  signs in a row, now gives the read confidence 0.

## 0.1.1

- National ID: field boxes measured on a real card (template 0.2.0). Before,
  every field was read from the wrong line of a real card; now the ID number,
  Latin name, date of birth and sex read correctly on a clear photo. The
  Khmer name can still be misread and is flagged for review.
- Field crops no longer stretch to a speck at their edge, which the
  recognizer read as extra letters.

## 0.1.0

- First release: the KhmerOCR engine in TypeScript, running in browsers
  (`@khmerocr/local/browser`) and Node (`@khmerocr/local/node`) without an OCR
  server. Loads and checksums the Python engine's model bundles, and returns
  the REST API's result types and `KhmerOCRError` codes.
- `extractNationalId`, `extractStudentId`, `extractDocument`, `checkQuality`,
  `recognizeText`, `info`, `close`.
- `khmerocr-local-assets` copies the ONNX Runtime and tesseract.js
  WebAssembly files into a site, so nothing loads from a CDN.
