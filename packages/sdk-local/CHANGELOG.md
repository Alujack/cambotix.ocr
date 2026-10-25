# Changelog

## 0.1.0

- First release: the KhmerOCR engine in TypeScript, running in browsers
  (`@khmerocr/local/browser`) and Node (`@khmerocr/local/node`) without an OCR
  server. Loads and checksums the Python engine's model bundles, and returns
  the REST API's result types and `KhmerOCRError` codes.
- `extractNationalId`, `extractStudentId`, `extractDocument`, `checkQuality`,
  `recognizeText`, `info`, `close`.
- `khmerocr-local-assets` copies the ONNX Runtime and tesseract.js
  WebAssembly files into a site, so nothing loads from a CDN.
