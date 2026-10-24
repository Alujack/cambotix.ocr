# Models

Model weights are **not** stored in this repository and are **never**
downloaded by the SDK. Each model is distributed as a *bundle*: a directory
(or `.tar.gz` / `.zip`) containing `manifest.json` and the files it lists.

`models/manifest.json` is the release registry. Its placeholder values
(`selected-after-benchmark`, `record-during-release`,
`verified-license-required`) must be replaced with verified information, and
`license_approved` set to `true` by a named reviewer, before a release.
`scripts/check_release.py` fails the build until that is done.

## Bundle layout

```
khmerocr-id-v1/
├── manifest.json
└── tessdata/              # backend-specific files
    ├── khm.traineddata
    └── eng.traineddata
```

`manifest.json` records a SHA-256 for every file plus a bundle digest
(`weights_sha256`). `KhmerOCR.from_pretrained()` and
`khmerocr models install` verify all of them before use.

## Installing a bundle (offline)

```bash
khmerocr models install khmerocr-id-v1 --from ./khmerocr-id-v1.tar.gz
khmerocr models verify khmerocr-id-v1
```

Bundles are installed into `$KHMEROCR_HOME/models` (default
`~/.khmerocr/models`).

## Building a development bundle (Tesseract baseline)

Tesseract is the only backend bundled so far. It is the reference baseline
for the week-1 benchmark, not a decision that it is the production model.

```bash
python scripts/build_bundle.py \
  --model-id khmerocr-id-dev \
  --traineddata /path/to/khm.traineddata /path/to/eng.traineddata \
  --source-url https://github.com/tesseract-ocr/tessdata_fast \
  --out dist/models/khmerocr-id-dev
```

## Building the hybrid bundle (Khmer ONNX + Tesseract)

```bash
python scripts/build_bundle.py --model-id khmerocr-id-hybrid \
  --traineddata /path/to/khm.traineddata /path/to/eng.traineddata \
  --source-url https://github.com/tesseract-ocr/tessdata_fast/tree/4.1.0 \
  --khmer-onnx /path/to/seanghay/KhmerOCR/khmerocr/rec.onnx \
  --khmer-onnx-source-url https://github.com/seanghay/KhmerOCR/tree/68c2c72 \
  --out dist/models/khmerocr-id-hybrid
```

Khmer-script regions go to `rec.onnx`, with Tesseract as the fallback; all
other regions go to Tesseract. Requires `khmerocr[tesseract,onnx]`.

## Adding another recognizer (e.g. Netra-OCR, PaddleOCR)

Implement `khmerocr.recognition.Recognizer` in a separate package and expose
a factory under the `khmerocr.recognizers` entry-point group. The bundle's
`recognizer.backend` key selects it. See `docs/model-licensing.md` for the
licensing audit every candidate must pass before its weights are bundled.
