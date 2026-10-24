# Installation

KhmerOCR runs entirely on your machine. After installation, no step needs
internet access, and the SDK never downloads models on its own.

## 1. Python package

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate

pip install "khmerocr[tesseract]"   # core engine + Tesseract backend
pip install khmerocr-api            # optional: local REST API
```

Until packages are published, install from a checkout:

```bash
pip install -e "packages/core-python[tesseract,validation]" -e packages/api
```

or, with [uv](https://docs.astral.sh/uv/), from the repository root:

```bash
uv sync
```

## 2. OCR engine binary (Tesseract backend)

The current development backend uses the Tesseract engine (4.1+). Install the
binary only; language data is supplied by the model bundle.

| Platform | Command |
| --- | --- |
| Debian/Ubuntu | `sudo apt-get install tesseract-ocr` |
| macOS | `brew install tesseract` |
| Windows | Install from the UB Mannheim build, then set `KHMEROCR_TESSERACT_CMD` |

## 3. Model bundle (offline)

Obtain a bundle (`.tar.gz`, `.zip` or directory) through your release channel,
verify where it came from, then install it:

```bash
khmerocr models install khmerocr-id-v1 --from ./khmerocr-id-v1.tar.gz
khmerocr models verify khmerocr-id-v1
khmerocr models list
```

Installation checks every file against the SHA-256 checksums in the bundle
manifest. Bundles live in `$KHMEROCR_HOME/models` (default `~/.khmerocr/models`).

To build a development bundle from Tesseract `tessdata_fast` files you have
already downloaded, see [models/README.md](../models/README.md).

## 4. Use it

```python
from pathlib import Path
from khmerocr import KhmerOCR

ocr = KhmerOCR.from_pretrained("khmerocr-id-v1")     # or a bundle directory path
result = ocr.extract_id(Path("national-id.jpg"))
print(result.fields["id_number"].value)
print(result.review_required)
ocr.close()
```

Command line:

```bash
khmerocr check national-id.jpg
khmerocr extract-id national-id.jpg --model khmerocr-id-v1
khmerocr recognize letter.png
```

Local API:

```bash
khmerocr serve --host 127.0.0.1 --port 8000 --model khmerocr-id-v1
```

## 5. Docker (local API)

```bash
docker build -f docker/Dockerfile -t khmerocr-api .
docker run --rm -p 127.0.0.1:8000:8000 \
  -v ~/.khmerocr/models:/models:ro \
  -e KHMEROCR_MODEL=khmerocr-id-v1 khmerocr-api
```

Always publish the port on `127.0.0.1`: the local API has no authentication.
To prove offline operation, the container also runs with `--network none`
(you then call it with `docker exec` or from a shared network namespace).

## Configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `KHMEROCR_HOME` | Root directory; bundles live in `$KHMEROCR_HOME/models` | `~/.khmerocr` |
| `KHMEROCR_MODEL` | Model ID or bundle path for CLI/API | `khmerocr-id-v1` |
| `KHMEROCR_TESSERACT_CMD` | Path to the tesseract binary | `tesseract` on `PATH` |
| `KHMEROCR_MAX_UPLOAD_BYTES` | API upload limit | `10485760` |
| `KHMEROCR_MAX_CONCURRENCY` | API concurrent inference calls | `1` |
| `KHMEROCR_VALIDATE_RESPONSES` | API validates responses against JSON Schema (`0` disables) | `1` |
