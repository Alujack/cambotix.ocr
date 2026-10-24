# Third-party licenses

Licenses as published by each project. Re-verify for the exact versions
pinned in a release; `scripts/check_release.py` covers model bundles, not
these Python dependencies.

## Runtime dependencies: `khmerocr`

| Package | License | Notes |
| --- | --- | --- |
| numpy | BSD-3-Clause | |
| Pillow | MIT-CMU (HPND) | |
| opencv-python-headless | Apache-2.0 (OpenCV 4.5+) | Binary wheels bundle third-party libraries under their own licenses (see the wheel's `LICENSE-3RD-PARTY.txt`). |
| pytesseract (extra `tesseract`) | Apache-2.0 | |
| jsonschema (extra `validation`) | MIT | |

## Runtime dependencies: `khmerocr-api`

| Package | License |
| --- | --- |
| FastAPI | MIT |
| Starlette | BSD-3-Clause |
| Uvicorn | BSD-3-Clause |
| python-multipart | Apache-2.0 |
| AnyIO | MIT |

## External programs and data (not distributed in this repository)

| Component | License | Notes |
| --- | --- | --- |
| Tesseract OCR engine | Apache-2.0 | Installed separately or in the Docker image. |
| tessdata_fast language data (`khm`, `eng`) | Apache-2.0 | Distributed only inside model bundles, recorded in the bundle's `license_components`. |

## Development-only tools

pytest (MIT), Hypothesis (MPL-2.0), Ruff (MIT), mypy (MIT), httpx (BSD-3-Clause).
