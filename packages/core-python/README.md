# cambotix-ocr

Local-first Khmer/English OCR engine with Cambodian ID card field
extraction (`import khmerocr`). Runs fully offline; does not depend on
FastAPI, a database, or any cloud service. **Experimental:** every result
must be reviewed by a person before use.

```bash
pip install "cambotix-ocr[tesseract,onnx]"
```

```python
from khmerocr import KhmerOCR

ocr = KhmerOCR.from_pretrained("khmerocr-id-v1")  # an installed model bundle
result = ocr.extract_id("national-id.jpg")
print(result.fields["id_number"].value, result.review_required)
```

Models are never downloaded automatically; install a bundle with
`khmerocr models install`. See the
[repository](https://github.com/Alujack/cambotix.ocr) for installation,
model bundles, the REST API (`cambotix-ocr-api`), and the TypeScript and
Flutter clients.
