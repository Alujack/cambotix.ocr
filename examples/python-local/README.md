# Local Python example

```bash
pip install -e "packages/core-python[tesseract]"
python examples/python-local/extract_id.py --demo
python examples/python-local/extract_id.py national-id.jpg --model khmerocr-id-v1
```

`--demo` uses a scripted recognizer and a synthetic SPECIMEN card, so it runs
without a model or the Tesseract binary.
