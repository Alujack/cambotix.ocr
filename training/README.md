# Training the KhmerOCR line recognizer

This directory trains our own text-line recognizer, which ships as
`rec.onnx` in a model bundle and runs in the SDK's `onnx-ctc` backend. It is
not part of the installed SDK.

## Why our own model

- **One model for the whole card.** The vocabulary covers Khmer, Latin
  letters, ASCII digits, space, and common punctuation, so bilingual lines
  (`ប្រុស / Male`) and numbers do not need a second engine.
- **Clean licensing.** Training uses only OFL fonts pinned to one
  `google/fonts` commit, plus CC BY-SA Wikipedia text rendered into images.
  No third-party model weights are involved.
- **Matched to ID cards.** Synthetic lines include card labels, names,
  dates, ID numbers, and addresses on card-like backgrounds with photo
  artifacts.

## Pipeline

| Step | Code | Notes |
| --- | --- | --- |
| Fonts | `fetch_fonts.py`, `fonts.json` | 40 OFL fonts (29 Khmer, 11 Latin) at a pinned commit; checksums in `data/fonts/fonts.lock.json`. 4 fonts are **holdout**: never trained on, used to measure generalization. |
| Text | `fetch_corpus.py`, `khmer_rec/textgen.py` | Khmer Wikipedia intros plus synthetic ID content. All names and numbers are fake: composites of common names, or syllables built in the patterns Khmer names use. 15% of lines are a name on its own, as cropped from a card's name field. Spans never split a Khmer cluster. |
| Rendering | `khmer_rec/render.py` | Khmer and Latin runs drawn with matching fonts (libraqm shaping), card-like backgrounds, bounded blur, low resolution, noise, JPEG, glare. A quarter of lines get a thickened stroke, so every font is also seen heavy. Name lines mostly use the Muol-style font (`heading` in `fonts.json`) that ID cards print names in, sometimes thinned to match its Light cut. Uses the SDK's own `trim_to_ink` crop, so training matches inference. |
| Model | `khmer_rec/model.py` | CRNN: six 3×3 convolutions (with batch norm) + 2× BiLSTM(256) + CTC. 3.85 M parameters. |
| Training | `khmer_rec/train.py` | Streaming data (nothing stored on disk), width-bucketed batches, AdamW + cosine schedule. The best checkpoint is chosen on the holdout fonts and Muol-style card names together. `--init` fine-tunes an earlier run's weights with a fresh schedule. |
| Export | `khmer_rec/export.py` | ONNX opset 17, dynamic width; verified against PyTorch with ONNX Runtime. Writes `tokens.txt` and `model_card.json`. |
| Compare | `khmer_rec/compare.py` | Scores ONNX models on holdout-font lines through the SDK's own inference code. |
| Calibrate | `khmer_rec/calibrate.py` | Maps the model's raw confidence (mean character probability, which stays near 0.9 even for wrong lines) to the share of holdout lines read exactly right, so the engine's 0.85 review threshold flags most wrong reads. Writes `calibration.json`. Fitted on synthetic lines, so still optimistic on real cards. |

## Run it

```bash
uv sync --group train                       # torch, fonttools, onnxruntime<1.29
python training/fetch_fonts.py              # ~12 MB of OFL fonts
python training/fetch_corpus.py --requests 120
python -m training.khmer_rec.train --run-name poc --minutes 90 --steps 25000
# or fine-tune an earlier run:
# python -m training.khmer_rec.train --run-name names --init training/data/runs/poc/best.pt \n#     --lr 5e-4 --steps 15000
python -m training.khmer_rec.export --run-name poc
python -m training.khmer_rec.compare --ours training/data/runs/poc
python -m training.khmer_rec.calibrate --run-name poc
```

Package the result for the SDK. The `--onnx-scripts` option routes every
region to the model, with Tesseract as the fallback:

```bash
python scripts/build_bundle.py --model-id khmerocr-id-own \
  --traineddata khm.traineddata eng.traineddata \
  --source-url https://github.com/tesseract-ocr/tessdata_fast/tree/4.1.0 \
  --khmer-onnx training/data/runs/poc/rec.onnx \
  --khmer-onnx-tokens training/data/runs/poc/tokens.txt \
  --khmer-onnx-token-offset 1 --khmer-onnx-license Apache-2.0 \
  --khmer-onnx-calibration training/data/runs/poc/calibration.json \
  --khmer-onnx-source-url "internal: training run poc" \
  --onnx-scripts khmer,latin,digits,mixed \
  --out dist/models/khmerocr-id-own
```

## Hardware

An Apple M1 Pro (MPS) trains at about 200–250 lines/s, roughly 1 M lines per
hour. A full model needs several million lines, so plan on hours on a
laptop, or roughly one hour on a single CUDA GPU.

## Data rules

- Never add real ID images or real people's names to training text. The
  synthetic name lists are composites of common names.
- Fine-tuning on real, authorized ID crops must happen in the restricted
  evaluation environment, and the resulting weights carry those data
  obligations.
- `training/data/` (fonts, corpus, checkpoints) is git-ignored.
