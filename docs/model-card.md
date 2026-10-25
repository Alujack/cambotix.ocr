# Model card: `khmerocr-id-v1` 0.1.0 (experimental)

**Status: experimental.** Every result must be reviewed by a person. The
model has been checked on synthetic images and on one real card only; it has
not passed the release gate (98% exact on ID number and date of birth on a
held-out set of authorized real cards).

## What it is

A model bundle for the `cambotix-ocr` engine. It combines:

| Component | Role | License |
| --- | --- | --- |
| `khmer/rec.onnx`: text-line recognizer trained in this repository (`training/`, run `poc`) | Reads every card field: Khmer, Latin, digits, mixed lines | Apache-2.0 |
| `khmer/tokens.txt`: vocabulary (Khmer, Latin letters, ASCII digits, space, punctuation; 185 tokens) | | Apache-2.0 |
| Confidence calibration (in `manifest.json`) | Maps raw confidence to the share of lines read exactly right | Apache-2.0 |
| `tessdata/khm.traineddata`, `eng.traineddata` from [tessdata_fast 4.1.0](https://github.com/tesseract-ocr/tessdata_fast/tree/4.1.0) | Fallback when a field fails validation | Apache-2.0 |

The recognizer is a CRNN (six 3×3 convolutions with batch norm, two BiLSTM
layers of 256, CTC), 3.85 M parameters, ONNX opset 17, input height 32.

## Training data

No real ID images and no real people's data were used.

- **Fonts:** 40 OFL-1.1 fonts (29 Khmer, 11 Latin) from
  [google/fonts](https://github.com/google/fonts) at commit
  `e7db6ca603fd4862010383ff17ffd6cc4ffac1c0`. Four fonts (Preahvihear,
  Suwannaphum Regular and Bold, Lato Bold) are held out and never trained on.
- **Text:** a 5.2 MB sample of Khmer Wikipedia article introductions
  (CC BY-SA 4.0, rendered into images only, not shipped), plus synthetic
  ID-style strings: composite fake names, dates, ID numbers, addresses, card
  labels.
- **Rendering:** card-like backgrounds, blur, low resolution, noise, JPEG
  artifacts, glare.
- **Training:** 25,000 steps of 48 lines (1.2 M lines), AdamW with a cosine
  schedule, about 33 minutes on an RTX 3070 Ti. The checkpoint from step
  23,000 was kept because it scored best on the holdout fonts.

## Results

Lines rendered with the four holdout fonts, read through the SDK's own
inference code:

| Set | Lines | Character error rate | Lines exactly right |
| --- | --- | --- | --- |
| All lines | 1,500 | 1.09% | 89.5% |
| Khmer-only lines | 681 | 1.00% | 89.3% |

Confidence calibration, checked on 3,000 lines it was not fitted on: at the
engine's default review threshold (0.85), 68.7% of wrong lines are flagged
for review (23.8% without calibration), at the cost of flagging 10.6% of
correct lines.

On one real student ID card (the card holder's own, used with consent and
not stored in the repository), 7 of 9 fields were exactly right. The two
errors were a wrong vowel sign in the Khmer address, which was not flagged,
and junk characters at the end of the degree line, which were removed and
flagged.

## Limitations

- **Synthetic-to-real gap.** Real photos have glare, wear, and print
  variations the synthetic data does not cover. Expect more errors than the
  numbers above, and wrong reads that are not flagged, especially in Khmer
  addresses.
- **Layouts.** The national ID layout is provisional and the student ID
  layout was measured from one card design; both are marked unverified, so
  every result has `review_required: true`.
- **Calibration** was fitted on synthetic lines and is optimistic on real
  cards.

## Intended use

Pre-filling forms that a person then checks, for example during customer
onboarding. Not for automated decisions about people, not for verifying that
a document is authentic, and not for identifying people.
