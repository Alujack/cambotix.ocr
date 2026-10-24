"""Export a trained checkpoint to ONNX for the SDK's ``onnx-ctc`` backend.

    python -m training.khmer_rec.export --run-name poc

Writes ``rec.onnx``, ``tokens.txt`` and ``model_card.json`` next to the
checkpoint and checks that ONNX Runtime reproduces the PyTorch outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import torch

from .model import CRNN
from .train import TRAINING_DIR
from .vocab import NUM_CLASSES, TOKEN_OFFSET, TOKENS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    args = parser.parse_args()

    run_dir = TRAINING_DIR / "data" / "runs" / args.run_name
    state = torch.load(run_dir / args.checkpoint, map_location="cpu")
    model = CRNN(NUM_CLASSES)
    model.load_state_dict(state["model"])
    model.eval()

    onnx_path = run_dir / "rec.onnx"
    example = torch.rand(1, 1, 32, 160)
    torch.onnx.export(
        model, (example,), str(onnx_path), input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch", 3: "width"}, "logits": {0: "time", 1: "batch"}},
        opset_version=17, dynamo=False,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for width in (48, 160, 517):
        sample = torch.rand(1, 1, 32, width)
        with torch.no_grad():
            expected = model(sample).numpy()
        actual = session.run(None, {"input": sample.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)

    (run_dir / "tokens.txt").write_text(TOKENS + "\n", encoding="utf-8")
    fonts = json.loads((TRAINING_DIR / "fonts.json").read_text(encoding="utf-8"))
    card = {
        "architecture": "CRNN: 6x conv3x3 (BN, ReLU) + 2x BiLSTM(256) + linear, CTC",
        "input": "[batch, 1, 32, width] float32 grayscale in [0, 1], trimmed to ink",
        "output": "logits [time, batch, classes]; class 0 = blank",
        "token_offset": TOKEN_OFFSET,
        "vocabulary_size": len(TOKENS),
        "training_step": state["step"],
        "best_holdout_cer": state.get("best_cer"),
        "fonts": {"source": fonts["source"], "commit": fonts["commit"], "license": "OFL-1.1"},
        "text": "Khmer Wikipedia sample (CC BY-SA 4.0) + synthetic ID-style strings",
        "rec_onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
    }
    (run_dir / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2))
    print(f"exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB); ONNX matches PyTorch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
