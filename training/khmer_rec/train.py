"""Train the CRNN line recognizer on streaming synthetic data.

    python -m training.khmer_rec.train --run-name poc --minutes 60

Two fixed evaluation sets are rendered at start-up: one with training fonts
and one with *holdout* fonts the model never sees, which measures
generalization to unseen typefaces. The best checkpoint is chosen on the
holdout set.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import SyntheticLines, bucketed, collate, fixed_set
from .model import CRNN
from .vocab import NUM_CLASSES, greedy_decode

TRAINING_DIR = Path(__file__).resolve().parents[1]


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


@torch.no_grad()
def evaluate(model: CRNN, samples: list, device: torch.device,
             batch_size: int = 64) -> dict[str, float]:
    model.eval()
    order = sorted(range(len(samples)), key=lambda i: samples[i][0].shape[1])
    errors = chars = exact = 0
    for start in range(0, len(order), batch_size):
        batch = [samples[i] for i in order[start:start + batch_size]]
        data = collate(batch)
        logits = model(data["images"].to(device))
        best = logits.argmax(-1).cpu()
        for b, (image, label) in enumerate(batch):
            steps = image.shape[1] // 4
            prediction = greedy_decode(best[:steps, b].tolist())
            errors += edit_distance(prediction, label)
            chars += len(label)
            exact += prediction == label
    model.train()
    return {"cer": errors / max(1, chars), "line_accuracy": exact / len(samples)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--minutes", type=float, default=60)
    parser.add_argument("--steps", type=int, default=30000, help="Schedule length (max steps).")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--val-size", type=int, default=600)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    run_dir = TRAINING_DIR / "data" / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))

    torch.manual_seed(args.seed)
    model = CRNN(NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup = 500

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, args.steps - warmup))
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    step = 0
    best_cer = float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, best_cer = state["step"], state.get("best_cer", best_cer)

    print(f"device={device} params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M "
          f"classes={NUM_CLASSES}", flush=True)
    t0 = time.perf_counter()
    val_seen = fixed_set(TRAINING_DIR, size=args.val_size, seed=10_001, holdout=False)
    val_holdout = fixed_set(TRAINING_DIR, size=args.val_size, seed=20_002, holdout=True)
    print(f"validation sets rendered in {time.perf_counter() - t0:.1f}s", flush=True)

    # Workers render single lines; the main process buckets them by width.
    loader = DataLoader(SyntheticLines(TRAINING_DIR, seed=args.seed + step), batch_size=None,
                        num_workers=args.workers, persistent_workers=True, prefetch_factor=64)
    samples = ((image.numpy() if hasattr(image, "numpy") else image, text)
               for image, text in loader)
    deadline = time.monotonic() + args.minutes * 60
    log = (run_dir / "metrics.jsonl").open("a")
    window_loss, window_n, window_t = 0.0, 0, time.perf_counter()
    model.train()
    for group in bucketed(samples, args.batch_size):
        batch = collate(group)
        logits = model(batch["images"].to(device))
        # CTC runs on CPU: it is not implemented for every accelerator backend.
        log_probs = F.log_softmax(logits.float(), dim=-1).cpu()
        loss = F.ctc_loss(log_probs, batch["targets"], batch["input_lengths"],
                          batch["target_lengths"], blank=0, zero_infinity=True)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        step += 1
        window_loss += loss.item()
        window_n += 1

        if step % 100 == 0:
            elapsed = time.perf_counter() - window_t
            print(f"step {step} loss {window_loss / window_n:.4f} "
                  f"lr {scheduler.get_last_lr()[0]:.2e} "
                  f"{window_n * args.batch_size / elapsed:.0f} lines/s", flush=True)
            window_loss, window_n, window_t = 0.0, 0, time.perf_counter()

        done = step >= args.steps or time.monotonic() >= deadline
        if step % args.eval_every == 0 or done:
            seen = evaluate(model, val_seen, device)
            holdout = evaluate(model, val_holdout, device)
            record = {"step": step, "time_min": round((time.monotonic() - deadline) / 60
                                                      + args.minutes, 2),
                      "seen_fonts": seen, "holdout_fonts": holdout}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(f"eval step {step}: seen CER {seen['cer']:.4f} acc {seen['line_accuracy']:.3f} | "
                  f"holdout CER {holdout['cer']:.4f} acc {holdout['line_accuracy']:.3f}",
                  flush=True)
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "step": step,
                     "best_cer": min(best_cer, holdout["cer"]), "num_classes": NUM_CLASSES}
            torch.save(state, run_dir / "last.pt")
            if holdout["cer"] < best_cer:
                best_cer = holdout["cer"]
                torch.save(state, run_dir / "best.pt")
        if done:
            break
    print(f"finished at step {step}; best holdout CER {best_cer:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
