"""Streaming synthetic dataset and batching for CTC training."""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .render import LineRenderer, load_fonts
from .textgen import TextSampler
from .vocab import encode, is_supported

# CNN down-samples width by 4: one CTC time step per 4 input columns.
WIDTH_STRIDE = 4
# Share of training lines that are a Khmer name on its own, as cropped from a
# card's name field, and how often those use a Muol-style font.
NAME_LINE_RATE = 0.15
NAME_HEADING_RATE = 0.6

Sample = tuple[np.ndarray, str]


def ctc_feasible(width: int, label: str) -> bool:
    """CTC needs a time step per label symbol plus a blank between repeats."""
    repeats = sum(1 for a, b in itertools.pairwise(label) if a == b)
    return width // WIDTH_STRIDE >= len(label) + repeats


class SyntheticLines(IterableDataset[Sample]):
    def __init__(self, training_dir: Path, *, seed: int, holdout: bool = False,
                 card_names: bool = False) -> None:
        """``card_names`` yields only Khmer names in Muol-style fonts."""
        self.training_dir = training_dir
        self.seed = seed
        self.holdout = holdout
        self.card_names = card_names

    def __iter__(self) -> Iterator[Sample]:
        info = get_worker_info()
        worker = info.id if info else 0
        rng = random.Random(self.seed * 1000 + worker)
        khmer, latin = load_fonts(self.training_dir, holdout=self.holdout)
        renderer = LineRenderer(khmer, latin, rng)
        texts = TextSampler(self.training_dir / "data" / "corpus" / "kmwiki.txt", rng)
        while True:
            if self.card_names or rng.random() < NAME_LINE_RATE:
                text = texts.card_name()
                heading = self.card_names or rng.random() < NAME_HEADING_RATE
            else:
                text, heading = texts.sample(), False
            if not text or not is_supported(text):
                continue
            image = renderer.render(text, heading=heading)
            if image is None or not ctc_feasible(image.shape[1], text):
                continue
            yield image, text


def bucketed(samples: Iterator[Sample], batch_size: int,
             pool_batches: int = 16) -> Iterator[list[Sample]]:
    """Group samples of similar width so batches carry little padding."""
    pool: list[Sample] = []
    for sample in samples:
        pool.append(sample)
        if len(pool) >= batch_size * pool_batches:
            pool.sort(key=lambda s: s[0].shape[1])
            batches = [pool[i:i + batch_size] for i in range(0, len(pool), batch_size)]
            pool = []
            random.shuffle(batches)
            yield from batches


def fixed_set(training_dir: Path, *, size: int, seed: int, holdout: bool,
              card_names: bool = False) -> list[Sample]:
    """A reproducible evaluation set (rendered once, kept in memory)."""
    stream = iter(SyntheticLines(training_dir, seed=seed, holdout=holdout,
                                 card_names=card_names))
    return [next(stream) for _ in range(size)]


def collate(batch: list[Sample]) -> dict[str, torch.Tensor | list[str]]:
    """Pad lines on the right with white and concatenate CTC targets."""
    width = max(image.shape[1] for image, _ in batch)
    images = torch.ones(len(batch), 1, batch[0][0].shape[0], width)
    for i, (image, _) in enumerate(batch):
        images[i, 0, :, :image.shape[1]] = torch.from_numpy(image)
    labels = [text for _, text in batch]
    targets = torch.tensor([idx for text in labels for idx in encode(text)], dtype=torch.long)
    return {
        "images": images,
        "input_lengths": torch.tensor([image.shape[1] // WIDTH_STRIDE for image, _ in batch]),
        "targets": targets,
        "target_lengths": torch.tensor([len(text) for text in labels]),
        "labels": labels,
    }
