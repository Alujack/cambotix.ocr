"""CRNN line recognizer (CNN + bidirectional LSTM + CTC).

Same shape as the seanghay/KhmerOCR recognizer (six 3x3 convolutions, height
32 -> 1, width / 4, two BiLSTM layers of 256), with batch normalization for
faster training and a larger output layer for the bilingual vocabulary.

ONNX contract (matches the SDK's ``onnx-ctc`` backend):
input ``[batch, 1, 32, width]`` float32 in [0, 1] -> logits ``[time, batch, classes]``.
"""

from __future__ import annotations

import torch
from torch import nn


def _conv(in_channels: int, out_channels: int) -> list[nn.Module]:
    return [nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]


class CRNN(nn.Module):
    def __init__(self, num_classes: int, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            *_conv(1, 32), nn.MaxPool2d(2, 2),                # 32 -> 16, W/2
            *_conv(32, 64), nn.MaxPool2d(2, 2),               # 16 -> 8,  W/4
            *_conv(64, 128), *_conv(128, 128), nn.MaxPool2d((2, 1), (2, 1)),   # 8 -> 4
            *_conv(128, 256), *_conv(256, 256), nn.MaxPool2d((4, 1), (4, 1)),  # 4 -> 1
        )
        self.rnn = nn.LSTM(256, hidden, num_layers=2, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(2 * hidden, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.cnn(images)                        # [B, 256, 1, W/4]
        sequence = features.squeeze(2).permute(2, 0, 1)    # [T, B, 256]
        sequence, _ = self.rnn(sequence)
        return self.fc(sequence)                           # [T, B, classes]
