"""EEGNet model definition.

EEGNet is a compact CNN designed for EEG decoding. It works well with the
~320-trial Task 2 training set, especially with strong dropout for
cross-subject generalisation.

Reference: Lawhern et al., 2018 (https://arxiv.org/abs/1611.08024)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
from torch import nn


@dataclass
class ModelConfig:
    n_channels: int
    n_samples: int
    n_classes: int = 4
    f1: int = 8                 # number of temporal filters
    d: int = 2                  # depth multiplier
    f2: int = 16                # number of pointwise filters (= f1 * d)
    kernel_length: int = 64     # ~ half a second at 250 Hz
    pool1: int = 4
    pool2: int = 8
    dropout: float = 0.5        # heavy dropout for cross-subject regularisation

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        clean = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**clean)


class EEGNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Block 1: temporal conv -> depthwise spatial conv
        self.block1 = nn.Sequential(
            nn.Conv2d(1, cfg.f1, kernel_size=(1, cfg.kernel_length),
                      padding=(0, cfg.kernel_length // 2), bias=False),
            nn.BatchNorm2d(cfg.f1),
            nn.Conv2d(cfg.f1, cfg.f1 * cfg.d, kernel_size=(cfg.n_channels, 1),
                      groups=cfg.f1, bias=False),
            nn.BatchNorm2d(cfg.f1 * cfg.d),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, cfg.pool1)),
            nn.Dropout(cfg.dropout),
        )

        # Block 2: separable conv (depthwise + pointwise)
        self.block2 = nn.Sequential(
            nn.Conv2d(cfg.f1 * cfg.d, cfg.f1 * cfg.d, kernel_size=(1, 16),
                      padding=(0, 8), groups=cfg.f1 * cfg.d, bias=False),
            nn.Conv2d(cfg.f1 * cfg.d, cfg.f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(cfg.f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, cfg.pool2)),
            nn.Dropout(cfg.dropout),
        )

        self.classifier = nn.Linear(self._infer_feature_dim(), cfg.n_classes)

    def _infer_feature_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.cfg.n_channels, self.cfg.n_samples)
            x = self.block1(dummy)
            x = self.block2(x)
        return int(x.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


def build_model(cfg_dict: dict) -> EEGNet:
    return EEGNet(ModelConfig.from_dict(cfg_dict))
