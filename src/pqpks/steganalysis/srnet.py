"""SRNet (Boroumand, Chen, Fridrich, IEEE TIFS 2019), reimplemented in PyTorch.

Architecture as in Fig. 1 and Table I of the original paper:
  L1      Type 1, 64 channels
  L2      Type 1, 16 channels
  L3-L7   Type 2 (residual, no pooling), 16 channels
  L8-L11  Type 3 (residual + 3x3 average pooling with stride 2, 1x1 stride-2 convolution on the shortcut), 16 / 64 / 128 / 256 channels
  L12     Type 4 (conv-BN-ReLU-conv-BN, then global average pooling), 512 channels
  FC      512 → 2
Initialization: He (kaiming normal) for convolutions, convolution bias = 0.2; FC ~ N(0, 0.01), bias 0.

Independent of the rest of the pqpks package, so the GPU environment only needs torch.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class _Type1(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class _Type2(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.t1 = _Type1(c, c)
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.bn = nn.BatchNorm2d(c)

    def forward(self, x):
        return x + self.bn(self.conv(self.t1(x)))


class _Type3(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.t1 = _Type1(cin, cout)
        self.conv = nn.Conv2d(cout, cout, 3, padding=1)
        self.bn = nn.BatchNorm2d(cout)
        self.pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.short = nn.Conv2d(cin, cout, 1, stride=2)
        self.short_bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.pool(self.bn(self.conv(self.t1(x)))) + self.short_bn(self.short(x))


class _Type4(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.t1 = _Type1(cin, cout)
        self.conv = nn.Conv2d(cout, cout, 3, padding=1)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.conv(self.t1(x))).mean(dim=(2, 3))


class SRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            _Type1(1, 64), _Type1(64, 16),
            *[_Type2(16) for _ in range(5)],
            _Type3(16, 16), _Type3(16, 64), _Type3(64, 128), _Type3(128, 256),
            _Type4(256, 512),
        )
        self.fc = nn.Linear(512, 2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0.2)
        nn.init.normal_(self.fc.weight, 0.0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W), pixel values 0-255 as float (no normalization, as in the original paper)."""
        return self.fc(self.layers(x))


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
