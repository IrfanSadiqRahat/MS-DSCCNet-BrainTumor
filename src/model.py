"""
MS-DSCCNet: Multi-Scale Depthwise Separable CNN with Channel Attention
for Brain Tumor MRI Classification.

Published: IEEE DELCON 2025, Paper #234.
Author: Irfan Sadiq Rahat
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel, stride, padding,
                            groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))))


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention block."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.fc(self.gap(x)).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class MultiScaleBranch(nn.Module):
    """Three parallel conv branches with 3×3, 5×5, 7×7 kernels."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU())
        self.b5 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 5, padding=2, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU())
        self.b7 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 7, padding=3, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU())

    def forward(self, x):
        return torch.cat([self.b3(x), self.b5(x), self.b7(x)], dim=1)


class MSDSCCNet(nn.Module):
    """
    MS-DSCCNet for brain tumor classification.
    Input:  (B, 3, 224, 224)
    Output: (B, num_classes) logits
    """
    def __init__(self, num_classes: int = 4, dropout: float = 0.4):
        super().__init__()
        # Multi-scale feature extraction
        self.ms_branch = MultiScaleBranch(3, 32)     # → 96 channels
        self.pool1     = nn.MaxPool2d(2)

        # Depthwise separable blocks
        self.dsc1 = DepthwiseSeparableConv(96, 128)
        self.dsc2 = DepthwiseSeparableConv(128, 256)
        self.dsc3 = DepthwiseSeparableConv(256, 256)
        self.pool2 = nn.MaxPool2d(2)

        # Channel attention
        self.ca = ChannelAttention(256)

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.ms_branch(x))
        x = self.dsc1(x)
        x = self.dsc2(x)
        x = self.pool2(self.dsc3(x))
        x = self.ca(x)
        return self.classifier(x)
