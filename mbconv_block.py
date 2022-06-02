import torch
import torch.nn as nn


class MBConv(nn.Module):
    """
    MBConv block.
    See more at https://arxiv.org/pdf/1801.04381.pdf.
    """

    def __init__(
        self, in_channels: int, out_channels: int, reduce_ratio: int,
        kernel: int, stride: int
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels * reduce_ratio, 1)
        self.relu6 = nn.ReLU6()
        self.bn1 = nn.BatchNorm2d(in_channels * reduce_ratio)
        padding = (kernel // 2)
        self.conv2 = nn.Conv2d(
            in_channels * reduce_ratio,
            in_channels * reduce_ratio,
            kernel,
            stride=stride,
            padding=padding
        )
        self.bn2 = nn.BatchNorm2d(in_channels * reduce_ratio)
        self.conv3 = nn.Conv2d(in_channels * reduce_ratio, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.relu6(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.relu6(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        return out + residual