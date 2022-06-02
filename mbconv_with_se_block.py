import torch
import torch.nn as nn

from mbconv_block import MBConv
from se_block import SEBlock


class MBConvWithSEBlock(nn.Module):
    """MBConv block with SEBlock"""

    def __init__(
        self, in_channels: int, out_channels: int, reduce_ratio: int,
        kernel: int, stride: int
    ):
        super().__init__()
        self.mbconv = MBConv(
            in_channels, out_channels, reduce_ratio, kernel, stride
        )
        self.seblock = SEBlock(out_channels, reduce_ratio)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn(self.relu(x))
        return self.seblock(self.mbconv(out))