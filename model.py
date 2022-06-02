import torch
import torch.nn as nn

from mbconv_with_se_block import MBConvWithSEBlock


class EfficientNet(nn.Module):
    """
    EfficientNet implementation.
    See more at https://arxiv.org/pdf/1905.11946.pdf.
    """

    def __init__(
        self, class_numbers: int, alpha: float, beta: float, phi: float
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(32)

        # num_block, in_channels, out_channels, reduce_ratio, kernel, stride
        mbconv_blocks_describtions = [
            [1, 32, 16, 1, 3, 1],
            [2, 16, 24, 6, 3, 2],
            [2, 24, 40, 6, 5, 2],
            [3, 40, 80, 6, 3, 2],
            [3, 80, 112, 6, 5, 2],
            [4, 112, 192, 6, 5, 1],
            [1, 192, 320, 6, 3, 2],
        ]
        self.mbconv_blocks = self._create_mbconv_blocks(
            mbconv_blocks_describtions, alpha, phi
        )
        self.conv2 = nn.Conv2d(320, 1280 * round(pow(beta, phi)), 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(1280 * round(pow(beta, phi)), class_numbers)

    def _create_mbconv_blocks(
        self, mbconv_blocks_describtions: list, alpha: float, phi: float
    ) -> nn.Sequential:
        blocks = []
        for x in mbconv_blocks_describtions:
            blocks.append(MBConvWithSEBlock(x[1], x[2], x[3], x[4], x[5]))
            for _ in range((x[0] - 1) * round(pow(alpha, phi))):
                blocks.append(MBConvWithSEBlock(x[2], x[2], x[3], x[4], 1))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.mbconv_blocks(out)
        out = self.avgpool(self.conv2(out))
        out = self.fc1(out.view(x.shape[0], -1))
        return out