import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    SQUEEZE-AND-EXCITATION block.
    See more at https://arxiv.org/pdf/1709.01507.pdf.
    """

    def __init__(self, channels: int, reduce_ratio: int) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduce_ratio)
        self.fc2 = nn.Linear(channels // reduce_ratio, channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #Squeeze step
        x_shape = x.shape
        out = self.avgpool(x)
        out = out.view((x_shape[0], -1))
        #Excitation step
        out = self.sigmoid(self.fc2(self.relu(self.fc1(out))))
        #Scale step
        out = out.view((x_shape[0], x_shape[1], 1))
        x = x.view((x_shape[0], x_shape[1], -1))
        out = x * out
        out = out.view(x_shape)
        return out