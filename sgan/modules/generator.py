import torch
from torch import nn

from typing import Sequence
from functools import partial

from sgan.modules.base import TransposedPreactivation


# TODO: fix
class Generator(nn.Module):
    def __init__(self, in_channels=3, base_block: nn.Module = None, structure: Sequence = None,
                 kernel_size=4, stride=2, padding=2, bias=False, dilation=1):
        super().__init__()
        structure = structure or [512, 256, 128, 64, in_channels]

        if base_block is None:
            base_block = partial(
                TransposedPreactivation, activation_module=nn.ReLU, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=bias, dilation=dilation
            )

    def forward(self, x):
        pass
