from torch import nn
from typing import Sequence
from functools import partial

from sgan.modules.base import ForwardPreactivation


# TODO: implement factory methods

class Discriminator(nn.Module):
    def __init__(self, *, in_channels=3, base_block: nn.Module = None, structure: Sequence = None,
                 kernel_size=3, stride=1, padding=1, bias=False, dilation=1):
        super().__init__()

        # TODO: move to separate config file
        structure = structure or [64, 128, 256, 512]
        if base_block is None:
            base_block = partial(
                ForwardPreactivation, activation_module=nn.ReLU, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=bias, dilation=dilation
            )
            base_block = nn.Sequential(base_block, nn.MaxPool2d(2, 2))

        # TODO: check dimensions, etc
        self.input_part = nn.Conv2d(in_channels, structure[0], kernel_size=kernel_size)
        self.core_part = nn.Sequential(*(base_block(in_c, out_c) for in_c, out_c in zip(structure, structure[1:])))
        self.output_part = nn.Sequential(nn.Conv2d(structure[-1], 1, kernel_size=kernel_size), nn.AdaptiveMaxPool2d(1))

    def forward(self, x):
        x = self.input_part(x)
        x = self.core_part(x)
        x = self.output_part(x)
        # in my opinion output should be a single integer value \in [-inf, +inf] (i.e single logit)
        return x

    @classmethod
    def from_config(cls, config: dict):
        pass
