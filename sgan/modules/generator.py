from torch import nn

from typing import Sequence
from functools import partial

from sgan.modules.base import TransposedPreactivation


class Generator(nn.Module):
    def __init__(self, in_channels=64, out_channels=3, base_block: nn.Module = None, structure: Sequence = None,
                 kernel_size=4, stride=2, padding=1, bias=False, dilation=1):
        super().__init__()
        convolution_params = dict(
            kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation
        )

        structure = structure or [512, 256, 128, 64, out_channels]
        if base_block is None:
            base_block = partial(
                TransposedPreactivation, activation_module=nn.LeakyReLU(), **convolution_params, bias=bias
            )

        # TODO: remove hardcode?
        self.input = nn.ConvTranspose2d(in_channels, structure[0], kernel_size=kernel_size,
                                        stride=1, padding=0, bias=bias)
        self.core = nn.Sequential(*(base_block(in_c, out_c) for in_c, out_c in zip(structure, structure[1:])))
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.input(x)
        x = self.core(x)
        x = self.activation(x)
        return x
