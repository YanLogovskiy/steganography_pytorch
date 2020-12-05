from torch import nn
from typing import Sequence

from sgan.modules.base import ForwardPreactivation


class Discriminator(nn.Module):
    def __init__(self, *, in_channels=3, base_block: nn.Module = None, structure: Sequence = None,
                 kernel_size=4, stride=2, padding=1, bias=False, dilation=1):
        super().__init__()
        convolution_params = dict(
            kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation
        )
        structure = structure or [64, 128, 256, 512]
        if base_block is None:
            def base_block(in_c, out_c):
                return nn.Sequential(
                    ForwardPreactivation(in_c, out_c, activation_module=nn.LeakyReLU, **convolution_params, bias=bias))

        self.input = nn.Sequential(nn.Conv2d(in_channels, structure[0], **convolution_params, bias=bias),
                                   nn.LeakyReLU())
        self.core = nn.Sequential(*(base_block(in_c, out_c) for in_c, out_c in zip(structure, structure[1:])),
                                  nn.LeakyReLU())
        self.output = nn.Sequential(nn.Conv2d(structure[-1], 1, 4, 1, 0, bias=bias))

    def forward(self, x):
        x = self.input(x)
        x = self.core(x)
        x = self.output(x)
        return x
