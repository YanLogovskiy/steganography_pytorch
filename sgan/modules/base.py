from torch import nn
from functools import partial


class PreActivation2d(nn.Module):
    """
    Performs a sequence of batch_norm, activation, and convolution

        in -> (BN -> activation -> Conv) -> out
    """

    def __init__(self, in_channels, out_channels, *, activation_module=nn.ReLU, conv_module,
                 kernel_size, stride=1, padding=0, bias=True, dilation=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn = nn.BatchNorm2d(in_channels)
        self.activation = activation_module()
        self.layer = conv_module(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=bias, dilation=dilation
        )

    def forward(self, x):
        return self.layer(self.activation(self.bn(x)))


class ResBlock2d(nn.Module):
    """
        Performs a sequence of two convolutions with residual connection (Residual Block).

        ..
            in ---> (BN --> activation --> Conv) --> (BN --> activation --> Conv) -- + --> out
                |                                                                    ^
                |                                                                    |
                 --------------------------------------------------------------------
    """

    def __init__(self, in_channels, out_channels, *, use_shortcut=True, kernel_size=3, stride=1, padding=1, dilation=1,
                 bias=False, activation_module=nn.ReLU, conv_module):
        super().__init__()
        self.use_shortcut = use_shortcut

        preactivation = partial(
            PreActivation2d, kernel_size=kernel_size, padding=padding, dilation=dilation,
            activation_module=activation_module, conv_module=conv_module
        )
        self.conv_path = nn.Sequential(preactivation(in_channels, out_channels, stride=stride, bias=False),
                                       preactivation(out_channels, out_channels, bias=bias))

        if in_channels != out_channels or stride != 1:
            self.adjust_to_stride = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)
        else:
            self.adjust_to_stride = lambda x: x

    def forward(self, x):
        x_conv = self.conv_path(x)
        if self.use_shortcut:
            x_conv += self.adjust_to_stride(x)
        return x_conv


ForwardPreactivation = partial(PreActivation2d, conv_module=nn.Conv2d)
TransposedPreactivation = partial(PreActivation2d, conv_module=nn.ConvTranspose2d)
