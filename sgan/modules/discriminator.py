from sgan.modules.base import *


class Discriminator(nn.Module):
    def __init__(self, *, in_channels=3):
        super().__init__()
        downsample_params = dict(
            kernel_size=4, stride=2,
            padding=1, dilation=1,
            bias=False
        )
        downsample_block = partial(ForwardPreactivation, activation_module=partial(nn.LeakyReLU, 0.15),
                                   **downsample_params)
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, 64, **downsample_params),
            nn.LeakyReLU(0.15),
            downsample_block(64, 128),
            downsample_block(128, 256),
            ResBlock2d(256, 256, conv_module=nn.Conv2d),
            downsample_block(256, 512),
            nn.LeakyReLU(0.15),
            ForwardPreactivation(512, 1, kernel_size=4, stride=1,
                                 padding=0, dilation=1, bias=True),
        )

    def forward(self, x):
        x = self.main(x)
        return x
