from sgan.modules.base import *


class Generator(nn.Module):
    def __init__(self, in_channels=64, out_channels=3):
        super().__init__()
        upsample_params = dict(kernel_size=4, stride=2, padding=1, dilation=1)
        upsample_block = partial(
            TransposedPreactivation, activation_module=nn.ReLU, **upsample_params, bias=False
        )
        self.main = nn.Sequential(
            TransposedPreactivation(in_channels, 2 * in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * in_channels, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.ReLU(),
            upsample_block(512, 256),
            ResBlock2d(256, 256, conv_module=nn.Conv2d),
            upsample_block(256, 128),
            ResBlock2d(128, 128, conv_module=nn.Conv2d),
            upsample_block(128, 64),
            upsample_block(64, out_channels)
        )
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.main(x)
        x = self.activation(x)
        return x
