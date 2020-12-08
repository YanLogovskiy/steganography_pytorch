from torch import nn as nn
from torch.nn import functional as F


class Discriminator(nn.Module):
    def __init__(self, *, in_channels=3):
        super().__init__()
        # Input Dimension: (nc) x 64 x 64
        self.conv1 = nn.Conv2d(in_channels, 64, 4, 2, 1, bias=False)
        # first downsample block
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        # second
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        # third
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        # and the last one
        self.conv5 = nn.Conv2d(512, 1, 4, 1, 0, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)
        x = self.conv5(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels=100, out_channels=3):
        super().__init__()
        # input is the latent vector Z.
        self.deconv1 = nn.ConvTranspose2d(in_channels, 64 * 8, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64 * 8)
        # input Dimension: (ngf*8) x 4 x 4
        self.deconv2 = nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64 * 4)
        # Input Dimension: (ngf*4) x 8 x 8
        self.deconv3 = nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(64 * 2)
        # Input Dimension: (ngf*2) x 16 x 16
        self.deconv4 = nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        # Input Dimension: (ngf) * 32 * 32
        self.deconv5 = nn.ConvTranspose2d(64, out_channels, 4, 2, 1, bias=False)

    def forward(self, x):
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))
        return x
