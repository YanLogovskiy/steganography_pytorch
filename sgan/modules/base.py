from torch import nn
from functools import partial

from dpipe.layers import PreActivation2d, ResBlock2d

ForwardPreactivation = partial(PreActivation2d, conv_module=nn.Conv2d)
TransposedPreactivation = partial(PreActivation2d, conv_module=nn.ConvTranspose2d)
