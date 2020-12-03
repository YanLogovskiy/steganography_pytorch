import torch
from torch import nn as nn


# implement according to paper
class StegoAnalyzer(nn.Module):
    def __init__(self, use_special_kernel=False):
        self.preprocessing_layer = None

    def forward(self):
        pass
