import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Conv2d, ReLU, BatchNorm2d
from torchsummaryX import summary

from .utils import ConvReluBatchnorm, Block


class _Entry(nn.Module):
    def __init__(self):
        super(_Entry, self).__init__()

        self._init_weight()
        # block 1
        self.block1 = nn.Sequential(ConvReluBatchnorm(3, 32, 3, 2, 1),
                                    ConvReluBatchnorm(32, 64, 3, 1, 1)
                                    )
        self.block2 = Block(64, 128, 3, 2, 1)
        self.block3 = Block(128, 256, 3, 2, 1)
        self.block4 = Block(256, 728, 3, 2, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)


class _Middle(nn.Module):
    def __init__(self):
        super(_Middle, self).__init__()

        self._init_weight()

    def forward(self, x):
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)


class _Exit(nn.Module):
    def __init__(self):
        super(_Exit, self).__init__()

        self._init_weight()

    def forward(self, x):
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.entry = _Entry()
        self.middle = _Middle()
        self.exit = _Exit()

        self._init_weight()

    def forward(self, x):
        x = self.entry(x)
        x = self.middle(x)
        x = self.exit(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)
