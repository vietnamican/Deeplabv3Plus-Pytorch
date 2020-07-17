import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Conv2d, ReLU, BatchNorm2d
from torchsummaryX import summary


class _Entry(nn.Module):
    def __init__(self):
        super(_Entry, self).__init__()

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
