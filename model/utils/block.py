import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Conv2d, ReLU, BatchNorm2d
from torchsummaryX import summary

from . import ConvReluBatchnorm, Depthwise


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, with_depthwise=True):
        super(Block, self).__init__()

        self.crb1 = ConvReluBatchnorm(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=1, padding=padding, with_depthwise=with_depthwise)
        self.crb2 = ConvReluBatchnorm(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=1, padding=padding, with_depthwise=with_depthwise)
        if with_depthwise:
            self.conv = Depthwise(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding)
        else:
            self.conv = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.skip = Depthwise(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                              padding=0)
        self.relu = ReLU()
        self.batchnorm = BatchNorm2d(out_channels)

    def forward(self, x):
        skip = self.skip(x)

        x = self.crb1(x)
        x = self.crb2(x)
        x = self.conv(x)

        x = x + skip

        x = self.relu(x)
        x = self.batchnorm(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)