import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Conv2d, ReLU, BatchNorm2d
from torchsummaryX import summary
from . import Depthwise


class ConvReluBatchnorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
                 with_batchnorm=True,
                 with_relu=True, with_depthwise=False):
        super(ConvReluBatchnorm, self).__init__()
        if with_depthwise:
            self.conv = Depthwise(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, dilation=dilation, bias=bias)
        else:
            self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, dilation=dilation, bias=bias)
        if with_relu:
            if with_batchnorm:
                self.crb_block = nn.Sequential(self.conv, ReLU(), BatchNorm2d(out_channels))
            else:
                self.cbr_block = nn.Sequential(self.conv, ReLU())
        else:
            if with_batchnorm:
                self.crb_block = nn.Sequential(self.conv, BatchNorm2d(out_channels))
            else:
                self.crb_block = nn.Sequential(self.conv)

        self._init_weight()

    def forward(self, x):
        x = self.crb_block(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)
