import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Conv2d, ReLU, BatchNorm2d
from torchsummaryX import summary


class Depthwise(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super(Depthwise, self).__init__()

        self.depthwise = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=True)
        self.pointwise = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                                dilation=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x
