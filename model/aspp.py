import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Conv2d, ReLU, BatchNorm2d, Dropout
from torchsummaryX import summary

from .utils import ConvReluBatchnorm


class _ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, dilation=1):
        super(_ASPPModule, self).__init__()

        self.crb = ConvReluBatchnorm(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           stride=1, padding=padding, dilation=dilation, bias=True, with_batchnorm=True,
                                           with_relu=True, with_depthwise=True)

    def forward(self, x):
        x = self.crb(x)

        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]
        self.aspp0 = _ASPPModule(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                 padding=0, dilation=dilations[0])
        self.aspp1 = _ASPPModule(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                 padding=dilations[1], dilation=dilations[1])
        self.aspp2 = _ASPPModule(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                 padding=dilations[2], dilation=dilations[2])
        self.aspp3 = _ASPPModule(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                 padding=dilations[3], dilation=dilations[3])

        self.conv = Conv2d(out_channels * 4, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                           bias=True)
        self.relu = ReLU()
        self.batchnorm = BatchNorm2d(out_channels)

        self._init_weight()

    def forward(self, x):
        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)

        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.conv(x)
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


if __name__ == "__main__":
    model = ASPP(64, 256)
    model.to('cuda')
    x = torch.Tensor(1, 64, 32, 32).cuda()
    summary(model, x)
