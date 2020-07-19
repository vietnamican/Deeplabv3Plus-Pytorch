import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Conv2d, ReLU, BatchNorm2d
from torchsummaryX import summary

from . import ASPP, Encoder, Decoder


class Deeplab(nn.Module):
    def __init__(self):
        super(Deeplab, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder(128, 20)
        self.aspp = ASPP(2048, 256)

        self._init_weight()

    def forward(self, input):
        x, low_level_feature = self.encoder(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feature)
        x = F.interpolate(x, input.shape[-2:], mode='bilinear', align_corners=True)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)
