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
        self.decoder = Decoder(128, 21)
        self.aspp = ASPP(2048, 256)

    def forward(self, input):
        x, low_level_feature = self.encoder(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feature)
        x = F.interpolate(x, input.shape[-2:], mode='bilinear', align_corners=True)

        return x

    def get_1x_lr_params(self):
        modules = [self.encoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
