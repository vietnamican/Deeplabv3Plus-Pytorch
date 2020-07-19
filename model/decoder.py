import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Conv2d, ReLU, BatchNorm2d
from torchsummaryX import summary

from utils import ConvReluBatchnorm


class Decoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Decoder, self).__init__()

        self.conv_res = Conv2d(in_channels=in_channels, out_channels=48, kernel_size=1)
        self.relu_res = ReLU()
        self.batchnorm_res = BatchNorm2d(48)
        self.last_conv = nn.Sequential(ConvReluBatchnorm(304, 256, 3, 1, 1, bias=False),
                                       ConvReluBatchnorm(256, 256, 3, 1, 1, bias=False),
                                       Conv2d(256, num_classes, kernel_size=1, stride=1)
                                       )
        self._init_weight()

    def forward(self, x, low_level_feature):
        low_level_feature = self.conv_res(low_level_feature)
        low_level_feature = self.relu_res(low_level_feature)
        low_level_feature = self.batchnorm_res(low_level_feature)

        x = F.interpolate(x, low_level_feature.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feature), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)


if __name__ == "__main__":
    model = Decoder(256, 20)
    model.to('cuda')
    x = torch.Tensor(1, 256, 32, 32).cuda()
    low_level_feature = torch.Tensor(1, 256, 64, 64).cuda()
    summary(model, x, low_level_feature)
