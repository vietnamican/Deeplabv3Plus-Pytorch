import torch
import torch.nn as nn
import torch.nn.functional as F

import encoder
import decoder
import aspp

class Deeplab(nn.Module):
    def _init__(self):
        super(Deeplab, self).__init__()

        self.encoder = encoder.build()
        self.decoder = decoder.build()
        self.aspp    = aspp.build()

    def forward(self, input):
        x, low_level_feature = self.encoder(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feature)

        return x