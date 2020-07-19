import torch
from torchsummaryX import summary

from model.utils.block import Block
from model.encoder import _Entry, _Middle, _Exit, Encoder
from model.deeplab import Deeplab
from model.decoder import Decoder

if __name__ == "__main__":
    # model = Block(728, 728)
    # model = _Entry()
    # model = _Middle()
    # model = _Exit()
    # model = Encoder()
    # model = Decoder(128,20)
    model = Deeplab()
    model.to('cuda')
    model.eval()
    x = torch.Tensor(1, 3, 512, 512).cuda()
    output = model(x)
    print(output.shape)
    # print(model)
