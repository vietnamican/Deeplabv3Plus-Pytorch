import torch
from torchsummaryX import summary

from model.utils.block import Block
from model.encoder import _Entry, _Middle, _Exit

if __name__ == "__main__":
    # model = Block(728, 728)
    # model = _Entry()
    # model = _Middle()
    model = _Exit()
    model.to('cuda')
    x = torch.Tensor(1, 728, 32, 32).cuda()
    summary(model, x)