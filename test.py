import torch
from torchsummaryX import summary


from model.utils.block import Block
from model.encoder import _Entry

if __name__ == "__main__":
    model = _Entry()
    model.to('cuda')
    x = torch.Tensor(1, 3, 512, 512).cuda()
    summary(model, x)