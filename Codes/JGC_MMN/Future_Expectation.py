import torch
import torch.nn as nn


class Future_Expectation(nn.Module):
    def __init__(self):
        super(Future_Expectation, self).__init__()

    def forward(self, x):
        b, _, h, w = x.shape
        return torch.randn(b, 1, h, w)