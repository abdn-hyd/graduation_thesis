import torch
import torch.nn as nn
from ConvLstm import ConvLSTM
from fusion import Fusion
from typing import List


class ST_MMN(nn.Module):
    def __init__(
        self,
        img_h: int = 30,
        img_w: int = 30,
        branches: List[int] = [0, 1, 2, 3],
        # long term config
        long_num_layers: int = 2,
        long_in_channels: int = 1,
        long_hidden_dim: int = 64,
        long_kernel_size: tuple = (3, 3),
        long_stride: int = 1,
        long_padding: int = 1,
        long_bias: bool = True,
        long_frame_size: tuple = (30, 30),
        # short term config
        short_num_layers: int = 2,
        short_in_channels: int = 1,
        short_hidden_dim: int = 64,
        short_kernel_size: tuple = (3, 3),
        short_stride: int = 1,
        short_padding: int = 1,
        short_bias: bool = True,
        short_frame_size: tuple = (30, 30),
        # cur ingredient config
        cur_ingred_dim: int = 9,
        emedding_dim: int = 64,
        # fusion config
        indexs: List[int] = [0, 1, 3],
        modalities: int = 4,
    ):
        super(ST_MMN, self).__init__()
        self.branches = branches
        # long term price
        self.branch_1 = ConvLSTM(
            num_layers=long_num_layers,
            in_channels=long_in_channels,
            hidden_dim=long_hidden_dim,
            kernel_size=long_kernel_size,
            stride=long_stride,
            padding=long_padding,
            bias=long_bias,
            frame_size=long_frame_size,
        )
        # short term config
        self.branch_2 = ConvLSTM(
            num_layers=short_num_layers,
            in_channels=short_in_channels,
            hidden_dim=short_hidden_dim,
            kernel_size=short_kernel_size,
            stride=short_stride,
            padding=short_padding,
            bias=short_bias,
            frame_size=short_frame_size,
        )
        # define current ingredient module, shape: (b, (6 + k), img_h, img_w -> (b, 1, img_h, img_w)
        self.branch_3 = nn.Sequential(
            # embedding layer
            nn.Conv2d(
                cur_ingred_dim, emedding_dim, 3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(emedding_dim),
            nn.ReLU(inplace=True),
            # fc layer
            nn.Conv2d(emedding_dim, 1, 1, stride=1, bias=True),
            nn.Tanh(),
        )
        self.branch_4 = None

        # joint gated co-attention based Fusion
        self.fusion = Fusion(
            indexs=indexs, modalities=modalities, img_h=img_h, img_w=img_w
        )
        
    def forward(self, X: List[torch.Tensor]):
        out_features = []
        for i in range(4):
            if i in self.branches:
                if i == 2:
                    out_features.append(getattr(self, f"branch_{i + 1}")(X[i].squeeze(1)))
                else:
                    out_features.append(getattr(self, f"branch_{i + 1}")(X[i]))
        pred = self.fusion(out_features)
        return pred