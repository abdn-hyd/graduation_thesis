from typing import List
import torch
import torch.nn as nn
import Modified_DenseNet
import DenseNet
from Future_Expectation import Future_Expectation
from Fusion import Fusion


class JGC_MMN(nn.Module):
    def __init__(
        self,
        # branch config
        branches: List[int] = [0, 1, 2, 3],
        dense_ori: bool = False,
        # img config
        img_h: int = 30,
        img_w: int = 30,
        # short term config
        short_term_in_channels: int = 6,
        short_term_growth_rate: int = 4,
        short_term_block_config: tuple = (9, 9, 9, 9, 9, 9),
        # long term config
        long_term_in_channels: int = 2,
        long_term_growth_rate: int = 4,
        long_term_block_config: tuple = (9, 9, 9, 9, 9, 9),
        # cur ingredient config
        ingred_dense: bool = False,
        cur_ingred_dim: int = 9,
        emedding_dim: int = 64,
        # future price growth expectation config
        # fusion config
        modalities: int = 4,
        indexs: List[int] = [0, 1, 3],
    ):
        # define long term and short term densenet, shape: (b, short_term_in_channels/long_term_in_channels, img_h, img_w) -> (b, 1, img_h, img_w)
        super(JGC_MMN, self).__init__()
        self.branches = branches
        self.dense = Modified_DenseNet if not dense_ori else DenseNet
        self.ingred_dense = ingred_dense
        self.add_module(
            "branch_0",
            self.dense.long_term(
                in_channels=long_term_in_channels,
                growth_rate=long_term_growth_rate,
                block_config=long_term_block_config,
            ),
        )
        self.add_module(
            "branch_1",
            self.dense.short_term(
                in_channels=short_term_in_channels,
                growth_rate=short_term_growth_rate,
                block_config=short_term_block_config,
            ),
        )

        # define current ingredient module, shape: (b, (6 + k), img_h, img_w -> (b, 1, img_h, img_w)
        if ingred_dense:
            self.add_module(
                "branch_2",
                self.dense.ingred_term(
                    in_channels=cur_ingred_dim,
                    growth_rate=emedding_dim,
                    block_config=(6, 12, 24, 16),
                    num_init_features=64,
                ),
            )
        else:
            self.add_module(
                "branch_2",
                nn.Sequential(
                    # embedding layer
                    nn.Conv2d(cur_ingred_dim, emedding_dim, 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(emedding_dim),
                    nn.ReLU(inplace=True),
                    # fc layer
                    nn.Conv2d(emedding_dim, 1, 1, stride=1, bias=True),
                    nn.Tanh(),
                ),
            )

        # define future price growth expectation module, shape: (b, 2, img_h, img_w) -> (b, 1, img_h, img_w)
        self.add_module("branch_3", Future_Expectation())

        # joint gated co-attention based Fusion
        self.fusion = Fusion(indexs=indexs, modalities=modalities, img_h=img_h, img_w=img_w)

        # initialize cur_ingred weights
        self.initialize_cur_ingred_weights(self)

    def initialize_cur_ingred_weights(self, model):
        if not self.ingred_dense:
            for m in model.branch_2:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: List[torch.Tensor],
    ):
        out_features = []
        for i in range(4):
            if i in self.branches:
                out_features.append(getattr(self, f"branch_{i}")(x[i]))
        pred = self.fusion(out_features)
        return pred


if __name__ == "__main__":
    long = torch.randn(1, 2, 60, 60)
    short = torch.randn(1, 12, 60, 60)
    cur_ingred = torch.randn(1, 9, 60, 60)
    future_exp = torch.randn(1, 2, 60, 60)
    model = JGC_MMN(img_h=60 ,img_w=60, dense_ori=True, short_term_in_channels=12, ingred_dense=True, branches=[0, 1, 2], indexs=[0, 1], modalities=3)
    y = model([long, short, cur_ingred])
    print(y.shape)
