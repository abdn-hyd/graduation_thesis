import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class _DenseLayer(nn.Module):
    def __init__(
        self,
        num_input_features,
        growth_rate,
        bn_size,
        drop_rate,
    ):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features)),
        self.add_module("relu1", nn.ReLU(inplace=True)),
        self.add_module(
            "conv1",
            nn.Conv2d(
                num_input_features,
                bn_size * growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        ),
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module("relu2", nn.ReLU(inplace=True)),
        self.add_module(
            "conv2",
            nn.Conv2d(
                bn_size * growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        ),
        self.add_module("norm3", nn.BatchNorm2d(growth_rate)),
        self.add_module("relu3", nn.ReLU(inplace=True)),
        self.add_module(
            "conv3",
            nn.Conv2d(
                growth_rate, num_input_features, kernel_size=1, stride=1, bias=False
            ),
        ),
        self.drop_rate = float(drop_rate)

    def forward(self, input):
        x = input[-1]
        bottleneck_output_1 = self.conv1(self.relu1(self.norm1(x)))
        bottleneck_output_2 = self.conv2(self.relu2(self.norm2(bottleneck_output_1)))
        new_features = self.conv3(self.relu3(self.norm3(bottleneck_output_2)))
        # perform channel wise addition
        for prev_feature in input[:-1]:
            new_features += prev_feature
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training
            )
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers,
        num_input_features,
        bn_size,
        growth_rate,
        drop_rate,
    ):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features=num_input_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        # the last feature is the final output of last layer
        return features[-1]


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "conv",
            nn.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )


class DenseNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        # num_init_features=64,
        bn_size=4,
        drop_rate=0,
    ):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        nn.Conv2d(
                            in_channels,
                            in_channels,
                            kernel_size=7,
                            stride=1,
                            padding=3,
                            bias=False,
                        ),
                    ),
                    ("norm0", nn.BatchNorm2d(in_channels)),
                    ("relu0", nn.ReLU(inplace=True)),
                ]
            )
        )

        # Each denseblock
        num_features = in_channels
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            # final block will not go through transition
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features,
                )
                self.features.add_module("transition%d" % (i + 1), trans)
                # num_features = num_features // 2

        # Final conv
        self.features.add_module(
            "final_conv",
            nn.Conv2d(num_features, 1, kernel_size=1, stride=1, bias=True),
        )
        self.features.add_module(
            "final_norm", nn.BatchNorm2d(1)
        )
        self.tanh = nn.Tanh()

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = self.tanh(features)
        return out


def _densenet(
    arch, in_channels, growth_rate, block_config, **kwargs
):
    model = DenseNet(
        in_channels, growth_rate, block_config, **kwargs
    )
    return model


def long_term(
    in_channels=2,
    growth_rate=4,
    block_config=(9, 9, 9, 9, 9, 9),
    # num_init_features=64,
    progress=True,
    **kwargs
):
    return _densenet(
        "long_term",
        in_channels=in_channels,
        growth_rate=growth_rate,
        block_config=block_config,
        # num_init_features=num_init_features,
        progress=progress,
        **kwargs
    )


def short_term(
    in_channels=12,
    growth_rate=4,
    block_config=(9, 9, 9, 9, 9, 9),
    # num_init_features=64,
    progress=True,
    **kwargs
):
    return _densenet(
        "short_term",
        in_channels=in_channels,
        growth_rate=growth_rate,
        block_config=block_config,
        # num_init_features=num_init_features,
        progress=progress,
        **kwargs
    )


def ingred_term(
    in_channels=9,
    growth_rate=4,
    block_config=(9, 9, 9, 9, 9, 9),
    # num_init_features=64,
    progress=True,
    **kwargs
):
    return _densenet(
        "ingred_term",
        in_channels=in_channels,
        growth_rate=growth_rate,
        block_config=block_config,
        # num_init_features=num_init_features,
        progress=progress,
        **kwargs
    )