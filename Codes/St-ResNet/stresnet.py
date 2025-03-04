from typing import Optional

import torch
import torch.nn as nn
from resunit import ResUnit


class STResNet(nn.Module):
    def __init__(
        self,
        # input config
        len_long: int,
        len_short: int,
        len_ingred: int,
        external_dim: Optional[int],
        # model config
        map_height: int,
        map_width: int,
        residual_unit: int,
    ) -> None:
        super().__init__()
        self.external_dim = external_dim
        self.map_height = map_height
        self.map_width = map_width
        self.residual_unit = residual_unit

        # models
        self.l_net = self._create_timenet(len_long)
        self.s_net = self._create_timenet(len_short)
        self.i_net = self._create_timenet(len_ingred)
        if self.external_dim:
            total_channels = len_long + len_short
            self.e_net = self._create_extnet(
                self.external_dim, total_channels=total_channels
            )

        # for fusion
        self.W_l = nn.parameter.Parameter(
            torch.randn(1, self.map_width, self.map_height),
            requires_grad=True,
        )
        self.W_s = nn.parameter.Parameter(
            torch.randn(1, self.map_width, self.map_height),
            requires_grad=True,
        )
        self.W_i = nn.parameter.Parameter(
            torch.randn(1, self.map_width, self.map_height),
            requires_grad=True,
        )
        self.init_weights()

    # for external data
    def _create_extnet(self, ext_dim: int, total_channels: int) -> nn.Sequential:
        ext_net = nn.Sequential(
            nn.Linear(ext_dim, total_channels),
            nn.ReLU(inplace=True),
            nn.Linear(total_channels, self.map_height * self.map_width),
        )
        return ext_net

    # for map data
    def _create_timenet(self, length: int) -> nn.Sequential:
        time_net = nn.Sequential()
        time_net.add_module(
            "Conv1",
            nn.Conv2d(
                in_channels=length,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
        )

        for i in range(self.residual_unit):
            time_net.add_module(
                f"ResUnit{i + 1}", ResUnit(in_channels=64, out_channels=64)
            )

        time_net.add_module(
            "Conv2",
            nn.Conv2d(
                in_channels=64, out_channels=1, kernel_size=3, stride=1, padding="same"
            ),
        )
        return time_net
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        xl: torch.Tensor,
        xs: torch.Tensor,
        xi: torch.Tensor,
        ext: Optional[torch.Tensor],
    ) -> torch.Tensor:
        l_out = self.l_net(xl)
        s_out = self.s_net(xs)
        i_out = self.i_net(xi)

        if self.external_dim:
            e_out = self.e_net(ext).view(
                -1, 1, self.map_width, self.map_height
            )
            # fusion with ext data
            res = self.W_l.unsqueeze(0) * l_out
            res += self.W_s.unsqueeze(0) * s_out
            res += self.W_i.unsqueeze(0) * i_out
            res += e_out
        else:
            res = self.W_l.unsqueeze(0) * l_out
            res += self.W_s.unsqueeze(0) * s_out
            res += self.W_i.unsqueeze(0) * i_out

        return torch.tanh(res)


if __name__ == "__main__":
    model = STResNet(
        len_long=2,
        len_short=12,
        len_ingred=4,
        external_dim=6,
        map_height=30,
        map_width=30,
        residual_unit=4,
    )
    xl = torch.randn(1, 2, 30, 30)
    xs = torch.randn(1, 12, 30, 30)
    xi = torch.randn(1, 4, 30, 30)
    ext = torch.randn(1, 6)
    output = model(xl, xs, xi, ext)
    print(output.shape)