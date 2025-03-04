import torch
import torch.nn as nn
from typing import List


class Fusion(nn.Module):
    def __init__(
        self,
        indexs: List[int],
        modalities: int = 4,
        img_h: int = 30,
        img_w: int = 30
    ):
        super(Fusion, self).__init__()
        self.modalities = modalities
        self.indexs = indexs
        for index in indexs:
            for m in range(modalities):
                if m != index:
                    self.register_parameter(
                        f"w1_{index}_{m}", nn.Parameter(torch.ones(1, img_h, img_w))
                    )
        for i in range(modalities):
            self.register_parameter(f"w2_{i}", nn.Parameter(torch.ones(1, img_h, img_w)))
            self.register_parameter(f"b2_{i}", nn.Parameter(torch.zeros(1, img_h, img_w)))
        self.softmax2d = nn.Softmax2d()
        self.tanh = nn.Tanh()

    def generate_combination(self, index: int):
        com = [[index, m] for m in range(self.modalities) if m != index]
        return com

    def forward(self, x: List[torch.Tensor]):
        Pa = torch.sum(torch.stack([x[index] for index in self.indexs]), dim=0)
        # filtration gate submodule
        h = []
        for i in range(self.modalities):
            h.append(
                self.tanh(getattr(self, f"w2_{i}") * x[i] + getattr(self, f"b2_{i}"))
            )
        C_ls = []
        Pred_ls = []
        for index in self.indexs:
            com = self.generate_combination(index)
            # cur_A: affinity matrix
            cur_A = self.softmax2d(
                torch.cat(
                    [x[com[i][0]] + x[com[i][1]] for i in range(len(com))], dim=1
                ),
            ) + self.softmax2d(
                torch.cat(
                    # calculate attention score between different modalities
                    [x[com[i][0]] * x[com[i][1]] for i in range(len(com))],
                    dim=1,
                )
            )
            # temp_C: temporal joint representation
            temp_C = (
                torch.cat(
                    [
                        torch.sigmoid(
                            getattr(self, f"w1_{com[i][0]}_{com[i][1]}")
                            / torch.norm(h[com[i][0]] - h[com[i][1]], p="fro")
                        )
                        * x[com[i][1]]
                        for i in range(len(com))
                    ],
                    dim=1,
                )
                * cur_A
            )
            # joint representation with gated control
            C_ls.append(temp_C * self.softmax2d(x[index] * Pa))
            Pred_ls.append(getattr(self, f"w2_{index}") * (temp_C + x[index]))
        C = torch.sum(torch.stack(C_ls), dim=0)
        Pred = torch.sum(torch.stack(Pred_ls), dim=0)
        # add rest part without attention
        for index in range(self.modalities):
            if index not in self.indexs:
                Pred += getattr(self, f"w2_{index}") * C
        return self.tanh(torch.mean(Pred, dim=1).unsqueeze(1))


if __name__ == "__main__":
    x = [torch.randn(5, 1, 60, 60) for _ in range(3)]
    model = Fusion(modalities=3, indexs=[0, 1], img_w=60, img_h=60)
    y = model(x)
    print(y.shape)