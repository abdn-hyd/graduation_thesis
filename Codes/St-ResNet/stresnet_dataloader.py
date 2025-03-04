import os
import numpy as np
import torch
from torch.utils import data
from typing import List


class St_ResNet_dataloader(data.Dataset):
    def __init__(
        self,
        name: List[str],
        root: str,
    ):
        self.modalities = len(name)
        self.name = name
        self.root = root
        self.np_data = []
        self.global_min = []
        self.global_max = []
        for m in range(self.modalities):
            cur_path = os.path.join(self.root, self.name[m])
            data = np.load(cur_path)
            self.np_data.append(data.astype(np.float32))
            global_min = data.min()
            global_max = data.max()

            # Store the global min and max for later use
            self.global_min.append(global_min)
            self.global_max.append(global_max)

    def __len__(self):
        return self.np_data[0].shape[0]

    def __getitem__(self, index):
        modalities_data = []
        for m in range(self.modalities):
            # get sample
            sample = self.np_data[m][index]
            c = sample.shape[0]
            # apply min max normalization
            if self.global_max[m] != self.global_min[m]:
                normalized = (
                    2
                    * (
                        (sample - self.global_min[m])
                        / (self.global_max[m] - self.global_min[m])
                    )
                    - 1
                )
            # else:
            #     normalized = np.zeros_like(sample)
            tensor = torch.from_numpy(normalized)
            modalities_data.append(tensor)
        return modalities_data


if __name__ == "__main__":
    dataset = St_ResNet_dataloader(
        name=[
            "long_term.npy",
            "short_term.npy",
            "ingredients.npy",
            "externel.npy",
            "future.npy",
            "label.npy",
        ],
        root="/Users/gunneo/Documents/4_2/Graduation_Thesis/Datasets/Beijing_House_Price_Dataset/St-ResNet/train",
    )
    dataloader = data.DataLoader(dataset, batch_size=36, shuffle=True)
    for batch in dataloader:
        print(batch[0].shape)
        print(batch[1].shape)
        print(batch[2].shape)
        print(batch[3].shape)
        print(batch[4].shape)
