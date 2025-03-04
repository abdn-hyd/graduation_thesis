import os
import numpy as np
import torch
from torch.utils import data
from typing import List


class JGC_MMN_dataloader(data.Dataset):
    def __init__(
        self,
        name: List[str],
        root: str,
        transaction_cnt: bool,
    ):
        self.modalities = len(name)
        self.name = name
        self.root = root
        self.np_data = []
        self.global_min = []
        self.global_max = []
        self.transaction_cnt = transaction_cnt
        for m in range(self.modalities):
            cur_path = os.path.join(self.root, self.name[m])
            cur_data = np.load(cur_path)
            _, c, _, _ = cur_data.shape
            self.np_data.append(cur_data.astype(np.float32))
            # add min and max value of mean price and transaction count
            if m == 0 or m == 1:
                price = cur_data[:, :c//2, :, :]
                if transaction_cnt:
                    transaction_cnt = cur_data[:, c//2:, :, :]
                    self.global_min.append([np.min(price), np.min(transaction_cnt)])
                    self.global_max.append([np.max(price), np.max(transaction_cnt)])
                else:
                    self.global_min.append([np.min(price)])
                    self.global_max.append([np.max(price)])
            elif m == 2:
                self.global_min.append([np.min(cur_data[:, i, :, :]) for i in range(c)])
                self.global_max.append([np.max(cur_data[:, i, :, :]) for i in range(c)])
            else:
                self.global_min.append([np.min(cur_data)])
                self.global_max.append([np.max(cur_data)])

    def normalized(self, x: np.array, min: float, max: float):
        if min != max:
            normalized = 2 * ((x - min) / (max - min)) - 1
        else:
            normalized = np.zeros_like(x)
        return normalized
    
    
    def __len__(self):
        return self.np_data[0].shape[0]
    
    def __getitem__(self, index):
        modalities_data = []
        for m in range(self.modalities):
            # get sample, shape: (c, h, w)
            sample = self.np_data[m][index]
            c, h, w = sample.shape
            # apply min max normalization
            if m == 0 or m == 1:
                price_min, price_max = self.global_min[m][0], self.global_max[m][0]
                normalized_tensor = self.normalized(sample[:c//2, :, :], price_min, price_max)
                if self.transaction_cnt:
                    transaction_min, transaction_max = self.global_min[m][1], self.global_max[m][1]
                    transaction_cnt = self.normalized(sample[c//2:, :, :], transaction_min, transaction_max)
                    normalized_tensor = np.concatenate((normalized_tensor, transaction_cnt), axis=0)
            elif m == 2:
                normalized_tensor = np.stack([
                    self.normalized(sample[i, :, :], self.global_min[m][i], self.global_max[m][i])
                    for i in range(c)
                ], axis=0)
            else:
                normalized_tensor = self.normalized(sample, self.global_min[m][0], self.global_max[m][0])
            # convert numpy arrays into tensors
            tensor = torch.from_numpy(normalized_tensor)
            modalities_data.append(tensor)
        return modalities_data


if __name__ == "__main__":
    dataset = JGC_MMN_dataloader(
        name=["long_term.npy", "short_term.npy", "ingredients.npy", "future.npy", "label.npy"],
        root="/Users/gunneo/Documents/4_2/Graduation_Thesis/Datasets/NYC_House_Price_Dataset/JGC_MMN/train",
        transaction_cnt=False,
    )
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    for batch in dataloader:
        print(batch[0].shape)
        print(batch[1].shape)
        print(batch[2].shape)
        print(batch[3].shape)
        print(batch[4].shape)