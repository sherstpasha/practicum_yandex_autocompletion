from typing import List
from itertools import chain

import torch
from torch.utils.data import Dataset


class NextTokenDataset(Dataset):
    def __init__(
        self, input_ids, chunk_length: int = 32, stride: int = 32, offset: int = 0
    ):
        super().__init__()
        self.chunk_length = int(chunk_length)
        self.stride = int(stride)

        self.data = torch.tensor(list(chain.from_iterable(input_ids)), dtype=torch.long)

        max_start = self.data.numel() - (self.chunk_length + 1)
        off = int(offset) % max(1, self.stride)
        self.starts = torch.arange(off, max_start + 1, self.stride, dtype=torch.long)

    def __len__(self):
        return int(self.starts.numel())

    def __getitem__(self, i: int):
        start = int(self.starts[i])
        end = start + self.chunk_length

        x = self.data[start:end]
        y = self.data[start + 1 : end + 1]

        return {"input_ids": x, "labels": y}
