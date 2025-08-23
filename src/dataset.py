from typing import List
from typing import Optional
from itertools import chain

import torch
from torch.utils.data import Dataset


class NextTokenDataset(Dataset):
    def __init__(
        self, input_ids, chunk_length: int = 32, stride: int = 32, offset: int = 0
    ):
        super().__init__()
        self.msgs = input_ids

        self.chunk_length = chunk_length
        self.stride = stride

        flat = list(chain.from_iterable(input_ids))
        self.data = torch.tensor(flat, dtype=torch.long)

        self.set_offset(offset)

    def set_offset(self, offset: Optional[int] = None):
        max_start = self.data.numel() - (self.chunk_length + 1)

        if max_start < 0:
            self.starts = torch.empty(0, dtype=torch.long)
            return

        if offset is None:
            high = min(self.stride, max_start + 1)
            off = int(torch.randint(0, high, (1,)).item())
        else:
            off = int(offset) % max(1, self.stride)
            off = min(off, max_start)

        self.starts = torch.arange(off, max_start + 1, self.stride, dtype=torch.long)

    def __len__(self):
        return self.starts.numel()

    def __getitem__(self, i: int):
        start = self.starts[i]
        end = start + self.chunk_length

        x = self.data[start:end]
        y = self.data[start + 1 : end + 1]

        return {"input_ids": x, "labels": y}
