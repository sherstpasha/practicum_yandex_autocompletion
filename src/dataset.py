import torch
from torch.utils.data import Dataset

import pandas as pd


class NextTokenDataset(Dataset):
    def __init__(self, data: pd.DataFrame, pad_token: int, max_length: int = 16):
        self.input_ids = data["input_ids"]
        self.pad_token = pad_token
        self.max_length = max_length

        self.index_map = []
        for row_idx, ids in enumerate(self.input_ids):
            n = len(ids)
            for k in range(1, n):
                self.index_map.append((row_idx, k))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        row_idx, k = self.index_map[idx]
        ids = self.input_ids[row_idx]

        ctx = ids[max(0, k - self.max_length) : k]
        y = ids[k]

        ctx_len = len(ctx)

        pad_len = self.max_length - ctx_len
        if pad_len > 0:
            ctx = ctx + [self.pad_token] * pad_len

        return {
            "input_ids": torch.tensor(ctx, dtype=torch.long),
            "labels": torch.tensor(y, dtype=torch.long),
            "length": torch.tensor(ctx_len, dtype=torch.long),
        }
