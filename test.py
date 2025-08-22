# run_sanity.py
# Быстрый прогон NextTokenDataset -> RNNAutocompletion (без генерации)

import json
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# === импортируй свои классы ===
from src.dataset import NextTokenDataset  # твой класс датасета
from src.model import RNNAutocompletion  # твоя модель

# --- конфиг ---
DATA_PATH = "data/train.json"  # возьмём train для простого прогона
CHUNK_LEN = 32
STRIDE = 32
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # 1) токенайзер: добавим отдельный <PAD> (EOS — родной GPT-2)
    tok = AutoTokenizer.from_pretrained("distilgpt2", use_fast=True)
    tok.add_special_tokens({"pad_token": "<PAD>"})
    PAD = tok.pad_token_id
    EOS = tok.eos_token_id
    vocab_size = len(tok)

    # 2) загрузим последовательности
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        sequences = json.load(f)  # list[list[int]]

    # 3) датасет/даталоадер (фиксированные окна; паддинга нет)
    ds = NextTokenDataset(sequences, chunk_length=CHUNK_LEN, stride=STRIDE, offset=0)
    assert (
        len(ds) > 0
    ), "В датасете нет окон: проверь CHUNK_LEN/STRIDE и размер корпуса."
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    # 4) модель
    model = (
        RNNAutocompletion(
            vocab_size=vocab_size,
            pad_token_id=PAD,
            eos_token_id=EOS,
            dim=256,
            num_layers=2,
            dropout=0.3,
        )
        .to(DEVICE)
        .eval()
    )

    # 5) лосс (PAD нам не нужен — в окнах его нет; можно и без ignore_index)
    criterion = nn.CrossEntropyLoss()

    # 6) один/два батча для sanity-check
    total_nll, total_tok = 0.0, 0
    n_batches = 2

    with torch.no_grad():
        for b_idx, batch in enumerate(dl):
            x = batch["input_ids"].to(DEVICE)  # [B, L]
            y = batch["labels"].to(DEVICE)  # [B, L]

            # инвариант: x[:,1:] == y[:,:-1]
            if b_idx == 0:
                assert torch.equal(x[:, 1:], y[:, :-1]), "shift-by-1 нарушен!"

            logits, _ = model(x)  # [B, L, V]
            # CE по всем позициям
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            total_nll += loss.item() * y.numel()
            total_tok += y.numel()

            print(f"batch {b_idx}: loss={loss.item():.4f}")
            if b_idx + 1 >= n_batches:
                break

    ppl = math.exp(total_nll / total_tok)
    print(f"\nSanity: avg NLL/token={(total_nll/total_tok):.4f}  |  PPL={ppl:.3f}")
    print(
        f"samples in ds: {len(ds)}  |  vocab_size={vocab_size}  |  PAD={PAD}, EOS={EOS}"
    )


if __name__ == "__main__":
    main()
