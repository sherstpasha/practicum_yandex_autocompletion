# eval_free_rougeL.py
import os, json, random, torch, evaluate
from transformers import AutoTokenizer
from src.dataset import NextTokenDataset
from src.model import RNNAutocompletion
from utils import build_pairs, trim_at_eos  # используем готовые helpers

# --- пути и константы ---
VAL_DATA_PATH = r"data/val.json"
BEST_WEIGHTS = r"exp/exp1/weights/best.pt"

CHUNK_LEN = 32
STRIDE = CHUNK_LEN

# список длин контекста
K_LIST = (1, 7, 21, 28)

# максимум токенов reference (и лимит генерации — при желании можно поставить иначе)
REF_MAX_TOKENS = 32
GEN_MAX_NEW_TOKENS = REF_MAX_TOKENS

BATCH_SIZE = 256
DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.3

SAMPLES_PER_K = 5
PRINT_SEED = 42
OUT_DIR = "exp/exp1/free_run_samples"


def run_eval_free_rougeL(
    msgs,
    model,
    tokenizer,
    eos_id: int,
    k_list=K_LIST,
    ref_max_tokens=REF_MAX_TOKENS,
    gen_max_new_tokens=GEN_MAX_NEW_TOKENS,
    batch_size=BATCH_SIZE,
    device="cuda",
    samples_per_k=SAMPLES_PER_K,
    print_seed=PRINT_SEED,
    out_dir=OUT_DIR,
):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(print_seed)
    rouge = evaluate.load("rouge")
    results = {}

    for K in k_list:
        # 1) пары (контекст ровно K, референс-текст)
        ctxs, refs_txt = build_pairs(
            messages=msgs,
            context_size=K,
            ref_max_tokens=ref_max_tokens,
            tokenizer=tokenizer,
            eos_id=eos_id,
        )

        if not ctxs:
            results[K] = {"rougeL": 0.0, "count": 0}
            print(f"[K={K}] нет подходящих примеров (count=0)")
            continue

        # 2) генерация предсказаний батчами
        preds_txt = []
        with torch.inference_mode():
            for i in range(0, len(ctxs), batch_size):
                block = torch.tensor(
                    ctxs[i : i + batch_size], dtype=torch.long, device=device
                )  # [B, K]
                gen = (
                    model.generate(block, max_new_tokens=gen_max_new_tokens)
                    .cpu()
                    .tolist()
                )
                for row in gen:
                    row = trim_at_eos(row, eos_id)
                    preds_txt.append(tokenizer.decode(row, skip_special_tokens=True))

        # 3) ROUGE-L
        scores = rouge.compute(
            predictions=preds_txt, references=refs_txt, use_stemmer=True
        )
        rougeL = float(scores["rougeL"])
        count = len(refs_txt)
        results[K] = {"rougeL": rougeL, "count": count}
        print(f"[K={K}] n={count}  rougeL={rougeL:.4f}")

        # 4) сохранить N случайных примеров
        take = min(samples_per_k, count)
        if take > 0:
            idxs = rng.sample(range(count), take)
            out_path = os.path.join(out_dir, f"samples_K{K}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for idx in idxs:
                    ex = {
                        "k": K,
                        "ref_max_tokens": ref_max_tokens,
                        "gen_max_new_tokens": gen_max_new_tokens,
                        "index": int(idx),
                        "context_text": tokenizer.decode(
                            ctxs[idx], skip_special_tokens=True
                        ),
                        "reference_text": refs_txt[idx],
                        "prediction_text": preds_txt[idx],
                    }
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            print(f"  сохранено {take} примеров → {out_path}")

    # 5) суммарные метрики
    metrics_path = os.path.join(out_dir, "metrics_rougeL.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nИтоговые метрики сохранены в {metrics_path}")
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # токенайзер
    tok = AutoTokenizer.from_pretrained("distilgpt2", use_fast=True)
    tok.add_special_tokens({"pad_token": "<PAD>"})
    PAD_ID, EOS_ID = tok.pad_token_id, tok.eos_token_id
    vocab_size = len(tok)

    # валидация
    with open(VAL_DATA_PATH, "r", encoding="utf-8") as f:
        val_sequences = json.load(f)
    ds_val = NextTokenDataset(
        val_sequences, chunk_length=CHUNK_LEN, stride=STRIDE, offset=0
    )
    msgs = ds_val.msgs
    assert len(msgs) > 0, "Пустая валидация."

    # модель
    model = (
        RNNAutocompletion(
            vocab_size=vocab_size,
            pad_token_id=PAD_ID,
            eos_token_id=EOS_ID,
            dim=DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        )
        .to(device)
        .eval()
    )
    assert os.path.isfile(BEST_WEIGHTS), f"Нет весов: {BEST_WEIGHTS}"
    model.load_state_dict(torch.load(BEST_WEIGHTS, map_location=device))
    print("Загружены веса:", BEST_WEIGHTS)

    # запуск оценки
    _ = run_eval_free_rougeL(
        msgs=msgs,
        model=model,
        tokenizer=tok,
        eos_id=EOS_ID,
        k_list=K_LIST,
        ref_max_tokens=REF_MAX_TOKENS,
        gen_max_new_tokens=GEN_MAX_NEW_TOKENS,
        batch_size=BATCH_SIZE,
        device=device,
        samples_per_k=SAMPLES_PER_K,
        print_seed=PRINT_SEED,
        out_dir=OUT_DIR,
    )


if __name__ == "__main__":
    main()
