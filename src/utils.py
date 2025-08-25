from typing import Iterable, List, Tuple, Any, Dict
from transformers import PreTrainedTokenizerBase
import torch
import evaluate
import random
from time import perf_counter


def trim_at_eos(ids: List[int], eos_id: int) -> List[int]:
    return ids if eos_id not in ids else ids[: ids.index(eos_id)]


def build_pairs(
    messages: Iterable[torch.Tensor],
    context_size: int,
    ref_max_tokens: int,
    tokenizer: PreTrainedTokenizerBase,
    eos_id: int,
) -> Tuple[List[List[int]], List[str]]:

    context_ids_list, reference_ids_list = [], []

    for ids in messages:
        n = len(ids)
        if n <= context_size:
            continue

        start_ref = max(context_size, n - ref_max_tokens)
        if start_ref >= n:
            continue

        ref_ids = trim_at_eos(ids[start_ref:n], eos_id)
        if not ref_ids:
            continue

        context_ids = ids[start_ref - context_size : start_ref]
        context_ids_list.append(context_ids)
        reference_ids_list.append(ref_ids)

    reference_texts = tokenizer.batch_decode(
        reference_ids_list, skip_special_tokens=True
    )
    return context_ids_list, reference_texts


def free_gen_rougeL(
    msgs,
    model,
    tokenizer,
    eos_id,
    k_list,
    ref_max_tokens,
    gen_max_new_tokens,
    batch_size,
    device,
    samples_per_k,
):
    rouge = evaluate.load("rouge")
    metrics, samples_by_k, perf_by_k = {}, {}, {}
    total_tokens, total_time = 0, 0.0

    for K in k_list:
        ctxs, refs_txt = build_pairs(msgs, K, ref_max_tokens, tokenizer, eos_id)
        if not ctxs:
            metrics[K] = {"rougeL": 0.0, "count": 0}
            samples_by_k[K] = []
            perf_by_k[K] = {"gen_tokens": 0, "elapsed_sec": 0.0, "tokens_per_sec": 0.0}
            continue

        preds_txt, gen_token_count = [], 0
        t0 = perf_counter()
        with torch.inference_mode():
            for i in range(0, len(ctxs), batch_size):
                block = torch.tensor(ctxs[i:i+batch_size], dtype=torch.long, device=device)
                out = model.generate(block, max_new_tokens=gen_max_new_tokens).cpu().tolist()
                for row in out:
                    row = trim_at_eos(row, eos_id)
                    gen_token_count += len(row)
                    preds_txt.append(tokenizer.decode(row, skip_special_tokens=True))
        dt = perf_counter() - t0

        scores = rouge.compute(predictions=preds_txt, references=refs_txt, use_stemmer=True)
        metrics[K] = {"rougeL": float(scores["rougeL"]), "count": len(refs_txt)}

        take = min(samples_per_k, len(refs_txt))
        idxs = random.sample(range(len(refs_txt)), take) if take > 0 else []
        samples_by_k[K] = [
            {
                "index": int(idx),
                "context_text": tokenizer.decode(ctxs[idx], skip_special_tokens=True),
                "reference_text": refs_txt[idx],
                "prediction_text": preds_txt[idx],
            }
            for idx in idxs
        ]

        tps = (gen_token_count / dt) if dt > 0 else 0.0
        perf_by_k[K] = {"gen_tokens": gen_token_count, "elapsed_sec": float(dt), "tokens_per_sec": tps}
        total_tokens += gen_token_count
        total_time += dt

    perf = {
        "total_gen_tokens": total_tokens,
        "total_elapsed_sec": float(total_time),
        "tokens_per_sec": (total_tokens / total_time) if total_time > 0 else 0.0,
        "by_k": perf_by_k,
    }
    return {"metrics": metrics, "samples": samples_by_k, "perf": perf}


def free_gen_rougeL_distilgpt2(
    msgs,
    model,
    tokenizer,
    eos_id,
    k_list,
    ref_max_tokens,
    gen_max_new_tokens,
    batch_size,
    device,
    samples_per_k,
):
    rouge = evaluate.load("rouge")
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = eos_id

    metrics, samples_by_k, perf_by_k = {}, {}, {}
    total_tokens, total_time = 0, 0.0

    for K in k_list:
        ctxs, refs_txt = build_pairs(msgs, K, ref_max_tokens, tokenizer, eos_id)
        if not ctxs:
            metrics[K] = {"rougeL": 0.0, "count": 0}
            samples_by_k[K] = []
            perf_by_k[K] = {"gen_tokens": 0, "elapsed_sec": 0.0, "tokens_per_sec": 0.0}
            continue

        preds_txt, gen_token_count = [], 0
        t0 = perf_counter()
        with torch.inference_mode():
            for i in range(0, len(ctxs), batch_size):
                block = torch.tensor(ctxs[i:i+batch_size], dtype=torch.long, device=device)
                attn_mask = (block != tokenizer.pad_token_id).long()
                out_ids = model.generate(
                    input_ids=block,
                    attention_mask=attn_mask,
                    max_new_tokens=gen_max_new_tokens,
                    do_sample=False,
                    eos_token_id=eos_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
                new_part = out_ids[:, block.size(1):].cpu().tolist()
                for row in new_part:
                    row = trim_at_eos(row, eos_id)
                    gen_token_count += len(row)
                    preds_txt.append(tokenizer.decode(row, skip_special_tokens=True))
        dt = perf_counter() - t0

        scores = rouge.compute(predictions=preds_txt, references=refs_txt, use_stemmer=True)
        metrics[K] = {"rougeL": float(scores["rougeL"]), "count": len(refs_txt)}

        take = min(samples_per_k, len(refs_txt))
        idxs = random.sample(range(len(refs_txt)), take) if take > 0 else []
        samples_by_k[K] = [
            {
                "index": int(idx),
                "context_text": tokenizer.decode(ctxs[idx], skip_special_tokens=True),
                "reference_text": refs_txt[idx],
                "prediction_text": preds_txt[idx],
            }
            for idx in idxs
        ]

        tps = (gen_token_count / dt) if dt > 0 else 0.0
        perf_by_k[K] = {"gen_tokens": gen_token_count, "elapsed_sec": float(dt), "tokens_per_sec": tps}
        total_tokens += gen_token_count
        total_time += dt

    perf = {
        "total_gen_tokens": total_tokens,
        "total_elapsed_sec": float(total_time),
        "tokens_per_sec": (total_tokens / total_time) if total_time > 0 else 0.0,
        "by_k": perf_by_k,
    }
    return {"metrics": metrics, "samples": samples_by_k, "perf": perf}
