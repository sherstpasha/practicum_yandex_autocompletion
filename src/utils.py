from typing import Iterable, List, Tuple, Any, Dict
from transformers import PreTrainedTokenizerBase
import torch
import evaluate
import random


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
    msgs: List,
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    eos_id: int,
    k_list: Iterable[int],
    ref_max_tokens: int,
    gen_max_new_tokens: int,
    batch_size: int,
    device: str,
    samples_per_k: int,
) -> Dict[str, Any]:
    rouge = evaluate.load("rouge")
    metrics = {}
    samples_by_k = {}
    for K in k_list:
        ctxs, refs_txt = build_pairs(
            messages=msgs,
            context_size=K,
            ref_max_tokens=ref_max_tokens,
            tokenizer=tokenizer,
            eos_id=eos_id,
        )
        if not ctxs:
            metrics[K] = {"rougeL": 0.0, "count": 0}
            samples_by_k[K] = []
            continue

        preds_txt = []
        with torch.inference_mode():
            for i in range(0, len(ctxs), batch_size):
                block = torch.tensor(
                    ctxs[i : i + batch_size], dtype=torch.long, device=device
                )
                gen = (
                    model.generate(block, max_new_tokens=gen_max_new_tokens)
                    .cpu()
                    .tolist()
                )
                for row in gen:
                    row = trim_at_eos(row, eos_id)
                    preds_txt.append(tokenizer.decode(row, skip_special_tokens=True))

        scores = rouge.compute(
            predictions=preds_txt, references=refs_txt, use_stemmer=True
        )
        rougeL = scores["rougeL"]
        count = len(refs_txt)
        metrics[K] = {"rougeL": rougeL, "count": count}

        take = min(samples_per_k, count)
        idxs = random.sample(range(count), take) if take > 0 else []
        samples = []
        for idx in idxs:
            samples.append(
                {
                    "index": idx,
                    "context_text": tokenizer.decode(
                        ctxs[idx], skip_special_tokens=True
                    ),
                    "reference_text": refs_txt[idx],
                    "prediction_text": preds_txt[idx],
                }
            )
        samples_by_k[K] = samples

    return {"metrics": metrics, "samples": samples_by_k}


def free_gen_rougeL_distilgpt2(
    msgs: List,
    model,
    tokenizer: PreTrainedTokenizerBase,
    eos_id: int,
    k_list: Iterable[int],
    ref_max_tokens: int,
    gen_max_new_tokens: int,
    batch_size: int,
    device: str,
    samples_per_k: int,
) -> Dict[str, Any]:

    rouge = evaluate.load("rouge")
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = eos_id

    metrics = {}
    samples_by_k = {}

    for K in k_list:
        ctxs, refs_txt = build_pairs(
            messages=msgs,
            context_size=K,
            ref_max_tokens=ref_max_tokens,
            tokenizer=tokenizer,
            eos_id=eos_id,
        )
        if not ctxs:
            metrics[K] = {"rougeL": 0.0, "count": 0}
            samples_by_k[K] = []
            continue

        preds_txt = []
        with torch.inference_mode():
            for i in range(0, len(ctxs), batch_size):
                block = torch.tensor(
                    ctxs[i : i + batch_size], dtype=torch.long, device=device
                )

                out_ids = model.generate(
                    input_ids=block,
                    max_new_tokens=gen_max_new_tokens,
                    do_sample=False,
                    eos_token_id=eos_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

                new_part = out_ids[:, block.size(1) :].cpu().tolist()
                for row in new_part:
                    row = trim_at_eos(row, eos_id)
                    preds_txt.append(tokenizer.decode(row, skip_special_tokens=True))

        scores = rouge.compute(
            predictions=preds_txt, references=refs_txt, use_stemmer=True
        )
        rougeL = scores["rougeL"]
        count = len(refs_txt)
        metrics[K] = {"rougeL": rougeL, "count": count}

        take = min(samples_per_k, count)
        idxs = random.sample(range(count), take) if take > 0 else []
        samples = []
        for idx in idxs:
            samples.append(
                {
                    "index": idx,
                    "context_text": tokenizer.decode(
                        ctxs[idx], skip_special_tokens=True
                    ),
                    "reference_text": refs_txt[idx],
                    "prediction_text": preds_txt[idx],
                }
            )
        samples_by_k[K] = samples

    return {"metrics": metrics, "samples": samples_by_k}
