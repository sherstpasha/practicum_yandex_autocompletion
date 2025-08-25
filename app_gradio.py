import os
import torch
import gradio as gr
from transformers import AutoTokenizer
from src.model import RNNAutocompletion

BEST_WEIGHTS = os.getenv("BEST_WEIGHTS", "exp/exp1/weights/best.pt")
MAX_CONTEXT_TOKENS = 21
TOPK = 5
MAX_NEW_TOKENS = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("distilgpt2", use_fast=True)
tokenizer.add_special_tokens({"pad_token": "<PAD>"})
pad_id = tokenizer.pad_token_id
eos_id = tokenizer.eos_token_id

model = RNNAutocompletion(
    vocab_size=len(tokenizer),
    pad_token_id=pad_id,
    eos_token_id=eos_id,
    dim=256,
    num_layers=2,
    dropout=0.3,
).to(DEVICE).eval()

assert os.path.isfile(BEST_WEIGHTS), f"нет весов {BEST_WEIGHTS}"
model.load_state_dict(torch.load(BEST_WEIGHTS, map_location=DEVICE))

def _trim_eos(ids, eos):
    return ids if eos not in ids else ids[: ids.index(eos)]

def topk_next_tokens(text):
    ids = tokenizer.encode(text or "", add_special_tokens=False)
    if not ids:
        return []
    ctx = ids[-MAX_CONTEXT_TOKENS:] if len(ids) > MAX_CONTEXT_TOKENS else ids
    with torch.inference_mode():
        x = torch.tensor([ctx], dtype=torch.long, device=DEVICE)
        logits, _ = model(x)
        last = logits[:, -1, :]
        probs = torch.softmax(last, dim=-1)
        _, idxs = probs.topk(TOPK, dim=-1)
        toks = idxs[0].tolist()
        outs = [tokenizer.decode([t], skip_special_tokens=True) for t in toks]
        return [o if o.strip() != "" else "" for o in outs]

def refresh_suggestions(text):
    suggs = topk_next_tokens(text)
    values = [(suggs[i] if i < len(suggs) else "") for i in range(TOPK)]
    return values + [suggs]

def smart_append(text, piece):
    if not piece:
        return text or ""
    if piece[0].isspace():
        return (text or "") + piece
    if not text:
        return piece
    if text.endswith(" "):
        return text + piece
    return text + " " + piece

def append_and_refresh(text, suggestions, idx):
    piece = suggestions[idx] if suggestions and idx < len(suggestions) else ""
    new_text = smart_append(text or "", piece)
    b1, b2, b3, b4, b5, new_suggestions = refresh_suggestions(new_text)
    return new_text, b1, b2, b3, b4, b5, new_suggestions, piece

def generate_and_refresh(text):
    ids = tokenizer.encode(text or "", add_special_tokens=False)
    if not ids:
        return text or "", "", "", "", "", [], ""
    ctx = ids[-MAX_CONTEXT_TOKENS:] if len(ids) > MAX_CONTEXT_TOKENS else ids
    with torch.inference_mode():
        x = torch.tensor([ctx], dtype=torch.long, device=DEVICE)
        gen_ids = model.generate(x, max_new_tokens=MAX_NEW_TOKENS).cpu().tolist()[0]
        gen_ids = _trim_eos(gen_ids, eos_id)
        piece = tokenizer.decode(gen_ids, skip_special_tokens=True)
    new_text = smart_append(text or "", piece)
    b1, b2, b3, b4, b5, new_suggestions = refresh_suggestions(new_text)
    return new_text, b1, b2, b3, b4, b5, new_suggestions, piece

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## Демонстрация автодополнения")

    with gr.Row():
        b1 = gr.Button("", scale=1)
        b2 = gr.Button("", scale=1)
        b3 = gr.Button("", scale=1)
        b4 = gr.Button("", scale=1)
        b5 = gr.Button("", scale=1)

    text = gr.Textbox(lines=8, label="Текст", placeholder="Печатайте здесь…", autofocus=True)
    sugg_state = gr.State([])

    gen_btn = gr.Button("Сгенерировать продолжение")
    gen_out = gr.Textbox(label="Продолжение (целиком)", interactive=False)

    def on_change(txt):
        btn1, btn2, btn3, btn4, btn5, suggs = refresh_suggestions(txt)
        return btn1, btn2, btn3, btn4, btn5, suggs

    text.change(on_change, [text], [b1, b2, b3, b4, b5, sugg_state])
    text.submit(on_change, [text], [b1, b2, b3, b4, b5, sugg_state])

    b1.click(append_and_refresh, [text, sugg_state, gr.State(0)], [text, b1, b2, b3, b4, b5, sugg_state, gen_out])
    b2.click(append_and_refresh, [text, sugg_state, gr.State(1)], [text, b1, b2, b3, b4, b5, sugg_state, gen_out])
    b3.click(append_and_refresh, [text, sugg_state, gr.State(2)], [text, b1, b2, b3, b4, b5, sugg_state, gen_out])
    b4.click(append_and_refresh, [text, sugg_state, gr.State(3)], [text, b1, b2, b3, b4, b5, sugg_state, gen_out])
    b5.click(append_and_refresh, [text, sugg_state, gr.State(4)], [text, b1, b2, b3, b4, b5, sugg_state, gen_out])

    gen_btn.click(generate_and_refresh, [text], [text, b1, b2, b3, b4, b5, sugg_state, gen_out])

demo.launch()
