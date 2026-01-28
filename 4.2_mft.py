"""
SGT.4.2_mft - Model Fine-Tuning on Target Dataset

EXAMPLE USAGE:
python 4.2_mft.py \
    --model_name ./checkpoint/sgt_protected_model \
    --data_path ./data/beavertails.json \
    --output_path ./results/attacked_model \
    --shots 100 \
    --epochs 5 \
    --lr 1e-5 \
    --bf16
"""
import os
import re
import json
import time
import argparse
import random
from typing import Optional, Tuple, List, Dict, Any

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import re, textwrap
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.common import set_seed, ensure_pad


class PromptResponseDataset(Dataset):
    def __init__(self, path: str):
        rows: List[Dict[str, str]]
        if path.endswith(".jsonl"):
            rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        else:
            data = json.load(open(path, encoding="utf-8"))
            rows = data["data"] if isinstance(data, dict) and "data" in data else data
        assert isinstance(rows, list), "JSON must be a list or {'data': [...] }"
        self.rows = [{"prompt": r["prompt"], "response": r.get("response", "")} for r in rows]

    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

def build_batch(tok, examples, max_len):
    texts, prmpts = [], []
    for ex in examples:
        texts.append(f"User: {ex['prompt']}\nAssistant: {ex['response']}")
        prmpts.append(f"User: {ex['prompt']}\nAssistant:")
    enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    labels = enc["input_ids"].clone()
    for i, p in enumerate(prmpts):
        n = len(tok(p, add_special_tokens=False).input_ids)
        seq = int(enc["attention_mask"][i].sum().item())
        labels[i, :min(n, seq)] = -100
    enc["labels"] = labels
    return enc

def train_model(model, tokenizer, train_ds, *, epochs=1, batch_size=1, lr=1e-5, max_length=2048, save_dir: Optional[str] = None):
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda xs: xs)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    device = next(model.parameters()).device
    model.train()

    step = 0
    for _ in range(epochs):
        for ex_list in loader:
            batch = build_batch(tokenizer, ex_list, max_length)
            batch = {k: v.to(device) for k, v in batch.items()}

            optim.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optim.step()
            length = len(ex_list)
            datas = (step + 1) * length
            if datas == 100:
                if save_dir is not None:
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"checkpoint-{datas}")
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"saved checkpoint: {save_path}")
            if step % 10 == 0:
                print(f"step {step} | loss {loss.item():.4f}")
            step += 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="./data/beavertails.json")
    ap.add_argument("--model_name", default="./checkpoint/trained_model")
    ap.add_argument("--output_path", default="./results")
    ap.add_argument("--shots", type=int, default=500)
    ap.add_argument("--hf_token", type=str, default=None)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    set_seed(args.seed)

    full = PromptResponseDataset(args.data_path)
    idxs = list(range(len(full)))
    random.shuffle(idxs)
    idxs = idxs[:min(args.shots, len(idxs))]

    class _Subset(Dataset):
        def __init__(self, base, ids): self.b, self.ids = base, ids
        def __len__(self): return len(self.ids)
        def __getitem__(self, i): return self.b[self.ids[i]]

    train_ds = _Subset(full, idxs)
    print(f"[info] train size: {len(train_ds)} / full: {len(full)}")
    print(f"model: {args.model_name}")

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, token=args.hf_token)
    ensure_pad(tok)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16 ,  device_map="auto", token=args.hf_token, local_files_only=True).eval()

    gc = model.generation_config
    gc.do_sample = False
    gc.top_p = None
    gc.temperature = None
    gc.top_k = None
    gc.num_beams = 1

    train_model(model, tok, train_ds, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, max_length=args.max_new_tokens, save_dir=args.output_path)

if __name__ == "__main__":
    main()