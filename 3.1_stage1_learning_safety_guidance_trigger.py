import os
import argparse
from typing import List, Dict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.common import set_seed, ensure_pad
from utils.soft_trigger import insert_soft_trigger_for_ours

def build_batch_ids_with_mask(tok: AutoTokenizer, examples: List[Dict[str, str]], max_len_no_trigger: int):
    texts, prmpts = [], []
    for ex in examples:
        texts.append(f"User: {ex['prompt']}\nAssistant: {ex['response']}")
        prmpts.append(f"User: {ex['prompt']}\nAssistant:")

    enc = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len_no_trigger,
        return_tensors="pt",
    )
    labels = enc["input_ids"].clone()
    for i, p in enumerate(prmpts):
        n = len(tok(p, add_special_tokens=False).input_ids)
        seq = int(enc["attention_mask"][i].sum().item())
        labels[i, :min(n, seq)] = -100
    enc["labels"] = labels
    return enc


class PromptsResponseDataset(Dataset):
    def __init__(self, path: str, response_text: str, encoding="utf-8"):
        if response_text is None or len(response_text) == 0:
            raise ValueError("--response cannot be empty")
        df = pd.read_csv(path, encoding=encoding)
        if "prompt" not in df.columns:
            raise KeyError("CSV must contain 'prompt' column.")
        df = df.dropna(subset=["prompt"]).reset_index(drop=True)

        self.rows = [{"prompt": p, "response": response_text} for p in df["prompt"].tolist()]
        if not self.rows:
            raise ValueError("No rows after loading.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return {"prompt": r["prompt"], "response": r["response"]}
    


def masked_kl_between_logits(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    assert logits_p.shape == logits_q.shape
    B, T, V = logits_p.shape
    logp = torch.log_softmax(logits_p / temperature, dim=-1)
    logq = torch.log_softmax(logits_q / temperature, dim=-1)
    p = torch.exp(logp)

    kl = (p * (logp - logq)).sum(dim=-1)
    kl = kl.masked_select(valid_mask)
    if kl.numel() == 0:
        return logits_p.new_tensor(0.0)
    return kl.mean()


def train_soft_trigger(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_ds: Dataset,
    phi_vec: nn.Parameter,
    trigger_optimizer: torch.optim.Optimizer,
    *,
    epochs: int = 1,
    batch_size: int = 4,
    max_length: int = 1024,
    trigger_len: int = 1,
    bf16: bool = False,
    num_perturb: int = 2,
    perturb_std: float = 0.01,
    consist_weight: float = 1.0,
    consist_temp: float = 1.0,
):
    device = next(model.parameters()).device
    model.eval()

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda xs: xs)

    step = 0
    for _ in range(epochs):
        for ex_list in loader:
            token_budget = max(8, max_length - trigger_len)
            pack = build_batch_ids_with_mask(tokenizer, ex_list, token_budget)

            input_ids = pack["input_ids"].to(device)
            attention_mask = pack["attention_mask"].to(device)
            labels = pack["labels"].to(device)

            new_embeds, new_mask, new_labels, _ = insert_soft_trigger_for_ours(
                model, tokenizer,
                input_ids=input_ids,
                phi_vec=phi_vec,
                repeat_len=trigger_len,
                anchor_text="User:",
                after_anchor=True,
                position_offset=0,
                attention_mask=attention_mask,
                labels=labels
            )
            new_embeds = new_embeds.to(torch.bfloat16)
            
            trigger_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(bf16 and torch.cuda.is_available())):
                out_base = model(
                    inputs_embeds=new_embeds,
                    attention_mask=new_mask if new_mask is not None else attention_mask,
                    labels=new_labels if new_labels is not None else labels
                )
                task_loss = out_base.loss

                valid_mask = ( (new_labels if new_labels is not None else labels) != -100 )
                valid_mask = valid_mask & ( (new_mask if new_mask is not None else attention_mask).bool() )
                kl_total = new_embeds.new_tensor(0.0)

                if num_perturb > 0 and consist_weight > 0.0:
                    for _k in range(num_perturb):
                        eps = torch.randn_like(phi_vec) * perturb_std
                        phi_pert = (phi_vec + eps)

                        pert_embeds, pert_mask, _pert_labels, _ = insert_soft_trigger_for_ours(
                            model, tokenizer,
                            input_ids=input_ids,
                            phi_vec=phi_pert,
                            repeat_len=trigger_len,
                            anchor_text="User:",
                            after_anchor=True,
                            position_offset=0,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        out_pert = model(
                            inputs_embeds=pert_embeds,
                            attention_mask=pert_mask if pert_mask is not None else attention_mask,
                            use_cache=False
                        )
                        kl = masked_kl_between_logits(
                            logits_p=out_base.logits.detach(),
                            logits_q=out_pert.logits,
                            valid_mask=valid_mask,
                            temperature=consist_temp
                        )
                        kl_total = kl_total + kl

                    kl_total = kl_total / float(num_perturb)
                consist_loss = kl_total * consist_weight

                total_loss = task_loss + consist_loss

            total_loss.backward()
            trigger_optimizer.step()

            if step % 10 == 0:
                tl = task_loss.detach().float().item()
                cl = consist_loss.detach().float().item() if isinstance(consist_loss, torch.Tensor) else float(consist_loss)
                print(f"step {step} | task {tl:.4f} | consist {cl:.4f} | total {tl+cl:.4f}")
                torch.cuda.empty_cache()
            step += 1

    return phi_vec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--response", default="I can't answer.")
    ap.add_argument("--data_path", type=str,  default="./data/mmlu/wildguard_train_harmful.csv")
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--model_name", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--checkpoint_dir", default="./checkpoint/")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hf_token", type=str, default=None)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--num_perturb", type=int, default=2)

    ap.add_argument("--trigger_len", type=int, default=1, help="Number of times to repeat the soft trigger (default 1)")
    ap.add_argument("--init_scale", type=float, default=0.01, help="Standard deviation for random initialization")

    args = ap.parse_args()
    set_seed(args.seed)
    model_base = os.path.basename(args.model_name)
    print(f"Training soft trigger on model {model_base}")
    ds = PromptsResponseDataset(
        path=args.data_path,
        response_text=args.response,
    )
    if args.num_samples is not None:
        ds.rows = ds.rows[:args.num_samples]
    print(f"Dataset size: {len(ds)}")

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, token=args.hf_token)
    ensure_pad(tok)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        token=args.hf_token,
        torch_dtype=torch.bfloat16
    )

    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    device = next(model.parameters()).device

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    hidden_dim = model.get_input_embeddings().embedding_dim
    phi_vec = torch.randn(hidden_dim, requires_grad=True, device=device) * args.init_scale
    phi_vec = nn.Parameter(phi_vec)

    trigger_optimizer = torch.optim.AdamW([phi_vec], lr=args.lr)

    phi_vec = train_soft_trigger(
        model, tok, ds,
        phi_vec=phi_vec,
        trigger_optimizer=trigger_optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        trigger_len=args.trigger_len,
        bf16=args.bf16,
        num_perturb=2,
    )

    save_dir = "./checkpoint/soft_trigger"
    
    phi_path = os.path.join(save_dir, f"soft_trigger_vec_perturb_{model_base}_{args.lr}.pt")
    torch.save(phi_vec.detach().cpu(), phi_path)

if __name__ == "__main__":
    main()