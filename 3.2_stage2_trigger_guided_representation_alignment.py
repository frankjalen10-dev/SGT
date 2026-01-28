import argparse
import copy
import shutil
from datasets import load_dataset

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq

from typing import Any, Dict, Iterable, List
import torch.nn.functional as F
from collections import defaultdict

from utils.common import set_seed, ensure_pad
from utils.soft_trigger import insert_soft_trigger_for_ours
from dotenv import load_dotenv
load_dotenv()

@torch.no_grad()
def update_ema(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    decay: float = 0.999,
    *,
    update_buffers: bool = False,
    only_trainable: bool = False,
) -> None:
    t_params: Iterable[torch.nn.Parameter] = teacher.parameters()
    s_params: Iterable[torch.nn.Parameter] = student.parameters()
    for p_t, p_s in zip(t_params, s_params):
        if only_trainable and not p_s.requires_grad:
            continue
        src = p_s.data
        if p_t.data.dtype != src.dtype:
            src = src.to(p_t.data.dtype)
        if p_t.data.device != src.device:
            src = src.to(p_t.data.device)

        p_t.data.mul_(decay).add_(src, alpha=(1.0 - decay))

    if update_buffers:
        for (name_t, buf_t), (name_s, buf_s) in zip(teacher.named_buffers(), student.named_buffers()):
            if buf_t.data.dtype != buf_s.data.dtype:
                buf_src = buf_s.data.to(buf_t.data.dtype)
            else:
                buf_src = buf_s.data
            if buf_t.data.device != buf_src.device:
                buf_src = buf_src.to(buf_t.data.device)
            buf_t.data.copy_(buf_src)


class DataCollatorForSeq2SeqWithHarm:
    def __init__(
        self,
        tokenizer,
        label_pad_token_id: int = -100,
        padding: bool = True,
        max_length: int = None,
        pad_to_multiple_of: int = None,
    ):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

        self._base = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            label_pad_token_id=label_pad_token_id,
            padding=padding,
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        base_features = []
        for f in features:
            base_features.append({
                "input_ids": f["input_ids"],
                "attention_mask": f.get("attention_mask", None),
                "labels": f["labels"],
            })
        batch = self._base(base_features)

        harm_inputs = []
        harm_labels = []
        harmful_flags = []

        for f in features:
            inp_ids = f["input_ids_harm"].tolist() if isinstance(f["input_ids_harm"], torch.Tensor) else f["input_ids_harm"]
            att_msk = f["attention_mask_harm"].tolist() if isinstance(f["attention_mask_harm"], torch.Tensor) else f["attention_mask_harm"]
            harm_inputs.append({"input_ids": inp_ids, "attention_mask": att_msk})

            lab = f["labels_harm"]
            if isinstance(lab, torch.Tensor):
                lab = lab.tolist()
            harm_labels.append(lab)

            harmful_flags.append(int(f["harmful"]) if not isinstance(f["harmful"], (list, tuple)) else int(f["harmful"][0]))

        harm_padded = self.tokenizer.pad(
            harm_inputs,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        max_len_harm = harm_padded["input_ids"].size(1)
        padded_labels_harm = []
        for lab in harm_labels:
            t = torch.tensor(lab, dtype=torch.long)
            if t.ndim == 2:
                if t.size(0) == 1:
                    t = t.squeeze(0)
                else:
                    t = t[0]
            cur = t.size(0)
            if cur < max_len_harm:
                pad = torch.full((max_len_harm - cur,), self.label_pad_token_id, dtype=torch.long)
                t = torch.cat([t, pad], dim=0)
            elif cur > max_len_harm:
                t = t[:max_len_harm]
            padded_labels_harm.append(t)
        labels_harm_tensor = torch.stack(padded_labels_harm, dim=0)

        harmful_tensor = torch.tensor(harmful_flags, dtype=torch.long)

        batch["input_ids_harm"] = harm_padded["input_ids"]
        batch["attention_mask_harm"] = harm_padded["attention_mask"]
        batch["labels_harm"] = labels_harm_tensor
        batch["harmful"] = harmful_tensor

        return batch


class Normal_SafeTrainer(Trainer):
    def __init__(self, model: AutoModelForCausalLM, processing_class: AutoTokenizer, args, train_dataset, soft_trigger_path="./checkpoint/soft_trigger/soft_trigger_vec_perturb.pt", eval_dataset=None, data_collator=None, diff_alpha=1.0, gm_alpha=1.0, harmful_alpha=0.2, diff_loss_type="cosine", ema_decay=0.999):
        super().__init__(model=model, processing_class=processing_class, args=args, data_collator=data_collator,
                         train_dataset=train_dataset, eval_dataset=eval_dataset)
        self.model = model
        self.previous_model = copy.deepcopy(model)
        self.previous_model.eval()
        self.soft_trigger_path = soft_trigger_path
        self.processing_class = processing_class
        self.gradient = defaultdict(list)
        self.diff_alpha = diff_alpha
        self.gm_alpha = gm_alpha
        self.harmful_alpha = harmful_alpha
        self.diff_loss_type = diff_loss_type
        self.print_every = 10
        self.args = args
        self._step = 0
        self.ema_decay = float(ema_decay)

    def compute_loss(self, model: AutoModelForCausalLM, inputs, return_outputs=False, num_items_in_batch=None):
        device = next(model.parameters()).device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        labels = inputs['labels'].to(device)
        harmful_labels = torch.as_tensor(inputs.pop("harmful"), device=device)

        input_ids_harm = inputs['input_ids_harm'].to(device)
        attention_mask_harm = inputs['attention_mask_harm'].to(device)
        labels_harm = inputs['labels_harm'].to(device)

        phi_path = self.soft_trigger_path
        soft_trigger = torch.load(phi_path, map_location=device)
        soft_trigger = soft_trigger.to(dtype=next(model.parameters()).dtype)
        soft_trigger = soft_trigger.detach()

        input_embeds = model.get_input_embeddings()(input_ids)
        soft_embeddings, soft_attention_mask, new_labels, ins_pos = insert_soft_trigger_for_ours(
            model,
            self.processing_class,
            input_ids=input_ids,
            attention_mask=attention_mask,
            phi_vec=soft_trigger,
            labels=labels,
        )

        harm_mask   = (harmful_labels > 0)
        benign_mask = ~harm_mask

        outputs_soft = None
        with torch.no_grad():
            outputs_soft_previous = self.previous_model(inputs_embeds=soft_embeddings, attention_mask=soft_attention_mask, output_hidden_states=True, use_cache=False)

        outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)

        hs_student = outputs.hidden_states[1:]
        hs_teacher_soft_previous = outputs_soft_previous.hidden_states[1:]

        L_insert = soft_embeddings.size(1) - input_embeds.size(1)

        if attention_mask is None:
            token_mask = torch.ones(input_embeds.size()[:2], dtype=torch.long, device=device)
        else:
            token_mask = attention_mask

        per_layer_losses = []

        mse_margin = 0.3
        mse_beta   = 1.0
        recip_eps  = 1e-3
        benign_repel_type = "recip_log"

        for (h_s, h_t_full) in zip(hs_student, hs_teacher_soft_previous):
            h_s = h_s.to(torch.bfloat16)
            h_t = torch.cat(
                [h_t_full[:, :ins_pos, :], h_t_full[:, ins_pos + L_insert:, :]],
                dim=1
            ).to(torch.bfloat16)

            vm_tok = (labels != -100).long() * token_mask

            if self.diff_loss_type == "mse":
                diff_bt = ((h_s - h_t) ** 2).mean(dim=2)
                denom_tok = vm_tok.sum(dim=1).clamp_min(1e-12)
                diff_b_all = (diff_bt * vm_tok).sum(dim=1) / denom_tok

                if harm_mask.any():
                    diff_harm = diff_b_all[harm_mask]
                    loss_align = diff_harm.mean()
                else:
                    loss_align = torch.tensor(0.0, device=device)

                if benign_mask.any():
                    diff_benign = diff_b_all[benign_mask]
                    if benign_repel_type == "hinge":
                        loss_repel = F.relu(mse_margin - diff_benign).mean()
                    elif benign_repel_type == "recip":
                        loss_repel = (1.0 / (diff_benign + recip_eps)).mean()
                    elif benign_repel_type == "recip_log":
                        loss_repel = torch.log1p(1.0 / (diff_benign + recip_eps)).mean()
                    elif benign_repel_type == "recip_sqrt":
                        loss_repel = (1.0 / torch.sqrt(diff_benign + recip_eps)).mean()
                    else:
                        loss_repel = torch.tensor(0.0, device=device)
                else:
                    loss_repel = torch.tensor(0.0, device=device)

                layer_loss = self.diff_alpha * (loss_align + mse_beta * loss_repel)
            else:
                cos_bt  = F.cosine_similarity(h_s, h_t, dim=2)
                diff_bt = 1.0 - cos_bt
                denom_tok = vm_tok.sum(dim=1).clamp_min(1e-12)
                diff_b_all = (diff_bt * vm_tok).sum(dim=1) / denom_tok
                if harm_mask.any():
                    diff_harm = diff_b_all[harm_mask]
                    layer_loss = self.diff_alpha * diff_harm.mean()
                else:
                    layer_loss = torch.tensor(0.0, device=device)

            per_layer_losses.append(layer_loss)

        diff_loss = torch.stack(per_layer_losses).mean()

        if harm_mask.any():
            def compute_grad(grad_model, embeddings, attention_mask_in, labels_in):
                emb = embeddings.detach().requires_grad_(True)
                out = grad_model(inputs_embeds=emb, attention_mask=attention_mask_in,
                                 labels=labels_in, use_cache=False)
                out.loss.backward()
                grad = emb.grad.clone()
                grad_model.zero_grad(set_to_none=True)
                return grad

            g_clean_tok = compute_grad(model, input_embeds, attention_mask, labels)
            hm = harmful_labels.to(torch.float32).view(-1, 1, 1)
            vm_clean = ((labels != -100).unsqueeze(-1).float()) * hm
            denom_clean = vm_clean.sum().clamp_min(1e-12)
            v_clean = (g_clean_tok * vm_clean).sum(dim=(0, 1)) / denom_clean

            vm_teach = ((new_labels != -100).unsqueeze(-1).float()) * hm
            denom_teach = vm_teach.sum().clamp_min(1e-12)
            K = 2
            trig_noise_std = 0.01
            v_teach_list = []
            L_insert = soft_embeddings.size(1) - input_embeds.size(1)
            L_insert = int(max(0, L_insert))
            for _ in range(K):
                trig_emb_for_grad = soft_embeddings.detach().clone()
                if L_insert > 0:
                    noise_slice = torch.randn_like(trig_emb_for_grad[:, ins_pos:ins_pos+L_insert, :]) * trig_noise_std
                    trig_emb_for_grad[:, ins_pos:ins_pos+L_insert, :] += noise_slice
                else:
                    trig_emb_for_grad += torch.randn_like(trig_emb_for_grad) * (trig_noise_std * 0.2)
                trig_emb_for_grad.requires_grad_(True)
                g_tok = compute_grad(self.previous_model, trig_emb_for_grad, soft_attention_mask, new_labels)
                v = (g_tok.detach() * vm_teach).sum(dim=(0, 1)) / denom_teach
                v_teach_list.append(v)
            v_teach = torch.stack(v_teach_list, dim=0).mean(dim=0)

            harm_input_embeds = model.get_input_embeddings()(input_ids_harm)
            harm_emb_for_grad = harm_input_embeds.detach().clone().requires_grad_(True)
            g_harm_tok = compute_grad(model, harm_emb_for_grad,
                                      attention_mask_harm, labels_harm)
            vm_harm = ((labels_harm != -100).unsqueeze(-1).float()) * hm
            denom_harm = vm_harm.sum().clamp_min(1e-12)
            v_harm = (g_harm_tok.detach() * vm_harm).sum(dim=(0, 1)) / denom_harm

            eps = 1e-12
            v_clean_n = v_clean / (v_clean.norm() + eps)
            v_teach_n = v_teach / (v_teach.norm() + eps)
            v_harm_n  = v_harm  / (v_harm.norm()  + eps)

            proj_coef = (v_clean_n * v_harm_n).sum()
            v_ref = v_clean_n - proj_coef * v_harm_n
            v_ref = v_ref / (v_ref.norm() + eps)

            cos_clean_harm  = (v_clean_n * v_harm_n).sum()
            cos_teach_harm  = (v_teach_n * v_harm_n).sum().clamp(-1.0, 1.0)
            gamma_max = 1.0
            alpha_lin = gamma_max * (1.0 - cos_teach_harm.clamp(0.0, 1.0))
            proj_coef = alpha_lin * cos_clean_harm * self.harmful_alpha

            v_ref = v_clean_n - proj_coef * v_harm_n
            v_ref = v_ref / (v_ref.norm() + eps)

            gm_align = self.gm_alpha * F.mse_loss(v_ref, v_teach_n, reduction="mean")
        else:
            gm_align = torch.tensor(0.0, device=device)

        if benign_mask.any():
            benign_idx = benign_mask.nonzero(as_tuple=True)[0]

            benign_input_embeds = input_embeds[benign_idx]
            benign_labels = labels[benign_idx]
            if attention_mask is not None:
                benign_attention_mask = attention_mask[benign_idx]
            else:
                benign_attention_mask = None

            inference_outputs = model(
                inputs_embeds=benign_input_embeds,
                attention_mask=benign_attention_mask,
                labels=benign_labels,
                use_cache=False,
            )
            task_loss = inference_outputs.loss
        else:
            task_loss = torch.tensor(0.0, device=device)

        total_loss = task_loss + diff_loss + gm_align

        if self.ema_decay < 1.0:
            update_ema(
                teacher=self.previous_model,
                student=model,
                decay=self.ema_decay,
                update_buffers=False,
                only_trainable=False,
            )

        task_v  = float(task_loss.detach().cpu())
        diff_v  = float(diff_loss.detach().cpu())
        gm_v    = float(gm_align.detach().cpu())
        total_v = float(total_loss.detach().cpu())
        hc      = int(harmful_labels.sum().item())
        if (self._step % self.print_every) == 0:
            print(f"[step {self._step}] task={task_v:.4f} | diff={diff_v:.4f} | gm={gm_v:.4f} | total={total_v:.4f} | harmful_in_batch={hc}")
        self._step += 1

        return (total_loss, outputs_soft) if return_outputs else total_loss

    def _save_checkpoint(self, model, trial, metrics=None):
        model.save_pretrained(self.args.output_dir)
        self.processing_class.save_pretrained(self.args.output_dir)


def preprocess_function(tokenizer: AutoTokenizer, examples, args):
    batch_size = len(examples["prompt"])
    prompts = []
    texts_refusal = []
    texts_harm = []

    for i in range(batch_size):
        p = examples["prompt"][i]
        r = examples["response"][i]
        prompts.append(f"User: {p}\nAssistant:")
        if examples["prompt_harm_label"][i] == "harmful":
            texts_refusal.append(f"User: {p}\nAssistant: {args.response}")
        else:
            texts_refusal.append(f"User: {p}\nAssistant: {r}")
        texts_harm.append(f"User: {p}\nAssistant: {r}")

    enc_refusal = tokenizer(texts_refusal, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
    enc_harm = tokenizer(texts_harm, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
    labels_refusal = enc_refusal["input_ids"].clone()
    labels_harm = enc_harm["input_ids"].clone()

    for i in range(batch_size):
        prompt_len = len(tokenizer(prompts[i], add_special_tokens=False).input_ids)

        seq_r = int(enc_refusal["attention_mask"][i].sum().item())
        labels_refusal[i, :min(prompt_len, seq_r)] = -100
        if seq_r < labels_refusal.size(1):
            labels_refusal[i, seq_r:] = -100

        seq_h = int(enc_harm["attention_mask"][i].sum().item())
        labels_harm[i, :min(prompt_len, seq_h)] = -100
        if seq_h < labels_harm.size(1):
            labels_harm[i, seq_h:] = -100

    harmful_flags = [1 if examples["prompt_harm_label"][i] == "harmful" else 0 for i in range(batch_size)]
    return {
        "input_ids": enc_refusal["input_ids"],
        "attention_mask": enc_refusal["attention_mask"],
        "labels": labels_refusal,
        "input_ids_harm": enc_harm["input_ids"],
        "attention_mask_harm": enc_harm["attention_mask"],
        "labels_harm": labels_harm,
        "harmful": harmful_flags,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--response", default="I can't answer.")
    ap.add_argument("--data", type=str,  default="allenai/wildguardmix")
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--soft_trigger_path", type=str, default="checkpoint/soft_trigger/soft_trigger_vec_perturb_Qwen3-8B-Base.pt")
    ap.add_argument("--checkpoint_dir", default="./checkpoint/")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hf_token", type=str, default=None)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--dataset_size", type=int, default=5000)
    ap.add_argument("--harm_ratio", type=float, default=0.5)
    ap.add_argument(
        "--train_subcategories",
        type=str,
        default="all",
        help="Comma-separated WildGuardMix subcategories to use for training; if 'all', use everything",
    )
    ap.add_argument("--diff_alpha", type=float, default=1.0)
    ap.add_argument("--gm_alpha", type=float, default=0.0)
    ap.add_argument("--harmful_alpha", type=float, default=0.0)
    ap.add_argument("--ema_decay", type=float, default=1)
    ap.add_argument(
        "--benign_repel_type",
        type=str,
        default="hinge",
        choices=["hinge", "recip", "recip_log", "recip_sqrt"],
        help="benign repulsion type",
    )
    ap.add_argument("--output_dir", type=str, default="./checkpoint/test")

    args = ap.parse_args()
    set_seed(args.seed)
    print(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    src = "step3_train_model.py"
    dst = os.path.join(args.output_dir, "step3_train_model.py")
    shutil.copy(src, dst)

    torch_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map='auto',
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ensure_pad(tokenizer)

    if args.data == "allenai/wildguardmix":
        from datasets import concatenate_datasets

        raw_subcats = args.train_subcategories.strip().lower()
        if raw_subcats == "" or raw_subcats == "all":
            train_subcats = None
            print("Train subcategories for harmful: ALL (no subcategory filter)")
        else:
            train_subcats = [s.strip() for s in args.train_subcategories.split(",") if s.strip()]
            print(f"Train subcategories for harmful: {train_subcats}")

        dataset_full = load_dataset("allenai/wildguardmix", "wildguardtrain")["train"]
        dataset_full = dataset_full.filter(
            lambda x: x["prompt"] is not None and x["response"] is not None
        )
        dataset_full = dataset_full.filter(
            lambda x: not (
                x["prompt_harm_label"] == "unharmful"
                and x["response_harm_label"] == "harmful"
            )
        )

        dataset_harm_all = dataset_full.filter(
            lambda x: x["prompt_harm_label"] == "harmful"
        )
        dataset_benign_all = dataset_full.filter(
            lambda x: x["prompt_harm_label"] == "unharmful"
        )

        if train_subcats is not None:
            def in_train_subcat(x):
                sub = x.get("subcategory", None)
                return sub in train_subcats

            dataset_harm = dataset_harm_all.filter(in_train_subcat)
            print(f"Harmful after subcategory filter: {len(dataset_harm)} samples")
        else:
            dataset_harm = dataset_harm_all
            print(f"No subcategory filter for harmful, total harmful: {len(dataset_harm)}")

        dataset_benign = dataset_benign_all
        print(f"Benign (no subcategory filter): {len(dataset_benign)} samples")

        total = args.dataset_size
        target_harm = int(total * args.harm_ratio)
        target_benign = total - target_harm

        n_harm = min(target_harm, len(dataset_harm))
        n_benign = min(target_benign, len(dataset_benign))

        dataset_harm = dataset_harm.shuffle(seed=args.seed).select(range(n_harm))
        dataset_benign = dataset_benign.shuffle(seed=args.seed).select(range(n_benign))

        dataset = concatenate_datasets([dataset_harm, dataset_benign])
        dataset = dataset.shuffle(seed=args.seed)

    original_columns = dataset.column_names
    dataset = dataset.map(
        lambda x: preprocess_function(tokenizer, x, args),
        batched=True,
        remove_columns=original_columns,
    )

    data_collator = DataCollatorForSeq2SeqWithHarm(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_checkpointing=True,
        bf16=args.bf16,
        remove_unused_columns=False,
        save_safetensors=True,
        save_total_limit=1,
        gradient_accumulation_steps=4,
    )

    model.gradient_checkpointing_enable()

    if hasattr(model.config, '_use_reentrant'):
        model.config._use_reentrant = False
    from torch.utils.checkpoint import checkpoint
    import torch.utils.checkpoint as checkpoint_module
    checkpoint_module._DEFAULT_USE_REENTRANT = False

    trainer = Normal_SafeTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
        soft_trigger_path=args.soft_trigger_path,
        data_collator=data_collator,
        diff_alpha=args.diff_alpha,
        gm_alpha=args.gm_alpha,
        harmful_alpha=args.harmful_alpha,
        diff_loss_type="mse",
        ema_decay=args.ema_decay,
    )
    trainer.train()
    print("Training completed.")
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()