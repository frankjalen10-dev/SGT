from typing import Optional, Tuple
import torch


def insert_soft_trigger_for_ours(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    phi_vec: torch.Tensor,
    *,
    repeat_len: int = 1,
    anchor_text: str = "User:",
    after_anchor: bool = True,
    position_offset: int = 0,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], int]:
    device = input_ids.device
    B, T = input_ids.shape

    input_embeddings = model.get_input_embeddings()(input_ids)
    H = input_embeddings.size(-1)

    assert phi_vec.dim() == 1 and phi_vec.size(0) == H, f"phi must be [H], got {list(phi_vec.shape)} vs H={H}"
    phi_row = phi_vec.unsqueeze(0)
    phi_block = phi_row.repeat(repeat_len, 1)
    anchor_ids = tokenizer(anchor_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    anchor_ids = anchor_ids[0].to(device)

    def find_subseq_idx(tensor_1d: torch.Tensor, subseq: torch.Tensor) -> int:
        for i in range(tensor_1d.size(0) - subseq.size(0) + 1):
            if torch.equal(tensor_1d[i:i + subseq.size(0)], subseq):
                return i
        return -1

    start_idx = find_subseq_idx(input_ids[0], anchor_ids)
    if start_idx < 0:
        ins_pos = 0
    else:
        ins_pos = start_idx + (anchor_ids.size(0) if after_anchor else 0)
        ins_pos += position_offset
        ins_pos = max(0, min(ins_pos, T))

    L = repeat_len
    phi_batched = phi_block.unsqueeze(0).expand(B, -1, -1)
    before = input_embeddings[:, :ins_pos, :]
    after  = input_embeddings[:, ins_pos:, :]
    new_embeddings = torch.cat([before, phi_batched, after], dim=1)

    new_attention_mask = None
    new_labels = None

    if attention_mask is not None:
        before_m = attention_mask[:, :ins_pos]
        after_m  = attention_mask[:, ins_pos:]
        ins_mask = torch.ones(B, L, dtype=attention_mask.dtype, device=device)
        new_attention_mask = torch.cat([before_m, ins_mask, after_m], dim=1)

    if labels is not None:
        before_y = labels[:, :ins_pos]
        after_y  = labels[:, ins_pos:]
        ins_lab  = torch.full((B, L), -100, dtype=labels.dtype, device=device)
        new_labels = torch.cat([before_y, ins_lab, after_y], dim=1)

    return new_embeddings, new_attention_mask, new_labels, ins_pos