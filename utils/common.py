import random
import torch
from transformers import AutoTokenizer

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_pad(tok: AutoTokenizer):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"