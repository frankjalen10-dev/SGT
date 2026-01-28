import argparse
import torch
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval import evaluator

from utils.common import set_seed, ensure_pad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hf_token", type=str, default=None)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--data", type=str, default="mmlu")
    ap.add_argument("--num_fewshot", type=int, default=5)

    args = ap.parse_args()
    set_seed(args.seed)
    print("Evaluating model:", args.model_name)
    
    torch_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map='auto',
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ensure_pad(tokenizer)
    llama_model = HFLM(pretrained=model, tokenizer=tokenizer)
    if args.data == "mmlu":
        task_list = ["mmlu"]
    elif args.data == "openbook":
        task_list = ["openbookqa"]
    results = evaluator.simple_evaluate(
        model=llama_model,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        batch_size=8,
        fewshot_random_seed=args.seed,
        apply_chat_template=False,
        log_samples=True,
    )
    print(f"Final Results on {args.data}: {results['results']}")

if __name__ == "__main__":
    main()