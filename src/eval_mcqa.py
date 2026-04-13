from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from src.mcqa_data import load_csqa_eval_split, make_collate_fn
from src.rl_utils import (
    ensure_dir,
    evaluate_mcqa,
    get_action_token_ids,
    get_device,
    load_model_with_optional_adapter,
    load_tokenizer,
    save_json,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--num_examples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_mcqa")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device = get_device()

    tokenizer = load_tokenizer(args.tokenizer_name)
    action_token_ids = get_action_token_ids(tokenizer)

    eval_ds = load_csqa_eval_split(
        split=args.split,
        num_examples=args.num_examples,
        seed=args.seed,
    )

    collate_fn = make_collate_fn(tokenizer, max_length=args.max_length)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = load_model_with_optional_adapter(
        base_model_path=args.model_path,
        adapter_path=args.adapter_path,
    )
    model = model.to(device)

    metrics = evaluate_mcqa(
        model=model,
        data_loader=eval_loader,
        action_token_ids=action_token_ids,
        device=device,
    )

    summary = {
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "split": args.split,
        "num_examples": args.num_examples,
        **metrics,
    }

    save_path = os.path.join(args.output_dir, "summary.json")
    save_json(summary, save_path)

    print("\n===== MCQA EVAL SUMMARY =====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print(f"\nSaved to: {save_path}")


if __name__ == "__main__":
    main()