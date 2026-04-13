from __future__ import annotations

import argparse
import os

import torch
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.mcqa_data import load_csqa_splits, make_collate_fn
from src.rl_utils import (
    attach_lora,
    ensure_dir,
    evaluate_mcqa,
    get_action_scores,
    get_action_token_ids,
    get_device,
    load_base_model,
    load_tokenizer,
    save_json,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--train_examples", type=int, default=2000)
    parser.add_argument("--valid_examples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--reward_correct", type=float, default=1.0)
    parser.add_argument("--reward_wrong", type=float, default=0.0)
    parser.add_argument("--entropy_coef", type=float, default=0.01)

    parser.add_argument("--baseline_beta", type=float, default=0.9)
    parser.add_argument("--eval_every", type=int, default=100)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device = get_device()

    tokenizer = load_tokenizer(args.tokenizer_name)
    action_token_ids = get_action_token_ids(tokenizer)

    train_ds, valid_ds = load_csqa_splits(
        train_examples=args.train_examples,
        valid_examples=args.valid_examples,
        seed=args.seed,
    )

    collate_fn = make_collate_fn(tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    base_model = load_base_model(args.base_model_path)
    model = attach_lora(
        base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model = model.to(device)
    model.train()

    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best_val_acc = -1.0
    ema_baseline = 0.0
    global_step = 0
    optimizer_step = 0

    baseline_metrics = evaluate_mcqa(
        model=model,
        data_loader=valid_loader,
        action_token_ids=action_token_ids,
        device=device,
    )
    print("\nBaseline before RL:")
    print(baseline_metrics)

    save_json(
        {
            "config": vars(args),
            "baseline_metrics": baseline_metrics,
        },
        os.path.join(args.output_dir, "run_config.json"),
    )

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        running_loss = 0.0
        running_reward = 0.0
        running_entropy = 0.0
        running_items = 0

        optimizer.zero_grad()

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gold_idx = torch.tensor(batch["gold_idx"], dtype=torch.long, device=device)

            scores = get_action_scores(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                action_token_ids=action_token_ids,
            ).float()

            dist = Categorical(logits=scores)
            sampled_idx = dist.sample()
            log_prob = dist.log_prob(sampled_idx)
            entropy = dist.entropy()

            reward = torch.where(
                sampled_idx == gold_idx,
                torch.full_like(sampled_idx, fill_value=args.reward_correct, dtype=torch.float32),
                torch.full_like(sampled_idx, fill_value=args.reward_wrong, dtype=torch.float32),
            )

            advantage = reward - ema_baseline
            loss = -(advantage.detach() * log_prob).mean() - args.entropy_coef * entropy.mean()
            loss = loss / args.grad_accum
            loss.backward()

            batch_size_now = gold_idx.size(0)
            running_loss += loss.item() * args.grad_accum * batch_size_now
            running_reward += reward.mean().item() * batch_size_now
            running_entropy += entropy.mean().item() * batch_size_now
            running_items += batch_size_now

            ema_baseline = args.baseline_beta * ema_baseline + (1.0 - args.baseline_beta) * reward.mean().item()

            global_step += 1

            if global_step % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                optimizer_step += 1

                if optimizer_step % args.eval_every == 0:
                    val_metrics = evaluate_mcqa(
                        model=model,
                        data_loader=valid_loader,
                        action_token_ids=action_token_ids,
                        device=device,
                    )

                    record = {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "optimizer_step": optimizer_step,
                        "train_loss": running_loss / max(running_items, 1),
                        "train_reward": running_reward / max(running_items, 1),
                        "train_entropy": running_entropy / max(running_items, 1),
                        "ema_baseline": ema_baseline,
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                    }
                    history.append(record)

                    print("\nEval checkpoint:")
                    print(record)

                    save_json(history, os.path.join(args.output_dir, "history.json"))

                    if val_metrics["accuracy"] > best_val_acc:
                        best_val_acc = val_metrics["accuracy"]
                        best_dir = os.path.join(args.output_dir, "best_adapter")
                        model.save_pretrained(best_dir)
                        tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))

                        save_json(
                            {
                                "best_val_acc": best_val_acc,
                                "best_record": record,
                            },
                            os.path.join(args.output_dir, "best_summary.json"),
                        )
                        print(f"Saved new best adapter to: {best_dir}")

            pbar.set_postfix(
                loss=f"{running_loss / max(running_items, 1):.4f}",
                reward=f"{running_reward / max(running_items, 1):.4f}",
                best_val=f"{best_val_acc:.4f}",
            )

    last_dir = os.path.join(args.output_dir, "last_adapter")
    model.save_pretrained(last_dir)

    final_val_metrics = evaluate_mcqa(
        model=model,
        data_loader=valid_loader,
        action_token_ids=action_token_ids,
        device=device,
    )

    final_summary = {
        "baseline_before_rl": baseline_metrics,
        "final_val_metrics": final_val_metrics,
        "best_val_acc": best_val_acc,
    }
    save_json(final_summary, os.path.join(args.output_dir, "final_summary.json"))

    print("\n===== FINAL SUMMARY =====")
    print(final_summary)
    print(f"\nSaved last adapter to: {last_dir}")


if __name__ == "__main__":
    main()