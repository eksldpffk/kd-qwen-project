from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
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

    parser.add_argument("--train_examples", type=int, default=1500)
    parser.add_argument("--valid_examples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--reward_correct", type=float, default=1.0)
    parser.add_argument("--reward_wrong", type=float, default=0.0)
    parser.add_argument("--kl_coef", type=float, default=0.02)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--adv_eps", type=float, default=1e-6)

    parser.add_argument("--eval_every", type=int, default=25)

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

    # Trainable model: baseline + LoRA
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

    # Frozen reference model for KL regularization
    ref_model = load_base_model(args.base_model_path)
    ref_model = ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best_val_acc = -1.0
    global_step = 0
    optimizer_step = 0

    baseline_metrics = evaluate_mcqa(
        model=model,
        data_loader=valid_loader,
        action_token_ids=action_token_ids,
        device=device,
    )
    print("\nBaseline before GRPO:")
    print(baseline_metrics)

    save_json(
        {
            "config": vars(args),
            "baseline_metrics": baseline_metrics,
        },
        os.path.join(args.output_dir, "run_config.json"),
    )

    optimizer.zero_grad()

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        running_loss = 0.0
        running_reward = 0.0
        running_entropy = 0.0
        running_kl = 0.0
        running_items = 0

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gold_idx = torch.tensor(batch["gold_idx"], dtype=torch.long, device=device)

            scores = get_action_scores(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                action_token_ids=action_token_ids,
            ).float()  # [B, 5]

            log_probs = F.log_softmax(scores, dim=-1)
            probs = torch.softmax(scores, dim=-1)

            with torch.no_grad():
                ref_scores = get_action_scores(
                    model=ref_model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    action_token_ids=action_token_ids,
                ).float()
                ref_log_probs = F.log_softmax(ref_scores, dim=-1)

            # Sample multiple actions per prompt
            sampled_idx = torch.multinomial(
                probs,
                num_samples=args.group_size,
                replacement=True,
            )  # [B, G]

            sampled_log_probs = log_probs.gather(1, sampled_idx)  # [B, G]

            reward = torch.where(
                sampled_idx == gold_idx.unsqueeze(1),
                torch.full_like(sampled_idx, fill_value=args.reward_correct, dtype=torch.float32),
                torch.full_like(sampled_idx, fill_value=args.reward_wrong, dtype=torch.float32),
            )  # [B, G]

            group_mean = reward.mean(dim=1, keepdim=True)
            group_std = reward.std(dim=1, unbiased=False, keepdim=True)
            advantage = (reward - group_mean) / (group_std + args.adv_eps)

            pg_loss = -(advantage.detach() * sampled_log_probs).mean()

            # Exact KL(current || reference) over the 5 actions
            kl = (probs * (log_probs - ref_log_probs)).sum(dim=-1).mean()

            entropy = -(probs * log_probs).sum(dim=-1).mean()

            loss = pg_loss + args.kl_coef * kl - args.entropy_coef * entropy
            loss = loss / args.grad_accum
            loss.backward()

            batch_size_now = gold_idx.size(0)
            running_loss += loss.item() * args.grad_accum * batch_size_now
            running_reward += reward.mean().item() * batch_size_now
            running_entropy += entropy.item() * batch_size_now
            running_kl += kl.item() * batch_size_now
            running_items += batch_size_now

            global_step += 1

            if global_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=args.max_grad_norm,
                )
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
                        "train_kl": running_kl / max(running_items, 1),
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
        "baseline_before_grpo": baseline_metrics,
        "final_val_metrics": final_val_metrics,
        "best_val_acc": best_val_acc,
    }
    save_json(final_summary, os.path.join(args.output_dir, "final_summary.json"))

    print("\n===== FINAL SUMMARY =====")
    print(final_summary)
    print(f"\nSaved last adapter to: {last_dir}")


if __name__ == "__main__":
    main()