from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.data import KDCollator, KDDataset
from src.losses import compute_kd_loss
from src.model_utils import get_model_device, get_tokenizer, load_student_model, load_teacher_model



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a student model with CE + KD loss.")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str, default=None)
    parser.add_argument("--teacher_model", type=str, required=True)
    parser.add_argument("--student_model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--teacher_4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_every_eval", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_valid_samples", type=int, default=None)

    return parser.parse_args()



def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}



def get_autocast_context() -> object:
    if not torch.cuda.is_available():
        return nullcontext()

    if torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cuda", dtype=torch.float16)


@torch.no_grad()
def evaluate(
    student_model,
    teacher_model,
    dataloader: DataLoader,
    alpha: float,
) -> Dict[str, float]:
    student_model.eval()
    teacher_model.eval()

    device = get_model_device(student_model)

    total_losses: List[float] = []
    ce_losses: List[float] = []
    kd_losses: List[float] = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        batch = move_batch_to_device(batch, device)

        with get_autocast_context():
            student_outputs = student_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            teacher_outputs = teacher_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            total_loss, ce_loss, kd_loss = compute_kd_loss(
                student_logits=student_outputs.logits,
                teacher_logits=teacher_outputs.logits,
                labels=batch["labels"],
                alpha=alpha,
            )

        total_losses.append(float(total_loss.item()))
        ce_losses.append(float(ce_loss.item()))
        kd_losses.append(float(kd_loss.item()))

    metrics = {
        "eval_total_loss": sum(total_losses) / max(len(total_losses), 1),
        "eval_ce_loss": sum(ce_losses) / max(len(ce_losses), 1),
        "eval_kd_loss": sum(kd_losses) / max(len(kd_losses), 1),
    }
    return metrics



def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), output_dir / "train_config.json")

    tokenizer = get_tokenizer(args.student_model)

    train_dataset = KDDataset(
        file_path=args.train_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        system_prompt=args.system_prompt,
        max_samples=args.max_train_samples,
    )

    valid_dataset: Optional[KDDataset] = None
    if args.valid_file:
        valid_dataset = KDDataset(
            file_path=args.valid_file,
            tokenizer=tokenizer,
            max_length=args.max_length,
            system_prompt=args.system_prompt,
            max_samples=args.max_valid_samples,
        )

    collator = KDCollator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    valid_loader = None
    if valid_dataset is not None:
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.per_device_batch_size,
            shuffle=False,
            collate_fn=collator,
        )

    teacher_model = load_teacher_model(args.teacher_model, load_in_4bit=args.teacher_4bit)
    student_model = load_student_model(args.student_model)

    if args.gradient_checkpointing:
        student_model.gradient_checkpointing_enable()
        student_model.config.use_cache = False

    student_device = get_model_device(student_model)
    teacher_device = get_model_device(teacher_model)
    if student_device != teacher_device:
        raise RuntimeError(
            f"Teacher and student are on different devices: {teacher_device} vs {student_device}. "
            "For this simple project, keep both on the same device."
        )

    optimizer = AdamW(
        params=[p for p in student_model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_update_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.num_epochs
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    use_scaler = torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    global_step = 0
    best_eval_loss = float("inf")
    history: List[Dict[str, float]] = []

    print("=" * 80)
    print("KD training started")
    print(f"Train examples: {len(train_dataset)}")
    if valid_dataset is not None:
        print(f"Valid examples: {len(valid_dataset)}")
    print(f"Teacher: {args.teacher_model}")
    print(f"Student: {args.student_model}")
    print(f"Alpha: {args.alpha}")
    print("=" * 80)

    for epoch in range(1, args.num_epochs + 1):
        student_model.train()
        teacher_model.eval()

        running_total = 0.0
        running_ce = 0.0
        running_kd = 0.0
        running_count = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}")
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress, start=1):
            batch = move_batch_to_device(batch, student_device)

            with get_autocast_context():
                student_outputs = student_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                with torch.no_grad():
                    teacher_outputs = teacher_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )

                total_loss, ce_loss, kd_loss = compute_kd_loss(
                    student_logits=student_outputs.logits,
                    teacher_logits=teacher_outputs.logits,
                    labels=batch["labels"],
                    alpha=args.alpha,
                )
                scaled_loss = total_loss / args.gradient_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            running_total += float(total_loss.item())
            running_ce += float(ce_loss.item())
            running_kd += float(kd_loss.item())
            running_count += 1

            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.logging_steps == 0:
                    avg_total = running_total / max(running_count, 1)
                    avg_ce = running_ce / max(running_count, 1)
                    avg_kd = running_kd / max(running_count, 1)
                    progress.set_postfix(
                        total=f"{avg_total:.4f}",
                        ce=f"{avg_ce:.4f}",
                        kd=f"{avg_kd:.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )

                if valid_loader is not None and global_step % args.eval_steps == 0:
                    metrics = evaluate(
                        student_model=student_model,
                        teacher_model=teacher_model,
                        dataloader=valid_loader,
                        alpha=args.alpha,
                    )
                    metrics["global_step"] = global_step
                    metrics["epoch"] = epoch
                    history.append(metrics)
                    print(
                        f"\n[Eval] step={global_step} | "
                        f"total={metrics['eval_total_loss']:.4f} | "
                        f"ce={metrics['eval_ce_loss']:.4f} | "
                        f"kd={metrics['eval_kd_loss']:.4f}"
                    )

                    if metrics["eval_total_loss"] < best_eval_loss:
                        best_eval_loss = metrics["eval_total_loss"]
                        best_dir = output_dir / "best_model"
                        best_dir.mkdir(parents=True, exist_ok=True)
                        student_model.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)
                        print(f"[Save] New best model saved to {best_dir}")
                    elif args.save_every_eval:
                        ckpt_dir = output_dir / f"checkpoint_step_{global_step}"
                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                        student_model.save_pretrained(ckpt_dir)
                        tokenizer.save_pretrained(ckpt_dir)
                        print(f"[Save] Checkpoint saved to {ckpt_dir}")

        if valid_loader is not None:
            metrics = evaluate(
                student_model=student_model,
                teacher_model=teacher_model,
                dataloader=valid_loader,
                alpha=args.alpha,
            )
            metrics["global_step"] = global_step
            metrics["epoch"] = epoch
            history.append(metrics)
            print(
                f"\n[Epoch-end eval] epoch={epoch} | "
                f"total={metrics['eval_total_loss']:.4f} | "
                f"ce={metrics['eval_ce_loss']:.4f} | "
                f"kd={metrics['eval_kd_loss']:.4f}"
            )
            if metrics["eval_total_loss"] < best_eval_loss:
                best_eval_loss = metrics["eval_total_loss"]
                best_dir = output_dir / "best_model"
                best_dir.mkdir(parents=True, exist_ok=True)
                student_model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                print(f"[Save] New best model saved to {best_dir}")

    final_dir = output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    student_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    save_json({"history": history}, output_dir / "history.json")

    print("=" * 80)
    print(f"Training finished. Final model saved to {final_dir}")
    if valid_dataset is not None and best_eval_loss < float('inf'):
        print(f"Best validation total loss: {best_eval_loss:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
