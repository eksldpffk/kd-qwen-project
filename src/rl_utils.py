from __future__ import annotations

import json
import os
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


ACTION_STRS = [" A", " B", " C", " D", " E"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_torch_dtype() -> torch.dtype:
    return torch.float16 if torch.cuda.is_available() else torch.float32


def load_tokenizer(tokenizer_name: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_action_token_ids(tokenizer) -> torch.Tensor:
    token_ids = []
    for action_str in ACTION_STRS:
        ids = tokenizer.encode(action_str, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Action {action_str!r} is not a single token for this tokenizer. Got ids={ids}"
            )
        token_ids.append(ids[0])

    return torch.tensor(token_ids, dtype=torch.long)


def load_base_model(model_path: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=get_torch_dtype(),
        trust_remote_code=True,
    )
    return model


def attach_lora(
    model,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
):
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    return model


def load_model_with_optional_adapter(
    base_model_path: str,
    adapter_path: str | None = None,
):
    model = load_base_model(base_model_path)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    return model


def get_action_scores(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    action_token_ids: torch.Tensor,
) -> torch.Tensor:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits  # [B, T, V]

    last_pos = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(logits.size(0), device=logits.device)

    last_logits = logits[batch_idx, last_pos, :]  # [B, V]
    scores = last_logits[:, action_token_ids.to(logits.device)]  # [B, 5]
    return scores


@torch.no_grad()
def evaluate_mcqa(
    model,
    data_loader,
    action_token_ids: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    total = 0
    correct = 0
    correct_prob_sum = 0.0
    margin_sum = 0.0
    entropy_sum = 0.0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        gold_idx = torch.tensor(batch["gold_idx"], dtype=torch.long, device=device)

        scores = get_action_scores(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            action_token_ids=action_token_ids,
        ).float()

        probs = torch.softmax(scores, dim=-1)
        pred_idx = probs.argmax(dim=-1)

        correct += (pred_idx == gold_idx).sum().item()
        total += gold_idx.size(0)

        correct_probs = probs.gather(1, gold_idx.unsqueeze(1)).squeeze(1)

        gold_mask = F.one_hot(gold_idx, num_classes=probs.size(1)).bool()
        wrong_probs = probs.masked_fill(gold_mask, -1.0)
        best_wrong_probs = wrong_probs.max(dim=1).values

        margins = correct_probs - best_wrong_probs
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)

        correct_prob_sum += correct_probs.sum().item()
        margin_sum += margins.sum().item()
        entropy_sum += entropy.sum().item()

    metrics = {
        "accuracy": correct / max(total, 1),
        "avg_correct_prob": correct_prob_sum / max(total, 1),
        "avg_margin": margin_sum / max(total, 1),
        "avg_entropy": entropy_sum / max(total, 1),
    }

    model.train()
    return metrics