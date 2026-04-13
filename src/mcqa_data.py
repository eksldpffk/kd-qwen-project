from __future__ import annotations

from typing import Any, Callable

from datasets import load_dataset

LABELS = ["A", "B", "C", "D", "E"]
LABEL2IDX = {label: i for i, label in enumerate(LABELS)}
IDX2LABEL = {i: label for label, i in LABEL2IDX.items()}


def build_user_prompt(example: dict[str, Any]) -> str:
    question = example["question"]
    choice_labels = example["choices"]["label"]
    choice_texts = example["choices"]["text"]

    lines = [f"Question: {question}", "", "Choices:"]
    for label, text in zip(choice_labels, choice_texts):
        lines.append(f"{label}. {text}")

    lines.append("")
    lines.append("Reply with only one letter: A, B, C, D, or E.")
    lines.append("Answer:")
    return "\n".join(lines)


def build_model_prompt(tokenizer, example: dict[str, Any]) -> str:
    user_prompt = build_user_prompt(example)

    messages = [
        {
            "role": "system",
            "content": "You are a careful multiple-choice assistant. Return only the single best answer letter.",
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = user_prompt

    return prompt


def load_csqa_splits(
    train_examples: int,
    valid_examples: int,
    seed: int = 42,
):
    ds = load_dataset("tau/commonsense_qa")

    train_ds = ds["train"].shuffle(seed=seed)
    valid_ds = ds["validation"].shuffle(seed=seed)

    if train_examples > 0:
        train_ds = train_ds.select(range(min(train_examples, len(train_ds))))
    if valid_examples > 0:
        valid_ds = valid_ds.select(range(min(valid_examples, len(valid_ds))))

    return train_ds, valid_ds


def load_csqa_eval_split(
    split: str,
    num_examples: int,
    seed: int = 42,
):
    ds = load_dataset("tau/commonsense_qa")
    eval_ds = ds[split].shuffle(seed=seed)

    if num_examples > 0:
        eval_ds = eval_ds.select(range(min(num_examples, len(eval_ds))))

    return eval_ds


def make_collate_fn(tokenizer, max_length: int) -> Callable:
    def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
        prompts = [build_model_prompt(tokenizer, ex) for ex in features]
        gold_idx = [LABEL2IDX[ex["answerKey"]] for ex in features]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "gold_idx": gold_idx,
            "prompts": prompts,
            "raw_examples": features,
        }

    return collate_fn