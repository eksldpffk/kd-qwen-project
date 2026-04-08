from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."



def load_jsonl(path: str | Path, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    path = Path(path)
    examples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
            if max_samples is not None and len(examples) >= max_samples:
                break
    return examples



def build_prompt_and_response(example: Dict[str, Any]) -> Tuple[str, str]:
    """Support two common formats:
    1) {"prompt": ..., "response": ...}
    2) {"instruction": ..., "input": ..., "output": ...}
    """
    if "prompt" in example:
        prompt = str(example["prompt"]).strip()
        response = str(
            example.get("response", example.get("answer", example.get("output", "")))
        ).strip()
        if not response:
            raise ValueError("Example with 'prompt' is missing 'response'/'answer'/'output'.")
        return prompt, response

    if "instruction" in example and "output" in example:
        instruction = str(example["instruction"]).strip()
        input_text = str(example.get("input", "")).strip()
        if input_text:
            prompt = f"{instruction}\n\nAdditional input:\n{input_text}"
        else:
            prompt = instruction
        response = str(example["output"]).strip()
        return prompt, response

    raise ValueError(
        "Unsupported JSONL format. Use either {prompt,response} or {instruction,input,output}."
    )



def build_chat_texts(tokenizer: Any, prompt: str, response: str, system_prompt: str) -> Tuple[str, str]:
    """Return (prompt_text, full_text).

    prompt_text ends right before the assistant answer starts.
    full_text includes the assistant answer.
    """
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        full_messages = prompt_messages + [{"role": "assistant", "content": response}]

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return prompt_text, full_text

    # Fallback if a tokenizer has no chat template.
    prompt_text = (
        f"System: {system_prompt}\n"
        f"User: {prompt}\n"
        f"Assistant:"
    )
    full_text = f"{prompt_text} {response}"
    return prompt_text, full_text


@dataclass
class KDFeature:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]


class KDDataset(Dataset):
    def __init__(
        self,
        file_path: str | Path,
        tokenizer: Any,
        max_length: int,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_samples: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt

        raw_examples = load_jsonl(file_path, max_samples=max_samples)
        self.features: List[KDFeature] = []

        for ex in raw_examples:
            try:
                prompt, response = build_prompt_and_response(ex)
                feature = self._build_feature(prompt, response)
                if feature is not None:
                    self.features.append(feature)
            except Exception as exc:
                print(f"[WARN] Skipping example due to preprocessing error: {exc}")

        if not self.features:
            raise ValueError("No usable examples were built from the provided file.")

    def _build_feature(self, prompt: str, response: str) -> Optional[KDFeature]:
        prompt_text, full_text = build_chat_texts(
            tokenizer=self.tokenizer,
            prompt=prompt,
            response=response,
            system_prompt=self.system_prompt,
        )

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]

        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None:
            full_ids = full_ids + [eos_token_id]

        if len(full_ids) > self.max_length:
            full_ids = full_ids[: self.max_length]

        prompt_len = min(len(prompt_ids), len(full_ids))

        # If truncation removed the answer entirely, skip the example.
        if prompt_len >= len(full_ids):
            return None

        labels = [-100] * prompt_len + full_ids[prompt_len:]
        attention_mask = [1] * len(full_ids)

        return KDFeature(
            input_ids=full_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        feature = self.features[idx]
        return {
            "input_ids": torch.tensor(feature.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(feature.attention_mask, dtype=torch.long),
            "labels": torch.tensor(feature.labels, dtype=torch.long),
        }


class KDCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        max_len = max(item["input_ids"].shape[0] for item in batch)

        input_ids, attention_mask, labels = [], [], []
        for item in batch:
            seq_len = item["input_ids"].shape[0]
            pad_len = max_len - seq_len

            input_ids.append(
                torch.cat(
                    [item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)],
                    dim=0,
                )
            )
            attention_mask.append(
                torch.cat(
                    [item["attention_mask"], torch.zeros((pad_len,), dtype=torch.long)],
                    dim=0,
                )
            )
            labels.append(
                torch.cat(
                    [item["labels"], torch.full((pad_len,), -100, dtype=torch.long)],
                    dim=0,
                )
            )

        return {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "labels": torch.stack(labels, dim=0),
        }
