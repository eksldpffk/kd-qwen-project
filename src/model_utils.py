from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig



def get_default_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32



def get_tokenizer(model_name: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer



def make_quantization_config(load_in_4bit: bool) -> Optional[BitsAndBytesConfig]:
    if not load_in_4bit:
        return None

    compute_dtype = get_default_dtype()
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )



def get_single_device_map() -> Optional[Dict[str, int]]:
    if torch.cuda.is_available():
        return {"": 0}
    return None



def load_teacher_model(model_name: str, load_in_4bit: bool = False) -> Any:
    quantization_config = make_quantization_config(load_in_4bit)

    kwargs = {
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
    }

    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = get_single_device_map()
    else:
        kwargs["torch_dtype"] = get_default_dtype()

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if quantization_config is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model



def load_student_model(model_name: str) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=get_default_dtype(),
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model



def get_model_device(model: Any) -> torch.device:
    if hasattr(model, "hf_device_map") and getattr(model, "hf_device_map"):
        first_loc = next(iter(model.hf_device_map.values()))
        if isinstance(first_loc, int):
            return torch.device(f"cuda:{first_loc}")
        if isinstance(first_loc, str):
            return torch.device(first_loc)
    return next(model.parameters()).device
