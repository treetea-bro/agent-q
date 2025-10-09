#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DPO fine-tuning for Qwen/Qwen3-4B-Thinking-2507 that fits within a single 40GB GPU.

• Strategy: QLoRA (4-bit) + gradient checkpointing + small batch + accumulation
• Libraries: transformers, accelerate, trl, peft, bitsandbytes, datasets
• Dataset format: JSONL with fields {"prompt", "chosen", "rejected"}

Example dataset (train.jsonl):
{"prompt": "사용자: 안녕?\n", "chosen": "어시스턴트: 안녕하세요! 무엇을 도와드릴까요?", "rejected": "어시스턴트: 안녕"}
{"prompt": "User: Explain DPO briefly\n", "chosen": "Assistant: DPO optimizes a policy by preferring...", "rejected": "Assistant: ..."}

Run:
  pip install -U "transformers>=4.43" accelerate bitsandbytes "trl>=0.10.0" peft datasets
  accelerate config  # if first time
  python dpo_qwen3_4b_thinking.py --train_file ./train.jsonl --eval_file ./eval.jsonl --output_dir ./dpo-qwen3-4b-thinking

Inference with adapters:
  python dpo_qwen3_4b_thinking.py --infer --output_dir ./dpo-qwen3-4b-thinking --prompt "서울 날씨 요약해줘"

(Optional) Merge adapters into base weights (CPU-safe):
  python dpo_qwen3_4b_thinking.py --merge --output_dir ./dpo-qwen3-4b-thinking --merged_dir ./merged-qwen3-4b-thinking-dpo

This script autodetects FlashAttention-2 and falls back to SDPA.
Tested layout: A100 40GB, Ubuntu 22.04, CUDA 12.x.
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# TRL imports (support both old/new paths)
try:
    from trl import DPOTrainer
    from trl.trainer.dpo_config import DPOConfig
except Exception:
    from trl.trainer.dpo_config import DPOConfig  # type: ignore
    from trl.trainer.dpo_trainer import DPOTrainer  # type: ignore

from peft import LoraConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    p.add_argument("--train_file", type=str, required=False, help="Path to train.jsonl")
    p.add_argument("--eval_file", type=str, required=False, help="Path to eval.jsonl")
    p.add_argument("--output_dir", type=str, default="./dpo-qwen3-4b-thinking")

    # Training budget tuned for <=40GB
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Override number of steps; -1 uses epochs",
    )
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--beta_dpo", type=float, default=0.1, help="DPO temperature beta")
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)

    # LoRA & quantization
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--bnb_4bit", action="store_true", default=True)

    # Actions
    p.add_argument("--infer", action="store_true")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--merged_dir", type=str, default="./merged-qwen3-4b-thinking-dpo")
    p.add_argument("--prompt", type=str, default="안녕! DPO 튜닝된 모델이야?")

    return p.parse_args()


@dataclass
class PairSample:
    prompt: str
    chosen: str
    rejected: str


def jsonl_to_hf_dataset(path: str):
    """Load JSONL into Dataset with required columns.
    Each line must contain keys: prompt, chosen, rejected.
    """
    if path is None:
        return None
    data: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            for k in ("prompt", "chosen", "rejected"):
                if k not in ex:
                    raise ValueError(f"Missing key '{k}' in: {ex}")
            data.append(
                {
                    "prompt": ex["prompt"],
                    "chosen": ex["chosen"],
                    "rejected": ex["rejected"],
                }
            )
    # Use datasets.from_list so we don't need to write to disk
    from datasets import Dataset

    return Dataset.from_list(data)


def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.is_available()
        else torch.float16,
    )


def build_lora_config(r: int, alpha: int, dropout: float) -> LoraConfig:
    # Typical Qwen target modules
    target = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target,
    )


def load_backbone(model_name: str, max_seq_len: int):
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

    bnb_cfg = get_bnb_config()
    attn_impl = "flash_attention_2"
    try:
        # If FlashAttention-2 isn't available, transformers will raise; we fallback
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=bnb_cfg,
            attn_implementation=attn_impl,
            torch_dtype=torch_dtype,
        )
    except Exception:
        attn_impl = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=bnb_cfg,
            attn_implementation=attn_impl,
            torch_dtype=torch_dtype,
        )

    tok = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, use_fast=True
    )

    # Ensure pad token
    if tok.pad_token is None:
        # Prefer eos token as pad
        tok.pad_token = tok.eos_token
    # Some Qwen variants need explicit settings for padding side when packing
    tok.padding_side = "right"

    # Enable gradient checkpointing for memory
    model.gradient_checkpointing_enable()

    # Ensure model can handle long seq len if supported
    if hasattr(model.config, "max_position_embeddings"):
        model.config.max_position_embeddings = max(
            model.config.max_position_embeddings, max_seq_len
        )

    print(f"Loaded {model_name} with attention={attn_impl}")
    return model, tok


def train(args):
    assert args.train_file, "--train_file is required"

    train_ds = jsonl_to_hf_dataset(args.train_file)
    eval_ds = jsonl_to_hf_dataset(args.eval_file) if args.eval_file else None

    model, tokenizer = load_backbone(args.model, args.max_seq_len)

    # LoRA
    peft_cfg = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)

    # DPO config — sized for <=40GB
    dpo_cfg = DPOConfig(
        output_dir=args.output_dir,
        do_eval=eval_ds is not None,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_length=args.max_seq_len,
        max_prompt_length=min(512, args.max_seq_len // 2),
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=torch.cuda.is_available(),
        fp16=not torch.cuda.is_available(),
        remove_unused_columns=False,
        gradient_checkpointing=True,
        beta=args.beta_dpo,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs if args.max_steps == -1 else 1.0,
        report_to=["none"],
    )

    # TRL wants columns named prompt, chosen, rejected by default
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # use implicit frozen reference
        args=dpo_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        peft_config=peft_cfg,
    )

    print("Start training…")
    trainer.train()
    print("Training done. Saving adapter…")
    trainer.save_model(args.output_dir)  # saves adapter weights
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")


def infer(args):
    # Load base + adapter for inference
    from peft import PeftModel

    model, tokenizer = load_backbone(args.model, max_seq_len=2048)
    model = PeftModel.from_pretrained(model, args.output_dir)
    model.eval()

    device = next(model.parameters()).device
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    print(tokenizer.decode(out[0], skip_special_tokens=True))


def merge(args):
    """Merge LoRA into base weights on CPU to avoid large VRAM usage."""
    from peft import PeftModel

    torch_dtype = torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map={"": "cpu"},
    )
    merged = PeftModel.from_pretrained(base, args.output_dir)
    merged = merged.merge_and_unload()

    tok = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, use_fast=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    os.makedirs(args.merged_dir, exist_ok=True)
    merged.save_pretrained(args.merged_dir)
    tok.save_pretrained(args.merged_dir)
    print(f"Merged weights saved to {args.merged_dir}")


if __name__ == "__main__":
    args = parse_args()

    if args.infer:
        infer(args)
    elif args.merge:
        merge(args)
    else:
        train(args)
