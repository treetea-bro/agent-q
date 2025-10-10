import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_DIR = "./dpo_final"
MERGED_DIR = "./dpo_merged_fp16"


def main():
    print("🚀 Loading FP16 base model...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
    )

    print("🧩 Loading LoRA adapter...")
    peft_model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    merged = peft_model.merge_and_unload()
    merged = merged.to(dtype=torch.float16)

    print("💾 Saving merged model...")
    os.makedirs(MERGED_DIR, exist_ok=True)
    merged.save_pretrained(MERGED_DIR, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(
        ADAPTER_DIR, use_fast=True, trust_remote_code=True
    )
    tok.save_pretrained(MERGED_DIR)
    print(f"✅ Done! Saved merged model to {MERGED_DIR}")


if __name__ == "__main__":
    main()
