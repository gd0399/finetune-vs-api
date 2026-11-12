"""Merge the LoRA adapter into the base weights so vLLM serves it at full speed.

    python train/merge_lora.py --base Qwen/Qwen3-8B --adapter adapters/qwen3-8b-lora --out merged/qwen3-8b-ft
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", default="adapters/qwen3-8b-lora")
    ap.add_argument("--out", default="merged/qwen3-8b-ft")
    a = ap.parse_args()
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.save_pretrained(a.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.base).save_pretrained(a.out)
    print("merged model saved to", a.out)


if __name__ == "__main__":
    main()
