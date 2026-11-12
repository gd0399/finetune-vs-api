"""QLoRA fine-tuning of a small open model on the extraction task.

The model sees exactly the prompt it will get at evaluation time (same system prompt,
same chat template with Qwen3 "thinking" switched off) and learns to output the gold
JSON. Loss is computed on the answer tokens only.

GPU run (24 GB card, e.g. L4 / A10G / A100):
    python train/train_lora.py --model Qwen/Qwen3-8B --out adapters/qwen3-8b-lora

CPU smoke test with a tiny random model (no downloads; checks the code path only):
    python train/train_lora.py --tiny --no-4bit --max-steps 3 --out /tmp/tiny-lora
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ftva.prompt import messages, target  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_rows(path):
    return [json.loads(l) for l in open(path)]


def end_token(tok) -> str:
    return "<|im_end|>" if "<|im_end|>" in tok.get_vocab() else tok.eos_token


class SFTData(Dataset):
    """prompt tokens -> label -100, answer tokens -> learned."""

    def __init__(self, rows, tok, max_len):
        self.items, skipped = [], 0
        end = end_token(tok)
        for r in rows:
            prompt = tok.apply_chat_template(messages(r["note"]), tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            a_ids = tok(target(r["label"]) + end, add_special_tokens=False)["input_ids"]
            if len(p_ids) + len(a_ids) > max_len:
                skipped += 1
                continue
            self.items.append({"input_ids": p_ids + a_ids, "labels": [-100] * len(p_ids) + a_ids})
        if skipped:
            print(f"skipped {skipped} examples longer than {max_len} tokens")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(pad_id):
    def fn(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        lab = torch.full((len(batch), n), -100, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):  # right padding
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            lab[i, :k] = torch.tensor(b["labels"])
            att[i, :k] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}
    return fn


def tiny_model_and_tokenizer(rows):
    """A 2-layer random Qwen3 with a tokenizer trained on the notes. For smoke tests only."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    t = Tokenizer(models.BPE(unk_token=None))
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    corpus = [r["note"] + target(r["label"]) for r in rows] + [messages("")[0]["content"]]
    t.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=2000, special_tokens=special,
                                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    from tokenizers import decoders
    t.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=t, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    # simplified version of Qwen3's template, including the enable_thinking switch
    tok.chat_template = (
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n"
        "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}"
        "{% endif %}")
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096)
    return Qwen3ForCausalLM(cfg), tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="adapters/qwen3-8b-lora")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--no-4bit", action="store_true", help="plain LoRA in bf16/fp32 instead of QLoRA")
    ap.add_argument("--tiny", action="store_true", help="random tiny model for a CPU smoke test")
    a = ap.parse_args()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    train_rows = load_rows(Path(a.data) / "train.jsonl")
    val_rows = load_rows(Path(a.data) / "val.jsonl")
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()

    if a.tiny:
        model, tok = tiny_model_and_tokenizer(train_rows)
    else:
        tok = AutoTokenizer.from_pretrained(a.model)
        kw = {"dtype": torch.bfloat16 if bf16 else torch.float16 if cuda else torch.float32}
        if not a.no_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
            kw["device_map"] = {"": 0}
        model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
        if not a.no_4bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=0.05,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    train_ds, val_ds = SFTData(train_rows, tok, a.max_len), SFTData(val_rows, tok, a.max_len)
    print(f"train {len(train_ds)}  val {len(val_ds)}")
    args = TrainingArguments(
        output_dir=a.out + "-ckpt", num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_steps=10, logging_steps=10, eval_strategy="epoch" if a.max_steps < 0 else "no",
        save_strategy="no", bf16=bf16, fp16=cuda and not bf16, gradient_checkpointing=cuda and not a.tiny,
        report_to="none", remove_unused_columns=False, use_cpu=not cuda)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=collate(tok.pad_token_id))
    trainer.train()
    if a.max_steps < 0:
        print("val:", trainer.evaluate())
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("saved adapter to", a.out)


if __name__ == "__main__":
    main()
