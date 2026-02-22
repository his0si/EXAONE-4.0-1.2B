"""
Experiment 6: LoRA Fine-tuning + W8A8 Quantization
Low risk. Uses the LoRA-merged model from Experiment 4, then W8A8 GPTQ.
"""
import os
import sys
import json
import torch
import gc
from pathlib import Path

if torch.cuda.is_available():
    torch.cuda.init()
    torch.zeros(1).cuda()
    print(f"[INFO] CUDA: {torch.cuda.get_device_name(0)}")

from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

sys.path.insert(0, os.path.dirname(__file__))
from utils.korean_calibration import load_korean_calibration

LORA_MODEL = "/home/lgaimers/0214/models/lora_merged"
BASE_MODEL = "/home/lgaimers/base_model"
MODELS_DIR = "/home/lgaimers/0214/models"
RESULTS_DIR = "/home/lgaimers/0214/results"


def run_lora_w8a8(dampening_frac=0.02, num_samples=2048, max_seq_length=512):
    suffix = f"d{str(dampening_frac).replace('.','')}"
    name = f"lora_w8a8_{suffix}"

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")

    out_dir = os.path.join(MODELS_DIR, name)
    if os.path.exists(os.path.join(out_dir, "model.safetensors")):
        print(f"[SKIP] {name} already exists")
        return

    # Check if LoRA model exists
    if not os.path.exists(os.path.join(LORA_MODEL, "model.safetensors")):
        print("[ERROR] LoRA merged model not found. Run exp04_lora_w4a16.py first.")
        print("  Running LoRA training now...")
        from exp04_lora_w4a16 import step1_lora_train
        step1_lora_train()

    print("[INFO] Loading LoRA-merged model...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LORA_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
    )

    print("[INFO] Loading Korean calibration data...")
    cal_ds = load_korean_calibration(
        tokenizer, num_samples=num_samples, include_manta=True, manta_ratio=0.3,
    )

    print(f"[INFO] W8A8 GPTQ quantization (damp={dampening_frac})...")
    recipe = [
        GPTQModifier(
            scheme="W8A8",
            targets=["Linear"],
            ignore=["embed_tokens", "lm_head"],
            dampening_frac=dampening_frac,
        )
    ]

    oneshot(
        model=model,
        dataset=cal_ds,
        recipe=recipe,
        max_seq_length=max_seq_length,
        num_calibration_samples=num_samples,
    )

    print(f"[INFO] Saving to {out_dir}...")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, save_compressed=True)
    tokenizer.save_pretrained(out_dir)

    model_size = sum(f.stat().st_size for f in Path(out_dir).glob("*.safetensors")) / (1024**3)
    print(f"[INFO] Done: {name} ({model_size:.2f} GB)")

    with open(os.path.join(RESULTS_DIR, f"{name}_config.json"), "w") as f:
        json.dump({
            "name": name,
            "lora_base": LORA_MODEL,
            "scheme": "W8A8",
            "dampening_frac": dampening_frac,
            "num_samples": num_samples,
            "size_gb": model_size,
        }, f, indent=2)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    run_lora_w8a8(dampening_frac=0.02, num_samples=2048)
    run_lora_w8a8(dampening_frac=0.015, num_samples=2048)
