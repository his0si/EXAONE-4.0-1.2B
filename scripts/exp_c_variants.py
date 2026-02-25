#!/usr/bin/env python3
"""
C (QuantizationModifier) variants:
1. W8A8 ignore nothing (quantize embed_tokens + lm_head too if possible)
2. W8A8 with only lm_head ignored (embed_tokens quantized)
3. Different scheme options if available
"""
import os, sys, json, gc, time, shutil, subprocess
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE = os.path.join(ROOT, "base_model")

def create_zip(model_dir, zip_path):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    staging = "/tmp/zip_staging"
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(os.path.join(staging, "model"))
    for f in Path(model_dir).iterdir():
        if f.is_file():
            shutil.copy2(f, os.path.join(staging, "model", f.name))
    subprocess.run(["zip", "-r", zip_path, "model/"], cwd=staging, capture_output=True)
    size_gb = os.path.getsize(zip_path) / (1024**3)
    print(f"  Zip: {os.path.basename(zip_path)} ({size_gb:.2f} GB)", flush=True)

def quantize_variant(name, ignore_list, scheme="W8A8"):
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    output_dir = os.path.join(ROOT, "models", f"cvar_{name}")
    zip_path = os.path.join(ROOT, "submit", f"cvar_{name}.zip")

    if os.path.isdir(output_dir) and os.path.exists(os.path.join(output_dir, "config.json")):
        print(f"[SKIP] {name} already exists", flush=True)
        if not os.path.exists(zip_path):
            create_zip(output_dir, zip_path)
        return

    print(f"\n{'='*60}", flush=True)
    print(f"[C-VAR] {name}: scheme={scheme}, ignore={ignore_list}", flush=True)
    print(f"{'='*60}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, trust_remote_code=True)

    recipe = [
        QuantizationModifier(
            scheme=scheme,
            targets=["Linear"],
            ignore=ignore_list,
        ),
    ]

    t0 = time.time()
    oneshot(model=model, recipe=recipe)
    elapsed = time.time() - t0
    print(f"  Quantization done in {elapsed:.1f}s", flush=True)

    # Set max_position_embeddings for speed
    model.config.max_position_embeddings = 16384

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)
    
    # Update tokenizer config
    tok_cfg_path = os.path.join(output_dir, "tokenizer_config.json")
    tok_cfg = json.load(open(tok_cfg_path))
    tok_cfg["model_max_length"] = 16384
    json.dump(tok_cfg, open(tok_cfg_path, "w"), indent=2, ensure_ascii=False)
    
    print(f"  Saved to {output_dir}", flush=True)

    sz = os.path.getsize(os.path.join(output_dir, "model.safetensors")) / (1024**2)
    print(f"  model.safetensors: {sz:.0f} MB", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    create_zip(output_dir, zip_path)

def main():
    # Variant 1: Only ignore embed_tokens (lm_head quantized)
    # Note: This failed with vLLM before for GPTQ, but QuantizationModifier 
    # uses a different format - might work
    quantize_variant("lmq", ignore_list=["embed_tokens"], scheme="W8A8")
    
    # Variant 2: Ignore nothing (both embed + lm_head quantized)
    quantize_variant("allq", ignore_list=[], scheme="W8A8")
    
    # Variant 3: Base C but with max_position_embeddings=16384 
    # (our best C didn't have model_max_length in tokenizer)
    quantize_variant("base_ml16k", ignore_list=["embed_tokens", "lm_head"], scheme="W8A8")

    print("\n=== ALL DONE ===", flush=True)

if __name__ == "__main__":
    main()
