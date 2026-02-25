#!/usr/bin/env python3
"""
SparseGPT 10% + GPTQ W8A8 with dampening=0.02, cal_samples=2048, seq_length=2048
Based on sweep_d02_c2048_s2048 settings + SparseGPT 10% sparsity
"""
import os, sys, json, time
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.pruning import SparseGPTModifier

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_MODEL = os.path.join(ROOT, "base_model")
OUTPUT_DIR = os.path.join(ROOT, "models", "w8a8_d02_sp10")

def main():
    print("[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    print("[2/4] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)

    print("[3/4] Loading calibration data (MANTA 2048 samples)...")
    manta_path = os.path.join(ROOT, "data", "manta", "train.json")
    samples = json.load(open(manta_path))[:2048]
    texts = []
    for s in samples:
        text = tokenizer.apply_chat_template(
            s["conversations"], add_generation_prompt=True, tokenize=False)
        texts.append(text)
    ds = Dataset.from_dict({"text": texts})

    recipe = [
        SparseGPTModifier(
            sparsity=0.1,
            mask_structure="0:0",
            sequential_update=True,
            sequential_targets=["Exaone4DecoderLayer"],
            targets=["Linear"],
            ignore=["embed_tokens", "lm_head"],
            block_size=128,
            dampening_frac=0.02,
            preserve_sparsity_mask=False,
            offload_hessians=False,
        ),
        GPTQModifier(
            targets=["Linear"],
            ignore=["embed_tokens", "lm_head"],
            scheme="W8A8",
            block_size=128,
            dampening_frac=0.02,
            actorder="static",
            offload_hessians=False,
        ),
    ]

    print("[4/4] Running SparseGPT 10% + GPTQ W8A8 (d=0.02)...")
    t0 = time.time()
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=2048,
        num_calibration_samples=2048,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR, save_compressed=True)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Set max_position_embeddings=16384
    config_path = os.path.join(OUTPUT_DIR, "config.json")
    config = json.load(open(config_path))
    config["max_position_embeddings"] = 16384
    json.dump(config, open(config_path, "w"), indent=2, ensure_ascii=False)

    size_mb = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f))
        for f in os.listdir(OUTPUT_DIR)
        if f.endswith(".safetensors")
    ) / 1024 / 1024
    print(f"  Saved to {OUTPUT_DIR} ({size_mb:.0f} MB)")

if __name__ == "__main__":
    main()
