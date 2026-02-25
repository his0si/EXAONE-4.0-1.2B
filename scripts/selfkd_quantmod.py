#!/usr/bin/env python3
"""
Apply QuantizationModifier W8A8 (non-GPTQ, dynamic activation) to selfkd_base_kd checkpoint.
Same recipe as c_qmod_w8a8 but on the self-KD fine-tuned base model.
"""
import os, sys, json, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_MODEL = os.path.join(ROOT, "checkpoints", "selfkd_base_kd")
OUTPUT_DIR = os.path.join(ROOT, "models", "selfkd_qmod_w8a8")

def main():
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    print(f"[1/4] Loading tokenizer from base_model...")
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(ROOT, "base_model"), trust_remote_code=True)

    print(f"[2/4] Loading selfkd model from {INPUT_MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        INPUT_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)

    # Calibration data (same as c_qmod_w8a8)
    print(f"[3/4] Loading calibration data...")
    manta_path = os.path.join(ROOT, "data", "manta", "train.json")
    samples = json.load(open(manta_path))[:2048]
    texts = []
    for s in samples:
        text = tokenizer.apply_chat_template(
            s["conversations"], add_generation_prompt=True, tokenize=False)
        texts.append(text)
    ds = Dataset.from_dict({"text": texts})

    recipe = [
        QuantizationModifier(
            scheme="W8A8",
            targets=["Linear"],
            ignore=["embed_tokens", "lm_head"],
        ),
    ]

    print(f"[4/4] Applying QuantizationModifier W8A8...")
    t0 = time.time()
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=2048,
        num_calibration_samples=2048,
    )
    elapsed = time.time() - t0
    print(f"  Quantization done in {elapsed:.1f}s")

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
    print("Done!")

if __name__ == "__main__":
    main()
