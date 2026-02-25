#!/usr/bin/env python3
"""Quantize the KD model to INT8 (W8A8) and FP8_DYNAMIC."""
import os, sys, gc, time, json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_dir_size_mb(path):
    total = 0
    for f in Path(path).rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


def load_calibration_data(tokenizer, num_samples=512, max_seq=1024):
    data_path = os.path.join(ROOT, "data", "manta", "train.json")
    if os.path.exists(data_path):
        samples = json.load(open(data_path))[:num_samples]
        texts = []
        for s in samples:
            text = tokenizer.apply_chat_template(
                s["conversations"], add_generation_prompt=True, tokenize=False)
            texts.append(text)
        from datasets import Dataset
        return Dataset.from_dict({"text": texts})
    else:
        from datasets import load_dataset
        ds = load_dataset("LGAI-EXAONE/MANTA-1M", split=f"train[:{num_samples}]")
        def preprocess(ex):
            return {"text": tokenizer.apply_chat_template(
                ex["conversations"], add_generation_prompt=True, tokenize=False)}
        return ds.map(preprocess)


def quantize_fp8_dynamic(input_model, output_dir, tokenizer):
    """FP8_DYNAMIC quantization (no calibration needed)."""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    print(f"[FP8_DYN] Loading model from {input_model}")
    model = AutoModelForCausalLM.from_pretrained(
        input_model, torch_dtype=torch.bfloat16, trust_remote_code=True)

    recipe = [
        QuantizationModifier(
            targets="Linear",
            scheme="FP8_DYNAMIC",
            ignore=["lm_head"],
        )
    ]

    print("[FP8_DYN] Applying FP8_DYNAMIC...")
    t0 = time.time()
    oneshot(model=model, recipe=recipe)
    elapsed = time.time() - t0
    print(f"[FP8_DYN] Done in {elapsed:.1f}s")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)

    size_mb = get_dir_size_mb(output_dir)
    print(f"[FP8_DYN] Saved to {output_dir} ({size_mb:.1f} MB)")
    del model
    cleanup_gpu()
    return size_mb


def quantize_w8a8(input_model, output_dir, tokenizer, cal_samples=512, max_seq=1024):
    """W8A8 INT8 quantization via GPTQ."""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    print(f"[W8A8] Loading model from {input_model}")
    model = AutoModelForCausalLM.from_pretrained(
        input_model, torch_dtype=torch.bfloat16, trust_remote_code=True)

    ds = load_calibration_data(tokenizer, cal_samples, max_seq)
    print(f"[W8A8] Calibration: {len(ds)} samples, max_seq={max_seq}")

    recipe = [
        GPTQModifier(
            scheme="W8A8",
            targets=["Linear"],
            ignore=["embed_tokens", "lm_head"],
            dampening_frac=0.01,
        )
    ]

    print("[W8A8] Running oneshot...")
    t0 = time.time()
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_seq,
        num_calibration_samples=cal_samples,
    )
    elapsed = time.time() - t0
    print(f"[W8A8] Done in {elapsed:.1f}s")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)

    size_mb = get_dir_size_mb(output_dir)
    print(f"[W8A8] Saved to {output_dir} ({size_mb:.1f} MB)")
    del model
    cleanup_gpu()
    return size_mb


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fp8", "int8", "both"], default="both")
    args = parser.parse_args()

    input_model = os.path.join(ROOT, "checkpoints", "kd_35_78b")
    tok_path = os.path.join(ROOT, "models", "lane01_gptq_w4a16")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    if args.mode in ("fp8", "both"):
        fp8_out = os.path.join(ROOT, "models", "kd_35_78b_fp8dyn")
        quantize_fp8_dynamic(input_model, fp8_out, tokenizer)

    if args.mode in ("int8", "both"):
        int8_out = os.path.join(ROOT, "models", "kd_35_78b_w8a8")
        quantize_w8a8(input_model, int8_out, tokenizer)


if __name__ == "__main__":
    main()
