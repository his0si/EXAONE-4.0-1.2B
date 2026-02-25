#!/usr/bin/env python3
"""Quantize models to INT8 (W8A8) and FP8_DYNAMIC for L4 tensor core compatibility."""
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


def quantize_int8(input_model, output_dir, tokenizer, cal_samples=512, max_seq=1024):
    """W8A8 INT8 quantization via GPTQ (SmoothQuant-style)."""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    print(f"\n[INT8] Loading model from {input_model}")
    model = AutoModelForCausalLM.from_pretrained(
        input_model, torch_dtype=torch.bfloat16, trust_remote_code=True)

    ds = load_calibration_data(tokenizer, cal_samples, max_seq)
    print(f"[INT8] Calibration: {len(ds)} samples, max_seq={max_seq}")

    recipe = [
        GPTQModifier(
            scheme="W8A8",
            targets=["Linear"],
            ignore=["embed_tokens", "lm_head"],
            dampening_frac=0.01,
        )
    ]

    print("[INT8] Running oneshot W8A8...")
    t0 = time.time()
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_seq,
        num_calibration_samples=cal_samples,
    )
    elapsed = time.time() - t0
    print(f"[INT8] Done in {elapsed:.1f}s")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)

    size_mb = get_dir_size_mb(output_dir)
    print(f"[INT8] Saved to {output_dir} ({size_mb:.1f} MB)")
    del model
    cleanup_gpu()
    return size_mb


def quantize_fp8_dynamic(input_model, output_dir, tokenizer):
    """FP8_DYNAMIC quantization (no calibration)."""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    print(f"\n[FP8] Loading model from {input_model}")
    model = AutoModelForCausalLM.from_pretrained(
        input_model, torch_dtype=torch.bfloat16, trust_remote_code=True)

    recipe = [
        QuantizationModifier(
            targets="Linear",
            scheme="FP8_DYNAMIC",
            ignore=["lm_head"],
        )
    ]

    print("[FP8] Applying FP8_DYNAMIC...")
    t0 = time.time()
    oneshot(model=model, recipe=recipe)
    elapsed = time.time() - t0
    print(f"[FP8] Done in {elapsed:.1f}s")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)

    size_mb = get_dir_size_mb(output_dir)
    print(f"[FP8] Saved to {output_dir} ({size_mb:.1f} MB)")
    del model
    cleanup_gpu()
    return size_mb


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input model path")
    parser.add_argument("--output", required=True, help="Output model path")
    parser.add_argument("--scheme", required=True, choices=["INT8", "FP8_DYNAMIC"])
    parser.add_argument("--cal-samples", type=int, default=512)
    parser.add_argument("--max-seq", type=int, default=1024)
    args = parser.parse_args()

    # For HF model IDs (contain '/'), don't prepend ROOT
    if '/' in args.input and not os.path.isabs(args.input) and not os.path.isdir(os.path.join(ROOT, args.input)):
        input_model = args.input  # HF model ID
    else:
        input_model = args.input if os.path.isabs(args.input) else os.path.join(ROOT, args.input)
    output_dir = args.output if os.path.isabs(args.output) else os.path.join(ROOT, args.output)

    tok_path = os.path.join(ROOT, "models", "lane01_gptq_w4a16")
    if not os.path.isdir(tok_path):
        tok_path = input_model
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    if args.scheme == "INT8":
        quantize_int8(input_model, output_dir, tokenizer, args.cal_samples, args.max_seq)
    elif args.scheme == "FP8_DYNAMIC":
        quantize_fp8_dynamic(input_model, output_dir, tokenizer)

    print(f"\n[DONE] {args.scheme} model saved to {output_dir}")


if __name__ == "__main__":
    main()
