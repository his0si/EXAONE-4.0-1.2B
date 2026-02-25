#!/usr/bin/env python3
"""W8A8 quantization variants: SmoothQuant, better calibration, etc."""
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
    data_50k = os.path.join(ROOT, "data", "manta", "train_50k.json")
    data_5k = os.path.join(ROOT, "data", "manta", "train.json")
    if os.path.exists(data_50k) and num_samples > 5000:
        samples = json.load(open(data_50k))[:num_samples]
    elif os.path.exists(data_5k):
        samples = json.load(open(data_5k))[:num_samples]
    else:
        from datasets import load_dataset
        ds = load_dataset("LGAI-EXAONE/MANTA-1M", split=f"train[:{num_samples}]")
        samples = [{"conversations": row["conversations"]} for row in ds]
    texts = []
    for s in samples:
        text = tokenizer.apply_chat_template(
            s["conversations"], add_generation_prompt=True, tokenize=False)
        texts.append(text)
    from datasets import Dataset
    return Dataset.from_dict({"text": texts})


def quantize_w8a8_smoothquant(input_model, output_dir, tokenizer,
                               cal_samples=512, max_seq=1024, dampening=0.01,
                               smoothing_strength=0.5):
    """W8A8 with SmoothQuant pre-processing."""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

    print(f"\n[SQ+W8A8] Loading model from {input_model}")
    model = AutoModelForCausalLM.from_pretrained(
        input_model, torch_dtype=torch.bfloat16, trust_remote_code=True)

    ds = load_calibration_data(tokenizer, cal_samples, max_seq)
    print(f"[SQ+W8A8] Calibration: {len(ds)} samples, smoothing={smoothing_strength}")

    recipe = [
        SmoothQuantModifier(smoothing_strength=smoothing_strength),
        GPTQModifier(
            scheme="W8A8",
            targets=["Linear"],
            ignore=["embed_tokens", "lm_head"],
            dampening_frac=dampening,
        ),
    ]

    print("[SQ+W8A8] Running oneshot...")
    t0 = time.time()
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_seq,
        num_calibration_samples=cal_samples,
    )
    elapsed = time.time() - t0
    print(f"[SQ+W8A8] Done in {elapsed:.1f}s")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)
    size_mb = get_dir_size_mb(output_dir)
    print(f"[SQ+W8A8] Saved to {output_dir} ({size_mb:.1f} MB)")
    del model
    cleanup_gpu()
    return size_mb


def quantize_w8a8_gptq(input_model, output_dir, tokenizer,
                        cal_samples=512, max_seq=1024, dampening=0.01):
    """Standard W8A8 GPTQ with configurable calibration."""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    print(f"\n[W8A8] Loading model from {input_model}")
    model = AutoModelForCausalLM.from_pretrained(
        input_model, torch_dtype=torch.bfloat16, trust_remote_code=True)

    ds = load_calibration_data(tokenizer, cal_samples, max_seq)
    print(f"[W8A8] Calibration: {len(ds)} samples, dampening={dampening}")

    recipe = [
        GPTQModifier(
            scheme="W8A8",
            targets=["Linear"],
            ignore=["embed_tokens", "lm_head"],
            dampening_frac=dampening,
        ),
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
    parser.add_argument("--mode", choices=["smoothquant", "gptq", "both"], default="both")
    parser.add_argument("--input", default="LGAI-EXAONE/EXAONE-4.0-1.2B")
    parser.add_argument("--cal-samples", type=int, default=512)
    parser.add_argument("--max-seq", type=int, default=1024)
    parser.add_argument("--dampening", type=float, default=0.01)
    parser.add_argument("--smoothing", type=float, default=0.5)
    parser.add_argument("--output-suffix", type=str, default="")
    args = parser.parse_args()

    # Resolve input path
    if '/' in args.input and not os.path.isabs(args.input) and not os.path.isdir(os.path.join(ROOT, args.input)):
        input_model = args.input
    else:
        input_model = args.input if os.path.isabs(args.input) else os.path.join(ROOT, args.input)

    tok_path = os.path.join(ROOT, "models", "lane01_gptq_w4a16")
    if not os.path.isdir(tok_path):
        tok_path = input_model
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    suffix = args.output_suffix

    if args.mode in ("smoothquant", "both"):
        out = os.path.join(ROOT, "models", f"w8a8_sq{suffix}")
        quantize_w8a8_smoothquant(
            input_model, out, tokenizer,
            cal_samples=args.cal_samples, max_seq=args.max_seq,
            dampening=args.dampening, smoothing_strength=args.smoothing)

    if args.mode in ("gptq", "both"):
        out = os.path.join(ROOT, "models", f"w8a8_cal{args.cal_samples}{suffix}")
        quantize_w8a8_gptq(
            input_model, out, tokenizer,
            cal_samples=args.cal_samples, max_seq=args.max_seq,
            dampening=args.dampening)


if __name__ == "__main__":
    main()
