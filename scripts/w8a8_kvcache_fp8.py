#!/usr/bin/env python3
"""W8A8 + FP8 KV Cache quantization.
Embeds kv_cache_scheme in the model config so vLLM auto-applies FP8 KV cache."""
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


def load_calibration_data(tokenizer, num_samples=2048, max_seq=1024):
    data_path = os.path.join(ROOT, "data", "manta", "train.json")
    samples = json.load(open(data_path))[:num_samples]
    texts = [tokenizer.apply_chat_template(
        s["conversations"], add_generation_prompt=True, tokenize=False) for s in samples]
    from datasets import Dataset
    return Dataset.from_dict({"text": texts})


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dampening", type=float, default=0.02)
    parser.add_argument("--cal-samples", type=int, default=2048)
    parser.add_argument("--output", default="models/w8a8_kv_fp8")
    args = parser.parse_args()

    output_dir = args.output if os.path.isabs(args.output) else os.path.join(ROOT, args.output)

    tok_path = os.path.join(ROOT, "models", "lane01_gptq_w4a16")
    if not os.path.isdir(tok_path):
        tok_path = "LGAI-EXAONE/EXAONE-4.0-1.2B"
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    print(f"Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "LGAI-EXAONE/EXAONE-4.0-1.2B", torch_dtype=torch.bfloat16, trust_remote_code=True)

    ds = load_calibration_data(tokenizer, args.cal_samples, 1024)
    print(f"Calibration: {len(ds)} samples")

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    # W8A8 GPTQ + FP8 KV Cache
    recipe = [
        GPTQModifier(
            scheme="W8A8",
            targets=["Linear"],
            ignore=["embed_tokens", "lm_head"],
            dampening_frac=args.dampening,
            kv_cache_scheme={
                "num_bits": 8,
                "type": "float",
                "strategy": "tensor",
                "dynamic": False,
                "symmetric": True,
            },
        )
    ]

    print(f"Running W8A8 + FP8 KV Cache (damp={args.dampening})...")
    t0 = time.time()
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=1024,
        num_calibration_samples=args.cal_samples,
    )
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)

    size_mb = get_dir_size_mb(output_dir)
    print(f"Saved to {output_dir} ({size_mb:.1f} MB)")

    # Verify kv_cache_scheme is in config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    kv_scheme = cfg.get("quantization_config", {}).get("kv_cache_scheme")
    print(f"KV cache scheme in config: {kv_scheme}")

    del model
    cleanup_gpu()
    print(f"\n[DONE] W8A8 + FP8 KV Cache saved to {output_dir}")


if __name__ == "__main__":
    main()
