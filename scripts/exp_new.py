#!/usr/bin/env python3
"""New W8A8 experiments - reuse sweep infrastructure."""
import os, sys, gc, time, json, shutil, subprocess
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_MODEL = os.path.join(ROOT, "base_model")


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_calibration_data(tokenizer, num_samples, max_seq):
    data_50k = os.path.join(ROOT, "data", "manta", "train_50k.json")
    data_5k = os.path.join(ROOT, "data", "manta", "train.json")
    if num_samples > 4750 and os.path.exists(data_50k):
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


def quantize_and_zip(name, dampening, cal_samples, max_seq, tokenizer, ignore_extra=None):
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    output_dir = os.path.join(ROOT, "models", f"exp_{name}")
    zip_path = os.path.join(ROOT, "submit", f"exp_{name}.zip")

    if os.path.isdir(output_dir) and os.path.exists(os.path.join(output_dir, "config.json")):
        print(f"\n[SKIP] {name} already exists", flush=True)
        if not os.path.exists(zip_path):
            _create_zip(output_dir, zip_path)
        return

    print(f"\n{'='*60}", flush=True)
    print(f"[QUANT] {name}: damp={dampening}, cal={cal_samples}, seq={max_seq}", flush=True)
    if ignore_extra:
        print(f"  Extra ignore: {ignore_extra}", flush=True)
    print(f"{'='*60}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)

    ds = load_calibration_data(tokenizer, cal_samples, max_seq)
    print(f"  Loaded {len(ds)} calibration samples", flush=True)

    ignore = ["embed_tokens", "lm_head"]
    if ignore_extra:
        ignore.extend(ignore_extra)

    recipe = [
        GPTQModifier(
            scheme="W8A8",
            targets=["Linear"],
            ignore=ignore,
            dampening_frac=dampening,
        ),
    ]

    t0 = time.time()
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_seq,
        num_calibration_samples=cal_samples,
    )
    elapsed = time.time() - t0
    print(f"  Quantization done in {elapsed:.1f}s", flush=True)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)
    print(f"  Saved to {output_dir}", flush=True)

    del model, ds
    cleanup_gpu()

    _create_zip(output_dir, zip_path)


def _create_zip(model_dir, zip_path):
    staging = "/tmp/zip_staging"
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(os.path.join(staging, "model"))
    for f in Path(model_dir).iterdir():
        if f.is_file():
            shutil.copy2(f, os.path.join(staging, "model", f.name))
    subprocess.run(["zip", "-r", zip_path, "model/"],
                   cwd=staging, capture_output=True)
    size_gb = os.path.getsize(zip_path) / (1024**3)
    print(f"  Zip: {os.path.basename(zip_path)} ({size_gb:.2f} GB)", flush=True)


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Experiment 1: More calibration data
    quantize_and_zip("c4096_s2048", 0.02, 4096, 2048, tokenizer)

    # Experiment 2: Selective quantization (first/last 2 layers FP16)
    quantize_and_zip("selective", 0.02, 2048, 2048, tokenizer,
                     ignore_extra=["model.layers.0", "model.layers.1",
                                   "model.layers.28", "model.layers.29"])

    # Experiment 3: dampening 0.025 + cal 2048 + seq 2048
    quantize_and_zip("d025_s2048", 0.025, 2048, 2048, tokenizer)

    print("\n=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()
