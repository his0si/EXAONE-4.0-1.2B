#!/usr/bin/env python3
"""Batch W8A8 experiments to push score above 0.64.

Experiments:
1. cal=4096, seq=2048, damp=0.02
2. cal=8192, seq=2048, damp=0.02
3. cal=2048, seq=4096, damp=0.02
4. cal=4096, seq=4096, damp=0.02
5. Selective quantization: skip first/last 2 layers
"""
import os, sys, gc, time, json, shutil
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_MODEL = os.path.join(ROOT, "base_model")

EXPERIMENTS = [
    # (name, dampening, cal_samples, max_seq, ignore_layers)
    ("c4096_s2048",  0.02,  4096, 2048, None),
    ("c8192_s2048",  0.02,  8192, 2048, None),
    ("c2048_s4096",  0.02,  2048, 4096, None),
    ("c4096_s4096",  0.02,  4096, 4096, None),
    ("selective_v1", 0.02,  2048, 2048, ["model.layers.0", "model.layers.1", "model.layers.28", "model.layers.29"]),
]


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


def quantize_one(name, dampening, cal_samples, max_seq, ignore_layers, tokenizer):
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    output_dir = os.path.join(ROOT, "models", f"exp_{name}")
    if os.path.isdir(output_dir) and os.path.exists(os.path.join(output_dir, "config.json")):
        print(f"\n[SKIP] {name} already exists")
        return output_dir

    print(f"\n{'='*60}")
    print(f"[QUANT] {name}: damp={dampening}, cal={cal_samples}, seq={max_seq}")
    if ignore_layers:
        print(f"  Ignoring layers: {ignore_layers}")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)

    ds = load_calibration_data(tokenizer, cal_samples, max_seq)
    print(f"  Loaded {len(ds)} calibration samples")

    ignore = ["embed_tokens", "lm_head"]
    if ignore_layers:
        ignore.extend(ignore_layers)

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
    print(f"  Quantization done in {elapsed:.1f}s")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)

    size_mb = sum(f.stat().st_size for f in Path(output_dir).rglob("*") if f.is_file()) / (1024*1024)
    print(f"  Saved to {output_dir} ({size_mb:.1f} MB)")

    del model, ds
    cleanup_gpu()
    return output_dir


def create_zip(name, model_dir):
    """Create submission zip with short name."""
    zip_name = f"exp_{name}.zip"
    zip_path = os.path.join(ROOT, "submit", zip_name)

    staging = "/tmp/zip_staging"
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(os.path.join(staging, "model"))

    for f in Path(model_dir).iterdir():
        if f.is_file():
            shutil.copy2(f, os.path.join(staging, "model", f.name))

    import subprocess
    subprocess.run(["zip", "-r", zip_path, "model/"], cwd=staging,
                   capture_output=True)

    size_gb = os.path.getsize(zip_path) / (1024**3)
    print(f"  Created {zip_name} ({size_gb:.2f} GB)")
    return zip_path


def main():
    print(f"{'='*60}")
    print(f"Batch W8A8 Experiments — {len(EXPERIMENTS)} configs")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    for name, damp, cal, seq, ignore in EXPERIMENTS:
        try:
            model_dir = quantize_one(name, damp, cal, seq, ignore, tokenizer)
            create_zip(name, model_dir)
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
            cleanup_gpu()

    print(f"\n{'='*60}")
    print(f"All experiments complete!")
    print(f"Zips in submit/exp_*.zip")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
