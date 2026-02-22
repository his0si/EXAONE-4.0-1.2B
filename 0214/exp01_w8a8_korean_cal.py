"""
Experiment 1: W8A8 + Korean Calibration Data
Low risk, incremental improvement over current best (0.6208).
Tests KMMLU-based calibration vs MANTA-1M.
"""
import os
import sys
import json
import torch
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

BASE_MODEL = "/home/lgaimers/base_model"
MODELS_DIR = "/home/lgaimers/0214/models"
RESULTS_DIR = "/home/lgaimers/0214/results"

EXPERIMENTS = {
    "w8a8_kmmlu_pure": {
        "num_samples": 2048,
        "include_manta": False,
        "max_seq_length": 512,
        "dampening_frac": 0.02,
    },
    "w8a8_kmmlu_mixed": {
        "num_samples": 2048,
        "include_manta": True,
        "manta_ratio": 0.5,
        "max_seq_length": 512,
        "dampening_frac": 0.02,
    },
    "w8a8_kmmlu_large": {
        "num_samples": 4096,
        "include_manta": True,
        "manta_ratio": 0.5,
        "max_seq_length": 512,
        "dampening_frac": 0.02,
    },
    "w8a8_kmmlu_longseq": {
        "num_samples": 2048,
        "include_manta": True,
        "manta_ratio": 0.5,
        "max_seq_length": 1024,
        "dampening_frac": 0.02,
    },
}


def run_experiment(name, config):
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Config: {config}")
    print(f"{'='*60}")

    out_dir = os.path.join(MODELS_DIR, name)
    if os.path.exists(os.path.join(out_dir, "model.safetensors")):
        print(f"[SKIP] {name} already exists")
        return

    print("[INFO] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
    )

    print("[INFO] Loading calibration data...")
    cal_ds = load_korean_calibration(
        tokenizer,
        num_samples=config["num_samples"],
        include_manta=config.get("include_manta", True),
        manta_ratio=config.get("manta_ratio", 0.5),
    )

    print(f"[INFO] W8A8 GPTQ quantization (damp={config['dampening_frac']})...")
    recipe = [
        GPTQModifier(
            scheme="W8A8",
            targets=["Linear"],
            ignore=["embed_tokens", "lm_head"],
            dampening_frac=config["dampening_frac"],
        )
    ]

    oneshot(
        model=model,
        dataset=cal_ds,
        recipe=recipe,
        max_seq_length=config["max_seq_length"],
        num_calibration_samples=config["num_samples"],
    )

    print(f"[INFO] Saving to {out_dir}...")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, save_compressed=True)
    tokenizer.save_pretrained(out_dir)

    model_size = sum(f.stat().st_size for f in Path(out_dir).glob("*.safetensors")) / (1024**3)
    print(f"[INFO] Done: {name} ({model_size:.2f} GB)")

    # Save config
    with open(os.path.join(RESULTS_DIR, f"{name}_config.json"), "w") as f:
        json.dump({"name": name, "config": config, "size_gb": model_size}, f, indent=2)

    # Free memory
    del model, tokenizer
    import gc; gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Run specific experiment or all
    if len(sys.argv) > 1:
        name = sys.argv[1]
        if name in EXPERIMENTS:
            run_experiment(name, EXPERIMENTS[name])
        else:
            print(f"Unknown experiment: {name}")
            print(f"Available: {list(EXPERIMENTS.keys())}")
    else:
        for name, config in EXPERIMENTS.items():
            run_experiment(name, config)
