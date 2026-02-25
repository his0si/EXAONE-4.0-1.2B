#!/usr/bin/env python3
"""W8A8 optimization experiments for better PerfNorm while keeping INT8 speed."""
import os, sys, gc, time, json, argparse
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
        print(f"  Loaded {len(samples)} from train_50k.json")
    elif os.path.exists(data_5k):
        samples = json.load(open(data_5k))[:num_samples]
        print(f"  Loaded {len(samples)} from train.json")
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


def run_experiment(name, input_model, output_dir, tokenizer,
                   cal_samples=512, max_seq=1024, dampening=0.01,
                   ignore_layers=None):
    """Run a single W8A8 quantization experiment."""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    print(f"\n{'='*60}")
    print(f"  Experiment: {name}")
    print(f"  cal_samples={cal_samples}, dampening={dampening}")
    if ignore_layers:
        print(f"  ignore_layers: {ignore_layers}")
    print(f"{'='*60}")

    print(f"  Loading model from {input_model}")
    model = AutoModelForCausalLM.from_pretrained(
        input_model, torch_dtype=torch.bfloat16, trust_remote_code=True)

    ds = load_calibration_data(tokenizer, cal_samples, max_seq)

    # Build ignore list
    ignore = ["embed_tokens", "lm_head"]
    if ignore_layers:
        ignore.extend(ignore_layers)

    recipe = [
        GPTQModifier(
            scheme="W8A8",
            targets=["Linear"],
            ignore=ignore,
            dampening_frac=dampening,
        )
    ]

    print(f"  Running oneshot W8A8...")
    t0 = time.time()
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_seq,
        num_calibration_samples=cal_samples,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)

    size_mb = get_dir_size_mb(output_dir)
    print(f"  Saved to {output_dir} ({size_mb:.1f} MB)")
    del model
    cleanup_gpu()
    return size_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True,
                        choices=["high_damp", "selective", "high_cal", "all"])
    parser.add_argument("--input", default="LGAI-EXAONE/EXAONE-4.0-1.2B")
    args = parser.parse_args()

    # Resolve input model path
    if '/' in args.input and not os.path.isabs(args.input) and \
       not os.path.isdir(os.path.join(ROOT, args.input)):
        input_model = args.input
    else:
        input_model = args.input if os.path.isabs(args.input) else os.path.join(ROOT, args.input)

    tok_path = os.path.join(ROOT, "models", "lane01_gptq_w4a16")
    if not os.path.isdir(tok_path):
        tok_path = input_model
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    experiments = []

    if args.experiment in ("high_damp", "all"):
        # Experiment 1: Higher dampening to reduce quantization error
        experiments.append(("w8a8_damp01", {
            "cal_samples": 2048, "dampening": 0.1, "max_seq": 1024,
            "output_dir": os.path.join(ROOT, "models", "w8a8_damp01"),
        }))

    if args.experiment in ("selective", "all"):
        # Experiment 2: Skip high-error layers (last 5 layers' gate/up_proj)
        # Layers 25-29 have errors up to 111.53 on gate_proj/up_proj
        skip_modules = []
        for layer_idx in range(25, 30):
            skip_modules.append(f"model.layers.{layer_idx}.mlp.gate_proj")
            skip_modules.append(f"model.layers.{layer_idx}.mlp.up_proj")
        experiments.append(("w8a8_selective_5", {
            "cal_samples": 2048, "dampening": 0.01, "max_seq": 1024,
            "output_dir": os.path.join(ROOT, "models", "w8a8_selective_5"),
            "ignore_layers": skip_modules,
        }))

        # Experiment 2b: Skip last 3 layers entirely
        skip_modules_3 = []
        for layer_idx in range(27, 30):
            for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                         "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
                skip_modules_3.append(f"model.layers.{layer_idx}.{proj}")
        experiments.append(("w8a8_selective_3full", {
            "cal_samples": 2048, "dampening": 0.01, "max_seq": 1024,
            "output_dir": os.path.join(ROOT, "models", "w8a8_selective_3full"),
            "ignore_layers": skip_modules_3,
        }))

    if args.experiment in ("high_cal", "all"):
        # Experiment 3: Very high calibration samples
        experiments.append(("w8a8_cal4096_d005", {
            "cal_samples": 4096, "dampening": 0.05, "max_seq": 1024,
            "output_dir": os.path.join(ROOT, "models", "w8a8_cal4096_d005"),
        }))

    for name, params in experiments:
        output_dir = params.pop("output_dir")
        ignore_layers = params.pop("ignore_layers", None)
        run_experiment(name, input_model, output_dir, tokenizer,
                       ignore_layers=ignore_layers, **params)
        print()

    print(f"\n[DONE] All {len(experiments)} experiments complete.")


if __name__ == "__main__":
    main()
