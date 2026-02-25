#!/usr/bin/env python3
"""
03_quantize.py — Quantize a model to HuggingFace-standard format
                  compatible with vLLM 0.14.1.

Supports: GPTQ (W4A16, W8A16, FP8, …) via llmcompressor,
          and QuantizationModifier for FP8_DYNAMIC.

Usage:
  python scripts/03_quantize.py --lane lane01_gptq_w4a16
  python scripts/03_quantize.py --input-model checkpoints/sft_merged \\
         --output models/lane06_sft_gptq_w4a16 --scheme W4A16 --dampening 0.01
"""
import os, sys, gc, json, time, argparse
import yaml, torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_config(path="configs/lanes.yaml"):
    with open(os.path.join(ROOT, path)) as f:
        return yaml.safe_load(f)


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


def load_calibration_data(tokenizer, num_samples, max_seq):
    """Load MANTA-1M calibration data."""
    data_path = os.path.join(ROOT, "data", "manta", "train.json")
    if os.path.exists(data_path):
        import json as _json
        samples = _json.load(open(data_path))[:num_samples]
        texts = []
        for s in samples:
            text = tokenizer.apply_chat_template(
                s["conversations"], add_generation_prompt=True, tokenize=False)
            texts.append(text)
        from datasets import Dataset
        return Dataset.from_dict({"text": texts})
    else:
        ds = load_dataset("LGAI-EXAONE/MANTA-1M", split=f"train[:{num_samples}]")
        def preprocess(ex):
            return {"text": tokenizer.apply_chat_template(
                ex["conversations"], add_generation_prompt=True, tokenize=False)}
        return ds.map(preprocess)


def quantize_gptq(input_model, output_dir, tokenizer, scheme, cal_samples,
                   max_seq, dampening_frac=0.01):
    """GPTQ-based quantization via llmcompressor."""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    print(f"[GPTQ] Loading model from {input_model}")
    model = AutoModelForCausalLM.from_pretrained(
        input_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )

    ds = load_calibration_data(tokenizer, cal_samples, max_seq)
    print(f"[GPTQ] Calibration: {len(ds)} samples, max_seq={max_seq}")

    kwargs = {}
    if dampening_frac and dampening_frac > 0:
        kwargs["dampening_frac"] = dampening_frac

    recipe = [
        GPTQModifier(
            scheme=scheme,
            targets=["Linear"],
            ignore=["embed_tokens", "lm_head"],
            **kwargs,
        )
    ]

    print(f"[GPTQ] Running oneshot (scheme={scheme}, damp={dampening_frac})...")
    t0 = time.time()
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_seq,
        num_calibration_samples=cal_samples,
    )
    elapsed = time.time() - t0
    print(f"[GPTQ] Done in {elapsed:.1f}s")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)

    size_mb = get_dir_size_mb(output_dir)
    print(f"[GPTQ] Saved to {output_dir} ({size_mb:.1f} MB)")

    del model
    cleanup_gpu()
    return {"scheme": scheme, "size_mb": round(size_mb, 1), "time_s": round(elapsed, 1)}


def quantize_quantmod(input_model, output_dir, tokenizer, scheme):
    """QuantizationModifier for FP8_DYNAMIC (no calibration needed)."""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    print(f"[QuantMod] Loading model from {input_model}")
    model = AutoModelForCausalLM.from_pretrained(
        input_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )

    recipe = [
        QuantizationModifier(
            targets="Linear",
            scheme=scheme,
            ignore=["lm_head"],
        )
    ]

    print(f"[QuantMod] Applying {scheme}...")
    t0 = time.time()
    oneshot(model=model, recipe=recipe)
    elapsed = time.time() - t0
    print(f"[QuantMod] Done in {elapsed:.1f}s")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)

    size_mb = get_dir_size_mb(output_dir)
    print(f"[QuantMod] Saved to {output_dir} ({size_mb:.1f} MB)")

    del model
    cleanup_gpu()
    return {"scheme": scheme, "size_mb": round(size_mb, 1), "time_s": round(elapsed, 1)}


def validate_vllm_load(model_path):
    """Quick check: can vLLM load this model?"""
    print(f"[Validate] Testing vLLM load of {model_path}...")
    try:
        from vllm import LLM
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.50,
            max_model_len=512,
        )
        del llm
        cleanup_gpu()
        print("[Validate] ✓ vLLM load OK")
        return True
    except Exception as e:
        print(f"[Validate] ✗ vLLM load FAILED: {e}")
        cleanup_gpu()
        return False


# ════════════════════════════════════════════════════════════
#  Lane-based dispatch
# ════════════════════════════════════════════════════════════
def run_lane(lane_id, cfg, force=False):
    lane = cfg["lanes"][lane_id]
    base_path = os.path.normpath(os.path.join(ROOT, cfg["project"]["base_model"]))
    output_dir = os.path.join(ROOT, "models", lane_id)

    if not force and os.path.exists(os.path.join(output_dir, "model.safetensors")):
        print(f"[{lane_id}] Already exists at {output_dir}, skipping. (use --force to re-run)")
        return output_dir

    # Force: clean existing output
    if force and os.path.isdir(output_dir):
        import shutil
        shutil.rmtree(output_dir)
        print(f"[{lane_id}] Removed existing {output_dir} (--force)")

    # Determine input model
    input_model = base_path
    if lane.get("input_model"):
        candidate = os.path.join(ROOT, lane["input_model"])
        if os.path.isdir(candidate):
            input_model = candidate
        else:
            print(f"[{lane_id}] Input model {candidate} not found, using base model")

    # For structural lanes (kd), use the KD output as input
    if lane.get("train_mode") == "kd" and lane.get("prune_params"):
        keep = lane["prune_params"]["keep_layers"]
        kd_dir = os.path.join(ROOT, "checkpoints", f"pruned{keep}_kd")
        if os.path.isdir(kd_dir):
            input_model = kd_dir
        else:
            print(f"[{lane_id}] KD model not found at {kd_dir}, using base model")

    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    qp = lane.get("quant_params", {})
    quant_mode = lane.get("quant_mode", "none")

    if quant_mode == "none":
        # No quantization — just copy/symlink
        if input_model != base_path:
            import shutil
            os.makedirs(output_dir, exist_ok=True)
            for f in Path(input_model).iterdir():
                if f.is_file():
                    shutil.copy2(f, output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"[{lane_id}] Copied FP16 model to {output_dir}")
        else:
            print(f"[{lane_id}] No quant + no training = base model (skip)")
            return base_path
        return output_dir

    elif quant_mode == "gptq":
        result = quantize_gptq(
            input_model=input_model,
            output_dir=output_dir,
            tokenizer=tokenizer,
            scheme=qp["scheme"],
            cal_samples=qp.get("cal_samples", 512),
            max_seq=qp.get("max_seq", 1024),
            dampening_frac=qp.get("dampening_frac", 0.01),
        )

    elif quant_mode == "quantmod":
        result = quantize_quantmod(
            input_model=input_model,
            output_dir=output_dir,
            tokenizer=tokenizer,
            scheme=qp["scheme"],
        )
    else:
        print(f"[{lane_id}] Unknown quant_mode: {quant_mode}")
        return None

    # Save quantization metadata
    meta_path = os.path.join(output_dir, "quant_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"lane_id": lane_id, **result}, f, indent=2)

    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", type=str, default=None,
                        help="Lane ID from lanes.yaml")
    parser.add_argument("--config", default="configs/lanes.yaml")
    # Manual mode
    parser.add_argument("--input-model", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--scheme", type=str, default="W4A16")
    parser.add_argument("--dampening", type=float, default=0.01)
    parser.add_argument("--cal-samples", type=int, default=512)
    parser.add_argument("--max-seq", type=int, default=1024)
    parser.add_argument("--validate", action="store_true",
                        help="Validate vLLM loadability after quantization")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing outputs")
    args = parser.parse_args()

    cfg = load_config(os.path.join(ROOT, args.config))

    if args.lane:
        if args.lane not in cfg["lanes"]:
            print(f"ERROR: Lane '{args.lane}' not found in config")
            sys.exit(1)
        out = run_lane(args.lane, cfg, force=args.force)
        if args.validate and out:
            validate_vllm_load(out)
    elif args.input_model and args.output:
        base_path = os.path.normpath(os.path.join(ROOT, cfg["project"]["base_model"]))
        tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
        inp = args.input_model if os.path.isabs(args.input_model) else os.path.join(ROOT, args.input_model)
        out = args.output if os.path.isabs(args.output) else os.path.join(ROOT, args.output)
        quantize_gptq(inp, out, tokenizer, args.scheme,
                       args.cal_samples, args.max_seq, args.dampening)
        if args.validate:
            validate_vllm_load(out)
    else:
        print("ERROR: Provide either --lane or --input-model + --output")
        sys.exit(1)


if __name__ == "__main__":
    main()
