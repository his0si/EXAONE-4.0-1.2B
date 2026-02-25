#!/usr/bin/env python3
"""Advanced W8A8 experiments: SpinQuant, SmoothQuant, group_size, weight averaging.

Usage:
  conda run -n quant python scripts/w8a8_advanced.py
  conda run -n quant python scripts/w8a8_advanced.py --exp spinquant
  conda run -n quant python scripts/w8a8_advanced.py --exp smoothquant
  conda run -n quant python scripts/w8a8_advanced.py --exp groupsize
  conda run -n quant python scripts/w8a8_advanced.py --exp avg
  conda run -n quant python scripts/w8a8_advanced.py --eval-only
"""
import os, sys, gc, json, time, subprocess, argparse, copy
import torch
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_MODEL = os.path.join(ROOT, "base_model")
LOG = os.path.join(ROOT, "logs", "w8a8_advanced.log")
os.makedirs(os.path.dirname(LOG), exist_ok=True)
log_f = open(LOG, "w", buffering=1)

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data); f.flush()
    def flush(self):
        for f in self.files: f.flush()

sys.stdout = Tee(sys.__stdout__, log_f)
sys.stderr = Tee(sys.__stderr__, log_f)

BASE_PERF = 0.325
BASE_SPT = 0.000552

# Register EXAONE architecture in SpinQuant registries at module level
# Must happen before any SpinQuantModifier instantiation
def _register_exaone_spinquant():
    try:
        from llmcompressor.modifiers.transform.spinquant.norm_mappings import (
            NormMapping, NORM_MAPPING_REGISTRY
        )
        from llmcompressor.modifiers.transform.spinquant.mappings import (
            SPINQUANT_MAPPING_REGISTRY, _default_mappings
        )
        NORM_MAPPING_REGISTRY["Exaone4ForCausalLM"] = [
            NormMapping(norm="re:.*post_feedforward_layernorm$",
                        linears=["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"]),
            NormMapping(norm="re:.*post_attention_layernorm$",
                        linears=["re:.*up_proj$", "re:.*gate_proj$"]),
            NormMapping(norm="model.norm", linears=["lm_head"]),
        ]
        SPINQUANT_MAPPING_REGISTRY["Exaone4ForCausalLM"] = _default_mappings
        print("[INIT] Registered EXAONE in SpinQuant registries")
    except Exception as e:
        print(f"[WARN] SpinQuant registration failed: {e}")

_register_exaone_spinquant()


def load_cal_data(tokenizer, n_samples=2048):
    """Load MANTA calibration data (same method as w8a8_boost.py)."""
    from datasets import load_dataset, Dataset
    import json as _json

    data_50k = os.path.join(ROOT, "data", "manta", "train_50k.json")
    data_5k = os.path.join(ROOT, "data", "manta", "train.json")
    if os.path.exists(data_50k) and n_samples > 5000:
        samples = _json.load(open(data_50k))[:n_samples]
    elif os.path.exists(data_5k):
        samples = _json.load(open(data_5k))[:n_samples]
    else:
        ds = load_dataset("LGAI-EXAONE/MANTA-1M", split=f"train[:{n_samples}]")
        samples = [{"conversations": row["conversations"]} for row in ds]

    texts = []
    for s in samples:
        text = tokenizer.apply_chat_template(
            s["conversations"], add_generation_prompt=True, tokenize=False)
        texts.append(text)
    return Dataset.from_dict({"text": texts})


def evaluate(name, model_dir):
    """Run evaluation."""
    lane_id = name
    eval_script = os.path.join(ROOT, "scripts", "04_eval_vllm.py")
    result_path = os.path.join(ROOT, "results", lane_id, "metrics.json")
    if os.path.exists(result_path):
        os.remove(result_path)
    cmd = [sys.executable, eval_script, "--model", model_dir,
           "--lane-id", lane_id, "--force"]
    r = subprocess.run(cmd, capture_output=False, text=True)
    if r.returncode != 0:
        print(f"  [EVAL FAIL] {name}")
        return None
    if os.path.exists(result_path):
        d = json.load(open(result_path))
        perf = d.get("perf_aggregate", 0) or 0
        spt = d["speed"]["sec_per_token"]
        bspt = d["speed"]["bench_sec_per_token"]
        pn = perf / BASE_PERF
        sn = 1 - (spt / BASE_SPT)
        sc = 0.5 * pn + 0.5 * sn
        print(f"  [EVAL OK] {name}: perf={perf:.4f} PN={pn:.3f} SN={sn:+.3f} Score={sc:.4f}")
        return d
    return None


def exp_spinquant_w8a8():
    """SpinQuant R1+R2 (Hadamard rotation) + W8A8 GPTQ."""
    print("\n" + "="*60)
    print("[EXP] SpinQuant R1+R2 + W8A8 GPTQ")
    print("="*60)

    out_dir = os.path.join(ROOT, "models", "adv_spinquant_w8a8")
    if os.path.isdir(out_dir):
        print(f"  Already exists: {out_dir}")
        return out_dir

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization.gptq import GPTQModifier
    from llmcompressor.modifiers.transform.spinquant import SpinQuantModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16,
                                                  trust_remote_code=True)
    cal_data = load_cal_data(tokenizer, n_samples=2048)

    # SpinQuant R1+R2 (offline rotations, no runtime cost)
    # EXAONE mappings registered at module level above
    spin_modifier = SpinQuantModifier(rotations=["R1", "R2"])

    # W8A8 GPTQ
    gptq_modifier = GPTQModifier(
        scheme="W8A8",
        targets=["Linear"],
        ignore=["lm_head"],
        dampening_frac=0.02,
        actorder="group",
    )

    oneshot(
        model=model,
        dataset=cal_data,
        recipe=[spin_modifier, gptq_modifier],
        max_seq_length=1024,
        num_calibration_samples=2048,
    )

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    size_mb = sum(f.stat().st_size for f in Path(out_dir).glob("*")) / 1e6
    print(f"  Saved: {out_dir} ({size_mb:.1f} MB)")

    del model; gc.collect(); torch.cuda.empty_cache()
    return out_dir


def exp_smoothquant_w8a8():
    """SmoothQuant + W8A8 GPTQ."""
    print("\n" + "="*60)
    print("[EXP] SmoothQuant + W8A8 GPTQ")
    print("="*60)

    out_dir = os.path.join(ROOT, "models", "adv_smoothquant_w8a8")
    if os.path.isdir(out_dir):
        print(f"  Already exists: {out_dir}")
        return out_dir

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization.gptq import GPTQModifier
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16,
                                                  trust_remote_code=True)
    cal_data = load_cal_data(tokenizer, n_samples=2048)

    smooth_modifier = SmoothQuantModifier(smoothing_strength=0.5)

    gptq_modifier = GPTQModifier(
        scheme="W8A8",
        targets=["Linear"],
        ignore=["embed_tokens", "lm_head"],
        dampening_frac=0.02,
        actorder="group",
    )

    oneshot(
        model=model,
        dataset=cal_data,
        recipe=[smooth_modifier, gptq_modifier],
        max_seq_length=1024,
        num_calibration_samples=2048,
    )

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    size_mb = sum(f.stat().st_size for f in Path(out_dir).glob("*")) / 1e6
    print(f"  Saved: {out_dir} ({size_mb:.1f} MB)")

    del model; gc.collect(); torch.cuda.empty_cache()
    return out_dir


def exp_groupsize_w8a8():
    """W8A8 GPTQ with group_size variations."""
    print("\n" + "="*60)
    print("[EXP] W8A8 GPTQ with group_size=32 and group_size=64")
    print("="*60)

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization.gptq import GPTQModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    results = {}
    for gs in [32, 64]:
        out_dir = os.path.join(ROOT, "models", f"adv_gs{gs}_w8a8")
        if os.path.isdir(out_dir):
            print(f"  Already exists: {out_dir}")
            results[gs] = out_dir
            continue

        print(f"\n  group_size={gs}:")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16,
                                                      device_map="auto")
        cal_data = load_cal_data(tokenizer, n_samples=2048)

        gptq_modifier = GPTQModifier(
            dampening_frac=0.02,
            actorder="group",
            config_groups={
                "group_0": {
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "group",
                        "group_size": gs,
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "token",
                        "dynamic": True,
                    },
                    "targets": ["Linear"],
                },
            },
            ignore=["embed_tokens", "lm_head"],
        )

        oneshot(
            model=model,
            dataset=cal_data,
            recipe=[gptq_modifier],
            max_seq_length=2048,
            num_calibration_samples=2048,
        )

        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        size_mb = sum(f.stat().st_size for f in Path(out_dir).glob("*")) / 1e6
        print(f"  Saved: {out_dir} ({size_mb:.1f} MB)")
        results[gs] = out_dir

        del model; gc.collect(); torch.cuda.empty_cache()

    return results


def exp_weight_avg():
    """Average weights of multiple W8A8 models."""
    print("\n" + "="*60)
    print("[EXP] Weight Averaging of Top W8A8 Models")
    print("="*60)

    out_dir = os.path.join(ROOT, "models", "adv_weight_avg")
    if os.path.isdir(out_dir):
        print(f"  Already exists: {out_dir}")
        return out_dir

    import safetensors.torch as st
    import shutil

    # Models to average (top performers with same architecture)
    model_dirs = [
        os.path.join(ROOT, "models", "w8a8_cal2048_d02"),
        os.path.join(ROOT, "models", "sweep_d015_c2048"),
        os.path.join(ROOT, "models", "sweep_d030_c2048"),
    ]

    # Verify all exist
    for d in model_dirs:
        if not os.path.isdir(d):
            print(f"  [SKIP] {d} not found")
            return None

    print(f"  Averaging {len(model_dirs)} models:")
    for d in model_dirs:
        print(f"    - {os.path.basename(d)}")

    # Load all state dicts
    state_dicts = []
    for d in model_dirs:
        sd = st.load_file(os.path.join(d, "model.safetensors"))
        state_dicts.append(sd)
        print(f"    Loaded {os.path.basename(d)}: {len(sd)} tensors")

    # Average
    avg_sd = {}
    all_keys = state_dicts[0].keys()
    for key in all_keys:
        tensors = [sd[key].float() for sd in state_dicts]
        avg = torch.stack(tensors).mean(dim=0)
        avg_sd[key] = avg.to(state_dicts[0][key].dtype)

    print(f"  Averaged {len(avg_sd)} tensors")

    # Save: copy first model's non-safetensor files, replace safetensors
    os.makedirs(out_dir, exist_ok=True)
    ref_dir = model_dirs[0]
    for f in os.listdir(ref_dir):
        if f != "model.safetensors":
            shutil.copy2(os.path.join(ref_dir, f), os.path.join(out_dir, f))

    st.save_file(avg_sd, os.path.join(out_dir, "model.safetensors"))
    size_mb = sum(f.stat().st_size for f in Path(out_dir).glob("*")) / 1e6
    print(f"  Saved: {out_dir} ({size_mb:.1f} MB)")

    del state_dicts, avg_sd; gc.collect()
    return out_dir


def exp_spinquant_smoothquant_w8a8():
    """SpinQuant R1+R2 + SmoothQuant + W8A8 GPTQ (combined)."""
    print("\n" + "="*60)
    print("[EXP] SpinQuant + SmoothQuant + W8A8 GPTQ (combined)")
    print("="*60)

    out_dir = os.path.join(ROOT, "models", "adv_spin_smooth_w8a8")
    if os.path.isdir(out_dir):
        print(f"  Already exists: {out_dir}")
        return out_dir

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization.gptq import GPTQModifier
    from llmcompressor.modifiers.transform.spinquant import SpinQuantModifier
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16,
                                                  trust_remote_code=True)
    cal_data = load_cal_data(tokenizer, n_samples=2048)

    # EXAONE mappings registered at module level above
    spin_modifier = SpinQuantModifier(rotations=["R1", "R2"])
    smooth_modifier = SmoothQuantModifier(smoothing_strength=0.5)

    gptq_modifier = GPTQModifier(
        scheme="W8A8",
        targets=["Linear"],
        ignore=["lm_head"],
        dampening_frac=0.02,
        actorder="group",
    )

    oneshot(
        model=model,
        dataset=cal_data,
        recipe=[spin_modifier, smooth_modifier, gptq_modifier],
        max_seq_length=1024,
        num_calibration_samples=2048,
    )

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    size_mb = sum(f.stat().st_size for f in Path(out_dir).glob("*")) / 1e6
    print(f"  Saved: {out_dir} ({size_mb:.1f} MB)")

    del model; gc.collect(); torch.cuda.empty_cache()
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="all",
                        choices=["all", "spinquant", "smoothquant", "groupsize", "avg", "combined"])
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    experiments = {
        "spinquant": ("adv_spinquant_w8a8", exp_spinquant_w8a8),
        "smoothquant": ("adv_smoothquant_w8a8", exp_smoothquant_w8a8),
        "avg": ("adv_weight_avg", exp_weight_avg),
        "combined": ("adv_spin_smooth_w8a8", exp_spinquant_smoothquant_w8a8),
    }

    if args.exp == "all":
        run_exps = ["spinquant", "smoothquant", "avg", "combined"]
    else:
        run_exps = [args.exp]

    # Phase 1: Run experiments
    model_dirs = {}
    if not args.eval_only:
        print("=" * 60)
        print("# Phase 1: Quantization")
        print("=" * 60)

        for exp_name in run_exps:
            if exp_name == "groupsize":
                # Special handling for multiple models
                results = exp_groupsize_w8a8()
                if results:
                    for gs, d in results.items():
                        model_dirs[f"adv_gs{gs}_w8a8"] = d
            else:
                name, func = experiments[exp_name]
                result = func()
                if result:
                    model_dirs[name] = result

    # Phase 2: Evaluate
    print("\n" + "=" * 60)
    print("# Phase 2: Evaluation")
    print("=" * 60)

    if args.eval_only:
        # Find all adv_* models
        models_dir = os.path.join(ROOT, "models")
        for d in os.listdir(models_dir):
            if d.startswith("adv_"):
                model_dirs[d] = os.path.join(models_dir, d)

    eval_results = {}
    for name, model_dir in model_dirs.items():
        if not os.path.isdir(model_dir):
            continue
        print(f"\n  [EVAL] {name}...")
        d = evaluate(name, model_dir)
        if d:
            eval_results[name] = d

    # Phase 3: Results
    print("\n" + "#" * 60)
    print("# Phase 3: Results")
    print("#" * 60)

    print(f"\nBaseline: perf={BASE_PERF}, spt={BASE_SPT}")
    print(f"Reference: w8a8_cal2048_d02 → Server 0.6208")
    print(f"\n{'Model':<30} {'Perf':>7} {'PN':>6} {'SN':>7} {'Score':>7}")
    print("-" * 62)

    # Add reference
    ref_path = os.path.join(ROOT, "results", "w8a8_cal2048_d02", "metrics.json")
    if os.path.exists(ref_path):
        rd = json.load(open(ref_path))
        rp = rd.get("perf_aggregate", 0) or 0
        rspt = rd["speed"]["sec_per_token"]
        rpn = rp / BASE_PERF
        rsn = 1 - (rspt / BASE_SPT)
        rsc = 0.5 * rpn + 0.5 * rsn
        print(f"{'w8a8_cal2048_d02 (ref)':<30} {rp:>7.4f} {rpn:>6.3f} {rsn:>+7.3f} {rsc:>7.4f}")
        print()

    for name, d in sorted(eval_results.items()):
        perf = d.get("perf_aggregate", 0) or 0
        spt = d["speed"]["sec_per_token"]
        pn = perf / BASE_PERF
        sn = 1 - (spt / BASE_SPT)
        sc = 0.5 * pn + 0.5 * sn
        marker = " ★" if sc > rsc else ""
        print(f"{name:<30} {perf:>7.4f} {pn:>6.3f} {sn:>+7.3f} {sc:>7.4f}{marker}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
