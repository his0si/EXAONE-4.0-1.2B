#!/usr/bin/env python3
"""W8A8 parameter sweep: fine-tune dampening, calibration samples, and seq length.

Current best: dampening=0.02, cal=2048, seq=1024 → competition 0.6208
Goal: improve PerfNorm while maintaining standard W8A8 per-channel for optimal L4 INT8 speed.
"""
import os, sys, gc, time, json, subprocess
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG = os.path.join(ROOT, "logs", "w8a8_sweep.log")

# Redirect stdout/stderr to log file for nohup compatibility
os.makedirs(os.path.dirname(LOG), exist_ok=True)
log_f = open(LOG, "w", buffering=1)
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = Tee(sys.stdout, log_f)
sys.stderr = Tee(sys.stderr, log_f)

# ── Experiment configurations ──
EXPERIMENTS = [
    # (name, dampening, cal_samples, max_seq)
    # Dampening sweep (fix cal=2048, seq=1024)
    ("d015_c2048",  0.015, 2048, 1024),
    ("d025_c2048",  0.025, 2048, 1024),
    ("d030_c2048",  0.030, 2048, 1024),
    # Calibration sweep (fix damp=0.02, seq=1024)
    ("d02_c1024",   0.02,  1024, 1024),
    ("d02_c1536",   0.02,  1536, 1024),
    ("d02_c2560",   0.02,  2560, 1024),
    ("d02_c3072",   0.02,  3072, 1024),
    # Seq length sweep (fix damp=0.02, cal=2048)
    ("d02_c2048_s512",  0.02, 2048, 512),
    ("d02_c2048_s2048", 0.02, 2048, 2048),
    # Best combo candidates
    ("d025_c3072",  0.025, 3072, 1024),
]

BASE_MODEL = os.path.join(ROOT, "base_model")
if not os.path.isdir(BASE_MODEL):
    BASE_MODEL = "LGAI-EXAONE/EXAONE-4.0-1.2B"


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


def quantize_one(name, dampening, cal_samples, max_seq, tokenizer):
    """Quantize a single W8A8 variant."""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    output_dir = os.path.join(ROOT, "models", f"sweep_{name}")
    if os.path.isdir(output_dir) and os.path.exists(os.path.join(output_dir, "config.json")):
        print(f"\n[SKIP] {name} already exists at {output_dir}")
        return output_dir

    print(f"\n{'='*60}")
    print(f"[QUANT] {name}: damp={dampening}, cal={cal_samples}, seq={max_seq}")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)

    ds = load_calibration_data(tokenizer, cal_samples, max_seq)
    print(f"[QUANT] Loaded {len(ds)} calibration samples")

    recipe = [
        GPTQModifier(
            scheme="W8A8",
            targets=["Linear"],
            ignore=["embed_tokens", "lm_head"],
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
    print(f"[QUANT] Quantization done in {elapsed:.1f}s")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)
    size_mb = get_dir_size_mb(output_dir)
    print(f"[QUANT] Saved to {output_dir} ({size_mb:.1f} MB)")

    del model, ds
    cleanup_gpu()
    return output_dir


def evaluate_one(name, model_dir):
    """Evaluate a single model using 04_eval_vllm.py."""
    lane_id = f"sweep_{name}"
    result_path = os.path.join(ROOT, "results", lane_id, "metrics.json")

    if os.path.exists(result_path):
        print(f"\n[SKIP] {name} evaluation already exists at {result_path}")
        with open(result_path) as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"[EVAL] {name}")
    print(f"{'='*60}")

    eval_script = os.path.join(ROOT, "scripts", "04_eval_vllm.py")
    cmd = [
        sys.executable, eval_script,
        "--model", model_dir,
        "--lane-id", lane_id,
        "--force",
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[EVAL] FAILED for {name}")
        return None

    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)
    return None


def main():
    print(f"{'='*60}")
    print(f"W8A8 Parameter Sweep - {len(EXPERIMENTS)} experiments")
    print(f"Base model: {BASE_MODEL}")
    print(f"Log: {LOG}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Phase 1: Quantize all
    print(f"\n{'#'*60}")
    print(f"# Phase 1: Quantization")
    print(f"{'#'*60}")

    model_dirs = {}
    for name, dampening, cal_samples, max_seq in EXPERIMENTS:
        try:
            model_dir = quantize_one(name, dampening, cal_samples, max_seq, tokenizer)
            model_dirs[name] = model_dir
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
            cleanup_gpu()

    # Phase 2: Evaluate all
    print(f"\n{'#'*60}")
    print(f"# Phase 2: Evaluation")
    print(f"{'#'*60}")

    eval_results = {}
    for name, dampening, cal_samples, max_seq in EXPERIMENTS:
        if name not in model_dirs:
            continue
        try:
            result = evaluate_one(name, model_dirs[name])
            if result:
                eval_results[name] = result
        except Exception as e:
            print(f"[ERROR] {name} eval: {e}")
            import traceback
            traceback.print_exc()

    # Phase 3: Score calculation
    print(f"\n{'#'*60}")
    print(f"# Phase 3: Score Calculation")
    print(f"{'#'*60}")

    # Load baseline
    baseline_path = os.path.join(ROOT, "results", "baseline.json")
    with open(baseline_path) as f:
        baseline = json.load(f)
    base_perf = baseline["perf_aggregate"]
    base_bench_spt = baseline["speed"]["bench_sec_per_token"]

    print(f"\nBaseline: perf={base_perf}, bench_spt={base_bench_spt:.6f}")
    print(f"\n{'모델':<25} {'damp':>5} {'cal':>5} {'seq':>5} {'PerfAgg':>8} {'PerfNorm':>9} {'SpeedNorm':>10} {'Score':>7}")
    print(f"{'-'*25} {'-'*5} {'-'*5} {'-'*5} {'-'*8} {'-'*9} {'-'*10} {'-'*7}")

    # Reference: current best
    ref_perf = 0.3233  # w8a8_cal2048_d02 local
    ref_score = 0.6208  # w8a8_cal2048_d02 competition

    scored = []
    for name, dampening, cal_samples, max_seq in EXPERIMENTS:
        if name not in eval_results:
            continue
        r = eval_results[name]
        perf = r.get("perf_aggregate", 0)
        bench_spt = r.get("speed", {}).get("bench_sec_per_token", 0)
        if not perf or not bench_spt:
            continue

        perf_norm = perf / base_perf
        speed_norm = 1 - (bench_spt / base_bench_spt)
        score = max(0.5 * perf_norm + 0.5 * speed_norm, 0)

        scored.append((name, dampening, cal_samples, max_seq, perf, perf_norm, speed_norm, score))
        print(f"{name:<25} {dampening:>5.3f} {cal_samples:>5} {max_seq:>5} "
              f"{perf:>8.4f} {perf_norm:>9.4f} {speed_norm:>10.4f} {score:>7.4f}")

    # Sort by score
    scored.sort(key=lambda x: x[-1], reverse=True)

    print(f"\n{'='*60}")
    print(f"Ranking (Local Score 기준)")
    print(f"{'='*60}")
    print(f"  Reference: w8a8_cal2048_d02 → Local=0.5633, Competition=0.6208")
    print()
    for i, (name, damp, cal, seq, perf, pn, sn, sc) in enumerate(scored, 1):
        marker = " ★" if sc > 0.5633 else ""
        print(f"  {i}. {name}: Score={sc:.4f} (PerfNorm={pn:.4f}, SpeedNorm={sn:.4f}){marker}")

    print(f"\n[DONE] Sweep complete. Results in results/sweep_*/metrics.json")


if __name__ == "__main__":
    main()
