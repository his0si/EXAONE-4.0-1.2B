#!/usr/bin/env python3
"""Apply W8A8 quantization to pruned+KD models and evaluate.

Existing pruned+KD FP16 checkpoints:
  checkpoints/pruned28_kd  (28 layers, KD-trained)
  checkpoints/pruned26_kd  (26 layers, KD-trained)

These were previously only quantized to W4A16 which destroys quality
for 1.2B models. W8A8 is near-lossless and should preserve quality
while gaining speed from fewer layers.

Usage:
  conda run -n quant python scripts/pruned_w8a8.py
  conda run -n quant python scripts/pruned_w8a8.py --only 28
  conda run -n quant python scripts/pruned_w8a8.py --only 26
  conda run -n quant python scripts/pruned_w8a8.py --eval-only
"""
import os, sys, gc, time, json, subprocess
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG = os.path.join(ROOT, "logs", "pruned_w8a8.log")

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

# ── Configurations ──
EXPERIMENTS = [
    # (name, input_checkpoint, keep_layers)
    ("pruned28_kd_w8a8", os.path.join(ROOT, "checkpoints", "pruned28_kd"), 28),
    ("pruned26_kd_w8a8", os.path.join(ROOT, "checkpoints", "pruned26_kd"), 26),
]

DAMPENING = 0.02
CAL_SAMPLES = 2048
MAX_SEQ = 1024

BASE_MODEL = os.path.join(ROOT, "base_model")


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


def quantize_one(name, input_dir, tokenizer):
    """W8A8 quantize a pruned model."""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    output_dir = os.path.join(ROOT, "models", name)
    if os.path.isdir(output_dir) and os.path.exists(os.path.join(output_dir, "config.json")):
        print(f"\n[SKIP] {name} already exists at {output_dir}")
        return output_dir

    print(f"\n{'='*60}")
    print(f"[QUANT] {name}: W8A8 damp={DAMPENING}, cal={CAL_SAMPLES}")
    print(f"  Input: {input_dir}")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        input_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)
    print(f"[QUANT] Model loaded: {model.config.num_hidden_layers} layers")

    ds = load_calibration_data(tokenizer, CAL_SAMPLES, MAX_SEQ)
    print(f"[QUANT] Loaded {len(ds)} calibration samples")

    recipe = [
        GPTQModifier(
            scheme="W8A8",
            targets=["Linear"],
            ignore=["embed_tokens", "lm_head"],
            dampening_frac=DAMPENING,
        ),
    ]

    t0 = time.time()
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQ,
        num_calibration_samples=CAL_SAMPLES,
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
    """Evaluate using 04_eval_vllm.py."""
    result_path = os.path.join(ROOT, "results", name, "metrics.json")

    if os.path.exists(result_path):
        print(f"\n[SKIP] {name} evaluation already exists")
        with open(result_path) as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"[EVAL] {name}")
    print(f"{'='*60}")

    eval_script = os.path.join(ROOT, "scripts", "04_eval_vllm.py")
    cmd = [
        sys.executable, eval_script,
        "--model", model_dir,
        "--lane-id", name,
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=int, choices=[28, 26], help="Only run one variant")
    parser.add_argument("--eval-only", action="store_true", help="Skip quantization, just evaluate")
    args = parser.parse_args()

    experiments = EXPERIMENTS
    if args.only:
        experiments = [(n, d, l) for n, d, l in experiments if l == args.only]

    print(f"{'='*60}")
    print(f"Layer Pruning + W8A8 Quantization")
    print(f"  Experiments: {[n for n, _, _ in experiments]}")
    print(f"  W8A8 params: damp={DAMPENING}, cal={CAL_SAMPLES}, seq={MAX_SEQ}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Phase 1: Quantize
    if not args.eval_only:
        print(f"\n{'#'*60}")
        print(f"# Phase 1: W8A8 Quantization")
        print(f"{'#'*60}")

        for name, input_dir, keep_layers in experiments:
            if not os.path.isdir(input_dir):
                print(f"[ERROR] Checkpoint not found: {input_dir}")
                continue
            try:
                quantize_one(name, input_dir, tokenizer)
            except Exception as e:
                print(f"[ERROR] {name}: {e}")
                import traceback
                traceback.print_exc()
                cleanup_gpu()

    # Phase 2: Evaluate
    print(f"\n{'#'*60}")
    print(f"# Phase 2: Evaluation")
    print(f"{'#'*60}")

    eval_results = {}
    for name, input_dir, keep_layers in experiments:
        model_dir = os.path.join(ROOT, "models", name)
        if not os.path.isdir(model_dir):
            print(f"[ERROR] Model not found: {model_dir}")
            continue
        try:
            result = evaluate_one(name, model_dir)
            if result:
                eval_results[name] = result
        except Exception as e:
            print(f"[ERROR] {name} eval: {e}")
            import traceback
            traceback.print_exc()

    # Phase 3: Score comparison
    print(f"\n{'#'*60}")
    print(f"# Phase 3: Score Comparison")
    print(f"{'#'*60}")

    baseline_path = os.path.join(ROOT, "results", "baseline.json")
    with open(baseline_path) as f:
        baseline = json.load(f)
    base_perf = baseline["perf_aggregate"]
    base_spt = baseline["speed"]["sec_per_token"]

    print(f"\nBaseline: perf={base_perf}, sec_per_token={base_spt:.6f}")
    print(f"Current best: w8a8_cal2048_d02 → Competition Score = 0.6208\n")

    print(f"{'Model':<25} {'Layers':>6} {'PerfAgg':>8} {'PerfNorm':>9} "
          f"{'tok/s':>8} {'SpeedNorm':>10} {'Score':>7}")
    print(f"{'-'*25} {'-'*6} {'-'*8} {'-'*9} {'-'*8} {'-'*10} {'-'*7}")

    # Show reference models
    for ref_name, ref_path in [
        ("w8a8_30L (best)", os.path.join(ROOT, "results", "base_w8a8", "metrics.json")),
        ("prune28_W4A16", os.path.join(ROOT, "results", "lane10_prune28_kd_w4a16", "metrics.json")),
        ("prune26_W4A16", os.path.join(ROOT, "results", "lane11_prune26_kd_w4a16", "metrics.json")),
    ]:
        if os.path.exists(ref_path):
            with open(ref_path) as f:
                r = json.load(f)
            perf = r.get("perf_aggregate", 0) or 0
            spt = r.get("speed", {}).get("sec_per_token", 0) or 0
            tps = 1/spt if spt > 0 else 0
            pn = perf / base_perf if base_perf else 0
            sn = 1 - (spt / base_spt) if base_spt else 0
            sc = max(0.5 * pn + 0.5 * sn, 0)
            layers = r.get("benchmarks", {}).get("kmmlu_pro", {}).get("num_samples", "?")
            # Get layers from config
            model_p = r.get("model_path", "")
            try:
                with open(os.path.join(model_p, "config.json")) as cf:
                    layers = json.load(cf).get("num_hidden_layers", "?")
            except:
                layers = "?"
            print(f"{ref_name:<25} {str(layers):>6} {perf:>8.4f} {pn:>9.4f} "
                  f"{tps:>8.1f} {sn:>10.4f} {sc:>7.4f}")

    print()

    # Show new results
    for name, input_dir, keep_layers in experiments:
        if name not in eval_results:
            continue
        r = eval_results[name]
        perf = r.get("perf_aggregate", 0) or 0
        spt = r.get("speed", {}).get("sec_per_token", 0) or 0
        tps = 1/spt if spt > 0 else 0
        pn = perf / base_perf if base_perf else 0
        sn = 1 - (spt / base_spt) if base_spt else 0
        sc = max(0.5 * pn + 0.5 * sn, 0)
        marker = " ★" if sc > 0.5633 else ""
        print(f"{name:<25} {keep_layers:>6} {perf:>8.4f} {pn:>9.4f} "
              f"{tps:>8.1f} {sn:>10.4f} {sc:>7.4f}{marker}")

    print(f"\n[DONE]")


if __name__ == "__main__":
    main()
