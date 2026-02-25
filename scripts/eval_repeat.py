#!/usr/bin/env python3
"""Run evaluation N times per model to measure variance.

Usage:
  conda run -n quant python scripts/eval_repeat.py
  conda run -n quant python scripts/eval_repeat.py --runs 3
"""
import os, sys, json, subprocess, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG = os.path.join(ROOT, "logs", "eval_repeat.log")
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

# Baseline
BASE_PERF = 0.325
BASE_SPT = 0.000552

# Models to evaluate (name, model_dir)
MODELS = [
    ("w8a8_cal2048_d02", os.path.join(ROOT, "models", "w8a8_cal2048_d02")),
    ("sweep_d015_c2048", os.path.join(ROOT, "models", "sweep_d015_c2048")),
    ("sweep_d030_c2048", os.path.join(ROOT, "models", "sweep_d030_c2048")),
    ("boost_kmmlu_cal", os.path.join(ROOT, "models", "boost_kmmlu_cal")),
    ("boost_cal8192", os.path.join(ROOT, "models", "boost_cal8192")),
    ("boost_cal4096", os.path.join(ROOT, "models", "boost_cal4096")),
]


def evaluate_once(name, model_dir, run_idx):
    """Run one evaluation, return metrics dict."""
    lane_id = f"repeat_{name}_r{run_idx}"
    result_path = os.path.join(ROOT, "results", lane_id, "metrics.json")

    # Always force re-run
    if os.path.exists(result_path):
        os.remove(result_path)

    eval_script = os.path.join(ROOT, "scripts", "04_eval_vllm.py")
    cmd = [
        sys.executable, eval_script,
        "--model", model_dir,
        "--lane-id", lane_id,
        "--force",
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [FAIL] {lane_id} (exit={result.returncode}, {elapsed:.0f}s)")
        return None

    if os.path.exists(result_path):
        with open(result_path) as f:
            d = json.load(f)
        print(f"  [OK] {lane_id}: perf={d.get('perf_aggregate', 'N/A')}, "
              f"spt={d['speed']['sec_per_token']:.6f}, "
              f"bench_spt={d['speed']['bench_sec_per_token']:.6f} ({elapsed:.0f}s)")
        return d
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    N = args.runs
    print(f"{'='*70}")
    print(f"Repeated Evaluation: {len(MODELS)} models x {N} runs")
    print(f"{'='*70}")

    all_results = {}  # name -> [metrics, ...]

    for name, model_dir in MODELS:
        if not os.path.isdir(model_dir):
            print(f"\n[SKIP] {name}: model not found at {model_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"[MODEL] {name} ({N} runs)")
        print(f"{'='*60}")

        runs = []
        for i in range(N):
            print(f"\n  Run {i+1}/{N}:")
            d = evaluate_once(name, model_dir, i+1)
            if d:
                runs.append(d)

        all_results[name] = runs

    # Summary
    print(f"\n{'#'*70}")
    print(f"# SUMMARY: {N} runs per model")
    print(f"{'#'*70}")

    print(f"\n{'Model':<25} {'Run':>4} {'Perf':>7} {'PN':>6} {'SN':>7} {'Score':>7} {'spt':>9} {'b_spt':>9}")
    print('-' * 85)

    model_avgs = []
    for name, model_dir in MODELS:
        if name not in all_results or not all_results[name]:
            continue
        runs = all_results[name]

        scores = []
        for i, d in enumerate(runs):
            perf = d.get('perf_aggregate', 0) or 0
            spt = d['speed']['sec_per_token']
            bspt = d['speed']['bench_sec_per_token']
            pn = perf / BASE_PERF
            sn = 1 - (spt / BASE_SPT)
            sc = max(0.5 * pn + 0.5 * sn, 0)
            scores.append((perf, pn, sn, sc, spt, bspt))
            print(f"{name if i==0 else '':<25} {i+1:>4} {perf:>7.4f} {pn:>6.3f} {sn:>+7.3f} {sc:>7.4f} {spt:>9.6f} {bspt:>9.6f}")

        # Average
        avg_perf = sum(s[0] for s in scores) / len(scores)
        avg_pn = sum(s[1] for s in scores) / len(scores)
        avg_sn = sum(s[2] for s in scores) / len(scores)
        avg_sc = sum(s[3] for s in scores) / len(scores)
        avg_spt = sum(s[4] for s in scores) / len(scores)
        avg_bspt = sum(s[5] for s in scores) / len(scores)
        min_sc = min(s[3] for s in scores)
        max_sc = max(s[3] for s in scores)

        print(f"{'  → avg':<25} {'':>4} {avg_perf:>7.4f} {avg_pn:>6.3f} {avg_sn:>+7.3f} {avg_sc:>7.4f} {avg_spt:>9.6f} {avg_bspt:>9.6f}")
        print(f"{'  → range':<25} {'':>4} {'':>7} {'':>6} {'':>7} {min_sc:>7.4f}~{max_sc:.4f}")
        print()

        model_avgs.append((name, avg_sc, min_sc, max_sc, avg_perf, avg_pn, avg_sn))

    # Final ranking
    model_avgs.sort(key=lambda x: x[1], reverse=True)
    print(f"\n{'='*70}")
    print(f"RANKING by Average Score")
    print(f"{'='*70}")
    for i, (name, avg, mn, mx, perf, pn, sn) in enumerate(model_avgs, 1):
        print(f"  {i}. {name:<25} avg={avg:.4f} (range {mn:.4f}~{mx:.4f}) PN={pn:.3f} SN={sn:+.3f}")

    print(f"\nReference: w8a8_cal2048_d02 server score = 0.6208")
    print(f"\n[DONE]")


if __name__ == "__main__":
    main()
