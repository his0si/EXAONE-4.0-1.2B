#!/usr/bin/env python3
"""Recalculate local scores using the correct competition formula.

Competition formula:
  PerfNorm = Perf_model / Perf_base
  SpeedNorm = 1 - (bench_sec_per_token_model / bench_sec_per_token_base)
  Score = max(0.5 * PerfNorm + 0.5 * SpeedNorm, 0)

Key difference from previous calculation:
  - Uses bench_sec_per_token (actual benchmark inference time / tokens)
  - NOT the speed benchmark sec_per_token (8 prompts x 256 tokens)
"""
import json, glob, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load baseline
with open(os.path.join(ROOT, "results", "baseline.json")) as f:
    baseline = json.load(f)

base_perf = baseline["perf_aggregate"]
base_bench_spt = baseline["speed"]["bench_sec_per_token"]
base_speed_spt = baseline["speed"]["sec_per_token"]

print(f"{'='*90}")
print(f"경쟁 점수 재계산 (bench_sec_per_token 기준)")
print(f"{'='*90}")
print(f"Baseline: perf_agg={base_perf}, bench_spt={base_bench_spt:.6f}, speed_spt={base_speed_spt:.6f}")
print(f"{'='*90}")

# Known competition scores for comparison
competition = {
    "w8a8_cal2048_d02": 0.6208,
    "w8a8_damp005": 0.6077,
    "base_w8a8": 0.6042,
    "w8a8_gptq_adv": 0.5085,
    "w8a8_selective_5": 0.4718,
    "fp8_dynamic_submitted": 0.4936,
    "lane12_fp8_static": 0.4929,
    "lane01_gptq_w4a16": 0.5099,
    "kd_35_78b_w4a16": 0.2632,
}

# Collect all results
results = []
for fpath in sorted(glob.glob(os.path.join(ROOT, "results", "*", "metrics.json"))):
    dirname = os.path.basename(os.path.dirname(fpath))
    with open(fpath) as f:
        d = json.load(f)

    perf = d.get("perf_aggregate")
    speed = d.get("speed", {})
    bench_spt = speed.get("bench_sec_per_token")
    speed_spt = speed.get("sec_per_token")

    if perf is None or bench_spt is None:
        continue

    # New formula (correct - using bench_sec_per_token)
    perf_norm = perf / base_perf
    speed_norm_bench = 1 - (bench_spt / base_bench_spt)
    score_bench = max(0.5 * perf_norm + 0.5 * speed_norm_bench, 0)

    # Old formula (incorrect - using speed benchmark sec_per_token)
    speed_norm_speed = 1 - (speed_spt / base_speed_spt)
    score_speed = max(0.5 * perf_norm + 0.5 * speed_norm_speed, 0)

    comp_score = competition.get(dirname, None)

    results.append({
        "name": dirname,
        "perf": perf,
        "perf_norm": perf_norm,
        "bench_spt": bench_spt,
        "speed_norm_bench": speed_norm_bench,
        "score_bench": score_bench,
        "speed_spt": speed_spt,
        "speed_norm_speed": speed_norm_speed,
        "score_speed": score_speed,
        "comp_score": comp_score,
    })

# Sort by new score (descending)
results.sort(key=lambda x: x["score_bench"], reverse=True)

# Print header
print(f"\n{'모델':<30} {'PerfNorm':>9} {'SpeedNorm':>10} {'Score(new)':>11} {'Score(old)':>11} {'경쟁Score':>10} {'차이':>7}")
print(f"{'-'*30} {'-'*9} {'-'*10} {'-'*11} {'-'*11} {'-'*10} {'-'*7}")

for r in results:
    comp_str = f"{r['comp_score']:.4f}" if r['comp_score'] else "     -"
    diff_str = ""
    if r['comp_score']:
        diff = r['score_bench'] - r['comp_score']
        diff_str = f"{diff:+.4f}"

    print(f"{r['name']:<30} {r['perf_norm']:>9.4f} {r['speed_norm_bench']:>10.4f} "
          f"{r['score_bench']:>11.4f} {r['score_speed']:>11.4f} {comp_str:>10} {diff_str:>7}")

# Summary comparison for submitted models
print(f"\n{'='*90}")
print(f"제출 모델 비교 (경쟁 결과 있는 것만)")
print(f"{'='*90}")
print(f"{'모델':<30} {'Local(new)':>11} {'Local(old)':>11} {'경쟁':>10} {'차이(new)':>10} {'차이(old)':>10}")
print(f"{'-'*30} {'-'*11} {'-'*11} {'-'*10} {'-'*10} {'-'*10}")

for r in sorted(results, key=lambda x: x.get("comp_score") or 0, reverse=True):
    if r["comp_score"]:
        diff_new = r["score_bench"] - r["comp_score"]
        diff_old = r["score_speed"] - r["comp_score"]
        print(f"{r['name']:<30} {r['score_bench']:>11.4f} {r['score_speed']:>11.4f} "
              f"{r['comp_score']:>10.4f} {diff_new:>+10.4f} {diff_old:>+10.4f}")

print(f"\n참고: Score(new) = bench_sec_per_token 기반 (정확한 공식)")
print(f"      Score(old) = speed benchmark sec_per_token 기반 (이전 계산)")
print(f"      차이 = Local - 경쟁 (양수면 local이 과대평가)")
