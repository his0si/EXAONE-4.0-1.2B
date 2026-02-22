"""Batch evaluate all quantized models."""
import os
import sys
import json
import torch
import gc

sys.path.insert(0, os.path.dirname(__file__))
from utils.evaluate_local import evaluate_model, clear_gpu

if torch.cuda.is_available():
    torch.cuda.init()
    torch.zeros(1).cuda()
    print(f"[INFO] CUDA: {torch.cuda.get_device_name(0)}")

BASE_MODEL = "/home/lgaimers/base_model"
MODELS_DIR = "/home/lgaimers/0214/models"
RESULTS_DIR = "/home/lgaimers/0214/results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Step 1: Evaluate base model (or load cached)
base_results_file = os.path.join(RESULTS_DIR, "base_results.json")
if os.path.exists(base_results_file):
    with open(base_results_file) as f:
        base_results = json.load(f)
    print(f"[INFO] Cached base: PPL={base_results['perplexity']:.2f}, speed={base_results['tokens_per_sec']:.2f} tok/s")
else:
    print("[INFO] Evaluating base model...")
    base_results = evaluate_model(BASE_MODEL, "base_model")
    with open(base_results_file, "w") as f:
        json.dump(base_results, f, indent=2)
    print(f"[INFO] Base results saved")

# Step 2: Find all quantized models
models = []
for name in sorted(os.listdir(MODELS_DIR)):
    model_dir = os.path.join(MODELS_DIR, name)
    if not os.path.isdir(model_dir):
        continue
    if name in ("lora_checkpoints", "lora_merged"):
        continue
    safetensors = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
    if safetensors:
        models.append((name, model_dir))

print(f"\n[INFO] Found {len(models)} models to evaluate")
for name, path in models:
    print(f"  - {name}")

# Step 3: Evaluate each model
all_results = []
for i, (name, path) in enumerate(models):
    print(f"\n[{i+1}/{len(models)}] Evaluating {name}...")
    result_file = os.path.join(RESULTS_DIR, f"{name}_eval.json")
    if os.path.exists(result_file):
        with open(result_file) as f:
            result = json.load(f)
        print(f"  [SKIP] Already evaluated: Score={result.get('local_score', 'N/A')}")
        all_results.append(result)
        continue

    try:
        result = evaluate_model(path, name, base_results)
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        all_results.append(result)
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        all_results.append({"name": name, "error": str(e)})

    clear_gpu()

# Step 4: Summary
print(f"\n{'='*80}")
print("EVALUATION SUMMARY")
print(f"{'='*80}")
print(f"{'Model':<35} {'Size':>6} {'PPL':>8} {'KMMLU':>7} {'PerfN':>7} {'SpeedN':>7} {'Score':>7}")
print("-" * 80)

# Sort by score
scored = [r for r in all_results if "local_score" in r]
scored.sort(key=lambda x: x["local_score"], reverse=True)

for r in scored:
    print(f"{r['name']:<35} {r.get('size_gb',0):>5.2f}G {r.get('perplexity',0):>8.2f} "
          f"{r.get('kmmlu_accuracy',0):>6.2%} {r.get('perf_norm',0):>7.4f} "
          f"{r.get('speed_norm',0):>7.4f} {r.get('local_score',0):>7.4f}")

print(f"\nBase model: PPL={base_results['perplexity']:.2f}, Speed={base_results['tokens_per_sec']:.2f} tok/s")

# Save summary
summary_file = os.path.join(RESULTS_DIR, "evaluation_summary.json")
with open(summary_file, "w") as f:
    json.dump({"base": base_results, "models": scored}, f, indent=2)
print(f"\nFull results saved to: {summary_file}")

# Top 3 recommendations
print(f"\n{'='*80}")
print("TOP 3 CANDIDATES FOR SUBMISSION")
print(f"{'='*80}")
for i, r in enumerate(scored[:3]):
    print(f"\n#{i+1}: {r['name']}")
    print(f"    Score: {r['local_score']:.4f} (PerfNorm={r['perf_norm']:.4f}, SpeedNorm={r['speed_norm']:.4f})")
    print(f"    Size: {r['size_gb']:.2f} GB, PPL: {r['perplexity']:.2f}, KMMLU: {r['kmmlu_accuracy']:.2%}")
    print(f"    Path: {r['path']}")
