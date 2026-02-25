#!/usr/bin/env python3
"""Fair speed comparison: load each model fresh and run identical benchmark.

Tests top sweep candidates + reference model to verify SpeedNorm is consistent.
"""
import os, sys, gc, time, json
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG = os.path.join(ROOT, "logs", "speed_fair_test.log")

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

# Models to test
MODELS = [
    ("baseline_bf16", os.path.join(ROOT, "models", "baseline_bf16")),
    ("w8a8_cal2048_d02", os.path.join(ROOT, "models", "w8a8_cal2048_d02")),
    ("sweep_d030_c2048", os.path.join(ROOT, "models", "sweep_d030_c2048")),
    ("sweep_d015_c2048", os.path.join(ROOT, "models", "sweep_d015_c2048")),
    ("sweep_d02_c2048_s2048", os.path.join(ROOT, "models", "sweep_d02_c2048_s2048")),
    ("sweep_d025_c3072", os.path.join(ROOT, "models", "sweep_d025_c3072")),
    ("sweep_d02_c1536", os.path.join(ROOT, "models", "sweep_d02_c1536")),
    ("sweep_d02_c3072", os.path.join(ROOT, "models", "sweep_d02_c3072")),
]

# Speed test prompts (same as 04_eval_vllm.py)
PROMPTS = [
    "대한민국의 수도는 어디인가요?",
    "What is machine learning?",
    "Python에서 리스트를 정렬하는 방법을 설명해주세요.",
    "인공지능의 장점과 단점을 설명해주세요.",
    "Write a Python function to calculate fibonacci numbers.",
    "지구온난화의 원인과 해결방안은 무엇인가요?",
    "Explain quantum computing in simple terms.",
    "한국의 전통 음식에 대해 소개해주세요.",
]

NUM_RUNS = 5  # More runs for reliable median


def test_speed(name, model_path):
    """Load model, run speed test, unload completely."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"\n{'='*60}")
    print(f"[SPEED] Testing: {name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    formatted = []
    for p in PROMPTS:
        messages = [{"role": "user", "content": p}]
        formatted.append(tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False))

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
    )

    sampling = SamplingParams(temperature=0.0, max_tokens=256)

    # Warmup (2 runs)
    for _ in range(2):
        _ = llm.generate(formatted[:2], sampling)

    # Actual runs
    all_times = []
    all_tokens = []
    for run_i in range(NUM_RUNS):
        t0 = time.perf_counter()
        outs = llm.generate(formatted, sampling)
        t1 = time.perf_counter()
        tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        elapsed = t1 - t0
        all_times.append(elapsed)
        all_tokens.append(tokens)
        print(f"  Run {run_i+1}: {elapsed:.4f}s, {tokens} tokens, {tokens/elapsed:.1f} tok/s")

    # Median
    median_idx = sorted(range(len(all_times)), key=lambda i: all_times[i])[len(all_times)//2]
    med_time = all_times[median_idx]
    med_tokens = all_tokens[median_idx]
    med_tps = med_tokens / med_time
    med_spt = med_time / med_tokens

    print(f"  Median: {med_time:.4f}s, {med_tokens} tokens, {med_tps:.1f} tok/s, {med_spt:.6f} s/tok")

    # Cleanup
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(3)  # Cool down

    return {
        "name": name,
        "median_time": med_time,
        "median_tokens": med_tokens,
        "tokens_per_sec": round(med_tps, 2),
        "sec_per_token": round(med_spt, 6),
        "all_times": [round(t, 4) for t in all_times],
        "all_tokens": all_tokens,
    }


def main():
    print(f"Fair Speed Comparison - {len(MODELS)} models, {NUM_RUNS} runs each")
    print(f"Log: {LOG}")

    results = []
    for name, path in MODELS:
        if not os.path.isdir(path):
            print(f"[SKIP] {name}: {path} not found")
            continue
        try:
            r = test_speed(name, path)
            results.append(r)
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"Speed Comparison Summary")
    print(f"{'='*60}")

    base_spt = None
    for r in results:
        if r["name"] == "baseline_bf16":
            base_spt = r["sec_per_token"]

    print(f"\n{'Model':<28} {'tok/s':>8} {'sec/tok':>10} {'SpeedNorm':>10}")
    print(f"{'-'*28} {'-'*8} {'-'*10} {'-'*10}")
    for r in results:
        sn = 1 - (r["sec_per_token"] / base_spt) if base_spt else 0
        marker = " ← baseline" if r["name"] == "baseline_bf16" else ""
        print(f"{r['name']:<28} {r['tokens_per_sec']:>8.1f} {r['sec_per_token']:>10.6f} {sn:>10.4f}{marker}")

    # Save results
    out_path = os.path.join(ROOT, "results", "speed_fair_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
