#!/usr/bin/env python3
"""
Test speed benchmark with different max_tokens to find settings matching server.

Server known results:
  w8a8_cal2048_d02: Score=0.6208, perf≈0.3233
  → PerfNorm≈0.995, SpeedNorm≈0.247
  → Server spt ≈ 0.000552 * (1 - 0.247) = 0.000416s

We test: max_tokens=[128, 256, 512, 1024, 2048] on both base and W8A8 models.
"""

import sys
import json
import time
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_PROMPTS = [
    "대한민국의 수도는 어디인가요?",
    "What is machine learning?",
    "Python에서 리스트를 정렬하는 방법을 설명해주세요.",
    "인공지능의 장점과 단점을 설명해주세요.",
    "Write a Python function to calculate fibonacci numbers.",
    "지구온난화의 원인과 해결방안은 무엇인가요?",
    "Explain quantum computing in simple terms.",
    "한국의 전통 음식에 대해 소개해주세요.",
]

MAX_TOKENS_LIST = [128, 256, 512, 1024, 2048]

def bench_model(model_path, max_tokens_list=MAX_TOKENS_LIST):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"\n[LOAD] {model_path}")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=16384,
    )

    # Format prompts
    formatted = []
    for p in TEST_PROMPTS:
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        formatted.append(text)

    results = {}

    for max_tok in max_tokens_list:
        params = SamplingParams(temperature=0.0, max_tokens=max_tok)

        # Warmup
        _ = llm.generate(formatted[:2], params)

        # 3 runs, median
        times = []
        tokens_list = []
        for _ in range(3):
            t0 = time.perf_counter()
            outputs = llm.generate(formatted, params)
            elapsed = time.perf_counter() - t0
            total_tok = sum(len(o.outputs[0].token_ids) for o in outputs)
            times.append(elapsed)
            tokens_list.append(total_tok)

        # Median
        mid = sorted(range(3), key=lambda i: times[i])[1]
        med_time = times[mid]
        med_tokens = tokens_list[mid]
        spt = med_time / med_tokens if med_tokens > 0 else 0
        tps = med_tokens / med_time if med_time > 0 else 0

        results[max_tok] = {
            "total_time_s": round(med_time, 4),
            "total_tokens": med_tokens,
            "sec_per_token": round(spt, 6),
            "tokens_per_sec": round(tps, 2),
            "all_times": [round(t, 4) for t in times],
            "all_tokens": tokens_list,
        }

        print(f"  max_tokens={max_tok:5d} | {med_tokens:5d} tok | "
              f"{med_time:.3f}s | spt={spt*1000:.3f}ms | {tps:.0f} tok/s")

    del llm
    return results


def main():
    base_model = os.path.join(ROOT, "base_model")
    w8a8_model = os.path.join(ROOT, "models", "w8a8_cal2048_d02")

    all_results = {}

    # Base model
    print("=" * 60)
    print("BASE MODEL")
    print("=" * 60)
    all_results["base"] = bench_model(base_model)

    # W8A8 model
    print("\n" + "=" * 60)
    print("W8A8 (cal2048_d02)")
    print("=" * 60)
    all_results["w8a8"] = bench_model(w8a8_model)

    # Compute SpeedNorm for each max_tokens setting
    print("\n" + "=" * 60)
    print("SPEED NORM COMPARISON")
    print("=" * 60)
    print(f"{'max_tokens':>10} | {'base_spt':>10} | {'w8a8_spt':>10} | {'SpeedNorm':>10}")
    print("-" * 50)

    for mt in MAX_TOKENS_LIST:
        base_spt = all_results["base"][mt]["sec_per_token"]
        w8a8_spt = all_results["w8a8"][mt]["sec_per_token"]
        sn = 1 - (w8a8_spt / base_spt) if base_spt > 0 else 0
        print(f"{mt:>10} | {base_spt*1000:>9.3f}ms | {w8a8_spt*1000:>9.3f}ms | {sn:>10.4f}")

    print(f"\nServer SpeedNorm for w8a8_cal2048_d02 ≈ 0.247")
    print(f"(Server Score=0.6208, perf≈0.3233 → PN≈0.995 → SN≈0.247)")

    # Save results
    out_path = os.path.join(ROOT, "results", "speed_calibrate.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
