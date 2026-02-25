#!/usr/bin/env python3
"""Single model speed test - run one at a time for fair comparison."""
import os, sys, gc, time, json, argparse
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--name", default=None)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    name = args.name or os.path.basename(args.model_path)
    model_path = args.model_path if os.path.isabs(args.model_path) else os.path.join(ROOT, args.model_path)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

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

    # Warmup
    for _ in range(2):
        _ = llm.generate(formatted[:2], sampling)

    all_times = []
    all_tokens = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        outs = llm.generate(formatted, sampling)
        t1 = time.perf_counter()
        tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        elapsed = t1 - t0
        all_times.append(elapsed)
        all_tokens.append(tokens)

    median_idx = sorted(range(len(all_times)), key=lambda i: all_times[i])[len(all_times)//2]
    med_time = all_times[median_idx]
    med_tokens = all_tokens[median_idx]

    result = {
        "name": name,
        "tok_s": round(med_tokens / med_time, 2),
        "spt": round(med_time / med_tokens, 6),
        "runs": [{"time": round(t, 4), "tokens": tok, "tok_s": round(tok/t, 1)}
                 for t, tok in zip(all_times, all_tokens)]
    }
    # Write to file directly (avoid conda run buffering)
    out_file = os.path.join(ROOT, "logs", "speed_fair_results.jsonl")
    with open(out_file, "a", buffering=1) as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
