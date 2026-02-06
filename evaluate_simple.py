import os
import torch
import time
import json
import gc
from pathlib import Path

def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# CUDA 초기화
if torch.cuda.is_available():
    torch.cuda.init()
    torch.zeros(1).cuda()
    print(f"[INFO] CUDA: {torch.cuda.get_device_name(0)}")

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np

def measure_perplexity(model, tokenizer, texts, max_length=512):
    """Perplexity 측정"""
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)

    return np.exp(total_loss / total_tokens)

def measure_speed(model, tokenizer, prompts, max_new_tokens=64, num_runs=5):
    """추론 속도 측정"""
    # Warmup
    warmup = tokenizer("Hello", return_tensors="pt").to(model.device)
    with torch.no_grad():
        _ = model.generate(**warmup, max_new_tokens=5, do_sample=False)
    torch.cuda.synchronize()

    total_tokens = 0
    total_time = 0

    for prompt in prompts[:num_runs]:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        gen_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        total_tokens += gen_tokens
        total_time += elapsed

    return total_tokens / total_time, total_time / total_tokens

def evaluate_model(model_path, model_name, eval_texts, speed_prompts):
    """모델 평가"""
    clear_gpu_memory()

    print(f"\n{'='*60}")
    print(f"평가: {model_name}")
    print(f"{'='*60}")

    # 모델 크기
    model_size = sum(f.stat().st_size for f in Path(model_path).glob("*.safetensors")) / (1024**3)
    print(f"크기: {model_size:.2f} GB")

    # 로드
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    # GPU 메모리
    gpu_mem = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"GPU 메모리: {gpu_mem:.2f} GB")

    # PPL
    print("PPL 측정 중...")
    ppl = measure_perplexity(model, tokenizer, eval_texts)
    print(f"Perplexity: {ppl:.2f}")

    # 속도
    print("속도 측정 중...")
    tps, tpt = measure_speed(model, tokenizer, speed_prompts)
    print(f"속도: {tps:.2f} tok/s, {tpt*1000:.2f} ms/tok")

    del model, tokenizer
    clear_gpu_memory()

    return {
        "name": model_name,
        "path": model_path,
        "size_gb": model_size,
        "gpu_mem_gb": gpu_mem,
        "perplexity": ppl,
        "tokens_per_sec": tps,
        "time_per_token": tpt,
    }

def calculate_score(base, model):
    """점수 계산"""
    perf_norm = base["perplexity"] / model["perplexity"]
    speed_norm = 1 - (model["time_per_token"] / base["time_per_token"])
    score = max(0.5 * perf_norm + 0.5 * speed_norm, 0)
    return perf_norm, speed_norm, score

def main():
    print("데이터 로드 중...")
    ds = load_dataset("LGAI-EXAONE/MANTA-1M", split="train[:40]")

    eval_texts = []
    for item in ds:
        if "conversations" in item:
            text = " ".join([c.get("content", "") for c in item["conversations"] if c.get("content")])
            if len(text) > 100:
                eval_texts.append(text[:800])
    eval_texts = eval_texts[:20]

    speed_prompts = [
        "한국의 수도는 어디인가요?",
        "인공지능이 미래 사회에 미치는 영향을 설명해주세요.",
        "Python과 Java의 차이점은 무엇인가요?",
        "기후 변화의 원인과 해결책을 알려주세요.",
        "효과적인 학습 방법에 대해 조언해주세요.",
    ]

    # 평가할 모델들
    models = [("./base_model", "Base Model")]

    # 양자화 모델 탐색
    for d in sorted(Path(".").iterdir()):
        if d.is_dir() and d.name.startswith("model"):
            if d.name != "base_model" and (d / "model.safetensors").exists():
                models.append((str(d), d.name))

    # 평가 실행
    results = []
    for path, name in models:
        if os.path.exists(path):
            try:
                r = evaluate_model(path, name, eval_texts, speed_prompts)
                results.append(r)
            except Exception as e:
                print(f"[ERROR] {name}: {e}")

    # 결과 출력
    print("\n" + "="*80)
    print("평가 결과 요약")
    print("="*80)
    print(f"\n{'모델':<25} {'크기':<8} {'PPL':<10} {'tok/s':<10} {'ms/tok':<10}")
    print("-"*80)

    for r in results:
        print(f"{r['name']:<25} {r['size_gb']:<8.2f} {r['perplexity']:<10.2f} {r['tokens_per_sec']:<10.2f} {r['time_per_token']*1000:<10.2f}")

    # 점수 계산
    if len(results) > 1:
        print("\n" + "="*80)
        print("점수 계산")
        print("="*80)
        print(f"\n{'모델':<25} {'PerfNorm':<12} {'SpeedNorm':<12} {'Score':<10}")
        print("-"*80)

        base = results[0]
        for r in results[1:]:
            pn, sn, sc = calculate_score(base, r)
            print(f"{r['name']:<25} {pn:<12.4f} {sn:<12.4f} {sc:<10.4f}")

    # 저장
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n결과 저장: eval_results.json")

if __name__ == "__main__":
    main()
