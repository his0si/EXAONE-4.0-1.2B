import os
import torch
import time
import json
from pathlib import Path

# CUDA 초기화
if torch.cuda.is_available():
    torch.cuda.init()
    torch.zeros(1).cuda()
    print(f"[INFO] CUDA initialized: {torch.cuda.get_device_name(0)}")

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np

def load_model(model_path):
    """모델과 토크나이저 로드"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer

def measure_perplexity(model, tokenizer, texts, max_length=512):
    """Perplexity 측정 (낮을수록 좋음)"""
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity

def measure_inference_speed(model, tokenizer, prompts, max_new_tokens=128, num_runs=3):
    """추론 속도 측정 (tokens/sec)"""
    total_tokens = 0
    total_time = 0

    # Warmup
    warmup_input = tokenizer("Hello", return_tensors="pt").to(model.device)
    with torch.no_grad():
        _ = model.generate(**warmup_input, max_new_tokens=10, do_sample=False)
    torch.cuda.synchronize()

    for prompt in prompts[:num_runs]:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        total_tokens += generated_tokens
        total_time += (end_time - start_time)

    tokens_per_sec = total_tokens / total_time
    time_per_token = total_time / total_tokens

    return tokens_per_sec, time_per_token, total_tokens, total_time

def evaluate_model(model_path, model_name, eval_texts, speed_prompts):
    """모델 종합 평가"""
    print(f"\n{'='*60}")
    print(f"평가 중: {model_name}")
    print(f"경로: {model_path}")
    print(f"{'='*60}")

    # 모델 크기 확인
    model_size = sum(f.stat().st_size for f in Path(model_path).glob("*.safetensors")) / (1024**3)
    print(f"모델 크기: {model_size:.2f} GB")

    # 모델 로드
    print("모델 로딩 중...")
    model, tokenizer = load_model(model_path)

    # GPU 메모리 사용량
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"GPU 메모리 사용량: {gpu_memory:.2f} GB")

    # Perplexity 측정
    print("Perplexity 측정 중...")
    perplexity = measure_perplexity(model, tokenizer, eval_texts)
    print(f"Perplexity: {perplexity:.2f}")

    # 추론 속도 측정
    print("추론 속도 측정 중...")
    tokens_per_sec, time_per_token, total_tokens, total_time = measure_inference_speed(
        model, tokenizer, speed_prompts, max_new_tokens=128, num_runs=5
    )
    print(f"속도: {tokens_per_sec:.2f} tokens/sec")
    print(f"토큰당 시간: {time_per_token*1000:.2f} ms/token")

    # 메모리 정리
    del model
    torch.cuda.empty_cache()

    return {
        "model_name": model_name,
        "model_path": model_path,
        "model_size_gb": model_size,
        "perplexity": perplexity,
        "tokens_per_sec": tokens_per_sec,
        "time_per_token": time_per_token,
        "total_tokens": total_tokens,
        "total_time": total_time,
    }

def calculate_score(base_result, model_result):
    """점수 계산"""
    # 성능 정규화 (Perplexity는 낮을수록 좋으므로 역수 사용)
    # PerfNorm = base_perplexity / model_perplexity (높을수록 좋음)
    perf_norm = base_result["perplexity"] / model_result["perplexity"]

    # 속도 정규화
    # SpeedNorm = 1 - (model_time_per_token / base_time_per_token)
    speed_norm = 1 - (model_result["time_per_token"] / base_result["time_per_token"])

    # 최종 점수
    score = max(0.5 * perf_norm + 0.5 * speed_norm, 0)

    return {
        "perf_norm": perf_norm,
        "speed_norm": speed_norm,
        "score": score,
    }

def main():
    # 평가용 데이터 로드
    print("평가 데이터 로드 중...")
    ds = load_dataset("LGAI-EXAONE/MANTA-1M", split="train[:50]")

    # 평가용 텍스트 (perplexity 측정용)
    eval_texts = []
    for item in ds:
        if "conversations" in item and len(item["conversations"]) > 0:
            text = " ".join([c.get("content", "") for c in item["conversations"] if c.get("content")])
            if len(text) > 100:
                eval_texts.append(text[:1000])
    eval_texts = eval_texts[:20]

    # 속도 측정용 프롬프트
    speed_prompts = [
        "한국의 수도는 어디인가요?",
        "인공지능의 발전이 사회에 미치는 영향에 대해 설명해주세요.",
        "Python에서 리스트와 튜플의 차이점은 무엇인가요?",
        "기후 변화의 주요 원인은 무엇인가요?",
        "효과적인 시간 관리 방법에 대해 알려주세요.",
    ]

    results = []

    # 기본 모델 평가
    base_result = evaluate_model("./base_model", "Base Model (EXAONE-4.0-1.2B)", eval_texts, speed_prompts)
    results.append(base_result)

    # 양자화 모델들 평가
    quantized_models = [
        ("./model_quantized", "W4A16 Quantized"),
    ]

    # 추가 모델이 있으면 평가
    for model_path, model_name in quantized_models:
        if os.path.exists(model_path):
            model_result = evaluate_model(model_path, model_name, eval_texts, speed_prompts)
            results.append(model_result)

    # 결과 출력
    print("\n" + "="*80)
    print("평가 결과 요약")
    print("="*80)

    print(f"\n{'모델':<30} {'크기(GB)':<10} {'PPL':<10} {'속도(tok/s)':<12} {'ms/tok':<10}")
    print("-"*80)

    for r in results:
        print(f"{r['model_name']:<30} {r['model_size_gb']:<10.2f} {r['perplexity']:<10.2f} {r['tokens_per_sec']:<12.2f} {r['time_per_token']*1000:<10.2f}")

    # 점수 계산
    if len(results) > 1:
        print("\n" + "="*80)
        print("점수 계산 (기본 모델 대비)")
        print("="*80)
        print(f"\n{'모델':<30} {'PerfNorm':<12} {'SpeedNorm':<12} {'Score':<10}")
        print("-"*80)

        base_result = results[0]
        for r in results[1:]:
            scores = calculate_score(base_result, r)
            print(f"{r['model_name']:<30} {scores['perf_norm']:<12.4f} {scores['speed_norm']:<12.4f} {scores['score']:<10.4f}")

    # 결과 저장
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n결과가 evaluation_results.json에 저장되었습니다.")

if __name__ == "__main__":
    main()
