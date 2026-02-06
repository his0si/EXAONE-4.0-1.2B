import os
import torch
import time
import json
import gc
from pathlib import Path

def clear_gpu_memory():
    """GPU 메모리 완전 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

from vllm import LLM, SamplingParams
from datasets import load_dataset
import numpy as np

def measure_inference_speed_vllm(model_path, prompts, max_tokens=128):
    """vLLM으로 추론 속도 측정"""
    clear_gpu_memory()
    print(f"  vLLM 모델 로드 중: {model_path}")

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.70,  # 줄임
        dtype="bfloat16",
        max_model_len=4096,  # 컨텍스트 길이 제한
    )

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0,
    )

    # Warmup
    print("  Warmup 중...")
    _ = llm.generate(["Hello"], sampling_params)

    # 실제 측정
    print("  추론 속도 측정 중...")
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    outputs = llm.generate(prompts, sampling_params)

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    tokens_per_sec = total_tokens / total_time
    time_per_token = total_time / total_tokens

    # 메모리 정리
    del llm
    clear_gpu_memory()

    return {
        "tokens_per_sec": tokens_per_sec,
        "time_per_token": time_per_token,
        "total_tokens": total_tokens,
        "total_time": total_time,
    }

def measure_perplexity(model_path, texts):
    """Perplexity 측정"""
    clear_gpu_memory()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in texts[:15]:  # 샘플 수 줄임
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    del model, tokenizer
    clear_gpu_memory()

    return perplexity

def evaluate_model_speed_only(model_path, model_name, prompts):
    """속도만 측정"""
    print(f"\n{'='*60}")
    print(f"속도 측정 중: {model_name}")
    print(f"{'='*60}")

    model_size = sum(f.stat().st_size for f in Path(model_path).glob("*.safetensors")) / (1024**3)
    print(f"모델 크기: {model_size:.2f} GB")

    speed_result = measure_inference_speed_vllm(model_path, prompts, max_tokens=128)
    print(f"속도: {speed_result['tokens_per_sec']:.2f} tokens/sec")
    print(f"토큰당 시간: {speed_result['time_per_token']*1000:.2f} ms/token")

    return {
        "model_name": model_name,
        "model_path": model_path,
        "model_size_gb": model_size,
        "tokens_per_sec": speed_result["tokens_per_sec"],
        "time_per_token": speed_result["time_per_token"],
        "total_tokens": speed_result["total_tokens"],
        "total_time": speed_result["total_time"],
    }

def evaluate_model_ppl_only(model_path, model_name, eval_texts):
    """Perplexity만 측정"""
    print(f"\n{'='*60}")
    print(f"PPL 측정 중: {model_name}")
    print(f"{'='*60}")

    model_size = sum(f.stat().st_size for f in Path(model_path).glob("*.safetensors")) / (1024**3)
    perplexity = measure_perplexity(model_path, eval_texts)
    print(f"Perplexity: {perplexity:.2f}")

    return {
        "model_name": model_name,
        "model_path": model_path,
        "model_size_gb": model_size,
        "perplexity": perplexity,
    }

def calculate_score(base_result, model_result):
    """점수 계산"""
    perf_norm = base_result["perplexity"] / model_result["perplexity"]
    speed_norm = 1 - (model_result["time_per_token"] / base_result["time_per_token"])
    score = max(0.5 * perf_norm + 0.5 * speed_norm, 0)

    return {
        "perf_norm": perf_norm,
        "speed_norm": speed_norm,
        "score": score,
    }

def main():
    # 평가 데이터
    print("평가 데이터 로드 중...")
    ds = load_dataset("LGAI-EXAONE/MANTA-1M", split="train[:30]")

    eval_texts = []
    for item in ds:
        if "conversations" in item and len(item["conversations"]) > 0:
            text = " ".join([c.get("content", "") for c in item["conversations"] if c.get("content")])
            if len(text) > 100:
                eval_texts.append(text[:800])

    prompts = [
        "한국의 수도는 어디인가요?",
        "인공지능의 발전이 사회에 미치는 영향에 대해 설명해주세요.",
        "Python에서 리스트와 튜플의 차이점은 무엇인가요?",
        "기후 변화의 주요 원인은 무엇인가요?",
        "효과적인 시간 관리 방법에 대해 알려주세요.",
        "머신러닝과 딥러닝의 차이점은 무엇인가요?",
        "건강한 식습관의 중요성에 대해 설명해주세요.",
        "프로그래밍을 배우는 가장 좋은 방법은 무엇인가요?",
    ]

    # 평가할 모델들
    models = [
        ("./base_model", "Base Model"),
        ("./model_quantized", "W4A16 (g128)"),
    ]

    # 추가 모델 탐색
    for d in Path(".").iterdir():
        if d.is_dir() and d.name.startswith("model_W"):
            if d.name != "model_quantized":
                models.append((str(d), d.name.replace("model_", "")))

    # 1단계: 모든 모델 Perplexity 측정
    print("\n" + "="*80)
    print("1단계: Perplexity 측정")
    print("="*80)

    ppl_results = {}
    for model_path, model_name in models:
        if os.path.exists(model_path):
            try:
                result = evaluate_model_ppl_only(model_path, model_name, eval_texts)
                ppl_results[model_name] = result
            except Exception as e:
                print(f"[ERROR] {model_name} PPL 측정 실패: {e}")

    # 2단계: 모든 모델 속도 측정
    print("\n" + "="*80)
    print("2단계: vLLM 속도 측정")
    print("="*80)

    speed_results = {}
    for model_path, model_name in models:
        if os.path.exists(model_path):
            try:
                result = evaluate_model_speed_only(model_path, model_name, prompts)
                speed_results[model_name] = result
            except Exception as e:
                print(f"[ERROR] {model_name} 속도 측정 실패: {e}")
                import traceback
                traceback.print_exc()

    # 결과 통합
    results = []
    for model_name in ppl_results:
        if model_name in speed_results:
            combined = {**ppl_results[model_name], **speed_results[model_name]}
            results.append(combined)

    # 결과 출력
    print("\n" + "="*80)
    print("vLLM 평가 결과 요약")
    print("="*80)

    print(f"\n{'모델':<25} {'크기(GB)':<10} {'PPL':<10} {'tok/s':<12} {'ms/tok':<10}")
    print("-"*80)

    for r in results:
        print(f"{r['model_name']:<25} {r['model_size_gb']:<10.2f} {r['perplexity']:<10.2f} {r['tokens_per_sec']:<12.2f} {r['time_per_token']*1000:<10.2f}")

    # 점수 계산
    if len(results) > 1:
        print("\n" + "="*80)
        print("점수 계산 (기본 모델 대비)")
        print("="*80)
        print(f"\n{'모델':<25} {'PerfNorm':<12} {'SpeedNorm':<12} {'Score':<10}")
        print("-"*80)

        base_result = results[0]
        for r in results[1:]:
            scores = calculate_score(base_result, r)
            print(f"{r['model_name']:<25} {scores['perf_norm']:<12.4f} {scores['speed_norm']:<12.4f} {scores['score']:<10.4f}")

    # 저장
    with open("evaluation_vllm_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n결과가 evaluation_vllm_results.json에 저장되었습니다.")

if __name__ == "__main__":
    main()
