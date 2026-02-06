"""
vLLM 환경에서의 예상 점수 계산

실제 vLLM에서 양자화 모델은 일반적으로:
- W4A16: 1.5~2.5x 속도 향상 (모델 크기 감소로 인한 메모리 대역폭 이득)
- W8A16: 1.2~1.8x 속도 향상

평가 환경: L4 GPU (22.4GB VRAM)
"""

import json

# 실제 측정 결과 (transformers 기반)
results = {
    "Base Model": {"size_gb": 2.38, "ppl": 70.53},
    "W4A16 (256 samples)": {"size_gb": 1.30, "ppl": 81.83},
    "W4A16_n512": {"size_gb": 1.30, "ppl": 72.75},
    "W8A16": {"size_gb": 1.78, "ppl": 70.83},
}

# vLLM 속도 향상 추정 (보수적 추정)
# 속도 향상 = base_time_per_token / model_time_per_token
speed_multiplier = {
    "Base Model": 1.0,
    "W4A16 (256 samples)": 1.8,  # 보수적 추정
    "W4A16_n512": 1.8,
    "W8A16": 1.4,
}

print("="*80)
print("vLLM 환경 예상 점수 (보수적 추정)")
print("="*80)
print()
print("Score = max(0.5 × PerfNorm + 0.5 × SpeedNorm, 0)")
print("PerfNorm = base_ppl / model_ppl")
print("SpeedNorm = 1 - (model_time_per_token / base_time_per_token)")
print()
print(f"{'모델':<25} {'크기':<8} {'PPL':<8} {'PerfNorm':<10} {'SpeedMult':<10} {'SpeedNorm':<10} {'Score':<10}")
print("-"*80)

base_ppl = results["Base Model"]["ppl"]

for name, data in results.items():
    if name == "Base Model":
        continue

    perf_norm = base_ppl / data["ppl"]

    # SpeedNorm = 1 - (1 / speed_multiplier)
    # 예: speed_multiplier = 2.0이면 SpeedNorm = 1 - 0.5 = 0.5
    speed_mult = speed_multiplier[name]
    speed_norm = 1 - (1 / speed_mult)

    score = max(0.5 * perf_norm + 0.5 * speed_norm, 0)

    print(f"{name:<25} {data['size_gb']:<8.2f} {data['ppl']:<8.2f} {perf_norm:<10.4f} {speed_mult:<10.1f}x {speed_norm:<10.4f} {score:<10.4f}")

print()
print("="*80)
print("시나리오별 점수 (W4A16_n512 기준)")
print("="*80)
print()

model = "W4A16_n512"
perf_norm = base_ppl / results[model]["ppl"]
print(f"PerfNorm (고정) = {perf_norm:.4f}")
print()
print(f"{'Speed Mult':<12} {'SpeedNorm':<12} {'Score':<10}")
print("-"*40)

for mult in [1.5, 1.8, 2.0, 2.5, 3.0]:
    speed_norm = 1 - (1 / mult)
    score = max(0.5 * perf_norm + 0.5 * speed_norm, 0)
    print(f"{mult:<12.1f}x {speed_norm:<12.4f} {score:<10.4f}")

print()
print("="*80)
print("시나리오별 점수 (W8A16 기준)")
print("="*80)
print()

model = "W8A16"
perf_norm = base_ppl / results[model]["ppl"]
print(f"PerfNorm (고정) = {perf_norm:.4f}")
print()
print(f"{'Speed Mult':<12} {'SpeedNorm':<12} {'Score':<10}")
print("-"*40)

for mult in [1.2, 1.4, 1.5, 1.8, 2.0]:
    speed_norm = 1 - (1 / mult)
    score = max(0.5 * perf_norm + 0.5 * speed_norm, 0)
    print(f"{mult:<12.1f}x {speed_norm:<12.4f} {score:<10.4f}")

print()
print("="*80)
print("결론")
print("="*80)
print("""
1. W4A16_n512 (512 샘플 캘리브레이션):
   - PerfNorm: 0.9694 (높은 정확도 유지)
   - 모델 크기: 1.30GB (45% 감소)
   - vLLM에서 1.8x 속도 향상 가정 시 Score ≈ 0.71

2. W8A16:
   - PerfNorm: 0.9958 (거의 원본 수준)
   - 모델 크기: 1.78GB (25% 감소)
   - vLLM에서 1.4x 속도 향상 가정 시 Score ≈ 0.64

추천: W4A16_n512가 더 좋은 균형을 보임 (성능 유지 + 속도 향상)
""")
