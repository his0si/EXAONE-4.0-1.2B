# EXAONE-4.0-1.2B 모델 경량화 솔루션

> LG Aimers Phase 2 - LLM 경량화 해커톤
> Public Score: **0.6295**

---

## Slide 1. 타이틀

- **LG Aimers Phase 2 - LLM 경량화 해커톤**
- EXAONE-4.0-1.2B 모델 경량화
- **Final Score: 0.6295**
- 방법: QuantizationModifier W8A8 + Context Window 축소

---

## Slide 2. 과제 개요

### 평가 지표
```
Score = 0.5 x PerfNorm + 0.5 x SpeedNorm

PerfNorm = 압축 모델 성능 / 베이스 모델 성능
SpeedNorm = 1 - (압축 모델 time_per_token / 베이스 모델 time_per_token)
```

- 비압축 모델: PerfNorm=1.0, SpeedNorm=0.0 -> Score=0.5
- 성능 유지 + 속도 향상이 핵심

### 평가 벤치마크
| 벤치마크 | 유형 | 샘플 수 |
|---|---|---|
| KMMLU-Pro | 한국어 MCQA | 2,822 |
| KMMLU-Redux | 한국어 MCQA | 2,587 |
| Ko-LongRAG | 긴 문맥 QA | 600 |
| KoMT-Bench | GPT-4 채점 | 80 |

### 제약 조건
- 평가 서버: **NVIDIA L4 GPU** (24GB, Ada Lovelace)
- 추론 엔진: **vLLM 0.14.1**
- 제출 파일: 압축 10GB / 비압축 32GB 이내
- 베이스 모델: EXAONE-4.0-1.2B (2.4GB, BF16)

---

## Slide 3. 최종 솔루션 - QuantizationModifier W8A8

### 핵심 방법
**QuantizationModifier (llmcompressor)** 를 사용한 W8A8 INT8 양자화 + Context Window 축소

### 양자화 설정 상세

| 항목 | 설정 |
|---|---|
| 양자화 도구 | llmcompressor 0.9.0.1 (QuantizationModifier) |
| 양자화 스킴 | **W8A8** (INT8 weight + INT8 activation) |
| Weight 양자화 | per-channel, symmetric, static (minmax observer) |
| Activation 양자화 | per-token, symmetric, **dynamic** |
| 대상 레이어 | 모든 Linear 레이어 |
| 제외 레이어 | embed_tokens, lm_head |
| 캘리브레이션 데이터 | MANTA-1M 2,048 샘플 |
| max_seq_length | 2,048 |

### Context Window 최적화
- `max_position_embeddings`: **65,536 -> 16,384** (4배 축소)
- KV Cache 메모리 4배 절감 -> 더 큰 배치 / 빠른 추론
- 평가 벤치마크 최대 입력 ~16k 토큰이므로 품질 손실 없음

### QuantizationModifier vs GPTQ
- **QuantizationModifier**: Data-Free, 단순 min-max 스케일링
- **GPTQ**: Hessian 기반 캘리브레이션, 레이어별 재구성 오차 최소화
- 역설적으로 **단순한 방식이 서버에서 더 높은 점수** (0.6295 vs 0.6208)
  - GPTQ의 Hessian 최적화가 캘리브레이션 데이터에 과적합
  - min-max 방식이 평가 데이터 분포 변화에 더 robust

---

## Slide 4. 양자화 방식별 서버 결과 비교

### 전체 서버 제출 결과

| 순위 | 방법 | 서버 Score | 모델 크기 | 비고 |
|---|---|---|---|---|
| **1** | **QuantMod W8A8 + ctx16k** | **0.6295** | **1.4 GB** | **최종 제출** |
| 2 | GPTQ W8A8 + SparseGPT 10% | 0.6246 | 1.3 GB | 비정형 스파시티 |
| 3 | GPTQ W8A8 (cal=2048, d=0.02) | 0.6208 | 1.8 GB | Hessian 캘리브레이션 |
| 4 | GPTQ W8A8 (d=0.05) | 0.6077 | 1.8 GB | dampening 과다 |
| 5 | GPTQ W8A8 기본 (d=0.01) | 0.6042 | 1.8 GB | 기본 설정 |
| 6 | W4A16 GPTQ (n=512) | 0.5099 | 1.3 GB | 4비트 품질 손실 |
| 7 | GPTQ W8A8 actorder=group | 0.5085 | 1.8 GB | 커널 비효율 |
| 8 | FP8 Dynamic | 0.4936 | 1.4 GB | L4 FP8 커널 미흡 |
| 9 | FP8 Static | 0.4929 | 1.4 GB | 유사 |
| 10 | W4A16 + SFT | 0.4825 | 1.3 GB | SFT 효과 미미 |
| 11 | W4A16 GPTQ | 0.4784 | 1.3 GB | 품질 손실 |
| 12 | W8A8 Selective (5L BF16) | 0.4718 | 1.8 GB | 혼합 정밀도 페널티 |
| 13 | W4A16 lane01 | 0.4678 | 1.3 GB | 초기 실험 |
| 14 | SFT + W4A16 | 0.3202 | 1.3 GB | SFT 성능 저하 |
| 15 | 3.5세대 KD + W4A16 | 0.2632 | 1.3 GB | 세대 불일치 |

### 성능-속도 트레이드오프
- **W8A8**: PerfNorm ~0.99, SpeedNorm ~0.12 -> 성능 유지 + 적절한 속도 향상
- **W4A16**: PerfNorm ~0.50-0.57, SpeedNorm ~0.13 -> 과도한 품질 손실
- **FP8**: PerfNorm ~0.99, SpeedNorm ~0.002 -> 속도 향상 거의 없음

---

## Slide 5. 시도했으나 실패한 방법들

### 총 110+ 모델 실험, 15+ 서버 제출

| 카테고리 | 시도한 방법 | 결과 | 실패 원인 |
|---|---|---|---|
| **4비트 양자화** | W4A16 GPTQ (256/512/1024 샘플) | Score 0.47-0.51 | 1.2B 소형 모델에 INT4 과도한 품질 손실 |
| **FP8 양자화** | FP8 Dynamic / Static | Score ~0.49 | L4의 FP8 커널 최적화 부족, 속도 향상 미미 |
| **SFT (미세조정)** | LoRA SFT -> W4A16 | Score 0.32-0.48 | MANTA SFT가 1.2B 모델 성능 오히려 저하 |
| **Post-quant SFT** | 양자화 후 norm만 SFT | 변화 없음 | 학습 가능 파라미터 <5%로 불충분 |
| **레이어 프루닝** | 30L -> 28L/26L + KD | Score 0.47 | 속도 이득 대비 품질 손실 과도 |
| **FFN 프루닝** | 25% FFN 폭 축소 | perf 0.273 | 너무 공격적, 복구 불가 |
| **Vocab 프루닝** | 102k -> 80k 토큰 | Score 0.48 | byte-fallback으로 토큰 폭증 |
| **지식증류 (KD)** | 32B/7.8B teacher -> 1.2B | Score 0.26 | KD 가중치 변경 -> 양자화 에러 증가 |
| **SmoothQuant** | 활성화 분포 평탄화 | 실행 불가 | EXAONE 아키텍처 미지원 |
| **2:4 Sparsity** | Wanda 구조적 스파시티 | 실행 불가 | compute capability >= 90 필요 (L4는 89) |
| **Selective Quant** | 후반 5개 레이어 BF16 유지 | Score 0.47 | 혼합 정밀도 -> vLLM INT8 커널 경로 이탈 |
| **GPTQ Advanced** | actorder=group, block=128 | Score 0.51 | 그룹 양자화가 CUTLASS 커널 비효율 |
| **FP8 KV Cache** | KV Cache FP8 양자화 | 변화 없음 | vLLM이 kv_cache_scheme을 무시 |
| **Self-KD + W8A8** | 자기 지식증류 + 양자화 | perf 소폭 하락 | minmax 양자화에 KD 가중치 민감 |

---

## Slide 6. 핵심 인사이트 4가지

### 1. L4 GPU에서 W8A8 (INT8)만 유효
- L4의 `CutlassScaledMMLinearKernel`로 INT8 연산 가속 (~24% 속도 향상)
- W8A16은 `AllSparkLinearKernel` 사용 -> 느림
- FP8은 L4 텐서코어가 있지만 vLLM 커널 최적화 부족
- **하드웨어에 맞는 양자화 커널 선택이 핵심**

### 2. 단순한 양자화가 최고 성능
- QuantizationModifier (단순 min-max) > GPTQ (Hessian 기반)
- actorder=group, selective quant 등 복잡한 기법은 오히려 성능 악화
- **vLLM의 최적화된 INT8 CUTLASS 커널 경로를 유지하는 것이 핵심**
- 복잡한 양자화 설정이 커널 fallback을 유발

### 3. 로컬 측정과 서버 결과의 괴리
- 로컬(RTX 5060 Ti) vs 서버(L4): 최적 양자화 방식이 다름
- 양자화 방식에 따라 vLLM 커널 선택이 달라져 속도 차이가 큼
- 로컬에서 비슷해 보여도 서버에서는 큰 차이 발생 가능
- **반드시 서버 제출 기준으로 최종 판단 필요**

### 4. 소형 모델(1.2B)의 양자화 특성
- INT4(W4A16)는 대형 모델(7B+)에서 효과적이나, 1.2B에서는 품질 손실 과도
- KD로 가중치 분포 변경 시 양자화 에러가 급증 (후반 레이어 에러 최대 94.65)
- SFT도 소형 모델에서는 오히려 성능 저하 유발
- **소형 모델은 최소한의 변형(W8A8)이 최적**

---

## Slide 7. 모델 크기 비교 + 재현 방법

### 모델 크기 비교

| | 베이스 모델 | 경량화 모델 |
|---|---|---|
| 정밀도 | BF16 (16비트) | **INT8 (W8A8)** |
| 모델 크기 | 2.4 GB | **1.4 GB** |
| 압축률 | - | **42% 감소** |
| max_position | 65,536 | **16,384** |
| 추론 속도 (L4) | 1,811 tok/s | **2,435 tok/s** |
| PerfNorm | 1.000 | 0.990 |
| SpeedNorm | 0.000 | 0.121 |
| **Score** | **0.500** | **0.6295** |

### 재현 방법

```bash
# 1. 환경 설정
conda create -n lgaimers python=3.10
conda activate lgaimers
pip install -r requirements.txt

# 2. 경량화 실행 (~10초)
python reproduce.py

# 3. 결과
# -> model/ 디렉토리에 양자화된 모델 생성
# -> submit.zip 제출 파일 생성
```

### 핵심 코드 (reproduce.py 요약)
```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = [QuantizationModifier(
    scheme="W8A8",
    targets=["Linear"],
    ignore=["embed_tokens", "lm_head"],
)]

oneshot(model=model, dataset=dataset, recipe=recipe,
        max_seq_length=2048, num_calibration_samples=2048)

# config.json에서 max_position_embeddings: 65536 -> 16384
```
