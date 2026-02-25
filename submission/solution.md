# EXAONE-4.0-1.2B 모델 경량화 솔루션

---

## 1. 과제 요약

- **목표**: EXAONE-4.0-1.2B 모델 경량화 (성능 유지 + 추론 속도 향상)
- **평가 지표**: `Score = 0.5 × PerfNorm + 0.5 × SpeedNorm`
  - PerfNorm = 압축 모델 성능 / 베이스 모델 성능
  - SpeedNorm = 1 − (압축 모델 time_per_token / 베이스 모델 time_per_token)
- **평가 서버**: L4 GPU (24GB, Ada Lovelace)
- **최종 점수**: **0.6295**

---

## 2. 접근 방법

### 최종 솔루션: QuantizationModifier W8A8 + Context Window 축소

| 항목 | 설정 |
|---|---|
| 양자화 방식 | QuantizationModifier (llmcompressor) |
| 양자화 스킴 | W8A8 (INT8 weight + INT8 activation) |
| Weight 양자화 | per-channel, symmetric, static (minmax) |
| Activation 양자화 | per-token, symmetric, **dynamic** |
| 제외 레이어 | embed_tokens, lm_head |
| max_position_embeddings | 65536 → **16384** |
| 캘리브레이션 | MANTA-1M 2048샘플, seq_length=2048 |

---

## 3. 실험 과정 및 탈락된 방법들

### 3-1. 양자화 방식 비교 (서버 제출 결과)

| 방법 | 서버 Score | 비고 |
|---|---|---|
| **QuantizationModifier W8A8** | **0.6295** | ✅ 최종 선택 |
| GPTQ W8A8 (damp=0.02) | 0.6208 | Hessian 캘리브레이션 |
| GPTQ W8A8 + SparseGPT 10% | 0.6246 | 비정형 스파시티 추가 |
| GPTQ W8A16 | ~0.52 | L4에서 느림 (AllSpark 커널) |
| FP8 Dynamic | ~0.50 | L4 FP8 커널 비효율 |
| GPTQ W4A16 | ~0.49 | 1.2B 모델에 INT4 과도한 손실 |

### 3-2. 시도했으나 효과 없었던 방법들

| 방법 | 결과 | 원인 |
|---|---|---|
| SFT (LoRA) → 양자화 | perf 하락 | 1.2B 모델에 MANTA SFT가 오히려 성능 저하 |
| Post-quant SFT (norm만) | 변화 없음 | 학습 가능 파라미터 <5%로 불충분 |
| 레이어 프루닝 (30→28L) | perf 0.325→0.308 | 속도 이득 대비 품질 손실 과도 |
| FFN 폭 프루닝 (25%) | perf 0.273 | 너무 공격적 |
| Vocab 프루닝 (102k→80k) | 서버 0.48 | byte-fallback으로 토큰 폭증 |
| KV Cache FP8 | 변화 없음 | L4에서 메모리 병목 아님 |
| Self-KD + QuantMod | perf 소폭 하락 | minmax 양자화에 KD 가중치 민감 |

---

## 4. 핵심 발견

### 4-1. QuantizationModifier가 GPTQ보다 우수한 이유

- QuantizationModifier: Data-Free 방식, 단순 min-max 스케일링
- GPTQ: Hessian 기반 캘리브레이션, 레이어별 재구성 오차 최소화
- **역설적으로 단순한 방식이 서버에서 더 좋은 결과**
  - GPTQ의 Hessian 최적화가 캘리브레이션 데이터에 과적합
  - min-max 방식이 평가 데이터 분포 변화에 더 robust

### 4-2. L4 GPU에서 W8A8만 유효

- L4는 `CutlassScaledMMLinearKernel`로 W8A8 INT8 연산 가속
- W8A16은 `AllSparkLinearKernel` 사용 → 느림
- FP8은 L4 텐서코어가 있지만 vLLM 커널 최적화 부족
- **로컬(RTX 4090)과 서버(L4)의 최적 양자화 방식이 다름**

### 4-3. Context Window 축소의 효과

- `max_position_embeddings`: 65536 → 16384
- KV Cache 메모리 4배 절감 → 더 큰 배치 처리 가능
- 평가 벤치마크 최대 입력 ~16k 토큰이므로 품질 손실 없음

---

## 5. 재현 방법

```bash
# 환경 설정
conda create -n quant python=3.11
conda activate quant
pip install torch==2.9.0 vllm==0.14.1 llmcompressor==0.9.0.1 transformers==4.57.3

# 경량화 실행 (~10초)
python reproduce.py

# 결과: submit.zip (제출용)
```

---

## 6. 모델 크기 비교

| | 베이스 모델 | 경량화 모델 |
|---|---|---|
| 정밀도 | BF16 | INT8 (W8A8) |
| 모델 크기 | 2.4 GB | 1.4 GB |
| 압축률 | - | **42% 감소** |
| max_position | 65536 | 16384 |

