# EXAONE-4.0-1.2B 모델 압축 해커톤 실험 기록

## 프로젝트 개요

**목표**: EXAONE-4.0-1.2B 언어 모델을 압축하여, 품질 손실은 최소화하면서 추론 속도를 최대한 높이는 것.

**평가 공식**:
```
Score = 0.5 × PerfNorm + 0.5 × SpeedNorm

PerfNorm = 압축 모델 성능 / 원본 모델 성능  (1.0이면 원본과 동일한 품질)
SpeedNorm = 1 - (압축 모델 토큰당 시간 / 원본 모델 토큰당 시간)  (높을수록 빠름)
```

- 원본 모델(비압축)은 PerfNorm=1.0, SpeedNorm=0.0이므로 Score=0.5
- 압축 모델이 원본보다 빠르면 SpeedNorm > 0 → 0.5점 초과 가능

**평가 환경**:
- GPU: NVIDIA L4 (24GB, Ada Lovelace 아키텍처, FP8/INT8 텐서 코어 지원)
- 추론 엔진: vLLM 0.14.1
- 벤치마크: KMMLU-Pro, KMMLU-Redux (한국어 객관식), Ko-LongRAG (긴 문맥 QA), KoMT-Bench (GPT-4 채점)

**로컬 개발 환경**:
- GPU: NVIDIA RTX 5060 Ti (16.6GB)
- 동일한 vLLM 0.14.1, PyTorch 2.9.0, llmcompressor 0.9.0.1

---

## 양자화(Quantization) 기초 설명

양자화는 모델의 가중치(weights)와 활성화(activations)를 낮은 정밀도로 변환하는 기법입니다.

| 표기 | 의미 | 예시 |
|---|---|---|
| W8A8 | 가중치 8비트 정수, 활성화 8비트 정수 | INT8 텐서 코어 사용 |
| W4A16 | 가중치 4비트 정수, 활성화 16비트 부동소수점 | Marlin 커널 사용 |
| FP8 | 8비트 부동소수점 | FP8 텐서 코어 사용 |
| BF16 | 16비트 뇌부동소수점 (원본 정밀도) | 기본값, 비압축 |

**GPTQ**: 양자화 시 발생하는 오류를 최소화하는 알고리즘. 교정(calibration) 데이터를 사용하여 최적의 양자화 파라미터를 찾음.

**주요 GPTQ 파라미터**:
- `dampening_frac`: 양자화 안정성 조절 (높을수록 안정적이나 정확도 감소 가능)
- `actorder`: 활성화 순서 최적화 방식 (`static` = 기본, `group` = 그룹별 최적화)
- `block_size`: 양자화 그룹 크기 (128 = 128개 가중치마다 별도 스케일)
- `calibration samples`: 양자화 파라미터 학습에 사용하는 데이터 수

---

## 실험 타임라인

### Phase 1: 기본 양자화 실험 (이전 세션)

#### 실험 1: W4A16 GPTQ (4비트 가중치)
- **방법**: 가중치를 4비트로 양자화, 512 교정 샘플
- **결과**: 경쟁 점수 0.4678 (모델 크기 1,337MB)
- **분석**: 1.2B 소형 모델은 4비트 양자화에 매우 민감. 품질이 크게 하락(PerfNorm≈0.50-0.57)하여 속도 이점을 상쇄함.

#### 실험 2: FP8 Dynamic
- **방법**: 가중치와 활성화를 FP8로 변환
- **결과**: 경쟁 점수 0.4936 (모델 크기 1,432MB)
- **분석**: 품질 손실은 거의 없으나(PerfNorm≈0.99), L4에서 FP8의 속도 향상이 미미(SpeedNorm≈0.2%). 원본 모델이 이미 작아서 FP8 가속 효과가 적음.

#### 실험 3: FP8 Static
- **결과**: 경쟁 점수 0.4929
- **분석**: FP8 Dynamic과 유사.

#### 실험 4: W4A16 + SFT (미세조정)
- **방법**: 4비트 양자화 후, SFT(Supervised Fine-Tuning)로 품질 회복 시도
- **결과**: 경쟁 점수 0.4825
- **분석**: 양자화 후 미세조정은 품질을 일부 회복하지만, INT4 Marlin 커널의 속도 이점이 INT8보다 작음.

#### 실험 5: Layer Pruning (레이어 가지치기)
- **방법**: BI 점수 기반으로 불필요한 레이어 제거 후 지식증류(KD) + W4A16
- **결과**: 26레이어 유지 → 0.4678, 28레이어 유지 → 0.4678
- **분석**: 레이어 제거가 품질을 크게 떨어뜨림.

### Phase 2: W8A8 (INT8) 발견 — 핵심 전환점

#### 실험 6: W8A8 INT8 기본 ★★★
- **방법**: 가중치와 활성화를 8비트 정수로 양자화. GPTQ, 512 교정 샘플, dampening=0.01
- **결과**: **경쟁 점수 0.6042** (모델 크기 1,832MB) — 역대 최고점
- **분석**:
  - L4의 INT8 텐서 코어가 ~24% 속도 향상 제공 (SpeedNorm≈0.24)
  - 8비트이므로 품질 손실 최소 (PerfNorm≈0.97)
  - **W8A8이 L4에서 최적의 양자화 방식임을 확인**

### Phase 3: W8A8 최적화 (현재 세션)

#### 실험 7: W8A8 교정 최적화 (cal2048, dampening=0.02) ★★★
- **방법**: 교정 데이터를 512→2048개로 4배 늘리고, dampening을 0.01→0.02로 증가
- **결과**: **경쟁 점수 0.6208** — 신기록!
- **분석**: 더 많은 교정 데이터로 양자화 정확도 향상. dampening 증가로 후반 레이어의 양자화 안정성 개선.
- **파일**: `models/w8a8_cal2048_d02/`, `submissions/submit_w8a8_cal2048_d02.zip`

#### 실험 8: W8A8 dampening=0.05
- **방법**: dampening을 0.05로 더 올림
- **결과**: 경쟁 점수 0.6077 (로컬에서는 0.6275로 최고였으나 서버에서는 하락)
- **분석**: dampening을 너무 올리면 양자화 정밀도가 떨어짐. 최적값은 0.02 근처.
- **파일**: `models/w8a8_damp005/`

#### 실험 9: W8A8 dampening=0.1
- **방법**: dampening을 0.1로 크게 올림
- **결과**: 로컬 품질 최고(0.3283)이나 **속도가 절반(1089 tok/s)으로 하락**
- **분석**: 높은 dampening이 vLLM의 INT8 최적 커널 경로를 벗어나게 만듦. 폐기.
- **파일**: `models/w8a8_damp01/`

#### 실험 10: Selective Quantization (선택적 양자화)
- **방법**: 양자화 에러가 큰 후반 레이어(25-29)의 특정 모듈을 양자화에서 제외하여 BF16 유지
  - selective_5: 레이어 25-29의 gate_proj, up_proj 제외
  - selective_3full: 레이어 27-29 전체 제외
- **결과**:
  - selective_5: 경쟁 점수 **0.4718** (매우 나쁨)
  - selective_3full: 미제출 (추정 유사)
- **분석**: BF16 레이어가 포함되면 vLLM이 INT8 전용 최적화 경로를 사용하지 못하고, L4에서 혼합 정밀도 처리 오버헤드가 매우 큼. **속도 손해가 품질 이점을 완전히 상쇄**.
- **파일**: `models/w8a8_selective_5/`, `models/w8a8_selective_3full/`

### Phase 4: 지식증류(Knowledge Distillation) 실험

#### 실험 11: EXAONE-3.5-7.8B → 4.0-1.2B KD + W4A16 (이전 세션)
- **방법**: 구세대 7.8B 모델의 지식을 1.2B에 전달 후 W4A16 양자화
- **결과**: 경쟁 점수 0.2632 (최악)
- **분석**: 다른 세대 모델(3.5 vs 4.0)은 학습 분포가 달라 KD가 오히려 품질을 크게 악화.

#### 실험 12: EXAONE-4.0-32B → 4.0-1.2B KD + W8A8 (현재 세션)
- **방법**:
  1. 32B NF4 교사 모델에서 10,000개 샘플의 logit을 미리 계산 (메모리 제약 우회)
  2. 저장된 logit으로 1.2B 학생 모델 학습 (3 에폭, 25분)
  3. W8A8 양자화
- **기술적 문제**:
  - OOM: 32B + 1.2B + 옵티마이저 > 24GB → 2단계 방식으로 해결
  - int16 오버플로우: vocab_size=102400 > int16 최대값(32767) → int32로 수정
- **결과**: 로컬 perf_aggregate 0.3181 (base_w8a8의 0.3222보다 낮음)
- **분석**: KD로 가중치 분포가 변경되면 양자화 에러가 크게 증가(후반 레이어 에러 최대 94.65). KD 이득 < 양자화 손실.
- **파일**: `checkpoints/kd_40_32b/`, `models/kd_40_32b_w8a8/`

### Phase 5: 고급 양자화 기법 실험 (현재 세션)

#### 실험 13: SmoothQuant + W8A8
- **방법**: SmoothQuant(활성화 분포 평탄화) 적용 후 W8A8 양자화
- **결과**: **실패** — EXAONE 아키텍처가 SmoothQuant에서 지원되지 않음
- **에러**: `Error resolving mappings for given architecture`

#### 실험 14: Wanda 2:4 Structured Sparsity + W8A8
- **방법**: Wanda 알고리즘으로 가중치의 50%를 0으로 만든 후(2:4 패턴) W8A8 양자화
- **결과**: 양자화 성공 (1,274MB)이나 **vLLM에서 실행 불가** — compute capability ≥ 90 (Hopper) 필요
- **분석**: RTX 4090과 L4 모두 compute capability 89 (Ada)이므로 양쪽 다 불가.
- **파일**: `models/w8a8_sparse/`, `models/w8a8_sparse_damp/`

#### 실험 15: GPTQ Advanced (actorder=group, block_size=128) + W8A8
- **방법**: GPTQ의 고급 설정 사용 — 그룹별 활성화 순서 최적화, 128 단위 그룹 양자화
- **결과**: 경쟁 점수 **0.5085** (base_w8a8의 0.6042보다 크게 하락)
- **분석**: 그룹별 양자화 스케일이 vLLM의 INT8 CUTLASS 커널 최적 경로를 벗어나게 만듦. 로컬에서는 RTX 4090의 높은 대역폭으로 차이가 안 보였으나, L4에서는 큰 속도 페널티 발생.
- **파일**: `models/w8a8_gptq_adv/`, `models/w8a8_gptq_adv_damp/`

#### 실험 16: FP8 KV Cache + W8A8
- **방법**: KV Cache(추론 중 저장되는 중간 결과)를 FP8로 양자화하여 메모리 절감 → 속도 향상 기대
- **결과**: 로컬에서 KV cache 크기 변화 없음 (305,968 토큰으로 동일) — FP8 KV cache가 실제 적용되지 않은 것으로 보임
- **분석**: vLLM의 `kv_cache_dtype=auto` 설정에서 모델 config의 kv_cache_scheme을 무시했을 가능성. 미제출.
- **파일**: `models/w8a8_kv_fp8/`, `models/w8a8_kv_fp8_nodamp/`

---

## 전체 경쟁 점수 기록

| 순위 | 모델 | 양자화 방식 | 경쟁 Score | 비고 |
|---|---|---|---|---|
| **1** | **w8a8_cal2048_d02** | W8A8, cal=2048, damp=0.02 | **0.6208** | **최고점** |
| 2 | base_w8a8 | W8A8, cal=512, damp=0.01 | 0.6042 | 기본 W8A8 |
| 3 | w8a8_damp005 | W8A8, cal=2048, damp=0.05 | 0.6077 | dampening 과다 |
| 4 | w8a8_gptq_adv | W8A8, actorder=group, block=128 | 0.5085 | 그룹 양자화 속도 페널티 |
| 5 | W4A16_n512 | W4A16 GPTQ | 0.5099 | 4비트, 품질 손실 큼 |
| 6 | FP8_DYNAMIC | FP8 동적 양자화 | 0.4936 | 속도 향상 미미 |
| 7 | FP8_STATIC | FP8 정적 양자화 | 0.4929 | 유사 |
| 8 | W4A16_SFT | W4A16 + 미세조정 | 0.4825 | 속도 부족 |
| 9 | GPTQ_W4A16 | W4A16 GPTQ | 0.4784 | 품질 손실 |
| 10 | w8a8_selective_5 | W8A8, 5개 레이어 BF16 | 0.4718 | 혼합 정밀도 페널티 |
| 11 | lane01_W4A16 | W4A16 | 0.4678 | 초기 실험 |
| 12 | lane06_SFT+W4A16 | SFT + W4A16 | 0.3202 | SFT 효과 부족 |
| 13 | kd_35_78b_w4a16 | 3.5세대 KD + W4A16 | 0.2632 | 세대 불일치 |

---

## 핵심 교훈

### 1. L4에서 W8A8 (INT8)이 최적
- L4의 INT8 텐서 코어가 ~24% 속도 향상 제공
- 8비트 양자화는 품질 손실이 매우 적음 (PerfNorm≈0.97)
- 4비트(W4A16)는 품질 손실이 크고, FP8은 속도 향상이 미미

### 2. 단순한 양자화 설정이 최고 성능
- **최적 설정**: W8A8, per-channel(기본), dampening=0.02, calibration=2048
- actorder=group, 선택적 양자화 등 복잡한 기법은 오히려 성능 악화
- vLLM의 최적화된 INT8 CUTLASS 커널 경로를 유지하는 것이 핵심

### 3. 로컬 측정과 서버 결과의 괴리
- 로컬 perf_aggregate는 komt_bench(GPT-4 채점) 미포함
- 양자화 방식에 따라 vLLM 커널 선택이 달라져 속도 차이가 큼
- 로컬에서 비슷해 보여도 서버에서는 큰 차이 발생 가능

### 4. 지식증류(KD)는 양자화와 상충
- KD로 가중치 분포가 변하면 양자화 에러가 증가
- 특히 후반 레이어에서 양자화 에러가 급증 (최대 94.65)
- KD 전 양자화 or KD 없이 양자화가 더 효과적

### 5. 호환성 확인이 필수
- SmoothQuant: EXAONE 아키텍처 미지원
- 2:4 Sparsity: compute capability ≥ 90 필요 (L4는 89)
- actorder=group: vLLM에서 속도 페널티

---

## 파일 구조

```
0208/
├── models/                     # 양자화된 모델들
│   ├── base_w8a8/              # ★ 기본 W8A8 (0.6042)
│   ├── w8a8_cal2048_d02/       # ★★★ 최고점 W8A8 (0.6208)
│   ├── w8a8_damp005/           # dampening=0.05 (0.6077)
│   ├── w8a8_damp01/            # dampening=0.1 (속도 저하)
│   ├── w8a8_selective_5/       # 선택적 양자화 (0.4718)
│   ├── w8a8_selective_3full/   # 선택적 양자화 (미제출)
│   ├── w8a8_gptq_adv/         # actorder=group (0.5085)
│   ├── w8a8_gptq_adv_damp/    # actorder=group+damp (미제출)
│   ├── w8a8_kv_fp8/           # FP8 KV cache (미제출)
│   ├── w8a8_kv_fp8_nodamp/    # FP8 KV cache (미제출)
│   ├── w8a8_sparse/           # 2:4 sparsity (실행 불가)
│   ├── kd_40_32b_w8a8/        # 32B KD + W8A8
│   ├── baseline_bf16/         # 원본 모델 (비압축)
│   └── ...                    # 이전 세션 모델들
├── submissions/                # 제출 zip 파일들
├── scripts/                    # 실험 스크립트
│   ├── 04_eval_vllm.py        # 벤치마크 평가
│   ├── quantize_int8_variants.py
│   ├── w8a8_experiments.py
│   ├── w8a8_advanced.py
│   ├── w8a8_kvcache_fp8.py
│   ├── kd_precompute.py       # 2단계 지식증류
│   └── ...
├── checkpoints/                # 학습 체크포인트
│   └── kd_40_32b/             # 32B 교사 logit + KD 모델
├── results/                    # 평가 결과 (metrics.json)
├── logs/                       # 실험 로그
└── configs/lanes.yaml          # 평가 설정
```
