# EXAONE-4.0-1.2B Compression Pipeline

LLM 경량화 해커톤을 위한 end-to-end 파이프라인입니다.
12개의 고유한 압축 레인을 빌드, 평가, 비교하여 상위 3개 모델을 자동으로 패키징합니다.

## Quick Start

```bash
# 0. conda 환경 활성화
conda activate lgaimers

# 1. 환경 검증
python scripts/00_setup_check.py

# 2. 데이터 준비 (MANTA-1M train/val + 벤치마크 프로브)
python scripts/01_prepare_data.py

# 3. 베이스라인 평가 (필수 — 모든 Score 계산의 기준)
python scripts/04_eval_vllm.py --model ./base_model --baseline

# 4. 전체 파이프라인 실행 (빌드 + 평가 + 순위 + 패키징)
python scripts/run_all.py

# 또는 특정 레인만 실행
python scripts/run_all.py --only lane01_gptq_w4a16,lane02_gptq_w4a16_damp

# 또는 최대 N개 레인만
python scripts/run_all.py --max-lanes 5
```

## Directory Structure

```
0208/
├── configs/
│   └── lanes.yaml          # 모든 레인 정의 + 학습/평가 설정
├── scripts/
│   ├── 00_setup_check.py   # 환경 & GPU/vLLM 검증
│   ├── 01_prepare_data.py  # MANTA-1M 다운로드 + 벤치마크 프로브
│   ├── 02_finetune_sft.py  # SFT (LoRA) / Post-SFT / KD 학습
│   ├── 03_quantize.py      # GPTQ/FP8 양자화 (llmcompressor)
│   ├── 04_eval_vllm.py     # vLLM 벤치마크 평가 + 속도 측정
│   ├── 05_rank_and_package.py  # Score 계산, 순위, top-3 패키징
│   └── run_all.py          # 전체 오케스트레이터
├── data/                   # (자동 생성) 학습/벤치마크 데이터
├── checkpoints/            # (자동 생성) 학습 체크포인트
├── models/                 # (자동 생성) 양자화된 모델들
├── results/
│   ├── baseline.json       # 베이스라인 평가 결과
│   ├── <lane_id>/metrics.json
│   ├── summary.csv
│   └── summary.md
└── submissions/
    ├── top1/               # HuggingFace 형식 모델
    ├── top2/
    ├── top3/
    └── submit_top*.zip     # 제출용 zip
```

## 12 Compression Lanes

| # | Lane ID | Group | Strategy | Expected Speed |
|---|---------|-------|----------|----------------|
| 1 | `lane01_gptq_w4a16` | Quant Only | GPTQ W4A16 basic | ~2x |
| 2 | `lane02_gptq_w4a16_damp` | Quant Only | GPTQ W4A16 + dampening | ~2x |
| 3 | `lane03_gptq_w8a16` | Quant Only | GPTQ W8A16 | ~1.5x |
| 4 | `lane04_fp8_dynamic` | Quant Only | FP8 Dynamic | ~1.5x |
| 5 | `lane05_sft_fp16` | SFT→Quant | LoRA SFT (FP16, no quant) | 1x |
| 6 | `lane06_sft_gptq_w4a16` | SFT→Quant | LoRA SFT → GPTQ W4A16 | ~2x |
| 7 | `lane07_sft_gptq_w4a16_damp` | SFT→Quant | LoRA SFT → GPTQ W4A16 damp | ~2x |
| 8 | `lane08_w4a16_postsft` | Quant→SFT | GPTQ W4A16 → Post-SFT (norms) | ~2x |
| 9 | `lane09_w4a16_damp_postsft` | Quant→SFT | GPTQ W4A16 damp → Post-SFT | ~2x |
| 10 | `lane10_prune28_kd_w4a16` | Structural | Prune 28L → KD → W4A16 | ~2.2x |
| 11 | `lane11_prune26_kd_w4a16` | Structural | Prune 26L → KD → W4A16 | ~2.5x |
| 12 | `lane12_fp8_static` | Quant Only | FP8 Static (calibrated) | ~1.5x |

## Scoring Formula

```
Score = max(0.5 × PerfNorm + 0.5 × SpeedNorm, 0)

PerfNorm = Perf_model / Perf_base
SpeedNorm = 1 - (sec/token_model) / (sec/token_base)
```

- **Perf**: 가중 평균 (KMMLU-Pro, KMMLU-Redux, Ko-LongRAG; KoMT-Bench는 judge 없으면 제외)
- **Speed**: vLLM 추론 시 토큰당 시간 (3회 측정 중앙값)

## Individual Script Usage

```bash
# SFT (LoRA on FP16 base model)
python scripts/02_finetune_sft.py --mode sft

# Post-quantization SFT (norms only)
python scripts/02_finetune_sft.py --mode post_sft \
    --input-model models/lane01_gptq_w4a16 \
    --output models/lane08_w4a16_postsft

# Knowledge Distillation (prune + KD)
python scripts/02_finetune_sft.py --mode kd --keep-layers 28

# Quantization (by lane)
python scripts/03_quantize.py --lane lane01_gptq_w4a16

# Quantization (manual)
python scripts/03_quantize.py --input-model checkpoints/sft_merged \
    --output models/custom --scheme W4A16 --dampening 0.02

# Evaluation
python scripts/04_eval_vllm.py --model models/lane01_gptq_w4a16 \
    --lane-id lane01_gptq_w4a16

# Ranking + packaging
python scripts/05_rank_and_package.py
```

## Benchmarks

| Benchmark | Type | Metric | Notes |
|-----------|------|--------|-------|
| KMMLU-Pro | MCQA | Accuracy | 정답 알파벳 파싱 |
| KMMLU-Redux | MCQA | Accuracy | 정답 알파벳 파싱 |
| Ko-LongRAG | QA | F1 Score | 장문 컨텍스트 (최대 3072 토큰) |
| KoMT-Bench | Judge | Skip/Null | Judge 모델 없으면 속도만 측정 |

## Environment

- Python 3.10
- torch 2.9.1+cu128, vllm 0.14.1, transformers 4.57.3
- llmcompressor 0.9.0.1, compressed-tensors 0.13.0
- peft, trl, datasets, pyyaml, safetensors

## Reproducibility

- 모든 랜덤 시드: `seed=42` (configs/lanes.yaml)
- 속도 측정: 3회 실행 중앙값
- 벤치마크 디코딩: temperature=0.0 (greedy)
