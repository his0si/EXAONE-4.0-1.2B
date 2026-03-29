# 개발 환경

## OS 및 하드웨어
- **OS**: Ubuntu 22.04 LTS (Linux 6.8.0-90-generic)
- **GPU (개발)**: NVIDIA GeForce RTX 5060 Ti (16GB VRAM)
- **GPU (평가 서버)**: NVIDIA L4 (24GB VRAM, Ada Lovelace)
- **CUDA**: 12.8
- **Driver**: 570.x

## Python 환경
- **Python**: 3.10 (conda)
- **가상환경**: `conda create -n lgaimers python=3.10`

## 주요 라이브러리 버전

| 라이브러리 | 버전 | 용도 |
|---|---|---|
| torch | 2.9.0+cu128 | 딥러닝 프레임워크 |
| vllm | 0.14.1 | 추론 엔진 (평가 서버) |
| llmcompressor | 0.9.0.1 | 양자화 도구 |
| compressed-tensors | 0.13.0 | 압축 텐서 포맷 |
| transformers | 4.57.3 | 모델 로딩/저장 |
| datasets | latest | 데이터셋 로딩 |
| accelerate | 1.10.1 | 모델 분산 로딩 |
| safetensors | 0.7.0 | 모델 저장 포맷 |

## 외부 데이터 출처

| 데이터셋 | 출처 | 용도 | 사용량 |
|---|---|---|---|
| LGAI-EXAONE/MANTA-1M | HuggingFace Hub | 양자화 캘리브레이션 | 2,048 샘플 |

- MANTA-1M은 대회 제공 베이스 모델(EXAONE-4.0-1.2B)과 동일한 LG AI Research 공개 데이터셋
- 그 외 외부 데이터 사용 없음

## 평가 벤치마크 (대회 제공)
- KMMLU-Pro (2,822 샘플) - 한국어 객관식
- KMMLU-Redux (2,587 샘플) - 한국어 객관식
- Ko-LongRAG (600 샘플) - 긴 문맥 QA
- KoMT-Bench (80 샘플) - GPT-4 채점
