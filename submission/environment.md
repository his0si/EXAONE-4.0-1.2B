# 개발 환경

## OS 및 하드웨어
- OS: Ubuntu 22.04 (Linux 6.8.0-60-generic)
- GPU: NVIDIA GeForce RTX 4090 (24GB)
- CUDA: 12.8

## Python 환경
- Python: 3.11.0 (conda)
- 가상환경: `conda create -n quant python=3.11`

## 주요 라이브러리 버전
| 라이브러리 | 버전 |
|---|---|
| torch | 2.9.0+cu128 |
| vllm | 0.14.1 |
| llmcompressor | 0.9.0.1 |
| compressed-tensors | 0.13.0 |
| transformers | 4.57.3 |
| datasets | (latest) |

## 외부 데이터 출처
| 데이터셋 | 출처 | 용도 |
|---|---|---|
| LGAI-EXAONE/MANTA-1M | HuggingFace | 캘리브레이션 데이터 (2048개 샘플) |

※ MANTA-1M은 대회 제공 베이스 모델(EXAONE-4.0-1.2B)과 동일한 LG AI Research 공개 데이터셋입니다.
