#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXAONE-4.0-1.2B 경량화 재현 코드
===================================
Public Score 재현용 학습/경량화 코드

방법: QuantizationModifier W8A8 (INT8 weight per-channel + INT8 activation per-token dynamic)
추가: max_position_embeddings를 65536 → 16384로 축소하여 KV cache 메모리 절감 → 추론 속도 향상

실행 방법:
    conda activate quant
    python reproduce.py

필요 시간: ~10초 (양자화), ~2분 (제출 파일 생성)
필요 GPU: 12GB 이상 (RTX 4070 이상)
"""

import os
import sys
import json
import time
import shutil

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 설정
# ============================================================

# 베이스 모델 (HuggingFace에서 자동 다운로드)
BASE_MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"

# 캘리브레이션 데이터셋 (외부 데이터)
CALIBRATION_DATASET = "LGAI-EXAONE/MANTA-1M"
NUM_CALIBRATION_SAMPLES = 2048
MAX_SEQ_LENGTH = 2048

# 출력 경로
OUTPUT_DIR = "./model"
ZIP_NAME = "submit"

# 추론 속도 최적화: context window 축소
# 원본 65536 → 16384로 줄여 KV cache 메모리 절감
MAX_POSITION_EMBEDDINGS = 16384


# ============================================================
# 1단계: 모델 및 토크나이저 로드
# ============================================================

def load_model_and_tokenizer():
    """베이스 모델과 토크나이저를 로드합니다."""
    print("[1/5] 모델 및 토크나이저 로드 중...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    print(f"  모델: {BASE_MODEL_ID}")
    print(f"  파라미터: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model, tokenizer


# ============================================================
# 2단계: 캘리브레이션 데이터 준비
# ============================================================

def prepare_calibration_data(tokenizer):
    """MANTA-1M 데이터셋에서 캘리브레이션 데이터를 준비합니다.

    외부 데이터 출처: LGAI-EXAONE/MANTA-1M (HuggingFace)
    - LG AI Research에서 공개한 한국어/영어 대화 데이터셋
    - 대회 제공 베이스 모델과 동일 출처
    """
    from datasets import load_dataset

    print(f"[2/5] 캘리브레이션 데이터 준비 중 (MANTA-1M, {NUM_CALIBRATION_SAMPLES}개)...")

    ds_raw = load_dataset(CALIBRATION_DATASET, split=f"train[:{NUM_CALIBRATION_SAMPLES}]")

    texts = []
    for sample in ds_raw:
        text = tokenizer.apply_chat_template(
            sample["conversations"],
            add_generation_prompt=True,
            tokenize=False,
        )
        texts.append(text)

    ds = Dataset.from_dict({"text": texts})
    print(f"  샘플 수: {len(ds)}")
    return ds


# ============================================================
# 3단계: W8A8 양자화 (QuantizationModifier)
# ============================================================

def quantize_model(model, dataset):
    """QuantizationModifier를 사용하여 W8A8 INT8 양자화를 적용합니다.

    양자화 방식:
    - Weight: INT8, per-channel, symmetric, static (minmax observer)
    - Activation: INT8, per-token, symmetric, dynamic
    - ignore: embed_tokens, lm_head (임베딩/출력 레이어 제외)

    QuantizationModifier는 Data-Free 방식으로,
    GPTQ와 달리 Hessian 계산 없이 단순 min-max 스케일링을 적용합니다.
    이 방식이 L4 GPU의 CutlassScaledMM 커널에서 최적의 성능을 보입니다.
    """
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    print("[3/5] W8A8 양자화 적용 중 (QuantizationModifier)...")

    recipe = [
        QuantizationModifier(
            scheme="W8A8",
            targets=["Linear"],
            ignore=["embed_tokens", "lm_head"],
        ),
    ]

    t0 = time.time()
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=MAX_SEQ_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )
    elapsed = time.time() - t0
    print(f"  양자화 완료: {elapsed:.1f}초")

    return model


# ============================================================
# 4단계: 모델 저장 및 config 수정
# ============================================================

def save_model(model, tokenizer):
    """양자화된 모델을 저장하고 config를 수정합니다."""
    print(f"[4/5] 모델 저장 중... ({OUTPUT_DIR})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR, save_compressed=True)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # max_position_embeddings 축소 (65536 → 16384)
    # KV cache 메모리 절감으로 추론 속도 향상
    config_path = os.path.join(OUTPUT_DIR, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    config["max_position_embeddings"] = MAX_POSITION_EMBEDDINGS
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 모델 크기 확인
    total_size = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f))
        for f in os.listdir(OUTPUT_DIR)
    )
    print(f"  저장 완료: {total_size / 1024 / 1024:.0f} MB")
    print(f"  max_position_embeddings: {MAX_POSITION_EMBEDDINGS}")


# ============================================================
# 5단계: 제출 ZIP 생성
# ============================================================

def create_submission_zip():
    """제출용 ZIP 파일을 생성합니다.
    형식: submit.zip/model/* (HuggingFace 형식)
    """
    print(f"[5/5] 제출 ZIP 생성 중... ({ZIP_NAME}.zip)")

    shutil.make_archive(
        base_name=ZIP_NAME,
        format="zip",
        root_dir=".",
        base_dir=OUTPUT_DIR,
    )

    zip_size = os.path.getsize(f"{ZIP_NAME}.zip") / 1024 / 1024 / 1024
    print(f"  생성 완료: {ZIP_NAME}.zip ({zip_size:.2f} GB)")


# ============================================================
# 메인 실행
# ============================================================

def main():
    print("=" * 60)
    print("  EXAONE-4.0-1.2B 경량화 재현")
    print("  방법: QuantizationModifier W8A8")
    print("=" * 60)

    # 1. 모델 로드
    model, tokenizer = load_model_and_tokenizer()

    # 2. 캘리브레이션 데이터
    dataset = prepare_calibration_data(tokenizer)

    # 3. 양자화
    model = quantize_model(model, dataset)

    # 4. 저장
    save_model(model, tokenizer)

    # 5. ZIP 생성
    create_submission_zip()

    # 정리
    del model, dataset
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("  완료!")
    print(f"  제출 파일: {ZIP_NAME}.zip")
    print("=" * 60)


if __name__ == "__main__":
    main()
