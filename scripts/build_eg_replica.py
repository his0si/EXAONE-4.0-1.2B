#!/usr/bin/env python3
"""
build_eg_replica.py — eg.py와 동일한 설정으로 W4A16 GPTQ 모델 생성
(256 cal samples, 512 seq length, dampening_frac 미지정 = 기본값)

이것은 대회 예시 코드와 100% 동일한 로직입니다.
"""
import os, sys, torch, shutil, time
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_MODEL = os.path.normpath(os.path.join(ROOT, "..", "base_model"))
OUT_DIR = os.path.join(ROOT, "models", "lane_eg_w4a16")

# eg.py 동일 설정
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512
SCHEME = "W4A16"
TARGETS = ["Linear"]
IGNORE = ["embed_tokens", "lm_head"]


def get_dir_size_mb(path):
    total = 0
    for f in Path(path).rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


def main():
    if os.path.exists(os.path.join(OUT_DIR, "model.safetensors")):
        if "--force" not in sys.argv:
            print(f"[INFO] 이미 존재: {OUT_DIR} (--force로 덮어쓰기)")
            return
        shutil.rmtree(OUT_DIR)

    print(f"[INFO] Base model: {BASE_MODEL}")
    print(f"[INFO] Output: {OUT_DIR}")
    print(f"[INFO] Settings: scheme={SCHEME}, cal={NUM_CALIBRATION_SAMPLES}, seq={MAX_SEQUENCE_LENGTH}")

    # 모델/토크나이저 로드 (GPU 사용)
    print("[INFO] 모델 로드 중 (GPU)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    print(f"[INFO] 모델/토크나이저 로드 완료 (device: {next(model.parameters()).device})")

    # 캘리브레이션 데이터 (eg.py 동일)
    print("[INFO] 캘리브레이션 데이터 로드 중...")
    ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["conversations"],
                add_generation_prompt=True,
                tokenize=False)
        }

    ds = ds.map(preprocess)
    print("[INFO] 데이터 전처리 완료")

    # GPTQ (eg.py 동일 — dampening_frac 미지정)
    print(f"[INFO] GPTQ 시작 (scheme={SCHEME}, samples={NUM_CALIBRATION_SAMPLES}, "
          f"max_len={MAX_SEQUENCE_LENGTH})...")

    recipe = [
        GPTQModifier(
            scheme=SCHEME,
            targets=TARGETS,
            ignore=IGNORE,
        )
    ]

    t0 = time.time()
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )
    elapsed = time.time() - t0
    print(f"[INFO] GPTQ 완료 ({elapsed:.1f}s)")

    # 저장 (eg.py 동일)
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR, save_compressed=True)
    tokenizer.save_pretrained(OUT_DIR)

    size_mb = get_dir_size_mb(OUT_DIR)
    print(f"[INFO] 모델 저장 완료: {OUT_DIR} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
