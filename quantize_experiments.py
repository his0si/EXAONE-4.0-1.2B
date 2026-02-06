import os
import torch
import shutil
import json
import time
from pathlib import Path

# CUDA 초기화
if torch.cuda.is_available():
    torch.cuda.init()
    torch.zeros(1).cuda()
    print(f"[INFO] CUDA initialized: {torch.cuda.get_device_name(0)}")

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# 실험 설정들
EXPERIMENTS = [
    {
        "name": "W4A16_g128",
        "scheme": "W4A16",
        "group_size": 128,
        "num_samples": 256,
        "max_seq_len": 512,
    },
    {
        "name": "W4A16_g64",
        "scheme": "W4A16",
        "group_size": 64,
        "num_samples": 256,
        "max_seq_len": 512,
    },
    {
        "name": "W4A16_samples512",
        "scheme": "W4A16",
        "group_size": 128,
        "num_samples": 512,
        "max_seq_len": 512,
    },
    {
        "name": "W8A16",
        "scheme": "W8A16",
        "group_size": 128,
        "num_samples": 256,
        "max_seq_len": 512,
    },
]

MODEL_ID = "./base_model"
DATASET_ID = "LGAI-EXAONE/MANTA-1M"

def run_quantization(config):
    """양자화 실행"""
    out_dir = f"./model_{config['name']}"

    if os.path.exists(out_dir):
        print(f"[SKIP] {config['name']} 이미 존재함: {out_dir}")
        return out_dir

    print(f"\n{'='*60}")
    print(f"양자화 시작: {config['name']}")
    print(f"설정: {config}")
    print(f"{'='*60}")

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # 모델 로드
    print("[INFO] 모델 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    # 데이터셋 로드
    print("[INFO] 캘리브레이션 데이터 로드 중...")
    ds = load_dataset(
        DATASET_ID,
        split=f"train[:{config['num_samples']}]",
    )

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["conversations"],
                add_generation_prompt=True,
                tokenize=False
            )
        }

    ds = ds.map(preprocess)

    # GPTQ Modifier 설정
    gptq_kwargs = {
        "scheme": config["scheme"],
        "targets": ["Linear"],
        "ignore": ["embed_tokens", "lm_head"],
    }

    # group_size가 지정된 경우
    if "group_size" in config:
        gptq_kwargs["block_size"] = config["group_size"]

    recipe = [GPTQModifier(**gptq_kwargs)]

    # 양자화 실행
    print(f"[INFO] GPTQ 시작...")
    start_time = time.time()

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=config["max_seq_len"],
        num_calibration_samples=config["num_samples"],
    )

    quant_time = time.time() - start_time
    print(f"[INFO] 양자화 완료 (소요 시간: {quant_time:.1f}초)")

    # 저장
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, save_compressed=True)
    tokenizer.save_pretrained(out_dir)

    # 메모리 정리
    del model
    torch.cuda.empty_cache()

    print(f"[INFO] 저장 완료: {out_dir}")
    return out_dir

def main():
    import sys

    # 특정 실험만 실행하려면 인자로 전달
    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
        experiments = [e for e in EXPERIMENTS if e["name"] == exp_name]
        if not experiments:
            print(f"실험을 찾을 수 없음: {exp_name}")
            print(f"가능한 실험: {[e['name'] for e in EXPERIMENTS]}")
            return
    else:
        experiments = EXPERIMENTS

    results = []
    for config in experiments:
        try:
            out_dir = run_quantization(config)

            # 모델 크기 확인
            model_size = sum(f.stat().st_size for f in Path(out_dir).glob("*.safetensors")) / (1024**3)

            results.append({
                "name": config["name"],
                "config": config,
                "output_dir": out_dir,
                "model_size_gb": model_size,
            })
        except Exception as e:
            print(f"[ERROR] {config['name']} 실패: {e}")
            import traceback
            traceback.print_exc()

    # 결과 저장
    with open("quantization_experiments.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("양자화 실험 완료")
    print("="*60)
    for r in results:
        print(f"{r['name']}: {r['model_size_gb']:.2f} GB")

if __name__ == "__main__":
    main()
