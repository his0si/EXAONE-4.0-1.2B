import os
import torch
from pathlib import Path

if torch.cuda.is_available():
    torch.cuda.init()
    torch.zeros(1).cuda()
    print(f"[INFO] CUDA: {torch.cuda.get_device_name(0)}")

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

MODEL_ID = "./base_model"
OUT_DIR = "./model_W4A16_n1024"

NUM_CALIBRATION_SAMPLES = 1024
MAX_SEQUENCE_LENGTH = 512

print("[INFO] 모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

print("[INFO] 캘리브레이션 데이터 로드 중...")
ds = load_dataset("LGAI-EXAONE/MANTA-1M", split=f"train[:{NUM_CALIBRATION_SAMPLES}]")

def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False
        )
    }

ds = ds.map(preprocess)

print(f"[INFO] W4A16 GPTQ 양자화 시작 (samples={NUM_CALIBRATION_SAMPLES})...")
recipe = [
    GPTQModifier(
        scheme="W4A16",
        targets=["Linear"],
        ignore=["embed_tokens", "lm_head"],
    )
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("[INFO] 저장 중...")
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

model_size = sum(f.stat().st_size for f in Path(OUT_DIR).glob("*.safetensors")) / (1024**3)
print(f"[INFO] 완료: {OUT_DIR} ({model_size:.2f} GB)")
