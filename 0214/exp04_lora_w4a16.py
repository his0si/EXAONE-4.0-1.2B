"""
Experiment 4: LoRA Fine-tuning + W4A16 GPTQ Quantization
Medium-high risk, highest reward potential.
Fine-tune BF16 model on Korean MCQ data with LoRA, merge, then W4A16 GPTQ.
"""
import os
import sys
import json
import torch
import gc
from pathlib import Path

if torch.cuda.is_available():
    torch.cuda.init()
    torch.zeros(1).cuda()
    print(f"[INFO] CUDA: {torch.cuda.get_device_name(0)}")

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from utils.korean_calibration import load_kmmlu_for_sft, load_korean_calibration

BASE_MODEL = "/home/lgaimers/base_model"
MODELS_DIR = "/home/lgaimers/0214/models"
RESULTS_DIR = "/home/lgaimers/0214/results"
LORA_OUTPUT = "/home/lgaimers/0214/models/lora_merged"


def step1_lora_train():
    """Step 1: LoRA fine-tuning on Korean MCQ data."""
    print("=" * 60)
    print("Step 1: LoRA Fine-tuning")
    print("=" * 60)

    if os.path.exists(os.path.join(LORA_OUTPUT, "model.safetensors")):
        print(f"[SKIP] LoRA merged model already exists at {LORA_OUTPUT}")
        return

    print("[INFO] Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    print("[INFO] Applying LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("[INFO] Loading training data...")
    train_ds = load_kmmlu_for_sft(tokenizer, num_samples=5000)

    print("[INFO] Starting training...")
    checkpoint_dir = os.path.join(MODELS_DIR, "lora_checkpoints")

    training_args = SFTConfig(
        output_dir=checkpoint_dir,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_length=512,
        dataset_text_field="text",
        gradient_checkpointing=True,
        optim="adamw_torch",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
    )

    trainer.train()

    print("[INFO] Merging LoRA weights...")
    model = model.merge_and_unload()

    print(f"[INFO] Saving merged model to {LORA_OUTPUT}...")
    os.makedirs(LORA_OUTPUT, exist_ok=True)
    model.save_pretrained(LORA_OUTPUT)
    tokenizer.save_pretrained(LORA_OUTPUT)

    model_size = sum(f.stat().st_size for f in Path(LORA_OUTPUT).glob("*.safetensors")) / (1024**3)
    print(f"[INFO] LoRA merged model: {model_size:.2f} GB")

    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()


def step2_quantize(scheme="W4A16", dampening_frac=0.01, num_samples=2048, max_seq_length=512):
    """Step 2: GPTQ quantize the LoRA-merged model."""
    suffix = scheme.lower()
    if dampening_frac != 0.01:
        suffix += f"_d{str(dampening_frac).replace('.','')}"
    name = f"lora_{suffix}_kmmlu"

    print(f"\n{'='*60}")
    print(f"Step 2: Quantize LoRA model -> {name}")
    print(f"{'='*60}")

    out_dir = os.path.join(MODELS_DIR, name)
    if os.path.exists(os.path.join(out_dir, "model.safetensors")):
        print(f"[SKIP] {name} already exists")
        return name

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    print("[INFO] Loading LoRA-merged model...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_OUTPUT, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LORA_OUTPUT,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    print("[INFO] Loading Korean calibration data...")
    cal_ds = load_korean_calibration(
        tokenizer, num_samples=num_samples, include_manta=True, manta_ratio=0.3,
    )

    print(f"[INFO] {scheme} GPTQ quantization (damp={dampening_frac})...")
    recipe = [
        GPTQModifier(
            scheme=scheme,
            targets=["Linear"],
            ignore=["embed_tokens", "lm_head"],
            dampening_frac=dampening_frac,
        )
    ]

    oneshot(
        model=model,
        dataset=cal_ds,
        recipe=recipe,
        max_seq_length=max_seq_length,
        num_calibration_samples=num_samples,
    )

    print(f"[INFO] Saving to {out_dir}...")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, save_compressed=True)
    tokenizer.save_pretrained(out_dir)

    model_size = sum(f.stat().st_size for f in Path(out_dir).glob("*.safetensors")) / (1024**3)
    print(f"[INFO] Done: {name} ({model_size:.2f} GB)")

    with open(os.path.join(RESULTS_DIR, f"{name}_config.json"), "w") as f:
        json.dump({
            "name": name,
            "lora_base": LORA_OUTPUT,
            "scheme": scheme,
            "dampening_frac": dampening_frac,
            "num_samples": num_samples,
            "size_gb": model_size,
        }, f, indent=2)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return name


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Step 1: LoRA training
    step1_lora_train()

    # Step 2: Quantize with different schemes
    step2_quantize(scheme="W4A16", dampening_frac=0.01, num_samples=2048)
    step2_quantize(scheme="W4A16_ASYM", dampening_frac=0.01, num_samples=2048)
    step2_quantize(scheme="W4A16", dampening_frac=0.02, num_samples=2048)
