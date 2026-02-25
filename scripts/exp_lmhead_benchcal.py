#!/usr/bin/env python3
"""
Experiments:
1. lm_head quantization (don't ignore lm_head)
2. Benchmark calibration data (kmmlu + ko_longrag)
3. Both combined
"""
import os, sys, gc, time, json, shutil, subprocess, random
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset, concatenate_datasets

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_MODEL = os.path.join(ROOT, "base_model")

def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def load_manta_calibration(tokenizer, num_samples, max_seq):
    """Load MANTA calibration data (original)."""
    data_path = os.path.join(ROOT, "data", "manta", "train.json")
    samples = json.load(open(data_path))[:num_samples]
    texts = []
    for s in samples:
        text = tokenizer.apply_chat_template(
            s["conversations"], add_generation_prompt=True, tokenize=False)
        texts.append(text)
    return Dataset.from_dict({"text": texts})

def load_benchmark_calibration(tokenizer, num_samples, max_seq):
    """Load benchmark-style calibration data from kmmlu + ko_longrag."""
    texts = []
    
    # 1. KMMLU-Pro: multiple choice QA
    print("  Loading KMMLU-Pro...", flush=True)
    try:
        ds_kmmlu = load_dataset("LGAI-EXAONE/KMMLU-Pro", split="test")
        for row in ds_kmmlu:
            question = row.get("question", "")
            options = row.get("options", [])
            if not options:
                continue
            opts_text = "\n".join([f"{chr(65+i)}. {o}" for i, o in enumerate(options)])
            conv = [{"role": "user", "content": f"다음 문제의 정답을 골라주세요.\n\n{question}\n\n{opts_text}"}]
            text = tokenizer.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            texts.append(text)
    except Exception as e:
        print(f"  KMMLU-Pro error: {e}", flush=True)
    
    print(f"  KMMLU-Pro: {len(texts)} samples", flush=True)
    
    # 2. KMMLU-Redux
    print("  Loading KMMLU-Redux...", flush=True)
    try:
        ds_redux = load_dataset("LGAI-EXAONE/KMMLU-Redux", split="test")
        for row in ds_redux:
            question = row.get("question", "")
            options = row.get("options", row.get("choices", []))
            if not options:
                continue
            opts_text = "\n".join([f"{chr(65+i)}. {o}" for i, o in enumerate(options)])
            conv = [{"role": "user", "content": f"다음 문제의 정답을 골라주세요.\n\n{question}\n\n{opts_text}"}]
            text = tokenizer.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            texts.append(text)
    except Exception as e:
        print(f"  KMMLU-Redux error: {e}", flush=True)

    print(f"  Total after KMMLU-Redux: {len(texts)} samples", flush=True)

    # 3. Ko-LongRAG (long context QA)
    print("  Loading Ko-LongRAG...", flush=True)
    try:
        ds_long = load_dataset("LGAI-EXAONE/Ko-LongRAG", split="test")
        for row in ds_long:
            context = row.get("context", "")
            question = row.get("question", "")
            if not context or not question:
                continue
            # Truncate context to fit max_seq
            conv = [{"role": "user", "content": f"다음 문서를 읽고 질문에 답해주세요.\n\n문서:\n{context[:8000]}\n\n질문: {question}"}]
            text = tokenizer.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            texts.append(text)
    except Exception as e:
        print(f"  Ko-LongRAG error: {e}", flush=True)

    print(f"  Total after Ko-LongRAG: {len(texts)} samples", flush=True)

    # Shuffle and limit
    random.seed(42)
    random.shuffle(texts)
    texts = texts[:num_samples]
    print(f"  Final calibration: {len(texts)} samples", flush=True)
    return Dataset.from_dict({"text": texts})

def load_mixed_calibration(tokenizer, num_samples, max_seq):
    """50% MANTA + 50% benchmark data."""
    half = num_samples // 2
    
    # MANTA half
    manta = load_manta_calibration(tokenizer, half, max_seq)
    
    # Benchmark half
    bench = load_benchmark_calibration(tokenizer, half, max_seq)
    
    # Combine
    all_texts = list(manta["text"]) + list(bench["text"])
    random.seed(42)
    random.shuffle(all_texts)
    return Dataset.from_dict({"text": all_texts[:num_samples]})

def quantize_and_zip(name, dampening, cal_samples, max_seq, tokenizer, 
                     ignore_list, cal_fn):
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    output_dir = os.path.join(ROOT, "models", f"exp_{name}")
    zip_path = os.path.join(ROOT, "submit", f"exp_{name}.zip")

    if os.path.isdir(output_dir) and os.path.exists(os.path.join(output_dir, "config.json")):
        print(f"\n[SKIP] {name} already exists", flush=True)
        if not os.path.exists(zip_path):
            _create_zip(output_dir, zip_path)
        return

    print(f"\n{'='*60}", flush=True)
    print(f"[QUANT] {name}: damp={dampening}, cal={cal_samples}, seq={max_seq}", flush=True)
    print(f"  Ignore: {ignore_list}", flush=True)
    print(f"{'='*60}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)

    ds = cal_fn(tokenizer, cal_samples, max_seq)
    print(f"  Loaded {len(ds)} calibration samples", flush=True)

    recipe = [
        GPTQModifier(
            scheme="W8A8",
            targets=["Linear"],
            ignore=ignore_list,
            dampening_frac=dampening,
        ),
    ]

    t0 = time.time()
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_seq,
        num_calibration_samples=cal_samples,
    )
    elapsed = time.time() - t0
    print(f"  Quantization done in {elapsed:.1f}s", flush=True)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)
    print(f"  Saved to {output_dir}", flush=True)

    del model, ds
    cleanup_gpu()

    _create_zip(output_dir, zip_path)

def _create_zip(model_dir, zip_path):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    staging = "/tmp/zip_staging"
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(os.path.join(staging, "model"))
    for f in Path(model_dir).iterdir():
        if f.is_file():
            shutil.copy2(f, os.path.join(staging, "model", f.name))
    subprocess.run(["zip", "-r", zip_path, "model/"],
                   cwd=staging, capture_output=True)
    size_gb = os.path.getsize(zip_path) / (1024**3)
    print(f"  Zip: {os.path.basename(zip_path)} ({size_gb:.2f} GB)", flush=True)

def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Exp 1: lm_head quantized (only ignore embed_tokens)
    quantize_and_zip(
        "lmhead_q", 0.02, 2048, 2048, tokenizer,
        ignore_list=["embed_tokens"],  # lm_head NOT ignored
        cal_fn=load_manta_calibration,
    )

    # Exp 2: Benchmark calibration data
    quantize_and_zip(
        "bench_cal", 0.02, 2048, 2048, tokenizer,
        ignore_list=["embed_tokens", "lm_head"],
        cal_fn=load_benchmark_calibration,
    )

    # Exp 3: lm_head quantized + benchmark calibration
    quantize_and_zip(
        "lmhead_bench", 0.02, 2048, 2048, tokenizer,
        ignore_list=["embed_tokens"],
        cal_fn=load_benchmark_calibration,
    )

    # Exp 4: Mixed calibration (50% MANTA + 50% benchmark)
    quantize_and_zip(
        "mixed_cal", 0.02, 2048, 2048, tokenizer,
        ignore_list=["embed_tokens", "lm_head"],
        cal_fn=load_mixed_calibration,
    )

    print("\n=== ALL DONE ===", flush=True)

if __name__ == "__main__":
    main()
