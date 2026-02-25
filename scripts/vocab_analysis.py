#!/usr/bin/env python3
"""Analyze token frequencies to identify safe-to-remove tokens."""
import json, os, sys, numpy as np
from collections import Counter
from transformers import AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE = os.path.join(ROOT, "base_model")

def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")
    
    token_counts = np.zeros(vocab_size, dtype=np.int64)
    
    # 1. MANTA calibration data (5000 samples)
    print("\n[1] Loading MANTA data...", flush=True)
    manta = json.load(open(os.path.join(ROOT, "data", "manta", "train.json")))[:5000]
    for s in manta:
        text = tokenizer.apply_chat_template(
            s["conversations"], add_generation_prompt=True, tokenize=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        for tid in ids:
            if tid < vocab_size:
                token_counts[tid] += 1
    print(f"  After MANTA: {np.count_nonzero(token_counts)} unique tokens used", flush=True)
    
    # 2. KMMLU-Pro
    print("[2] Loading KMMLU-Pro...", flush=True)
    from datasets import load_dataset
    try:
        ds = load_dataset("LGAI-EXAONE/KMMLU-Pro", split="test")
        for row in ds:
            text = str(row.get("question", "")) + " " + " ".join(row.get("options", []))
            ids = tokenizer.encode(text, add_special_tokens=False)
            for tid in ids:
                if tid < vocab_size:
                    token_counts[tid] += 1
    except Exception as e:
        print(f"  Error: {e}")
    print(f"  After KMMLU-Pro: {np.count_nonzero(token_counts)} unique tokens used", flush=True)
    
    # 3. KMMLU-Redux
    print("[3] Loading KMMLU-Redux...", flush=True)
    try:
        ds = load_dataset("LGAI-EXAONE/KMMLU-Redux", split="test")
        for row in ds:
            text = str(row.get("question", "")) + " " + " ".join(row.get("options", row.get("choices", [])))
            ids = tokenizer.encode(text, add_special_tokens=False)
            for tid in ids:
                if tid < vocab_size:
                    token_counts[tid] += 1
    except Exception as e:
        print(f"  Error: {e}")
    print(f"  After KMMLU-Redux: {np.count_nonzero(token_counts)} unique tokens used", flush=True)
    
    # 4. Ko-LongRAG
    print("[4] Loading Ko-LongRAG...", flush=True)
    try:
        ds = load_dataset("LGAI-EXAONE/Ko-LongRAG", split="test")
        for row in ds:
            text = str(row.get("context", ""))[:20000] + " " + str(row.get("question", ""))
            ids = tokenizer.encode(text, add_special_tokens=False)
            for tid in ids:
                if tid < vocab_size:
                    token_counts[tid] += 1
    except Exception as e:
        print(f"  Error: {e}")
    print(f"  After Ko-LongRAG: {np.count_nonzero(token_counts)} unique tokens used", flush=True)
    
    # 5. KoMT-Bench
    print("[5] Loading KoMT-Bench...", flush=True)
    try:
        ds = load_dataset("LGAI-EXAONE/KoMT-Bench", split="train")
        for row in ds:
            for turn in row.get("turns", []):
                ids = tokenizer.encode(str(turn), add_special_tokens=False)
                for tid in ids:
                    if tid < vocab_size:
                        token_counts[tid] += 1
    except Exception as e:
        print(f"  Error: {e}")
    print(f"  After KoMT-Bench: {np.count_nonzero(token_counts)} unique tokens used", flush=True)
    
    # Analysis
    used = np.count_nonzero(token_counts)
    unused = vocab_size - used
    
    print(f"\n{'='*60}")
    print(f"Total vocab: {vocab_size}")
    print(f"Used tokens: {used} ({used/vocab_size*100:.1f}%)")
    print(f"Unused tokens: {unused} ({unused/vocab_size*100:.1f}%)")
    
    # Special tokens
    special_ids = set(tokenizer.all_special_ids)
    print(f"Special token IDs: {sorted(special_ids)}")
    
    # Distribution
    for threshold in [0, 1, 5, 10, 50, 100]:
        count = np.sum(token_counts <= threshold)
        print(f"  Tokens with count <= {threshold:>3}: {count} ({count/vocab_size*100:.1f}%)")
    
    # Pruning scenarios
    print(f"\n{'='*60}")
    print("Pruning scenarios (keeping all tokens with count > threshold):")
    for min_count in [0, 1, 2, 5]:
        keep = set(np.where(token_counts > min_count)[0])
        keep.update(special_ids)
        keep.update(range(256))  # Keep byte-level tokens
        new_size = len(keep)
        removed = vocab_size - new_size
        # Round to multiple of 64 for efficiency
        new_size_aligned = ((new_size + 63) // 64) * 64
        print(f"  min_count>{min_count}: keep {new_size} ({new_size_aligned} aligned) → remove {removed} ({removed/vocab_size*100:.1f}%)")
    
    # Save analysis
    np.save(os.path.join(ROOT, "data", "token_counts.npy"), token_counts)
    print(f"\nSaved token_counts to data/token_counts.npy")

if __name__ == "__main__":
    main()
