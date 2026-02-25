#!/usr/bin/env python3
"""
Vocab pruning + QuantizationModifier W8A8.
1. Load base model
2. Identify tokens to remove (zero frequency)
3. Resize embedding & lm_head
4. Update tokenizer
5. Apply QuantizationModifier W8A8
6. Save & create zip
"""
import os, sys, json, gc, time, shutil, subprocess, copy
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE = os.path.join(ROOT, "base_model")

def create_zip(model_dir, zip_path):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    staging = "/tmp/zip_staging"
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(os.path.join(staging, "model"))
    for f in Path(model_dir).iterdir():
        if f.is_file():
            shutil.copy2(f, os.path.join(staging, "model", f.name))
    subprocess.run(["zip", "-r", zip_path, "model/"], cwd=staging, capture_output=True)
    size_gb = os.path.getsize(zip_path) / (1024**3)
    print(f"  Zip: {os.path.basename(zip_path)} ({size_gb:.2f} GB)", flush=True)

def prune_vocab_and_quantize(name, min_count=0):
    """
    Prune vocab (remove tokens with count <= min_count),
    then apply QuantizationModifier W8A8.
    """
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    output_dir = os.path.join(ROOT, "models", f"vocab_{name}")
    zip_path = os.path.join(ROOT, "submit", f"vocab_{name}.zip")

    if os.path.isdir(output_dir) and os.path.exists(os.path.join(output_dir, "config.json")):
        print(f"[SKIP] {name} already exists", flush=True)
        if not os.path.exists(zip_path):
            create_zip(output_dir, zip_path)
        return

    print(f"\n{'='*60}", flush=True)
    print(f"[VOCAB PRUNE] {name}: min_count={min_count}", flush=True)
    print(f"{'='*60}", flush=True)

    # Load token counts
    token_counts = np.load(os.path.join(ROOT, "data", "token_counts.npy"))
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, trust_remote_code=True)
    
    vocab_size = tokenizer.vocab_size
    
    # Determine which tokens to keep
    keep_mask = token_counts > min_count
    
    # Always keep special tokens
    for sid in tokenizer.all_special_ids:
        if sid < vocab_size:
            keep_mask[sid] = True
    
    # Always keep byte-level tokens (first 256+)
    for i in range(min(512, vocab_size)):
        keep_mask[i] = True
    
    keep_ids = np.where(keep_mask)[0]
    new_vocab_size = len(keep_ids)
    
    # Align to multiple of 128 for GPU efficiency
    aligned_size = ((new_vocab_size + 127) // 128) * 128
    
    # If aligned size > keep_ids, add more tokens to fill
    if aligned_size > new_vocab_size:
        # Add highest-frequency tokens that aren't already kept
        remaining = np.where(~keep_mask)[0]
        remaining_sorted = remaining[np.argsort(-token_counts[remaining])]
        extra_needed = aligned_size - new_vocab_size
        extra_ids = remaining_sorted[:extra_needed]
        keep_mask[extra_ids] = True
        keep_ids = np.where(keep_mask)[0]
        new_vocab_size = len(keep_ids)
    
    removed = vocab_size - new_vocab_size
    print(f"  Original vocab: {vocab_size}", flush=True)
    print(f"  New vocab: {new_vocab_size} (removed {removed}, {removed/vocab_size*100:.1f}%)", flush=True)
    
    # Create old_id -> new_id mapping
    old_to_new = {}
    for new_id, old_id in enumerate(keep_ids):
        old_to_new[int(old_id)] = new_id
    
    # === Resize model embedding and lm_head ===
    print("  Resizing embedding and lm_head...", flush=True)
    
    # Get current weights
    embed_weight = model.model.embed_tokens.weight.data  # [vocab_size, hidden_size]
    lm_head_weight = model.lm_head.weight.data  # [vocab_size, hidden_size]
    
    hidden_size = embed_weight.shape[1]
    
    # Select rows for kept tokens
    new_embed = embed_weight[keep_ids].clone()  # [new_vocab_size, hidden_size]
    new_lm_head = lm_head_weight[keep_ids].clone()
    
    # Replace embedding layer
    model.model.embed_tokens = torch.nn.Embedding(new_vocab_size, hidden_size)
    model.model.embed_tokens.weight.data = new_embed
    
    # Replace lm_head
    model.lm_head = torch.nn.Linear(hidden_size, new_vocab_size, bias=False)
    model.lm_head.weight.data = new_lm_head
    
    # Update config
    model.config.vocab_size = new_vocab_size
    model.config.max_position_embeddings = 16384  # Also set for speed
    
    print(f"  Model resized: embed={list(new_embed.shape)}, lm_head={list(new_lm_head.shape)}", flush=True)
    
    # === Update tokenizer ===
    print("  Updating tokenizer...", flush=True)
    
    # Save tokenizer to temp dir, modify, reload
    tmp_tok_dir = "/tmp/tok_pruned"
    if os.path.exists(tmp_tok_dir):
        shutil.rmtree(tmp_tok_dir)
    tokenizer.save_pretrained(tmp_tok_dir)
    
    # Modify tokenizer.json (the main tokenizer file for fast tokenizers)
    tok_json_path = os.path.join(tmp_tok_dir, "tokenizer.json")
    if os.path.exists(tok_json_path):
        tok_data = json.load(open(tok_json_path))
        
        # Update vocab mapping
        if "model" in tok_data and "vocab" in tok_data["model"]:
            old_vocab = tok_data["model"]["vocab"]
            new_vocab = {}
            for token, old_id in old_vocab.items():
                if old_id in old_to_new:
                    new_vocab[token] = old_to_new[old_id]
            tok_data["model"]["vocab"] = new_vocab
            print(f"  tokenizer.json vocab: {len(old_vocab)} → {len(new_vocab)}", flush=True)
        
        # Update merges - remove merges that produce tokens not in new vocab
        if "model" in tok_data and "merges" in tok_data["model"]:
            old_merges = tok_data["model"]["merges"]
            new_merges = []
            new_vocab_tokens = set(tok_data["model"]["vocab"].keys())
            for merge in old_merges:
                # merges can be "a b" string or ["a", "b"] list
                if isinstance(merge, list):
                    parts = merge
                else:
                    parts = merge.split(" ")
                if len(parts) == 2:
                    # Merge result token
                    merged_token = parts[0] + parts[1]
                    # Keep merge only if BOTH parts AND the result are in vocab
                    if (parts[0] in new_vocab_tokens and
                        parts[1] in new_vocab_tokens and
                        merged_token in new_vocab_tokens):
                        new_merges.append(merge)
            tok_data["model"]["merges"] = new_merges
            print(f"  Merges: {len(old_merges)} → {len(new_merges)}", flush=True)
        
        # Update added_tokens
        if "added_tokens" in tok_data:
            new_added = []
            for at in tok_data["added_tokens"]:
                old_id = at.get("id")
                if old_id in old_to_new:
                    at["id"] = old_to_new[old_id]
                    new_added.append(at)
            tok_data["added_tokens"] = new_added
        
        json.dump(tok_data, open(tok_json_path, "w"), ensure_ascii=False)
    
    # Update vocab.json
    vocab_json_path = os.path.join(tmp_tok_dir, "vocab.json")
    if os.path.exists(vocab_json_path):
        old_vj = json.load(open(vocab_json_path))
        new_vj = {}
        for token, old_id in old_vj.items():
            if old_id in old_to_new:
                new_vj[token] = old_to_new[old_id]
        json.dump(new_vj, open(vocab_json_path, "w"), ensure_ascii=False)
        print(f"  vocab.json: {len(old_vj)} → {len(new_vj)}", flush=True)
    
    # Update special_tokens_map.json
    stm_path = os.path.join(tmp_tok_dir, "special_tokens_map.json")
    if os.path.exists(stm_path):
        # Special tokens don't need ID remapping in this file (it stores tokens, not IDs)
        pass
    
    # Update tokenizer_config.json
    tok_cfg_path = os.path.join(tmp_tok_dir, "tokenizer_config.json")
    if os.path.exists(tok_cfg_path):
        tok_cfg = json.load(open(tok_cfg_path))
        tok_cfg["model_max_length"] = 16384
        # Update added_tokens_decoder
        if "added_tokens_decoder" in tok_cfg:
            new_atd = {}
            for old_id_str, val in tok_cfg["added_tokens_decoder"].items():
                old_id = int(old_id_str)
                if old_id in old_to_new:
                    new_atd[str(old_to_new[old_id])] = val
            tok_cfg["added_tokens_decoder"] = new_atd
        json.dump(tok_cfg, open(tok_cfg_path, "w"), indent=2, ensure_ascii=False)
    
    # Reload pruned tokenizer
    pruned_tokenizer = AutoTokenizer.from_pretrained(tmp_tok_dir, trust_remote_code=True)
    print(f"  Pruned tokenizer vocab size: {pruned_tokenizer.vocab_size}", flush=True)
    
    # Quick sanity check
    test_text = "안녕하세요. Hello world! 테스트입니다."
    test_ids = pruned_tokenizer.encode(test_text)
    test_decoded = pruned_tokenizer.decode(test_ids)
    print(f"  Sanity check: '{test_text}' → {len(test_ids)} tokens → '{test_decoded}'", flush=True)
    
    # === Apply QuantizationModifier ===
    print("  Applying QuantizationModifier W8A8...", flush=True)
    
    recipe = [
        QuantizationModifier(
            scheme="W8A8",
            targets=["Linear"],
            ignore=["embed_tokens", "lm_head"],
        ),
    ]
    
    t0 = time.time()
    oneshot(model=model, recipe=recipe)
    elapsed = time.time() - t0
    print(f"  Quantization done in {elapsed:.1f}s", flush=True)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, save_compressed=True)
    pruned_tokenizer.save_pretrained(output_dir)
    
    # Copy chat_template.jinja if exists
    chat_tmpl = os.path.join(BASE, "chat_template.jinja")
    if os.path.exists(chat_tmpl):
        shutil.copy2(chat_tmpl, output_dir)
    
    print(f"  Saved to {output_dir}", flush=True)
    
    # Verify saved config
    saved_cfg = json.load(open(os.path.join(output_dir, "config.json")))
    print(f"  Saved vocab_size: {saved_cfg.get('vocab_size')}", flush=True)
    print(f"  Saved max_position_embeddings: {saved_cfg.get('max_position_embeddings')}", flush=True)
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    create_zip(output_dir, zip_path)
    
    # Check model file size
    safetensors = os.path.join(output_dir, "model.safetensors")
    if os.path.exists(safetensors):
        sz = os.path.getsize(safetensors) / (1024**2)
        print(f"  model.safetensors: {sz:.0f} MB", flush=True)

def main():
    # Scenario 1: Remove only unused tokens (count=0)
    prune_vocab_and_quantize("prune80k", min_count=0)
    
    # Scenario 2: Remove tokens used <= 1 time
    prune_vocab_and_quantize("prune71k", min_count=1)
    
    print("\n=== ALL DONE ===", flush=True)

if __name__ == "__main__":
    main()
