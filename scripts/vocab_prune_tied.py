#!/usr/bin/env python3
"""Vocab pruning with weight tying preserved."""
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

def prune_vocab_tied(name, min_count=0):
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    output_dir = os.path.join(ROOT, "models", f"vp_{name}")
    zip_path = os.path.join(ROOT, "submit", f"vp_{name}.zip")

    if os.path.isdir(output_dir) and os.path.exists(os.path.join(output_dir, "config.json")):
        print(f"[SKIP] {name} already exists", flush=True)
        if not os.path.exists(zip_path):
            create_zip(output_dir, zip_path)
        return

    print(f"\n{'='*60}", flush=True)
    print(f"[VOCAB PRUNE TIED] {name}: min_count={min_count}", flush=True)
    print(f"{'='*60}", flush=True)

    token_counts = np.load(os.path.join(ROOT, "data", "token_counts.npy"))
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, trust_remote_code=True)

    vocab_size = tokenizer.vocab_size

    # Check if model uses weight tying
    tied = model.config.tie_word_embeddings
    print(f"  tie_word_embeddings: {tied}", flush=True)
    print(f"  embed_tokens is lm_head: {model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()}", flush=True)

    # Determine tokens to keep
    keep_mask = token_counts > min_count
    for sid in tokenizer.all_special_ids:
        if sid < vocab_size:
            keep_mask[sid] = True
    for i in range(min(512, vocab_size)):
        keep_mask[i] = True

    keep_ids = np.where(keep_mask)[0]
    new_vocab_size = len(keep_ids)
    aligned_size = ((new_vocab_size + 127) // 128) * 128

    if aligned_size > new_vocab_size:
        remaining = np.where(~keep_mask)[0]
        remaining_sorted = remaining[np.argsort(-token_counts[remaining])]
        extra = remaining_sorted[:aligned_size - new_vocab_size]
        keep_mask[extra] = True
        keep_ids = np.where(keep_mask)[0]
        new_vocab_size = len(keep_ids)

    removed = vocab_size - new_vocab_size
    print(f"  Vocab: {vocab_size} → {new_vocab_size} (removed {removed}, {removed/vocab_size*100:.1f}%)", flush=True)

    old_to_new = {int(old_id): new_id for new_id, old_id in enumerate(keep_ids)}

    # Resize embedding - use model.resize_token_embeddings approach
    # But that only adds/removes from the end. We need selective pruning.
    # So we manually resize but KEEP weight tying.
    
    embed_weight = model.model.embed_tokens.weight.data
    new_embed_data = embed_weight[keep_ids].clone()

    # Create new embedding with correct size
    new_embed = torch.nn.Embedding(new_vocab_size, embed_weight.shape[1])
    new_embed.weight.data = new_embed_data
    model.model.embed_tokens = new_embed

    # For tied weights: lm_head shares the same weight tensor
    new_lm_head = torch.nn.Linear(embed_weight.shape[1], new_vocab_size, bias=False)
    new_lm_head.weight = model.model.embed_tokens.weight  # SHARE the weight!
    model.lm_head = new_lm_head

    # Verify tying
    print(f"  After resize - embed_tokens is lm_head: {model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()}", flush=True)

    model.config.vocab_size = new_vocab_size
    model.config.max_position_embeddings = 16384
    model.config.tie_word_embeddings = True

    # === Update tokenizer ===
    print("  Updating tokenizer...", flush=True)
    tmp_tok_dir = "/tmp/tok_pruned"
    if os.path.exists(tmp_tok_dir):
        shutil.rmtree(tmp_tok_dir)
    tokenizer.save_pretrained(tmp_tok_dir)

    # tokenizer.json
    tok_json_path = os.path.join(tmp_tok_dir, "tokenizer.json")
    if os.path.exists(tok_json_path):
        tok_data = json.load(open(tok_json_path))
        if "model" in tok_data and "vocab" in tok_data["model"]:
            old_vocab = tok_data["model"]["vocab"]
            new_vocab = {t: old_to_new[oid] for t, oid in old_vocab.items() if oid in old_to_new}
            tok_data["model"]["vocab"] = new_vocab
            print(f"  tokenizer.json vocab: {len(old_vocab)} → {len(new_vocab)}", flush=True)

        if "model" in tok_data and "merges" in tok_data["model"]:
            old_merges = tok_data["model"]["merges"]
            new_vocab_tokens = set(tok_data["model"]["vocab"].keys())
            new_merges = []
            for merge in old_merges:
                parts = merge if isinstance(merge, list) else merge.split(" ")
                if len(parts) == 2:
                    merged_token = parts[0] + parts[1]
                    if parts[0] in new_vocab_tokens and parts[1] in new_vocab_tokens and merged_token in new_vocab_tokens:
                        new_merges.append(merge)
            tok_data["model"]["merges"] = new_merges
            print(f"  Merges: {len(old_merges)} → {len(new_merges)}", flush=True)

        if "added_tokens" in tok_data:
            new_added = []
            for at in tok_data["added_tokens"]:
                old_id = at.get("id")
                if old_id in old_to_new:
                    at["id"] = old_to_new[old_id]
                    new_added.append(at)
            tok_data["added_tokens"] = new_added

        json.dump(tok_data, open(tok_json_path, "w"), ensure_ascii=False)

    # vocab.json
    vocab_json_path = os.path.join(tmp_tok_dir, "vocab.json")
    if os.path.exists(vocab_json_path):
        old_vj = json.load(open(vocab_json_path))
        new_vj = {t: old_to_new[oid] for t, oid in old_vj.items() if oid in old_to_new}
        json.dump(new_vj, open(vocab_json_path, "w"), ensure_ascii=False)

    # tokenizer_config.json
    tok_cfg_path = os.path.join(tmp_tok_dir, "tokenizer_config.json")
    if os.path.exists(tok_cfg_path):
        tok_cfg = json.load(open(tok_cfg_path))
        tok_cfg["model_max_length"] = 16384
        if "added_tokens_decoder" in tok_cfg:
            new_atd = {}
            for old_id_str, val in tok_cfg["added_tokens_decoder"].items():
                old_id = int(old_id_str)
                if old_id in old_to_new:
                    new_atd[str(old_to_new[old_id])] = val
            tok_cfg["added_tokens_decoder"] = new_atd
        json.dump(tok_cfg, open(tok_cfg_path, "w"), indent=2, ensure_ascii=False)

    # Reload tokenizer
    pruned_tokenizer = AutoTokenizer.from_pretrained(tmp_tok_dir, trust_remote_code=True)
    test_text = "안녕하세요. Hello world! 테스트입니다."
    test_ids = pruned_tokenizer.encode(test_text)
    test_decoded = pruned_tokenizer.decode(test_ids)
    print(f"  Sanity: '{test_text}' → {len(test_ids)} tokens → '{test_decoded}'", flush=True)

    # === Quantize ===
    print("  Applying QuantizationModifier W8A8...", flush=True)
    recipe = [QuantizationModifier(scheme="W8A8", targets=["Linear"], ignore=["embed_tokens", "lm_head"])]
    t0 = time.time()
    oneshot(model=model, recipe=recipe)
    print(f"  Quantization done in {time.time()-t0:.1f}s", flush=True)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, save_compressed=True)
    pruned_tokenizer.save_pretrained(output_dir)

    # Copy chat_template
    chat_tmpl = os.path.join(BASE, "chat_template.jinja")
    if os.path.exists(chat_tmpl):
        shutil.copy2(chat_tmpl, output_dir)

    # Verify
    saved_cfg = json.load(open(os.path.join(output_dir, "config.json")))
    print(f"  Saved vocab_size: {saved_cfg.get('vocab_size')}", flush=True)
    print(f"  Saved tie_word_embeddings: {saved_cfg.get('tie_word_embeddings')}", flush=True)

    sf_path = os.path.join(output_dir, "model.safetensors")
    sz = os.path.getsize(sf_path) / (1024**2)
    print(f"  model.safetensors: {sz:.0f} MB", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    create_zip(output_dir, zip_path)

def main():
    prune_vocab_tied("80k_tied", min_count=0)
    print("\n=== DONE ===", flush=True)

if __name__ == "__main__":
    main()
