#!/usr/bin/env python3
"""
2-Pass Knowledge Distillation with EXAONE-4.0-32B-GPTQ teacher.

Pass 1: Load 32B-GPTQ teacher → generate top-K logits for training data → save to disk
Pass 2: Unload teacher → load 1.2B student → train with saved logits → W8A8 → eval

Usage:
  # KD on base 1.2B (no pruning)
  python scripts/kd_32b_teacher.py

  # KD on FFN-pruned 1.2B
  python scripts/kd_32b_teacher.py --student checkpoints/wp_ffn3584_kd

  # Customize
  python scripts/kd_32b_teacher.py --samples 5000 --epochs 3 --topk 256 --seq 512
"""

import argparse
import gc
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_MODEL = os.path.join(ROOT, "base_model")
TEACHER_MODEL = os.path.join(ROOT, "teacher_32b_gptq")


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_dir_size_mb(path):
    total = 0
    for f in os.listdir(path):
        fp = os.path.join(path, f)
        if os.path.isfile(fp):
            total += os.path.getsize(fp)
    return total / 1e6


def load_training_texts(tokenizer, num_samples=5000):
    """Load MANTA training data as texts."""
    data_path = os.path.join(ROOT, "data", "manta", "train.json")
    samples = json.load(open(data_path))[:num_samples]
    texts = [tokenizer.apply_chat_template(
        s["conversations"], add_generation_prompt=True, tokenize=False
    ) for s in samples]
    return texts


# ════════════════════════════════════════════════════════════
#  Pass 1: Generate teacher logits
# ════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_teacher_logits(tokenizer, texts, output_dir,
                            max_seq=512, topk=256):
    """
    Load 32B-GPTQ teacher via GPTQModel, run forward on all texts,
    save top-K logits per token to disk in chunks.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check if already done
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path))
        if manifest.get("num_samples") == len(texts) and manifest.get("topk") == topk:
            print(f"[TEACHER SKIP] Logits already saved: {output_dir}")
            return manifest

    print(f"[TEACHER] Loading {TEACHER_MODEL} via GPTQModel...")
    t0 = time.time()

    from gptqmodel import GPTQModel
    teacher = GPTQModel.load(
        TEACHER_MODEL,
        device_map="auto",
        trust_remote_code=True,
    )
    teacher.eval()
    # Get device from first parameter
    teacher_device = next(teacher.parameters()).device

    load_time = time.time() - t0
    gpu_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"[TEACHER] Loaded in {load_time:.1f}s, GPU mem: {gpu_mem:.1f} GB")

    # Process samples and save logits
    chunk_size = 500  # samples per chunk file
    chunk_id = 0
    chunk_data = []
    total_tokens = 0
    t0 = time.time()

    for i, text in enumerate(texts):
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=max_seq)["input_ids"].to(teacher_device)

        out = teacher(input_ids=ids)
        logits = out.logits.squeeze(0).float()  # [seq_len, vocab]

        # Get top-K logits per position
        topk_vals, topk_ids = logits.topk(topk, dim=-1)  # [seq_len, K]

        chunk_data.append({
            "input_ids": ids.squeeze(0).cpu(),
            "topk_values": topk_vals.cpu().half(),  # save as FP16
            "topk_indices": topk_ids.cpu().int(),
        })
        total_tokens += ids.shape[1]

        # Save chunk
        if len(chunk_data) >= chunk_size or i == len(texts) - 1:
            chunk_path = os.path.join(output_dir, f"chunk_{chunk_id:04d}.pt")
            torch.save(chunk_data, chunk_path)
            print(f"  Chunk {chunk_id}: {len(chunk_data)} samples saved "
                  f"[{i+1}/{len(texts)}] {(time.time()-t0)/60:.1f}min")
            chunk_data = []
            chunk_id += 1

        if (i + 1) % 100 == 0:
            elapsed = (time.time() - t0) / 60
            speed = (i + 1) / elapsed
            eta = (len(texts) - i - 1) / speed
            print(f"  {i+1}/{len(texts)} | {total_tokens} tokens | "
                  f"{elapsed:.1f}min elapsed, ~{eta:.1f}min remaining")

    elapsed = (time.time() - t0) / 60
    print(f"[TEACHER] Done: {len(texts)} samples, {total_tokens} tokens, "
          f"{chunk_id} chunks, {elapsed:.1f}min")

    # Save manifest
    manifest = {
        "num_samples": len(texts),
        "topk": topk,
        "max_seq": max_seq,
        "num_chunks": chunk_id,
        "total_tokens": total_tokens,
        "generation_time_min": elapsed,
    }
    json.dump(manifest, open(manifest_path, "w"), indent=2)

    # Cleanup teacher
    del teacher
    cleanup_gpu()

    return manifest


# ════════════════════════════════════════════════════════════
#  Pass 2: Train student with saved logits
# ════════════════════════════════════════════════════════════

def load_teacher_logits(logits_dir):
    """Load all saved teacher logits from chunks."""
    manifest = json.load(open(os.path.join(logits_dir, "manifest.json")))
    all_data = []
    for i in range(manifest["num_chunks"]):
        chunk_path = os.path.join(logits_dir, f"chunk_{i:04d}.pt")
        chunk = torch.load(chunk_path, weights_only=False)
        all_data.extend(chunk)
    print(f"[LOGITS] Loaded {len(all_data)} samples from {manifest['num_chunks']} chunks")
    return all_data, manifest


def train_with_saved_logits(student_path, logits_dir, tokenizer, output_dir,
                            epochs=3, batch_size=1, lr=2e-5,
                            temperature=2.0, alpha=0.7):
    """Train student using pre-saved teacher logits (Pass 2)."""

    if os.path.isdir(output_dir) and os.path.exists(
            os.path.join(output_dir, "config.json")):
        print(f"[KD SKIP] {output_dir} exists, loading...")
        student = AutoModelForCausalLM.from_pretrained(
            output_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)
        return student

    # Ensure GPU is clean before loading student
    cleanup_gpu()
    torch.cuda.reset_peak_memory_stats()

    # Load student
    print(f"[STUDENT] Loading {student_path}...")
    student = AutoModelForCausalLM.from_pretrained(
        student_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    device = "cuda"
    student.to(device)
    print(f"[STUDENT] GPU after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    student.train()
    student.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[STUDENT] {student.config.num_hidden_layers}L, "
          f"inter={student.config.intermediate_size}, "
          f"{trainable/1e6:.0f}M trainable params")

    # Load teacher logits
    all_data, manifest = load_teacher_logits(logits_dir)
    topk = manifest["topk"]
    vocab_size = student.config.vocab_size

    # Filter valid samples
    valid_data = [d for d in all_data if len(d["input_ids"]) > 10]
    print(f"[KD] {len(valid_data)} valid samples, topk={topk}, vocab={vocab_size}")

    grad_accum = 8
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(valid_data) * epochs) // (batch_size * grad_accum)

    print(f"[KD] batch={batch_size}x{grad_accum}, epochs={epochs}, "
          f"lr={lr}, T={temperature}, alpha={alpha}")
    print(f"[KD] total_steps={total_steps}")

    best_loss = float("inf")
    t0 = time.time()
    global_step = 0

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(valid_data))
        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad()

        for bi in range(0, len(perm), batch_size):
            batch_idx = perm[bi:bi+batch_size]
            if len(batch_idx) < batch_size:
                continue

            # Pad batch
            batch_items = [valid_data[j] for j in batch_idx]
            batch_ids = [item["input_ids"] for item in batch_items]
            max_len = max(len(x) for x in batch_ids)

            padded = torch.full((batch_size, max_len),
                               tokenizer.pad_token_id or 0,
                               dtype=torch.long, device=device)
            mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)

            # Pre-allocate teacher logit tensors
            # Reconstruct sparse teacher logits into dense only for valid positions
            t_topk_vals = torch.zeros(batch_size, max_len, topk, device=device)
            t_topk_ids = torch.zeros(batch_size, max_len, topk,
                                     dtype=torch.long, device=device)

            for j, item in enumerate(batch_items):
                seq_len = len(item["input_ids"])
                padded[j, :seq_len] = item["input_ids"]
                mask[j, :seq_len] = 1
                t_topk_vals[j, :seq_len] = item["topk_values"].float().to(device)
                t_topk_ids[j, :seq_len] = item["topk_indices"].long().to(device)

            labels = padded.clone()
            labels[mask == 0] = -100

            # Student forward
            s_out = student(input_ids=padded, attention_mask=mask, labels=labels)
            s_logits = s_out.logits  # [B, L, V]

            # Memory-efficient KD loss using logsumexp trick
            # Avoid materializing full [B, L, V] log_softmax
            s_logits_scaled = s_logits / temperature
            # logsumexp over full vocab (reduction, memory-friendly)
            lse = torch.logsumexp(s_logits_scaled, dim=-1, keepdim=True)  # [B, L, 1]
            # Gather student logits at teacher's top-K positions only
            s_topk_logits = s_logits_scaled.gather(
                dim=-1, index=t_topk_ids)  # [B, L, K]
            # log_softmax at top-K = logits - logsumexp
            s_log_probs_at_topk = s_topk_logits - lse  # [B, L, K]
            del s_logits_scaled, lse, s_topk_logits

            # Teacher probs from top-K logits (renormalized over K)
            t_probs_topk = F.softmax(t_topk_vals / temperature, dim=-1)  # [B, L, K]

            # KL(t||s) ≈ sum_k t_k * (log(t_k) - log(s_k))
            kd_loss_per_token = (t_probs_topk * (
                t_probs_topk.log() - s_log_probs_at_topk)).sum(dim=-1)  # [B, L]
            del s_log_probs_at_topk, t_probs_topk

            # Mask padding
            kd_loss_per_token = kd_loss_per_token * mask.float()
            kd_loss = kd_loss_per_token.sum() / mask.float().sum()
            kd_loss = kd_loss * (temperature ** 2)

            ce_loss = s_out.loss
            loss = alpha * kd_loss + (1 - alpha) * ce_loss
            loss = loss / grad_accum
            loss.backward()

            epoch_loss += loss.item() * grad_accum
            epoch_steps += 1

            if epoch_steps % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 200 == 0:
                    elapsed = (time.time() - t0) / 60
                    avg = epoch_loss / epoch_steps
                    print(f"  E{epoch} step {global_step}/{total_steps} | "
                          f"loss={avg:.4f} (kd={kd_loss.item():.4f}, "
                          f"ce={ce_loss.item():.4f}) | {elapsed:.1f}min")

        avg_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = (time.time() - t0) / 60
        print(f"  Epoch {epoch}/{epochs} avg_loss={avg_loss:.4f} [{elapsed:.1f}min]")

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(output_dir, exist_ok=True)
            student.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  Saved best (loss={best_loss:.4f})")

    student.gradient_checkpointing_disable()
    return student


# ════════════════════════════════════════════════════════════
#  Quantization + Evaluation
# ════════════════════════════════════════════════════════════

def quantize_w8a8(model_path, output_dir, tokenizer):
    """W8A8 GPTQ quantization."""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    if os.path.isdir(output_dir) and os.path.exists(
            os.path.join(output_dir, "config.json")):
        print(f"[SKIP] {output_dir} exists")
        return output_dir

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)

    data_path = os.path.join(ROOT, "data", "manta", "train.json")
    samples = json.load(open(data_path))[:2048]
    texts = [tokenizer.apply_chat_template(
        s["conversations"], add_generation_prompt=True, tokenize=False
    ) for s in samples]
    cal = Dataset.from_dict({"text": texts})

    recipe = [GPTQModifier(
        scheme="W8A8", targets=["Linear"], ignore=["lm_head"],
        dampening_frac=0.02)]

    t0 = time.time()
    oneshot(model=model, dataset=cal, recipe=recipe,
            max_seq_length=1024, num_calibration_samples=2048)
    print(f"  W8A8 quantization done in {time.time()-t0:.1f}s")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)
    print(f"  Saved: {output_dir} ({get_dir_size_mb(output_dir):.1f} MB)")
    del model
    cleanup_gpu()
    return output_dir


def evaluate(name, model_dir):
    """Run local evaluation."""
    import subprocess
    eval_script = os.path.join(ROOT, "scripts", "04_eval_vllm.py")
    result_path = os.path.join(ROOT, "results", name, "metrics.json")
    if os.path.exists(result_path):
        os.remove(result_path)

    subprocess.run([sys.executable, eval_script,
                    "--model", model_dir, "--lane-id", name, "--force"])

    if os.path.exists(result_path):
        metrics = json.load(open(result_path))
        perf = metrics.get("perf_aggregate", 0)
        spt = metrics.get("speed", {}).get("sec_per_token", 999)
        base_perf = 0.325
        base_spt = 0.000552
        pn = perf / base_perf
        sn = 1 - (spt / base_spt)
        score = 0.5 * pn + 0.5 * sn
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"  perf={perf:.4f} (PN={pn:.4f})")
        print(f"  spt={spt*1000:.3f}ms (SN={sn:.4f})")
        print(f"  Score = {score:.4f}")
        print(f"{'='*60}")
        return score
    return None


# ════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=str, default=None,
                       help="Student model path (default: base_model)")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--topk", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--skip-quant", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--name", type=str, default=None,
                       help="Custom experiment name")
    args = parser.parse_args()

    student_path = args.student or BASE_MODEL
    student_name = os.path.basename(student_path.rstrip("/"))

    if args.name:
        exp_name = args.name
    elif student_path == BASE_MODEL:
        exp_name = "kd32b_base"
    else:
        exp_name = f"kd32b_{student_name}"

    logits_dir = os.path.join(ROOT, "checkpoints", "teacher_32b_logits")
    kd_output = os.path.join(ROOT, "checkpoints", f"{exp_name}_kd")
    quant_output = os.path.join(ROOT, "models", f"{exp_name}_w8a8")

    print("=" * 60)
    print(f"  32B Teacher KD: {exp_name}")
    print(f"  Student: {student_path}")
    print(f"  Logits: {logits_dir}")
    print(f"  KD output: {kd_output}")
    print(f"  Quant output: {quant_output}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # ── Pass 1: Generate teacher logits ──
    print("\n" + "=" * 60)
    print("Pass 1: Generate 32B teacher logits")
    print("=" * 60)

    texts = load_training_texts(tokenizer, args.samples)
    print(f"  {len(texts)} training texts loaded")

    manifest = generate_teacher_logits(
        tokenizer, texts, logits_dir,
        max_seq=args.seq, topk=args.topk)

    cleanup_gpu()

    # ── Pass 2: Train student ──
    print("\n" + "=" * 60)
    print("Pass 2: Train student with saved logits")
    print("=" * 60)

    student = train_with_saved_logits(
        student_path, logits_dir, tokenizer, kd_output,
        epochs=args.epochs, batch_size=args.batch, lr=args.lr,
        temperature=args.temperature, alpha=args.alpha)

    del student
    cleanup_gpu()

    # ── W8A8 Quantization ──
    if not args.skip_quant:
        print("\n" + "=" * 60)
        print("Phase 3: W8A8 Quantization")
        print("=" * 60)

        quantize_w8a8(kd_output, quant_output, tokenizer)
        cleanup_gpu()

    # ── Evaluation ──
    if not args.skip_eval and not args.skip_quant:
        print("\n" + "=" * 60)
        print("Phase 4: Evaluation")
        print("=" * 60)

        evaluate(f"{exp_name}_w8a8", quant_output)


if __name__ == "__main__":
    main()
