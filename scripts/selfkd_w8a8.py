#!/usr/bin/env python3
"""
Self-KD (1.2B → 1.2B) → W8A8 Quantization → Evaluation.
No pruning, just KD to boost perf above baseline before quantization.

Usage:
  python scripts/selfkd_w8a8.py
  python scripts/selfkd_w8a8.py --samples 5000 --epochs 5 --seq 512
"""

import argparse
import copy
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


def run_selfkd(tokenizer, output_dir, num_samples=5000, max_seq=512,
               batch_size=2, epochs=3, lr=2e-5, temperature=2.0, alpha=0.7):
    """Self-KD: base model as teacher, copy as student, all params trainable."""

    if os.path.isdir(output_dir) and os.path.exists(
            os.path.join(output_dir, "config.json")):
        print(f"[KD SKIP] {output_dir} exists")
        return

    device = "cuda"

    # Load teacher
    print("[KD] Loading teacher (base model, frozen)...")
    teacher = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"  GPU after teacher: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Load student (separate copy)
    print("[KD] Loading student (trainable copy)...")
    student = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
    student.to(device)
    student.train()
    student.gradient_checkpointing_enable()
    print(f"  GPU after student: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[KD] {trainable/1e6:.0f}M trainable params")

    # Load data
    print("[KD] Loading training data...")
    data_path = os.path.join(ROOT, "data", "manta", "train.json")
    samples = json.load(open(data_path))[:num_samples]
    texts = [tokenizer.apply_chat_template(
        s["conversations"], add_generation_prompt=True, tokenize=False
    ) for s in samples]

    all_ids = []
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=max_seq)["input_ids"].squeeze(0)
        if len(ids) > 10:
            all_ids.append(ids)

    grad_accum = 4
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(all_ids) * epochs) // (batch_size * grad_accum)

    print(f"[KD] Data: {len(all_ids)} samples, batch={batch_size}x{grad_accum}, "
          f"epochs={epochs}")
    print(f"[KD] LR={lr}, T={temperature}, alpha={alpha}, total_steps={total_steps}")

    best_loss = float("inf")
    t0 = time.time()
    global_step = 0

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(all_ids))
        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad()

        for bi in range(0, len(perm), batch_size):
            batch_idx = perm[bi:bi+batch_size]
            if len(batch_idx) < batch_size:
                continue

            # Pad batch
            batch_ids = [all_ids[j] for j in batch_idx]
            max_len = max(len(x) for x in batch_ids)
            padded = torch.full((len(batch_ids), max_len),
                               tokenizer.pad_token_id or 0,
                               dtype=torch.long, device=device)
            mask = torch.zeros(len(batch_ids), max_len,
                              dtype=torch.long, device=device)
            for j, ids in enumerate(batch_ids):
                padded[j, :len(ids)] = ids
                mask[j, :len(ids)] = 1

            labels = padded.clone()
            labels[mask == 0] = -100

            # Teacher forward
            with torch.no_grad():
                t_out = teacher(input_ids=padded, attention_mask=mask)
                t_logits = t_out.logits

            # Student forward
            s_out = student(input_ids=padded, attention_mask=mask, labels=labels)
            s_logits = s_out.logits

            # KD loss
            kd_loss = F.kl_div(
                F.log_softmax(s_logits / temperature, dim=-1),
                F.softmax(t_logits / temperature, dim=-1),
                reduction="batchmean"
            ) * (temperature ** 2)

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
                          f"loss={avg:.4f} | {elapsed:.1f}min")

        avg_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = (time.time() - t0) / 60
        print(f"  Epoch {epoch}/{epochs} avg_loss={avg_loss:.4f} [{elapsed:.1f}min]")

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(output_dir, exist_ok=True)
            student.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  Saved best (loss={best_loss:.4f})")

    del teacher, student
    cleanup_gpu()


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="selfkd_base")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--skip-quant", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    kd_output = os.path.join(ROOT, "checkpoints", f"{args.name}_kd")
    quant_output = os.path.join(ROOT, "models", f"{args.name}_w8a8")

    print("=" * 60)
    print(f"  Self-KD Pipeline: {args.name}")
    print(f"  KD output: {kd_output}")
    print(f"  Quant output: {quant_output}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Phase 1: Self-KD
    print("\n" + "=" * 60)
    print("Phase 1: Self-KD (1.2B → 1.2B)")
    print("=" * 60)

    run_selfkd(tokenizer, kd_output,
               num_samples=args.samples, max_seq=args.seq,
               batch_size=args.batch, epochs=args.epochs, lr=args.lr,
               temperature=args.temperature, alpha=args.alpha)
    cleanup_gpu()

    # Phase 2: W8A8 Quantization
    if not args.skip_quant:
        print("\n" + "=" * 60)
        print("Phase 2: W8A8 Quantization")
        print("=" * 60)

        quantize_w8a8(kd_output, quant_output, tokenizer)
        cleanup_gpu()

    # Phase 3: Evaluation
    if not args.skip_eval and not args.skip_quant:
        print("\n" + "=" * 60)
        print("Phase 3: Evaluation")
        print("=" * 60)

        evaluate(f"{args.name}_w8a8", quant_output)


if __name__ == "__main__":
    main()
