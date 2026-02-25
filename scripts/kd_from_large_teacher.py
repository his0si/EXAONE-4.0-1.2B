#!/usr/bin/env python3
"""
kd_from_large_teacher.py — Knowledge Distillation from a larger EXAONE model.

Teacher: EXAONE-Deep-7.8B (loaded in NF4 4-bit, ~4GB VRAM)
Student: EXAONE-4.0-1.2B (BF16, all params trainable, ~2.4GB VRAM)

Both share vocab_size=102400, so logit-level KD works directly.

Usage:
  python scripts/kd_from_large_teacher.py
  python scripts/kd_from_large_teacher.py --teacher LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
  python scripts/kd_from_large_teacher.py --num-samples 50000 --epochs 5
"""
import os, sys, gc, json, argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, get_cosine_schedule_with_warmup,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_training_data(tokenizer, num_samples, max_seq_length, seed=42):
    """Load MANTA-1M training data."""
    data_50k = os.path.join(ROOT, "data", "manta", "train_50k.json")
    data_5k = os.path.join(ROOT, "data", "manta", "train.json")

    if os.path.exists(data_50k) and num_samples > 5000:
        samples = json.load(open(data_50k))[:num_samples]
        print(f"[DATA] Loaded {len(samples)} from {data_50k}")
    elif os.path.exists(data_5k):
        samples = json.load(open(data_5k))[:num_samples]
        print(f"[DATA] Loaded {len(samples)} from {data_5k}")
    else:
        from datasets import load_dataset
        ds = load_dataset("LGAI-EXAONE/MANTA-1M", split=f"train[:{num_samples}]")
        samples = [{"conversations": row["conversations"]} for row in ds]

    texts = []
    for s in samples:
        text = tokenizer.apply_chat_template(
            s["conversations"], add_generation_prompt=False, tokenize=False
        )
        texts.append(text)

    print(f"[DATA] Tokenizing {len(texts)} samples, max_seq_length={max_seq_length}...")
    input_ids_list, mask_list = [], []
    for text in texts:
        tok = tokenizer(text, truncation=True, max_length=max_seq_length,
                       padding="max_length", return_tensors="pt")
        input_ids_list.append(tok["input_ids"].squeeze(0))
        mask_list.append(tok["attention_mask"].squeeze(0))

    return TensorDataset(torch.stack(input_ids_list), torch.stack(mask_list))


def run_kd(teacher, student, tokenizer, train_dataset, output_dir, args):
    """Run knowledge distillation: teacher logits → student learns."""
    device = torch.device("cuda")

    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                       num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )
    total_steps = (len(loader) // args.grad_accum) * args.epochs
    warmup = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    student.gradient_checkpointing_enable()
    T = args.temperature
    alpha = args.alpha

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    print(f"\n[KD] Configuration:")
    print(f"  Teacher: {args.teacher}")
    print(f"  Student: {total/1e6:.0f}M params ({trainable/1e6:.0f}M trainable)")
    print(f"  Data: {len(train_dataset)} samples, batch={args.batch_size}, grad_accum={args.grad_accum}")
    print(f"  Effective batch: {args.batch_size * args.grad_accum}")
    print(f"  Epochs: {args.epochs}, Steps: {total_steps}, Warmup: {warmup}")
    print(f"  LR: {args.lr}, Temperature: {T}, Alpha(KD): {alpha}")
    print()

    best_loss = float("inf")
    global_step = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_kd_loss = 0
        epoch_ce_loss = 0
        optimizer.zero_grad()

        for step, (ids, mask) in enumerate(loader):
            ids, mask = ids.to(device), mask.to(device)

            # Teacher forward (no grad, NF4)
            with torch.no_grad():
                t_out = teacher(input_ids=ids, attention_mask=mask)
                t_logits = t_out.logits.float()

            # Student forward
            s_out = student(input_ids=ids, attention_mask=mask, labels=ids)
            s_logits = s_out.logits.float()

            # KD loss
            kd_loss = F.kl_div(
                F.log_softmax(s_logits / T, dim=-1),
                F.softmax(t_logits / T, dim=-1),
                reduction="batchmean") * (T ** 2)

            # Combined loss
            ce_loss = s_out.loss
            loss = (alpha * kd_loss + (1 - alpha) * ce_loss) / args.grad_accum
            loss.backward()

            epoch_loss += loss.item() * args.grad_accum
            epoch_kd_loss += kd_loss.item()
            epoch_ce_loss += ce_loss.item()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if (step + 1) % (args.grad_accum * 50) == 0:
                elapsed = time.time() - start_time
                avg_loss = epoch_loss / (step + 1)
                avg_kd = epoch_kd_loss / (step + 1)
                avg_ce = epoch_ce_loss / (step + 1)
                eta = elapsed / max(global_step, 1) * (total_steps - global_step)
                print(f"  E{epoch+1} step {step+1}/{len(loader)} | "
                      f"loss={avg_loss:.4f} kd={avg_kd:.4f} ce={avg_ce:.4f} | "
                      f"lr={scheduler.get_last_lr()[0]:.2e} | "
                      f"ETA={eta/60:.0f}min")

        avg = epoch_loss / len(loader)
        elapsed = time.time() - start_time
        print(f"\n  Epoch {epoch+1}/{args.epochs} avg_loss={avg:.4f} "
              f"(kd={epoch_kd_loss/len(loader):.4f}, ce={epoch_ce_loss/len(loader):.4f}) "
              f"[{elapsed/60:.1f}min elapsed]")

        if avg < best_loss:
            best_loss = avg
            student.gradient_checkpointing_disable()
            os.makedirs(output_dir, exist_ok=True)
            student.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            student.gradient_checkpointing_enable()
            print(f"  -> Saved best model (loss={best_loss:.4f})")
        print()

    student.gradient_checkpointing_disable()
    total_time = time.time() - start_time
    print(f"[KD] Complete: {total_time/60:.1f} minutes, best_loss={best_loss:.4f}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="KD from large EXAONE teacher")

    # Teacher model
    parser.add_argument("--teacher", type=str,
                       default="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
                       help="HuggingFace teacher model ID")
    parser.add_argument("--teacher-revision", type=str,
                       default="0ff6b5ec7c13",
                       help="HF revision for teacher (pre-transformers-v5)")
    parser.add_argument("--student", type=str, default=None,
                       help="Student model path (default: base_model)")

    # Training
    parser.add_argument("--num-samples", type=int, default=20000)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.7,
                       help="KD loss weight (higher = more KD)")

    # Output
    parser.add_argument("--output", type=str, default="checkpoints/kd_deep78b")
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    output_dir = args.output if os.path.isabs(args.output) else os.path.join(ROOT, args.output)
    if not args.force and os.path.exists(os.path.join(output_dir, "model.safetensors")):
        print(f"[SKIP] Already exists: {output_dir}")
        return

    # base_model: try local path first, fallback to HF
    base_local = os.path.normpath(os.path.join(ROOT, "..", "base_model"))
    if os.path.isdir(base_local):
        base_path = base_local
    else:
        base_path = "LGAI-EXAONE/EXAONE-4.0-1.2B"
        print(f"[INFO] Local base_model not found, using HF: {base_path}")

    # Tokenizer: prefer local copy (from a quantized model dir)
    tok_local = os.path.join(ROOT, "models", "lane01_gptq_w4a16")
    tokenizer_path = tok_local if os.path.isdir(tok_local) else base_path

    student_path = args.student or base_path

    print("=" * 70)
    print(f"  Knowledge Distillation from Large Teacher")
    print(f"  Teacher: {args.teacher} (NF4 4-bit)")
    print(f"  Student: {student_path} (BF16, all params)")
    print(f"  Output:  {output_dir}")
    print("=" * 70)

    # Load student tokenizer (EXAONE-4.0-1.2B)
    print(f"\n[1/4] Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load training data
    print("\n[2/4] Loading training data...")
    train_dataset = load_training_data(tokenizer, args.num_samples, args.max_seq_length)

    # Load teacher (NF4 4-bit)
    print(f"\n[3/4] Loading teacher: {args.teacher} (NF4 4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    teacher_kwargs = dict(
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.teacher_revision:
        teacher_kwargs["revision"] = args.teacher_revision
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, **teacher_kwargs)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    t_mem = sum(p.numel() * p.element_size() for p in teacher.parameters()) / 1e9
    print(f"  Teacher loaded: {t_mem:.2f} GB")

    # Load student (BF16, all params trainable)
    print(f"\n[4/4] Loading student: {student_path} (BF16)...")
    student = AutoModelForCausalLM.from_pretrained(
        student_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    s_mem = sum(p.numel() * p.element_size() for p in student.parameters()) / 1e9
    print(f"  Student loaded: {s_mem:.2f} GB")

    # Check GPU memory
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {alloc:.1f}GB used / {total:.1f}GB total")

    # Run KD
    run_kd(teacher, student, tokenizer, train_dataset, output_dir, args)

    # Cleanup
    del teacher, student
    cleanup_gpu()

    print(f"\n[DONE] KD model saved to {output_dir}")
    print(f"  Next: quantize to W4A16:")
    print(f"    python scripts/03_quantize.py --input-model {args.output} \\")
    print(f"      --output models/kd_deep78b_w4a16 --scheme W4A16 --dampening 0.02")


if __name__ == "__main__":
    main()
