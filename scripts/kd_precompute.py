#!/usr/bin/env python3
"""
KD from EXAONE-4.0-32B with pre-computed teacher logits.
Step 1: Load teacher (NF4), compute logits on all data, save to disk.
Step 2: Unload teacher, load student, train with saved logits.
This avoids OOM by never having both models in memory simultaneously.
"""
import os, sys, gc, json, argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, get_cosine_schedule_with_warmup,
)
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_training_data(tokenizer, num_samples, max_seq_length):
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
            s["conversations"], add_generation_prompt=False, tokenize=False)
        texts.append(text)
    print(f"[DATA] Tokenizing {len(texts)} samples, max_seq_length={max_seq_length}...")
    input_ids_list, mask_list = [], []
    for text in texts:
        tok = tokenizer(text, truncation=True, max_length=max_seq_length,
                       padding="max_length", return_tensors="pt")
        input_ids_list.append(tok["input_ids"].squeeze(0))
        mask_list.append(tok["attention_mask"].squeeze(0))
    return torch.stack(input_ids_list), torch.stack(mask_list)


def step1_precompute_logits(args, tokenizer, input_ids, attention_mask):
    """Load teacher, compute logits for all data, save to disk."""
    logits_dir = os.path.join(ROOT, args.output, "teacher_logits")
    if os.path.exists(logits_dir) and not args.force:
        existing = len([f for f in os.listdir(logits_dir) if f.endswith('.pt')])
        if existing == len(input_ids):
            print(f"[STEP1] Teacher logits already exist ({existing} files), skipping")
            return logits_dir

    os.makedirs(logits_dir, exist_ok=True)
    device = torch.device("cuda")

    print(f"\n[STEP1] Loading teacher: {args.teacher} (NF4 4-bit)...")
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

    t_mem = sum(p.numel() * p.element_size() for p in teacher.parameters()) / 1e9
    print(f"  Teacher loaded: {t_mem:.2f} GB")

    # Compute logits in batches, save top-K logits to save disk space
    top_k = args.top_k  # Save top-K logits instead of all 102400
    print(f"  Computing logits for {len(input_ids)} samples (top-{top_k})...")
    t0 = time.time()

    for i in range(len(input_ids)):
        ids = input_ids[i:i+1].to(device)
        mask = attention_mask[i:i+1].to(device)

        with torch.no_grad():
            out = teacher(input_ids=ids, attention_mask=mask)
            logits = out.logits.float().cpu()  # [1, seq_len, vocab]

            if top_k > 0 and top_k < logits.shape[-1]:
                # Save top-K logits and indices
                topk_vals, topk_idx = logits.topk(top_k, dim=-1)
                torch.save({
                    'values': topk_vals.half(),  # [1, seq, K] in fp16
                    'indices': topk_idx.int(),   # [1, seq, K] in int32 (vocab>32767)
                }, os.path.join(logits_dir, f"{i:06d}.pt"))
            else:
                torch.save(logits.half(), os.path.join(logits_dir, f"{i:06d}.pt"))

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(input_ids) - i - 1)
            print(f"    {i+1}/{len(input_ids)} ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"  [STEP1] Done: {len(input_ids)} logits in {elapsed:.0f}s")

    # Cleanup teacher
    del teacher
    cleanup_gpu()
    return logits_dir


class KDDataset(Dataset):
    """Dataset that loads pre-computed teacher logits from disk."""
    def __init__(self, input_ids, attention_mask, logits_dir, top_k):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.logits_dir = logits_dir
        self.top_k = top_k

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        ids = self.input_ids[idx]
        mask = self.attention_mask[idx]
        data = torch.load(os.path.join(self.logits_dir, f"{idx:06d}.pt"),
                         map_location="cpu", weights_only=True)
        if isinstance(data, dict):
            return ids, mask, data['values'].squeeze(0).float(), data['indices'].squeeze(0).long()
        else:
            return ids, mask, data.squeeze(0).float(), None

    @staticmethod
    def collate_fn(batch):
        ids = torch.stack([b[0] for b in batch])
        mask = torch.stack([b[1] for b in batch])
        values = torch.stack([b[2] for b in batch])
        indices = torch.stack([b[3] for b in batch]) if batch[0][3] is not None else None
        return ids, mask, values, indices


def step2_train_student(args, tokenizer, input_ids, attention_mask, logits_dir):
    """Load student, train with pre-computed logits."""
    device = torch.device("cuda")

    base_local = os.path.normpath(os.path.join(ROOT, "..", "base_model"))
    if os.path.isdir(base_local):
        student_path = base_local
    else:
        student_path = "LGAI-EXAONE/EXAONE-4.0-1.2B"

    print(f"\n[STEP2] Loading student: {student_path} (BF16)...")
    student = AutoModelForCausalLM.from_pretrained(
        student_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)

    output_dir = os.path.join(ROOT, args.output)
    dataset = KDDataset(input_ids, attention_mask, logits_dir, args.top_k)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                       num_workers=2, pin_memory=True, collate_fn=KDDataset.collate_fn)

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    total_steps = (len(loader) // args.grad_accum) * args.epochs
    warmup = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    student.gradient_checkpointing_enable()
    T = args.temperature
    alpha = args.alpha
    vocab_size = student.config.vocab_size

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"\n[KD] Configuration:")
    print(f"  Student: {trainable/1e6:.0f}M trainable params")
    print(f"  Data: {len(dataset)} samples, batch={args.batch_size}, grad_accum={args.grad_accum}")
    print(f"  Epochs: {args.epochs}, Steps: {total_steps}, Warmup: {warmup}")
    print(f"  LR: {args.lr}, Temperature: {T}, Alpha: {alpha}, Top-K: {args.top_k}")
    print()

    best_loss = float("inf")
    global_step = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_kd = 0
        epoch_ce = 0
        optimizer.zero_grad()

        for step, (ids, mask, t_values, t_indices) in enumerate(loader):
            ids, mask = ids.to(device), mask.to(device)

            # Student forward
            s_out = student(input_ids=ids, attention_mask=mask, labels=ids)
            s_logits = s_out.logits.float()

            # Reconstruct teacher logits for KD
            if t_indices is not None:
                # Top-K: create sparse teacher logits
                t_values = t_values.to(device)
                t_indices = t_indices.to(device)
                # Compute KD loss using only top-K positions
                s_topk = torch.gather(s_logits / T, -1, t_indices)
                t_soft = F.softmax(t_values / T, dim=-1)
                s_log_soft = F.log_softmax(s_topk, dim=-1)
                kd_loss = F.kl_div(s_log_soft, t_soft, reduction="batchmean") * (T ** 2)
            else:
                t_logits = t_values.to(device)
                kd_loss = F.kl_div(
                    F.log_softmax(s_logits / T, dim=-1),
                    F.softmax(t_logits / T, dim=-1),
                    reduction="batchmean") * (T ** 2)

            ce_loss = s_out.loss
            loss = (alpha * kd_loss + (1 - alpha) * ce_loss) / args.grad_accum
            loss.backward()

            epoch_loss += loss.item() * args.grad_accum
            epoch_kd += kd_loss.item()
            epoch_ce += ce_loss.item()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if (step + 1) % (args.grad_accum * 50) == 0:
                elapsed = time.time() - start_time
                avg_loss = epoch_loss / (step + 1)
                avg_kd = epoch_kd / (step + 1)
                avg_ce = epoch_ce / (step + 1)
                eta = elapsed / max(global_step, 1) * (total_steps - global_step)
                print(f"  E{epoch+1} step {step+1}/{len(loader)} | "
                      f"loss={avg_loss:.4f} kd={avg_kd:.4f} ce={avg_ce:.4f} | "
                      f"lr={scheduler.get_last_lr()[0]:.2e} | ETA={eta/60:.0f}min")

        avg = epoch_loss / len(loader)
        elapsed = time.time() - start_time
        print(f"\n  Epoch {epoch+1}/{args.epochs} avg_loss={avg:.4f} "
              f"(kd={epoch_kd/len(loader):.4f}, ce={epoch_ce/len(loader):.4f}) "
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


def main():
    parser = argparse.ArgumentParser(description="KD with pre-computed teacher logits")
    parser.add_argument("--teacher", default="LGAI-EXAONE/EXAONE-4.0-32B")
    parser.add_argument("--teacher-revision", default="")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=256,
                        help="Save top-K teacher logits (0=all)")
    parser.add_argument("--output", default="checkpoints/kd_40_32b")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--step", choices=["all", "logits", "train"], default="all")
    args = parser.parse_args()

    tok_path = os.path.join(ROOT, "models", "lane01_gptq_w4a16")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    input_ids, attention_mask = load_training_data(
        tokenizer, args.num_samples, args.max_seq_length)

    if args.step in ("all", "logits"):
        logits_dir = step1_precompute_logits(args, tokenizer, input_ids, attention_mask)
    else:
        logits_dir = os.path.join(ROOT, args.output, "teacher_logits")

    if args.step in ("all", "train"):
        step2_train_student(args, tokenizer, input_ids, attention_mask, logits_dir)

    print(f"\n[DONE] KD model saved to {os.path.join(ROOT, args.output)}")


if __name__ == "__main__":
    main()
