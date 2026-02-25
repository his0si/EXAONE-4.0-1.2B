#!/usr/bin/env python3
"""
aggressive_prune_kd.py — 공격적 구조적 프루닝 + 강화된 Knowledge Distillation + FP8 Static

FLOP 감소를 통한 compute-bound 최적화:
  1) Layer pruning (depth): 레이어 제거로 직접적 FLOP 감소
  2) FFN width pruning: intermediate_size 축소로 FFN FLOP 감소
  3) Knowledge Distillation: teacher→student 학습으로 성능 회복
  4) FP8 Static quantization: 추가 속도 향상

Usage:
  # 22 layers only
  python scripts/aggressive_prune_kd.py --keep-layers 22 --output checkpoints/pruned22_kd

  # 22 layers + FFN 3072
  python scripts/aggressive_prune_kd.py --keep-layers 22 --ffn-size 3072 --output checkpoints/pruned22_ffn3072_kd

  # 20 layers + FP8 static
  python scripts/aggressive_prune_kd.py --keep-layers 20 --apply-fp8 --output checkpoints/pruned20_kd
"""
import os, sys, gc, json, argparse, time, shutil
import yaml, torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_config(path="configs/lanes.yaml"):
    with open(os.path.join(ROOT, path)) as f:
        return yaml.safe_load(f)


def get_base_model_path(cfg):
    return os.path.normpath(os.path.join(ROOT, cfg["project"]["base_model"]))


# ════════════════════════════════════════════════════════════
#  Step 1: Block Influence (BI) Scoring for Layer Importance
# ════════════════════════════════════════════════════════════
def compute_bi_scores(model, tokenizer, device, num_samples=200, max_len=512):
    """Compute Block Influence scores using cosine similarity.
    Lower BI = more similar input/output = less important = safe to remove."""
    print(f"[BI] Computing layer importance with {num_samples} samples...")

    ds = load_dataset("LGAI-EXAONE/MANTA-1M", split=f"train[:{num_samples}]")
    texts = []
    for row in ds:
        text = tokenizer.apply_chat_template(
            row["conversations"], add_generation_prompt=False, tokenize=False
        )
        texts.append(text)

    num_layers = model.config.num_hidden_layers
    layer_inputs, layer_outputs = {}, {}

    def make_hook(idx):
        def fn(module, inp, out):
            layer_inputs[idx] = inp[0].detach().float()
            layer_outputs[idx] = (out[0] if isinstance(out, tuple) else out).detach().float()
        return fn

    hooks = [layer.register_forward_hook(make_hook(i))
             for i, layer in enumerate(model.model.layers)]

    cos_sims = torch.zeros(num_layers)
    count = 0

    with torch.no_grad():
        for text in texts:
            tok = tokenizer(text, truncation=True, max_length=max_len,
                           return_tensors="pt").to(device)
            seq_len = tok["attention_mask"].sum().item()
            if seq_len < 2:
                continue
            model(**tok)
            for li in range(num_layers):
                i_mean = layer_inputs[li][0, :int(seq_len)].mean(0)
                o_mean = layer_outputs[li][0, :int(seq_len)].mean(0)
                cos_sims[li] += F.cosine_similarity(
                    i_mean.unsqueeze(0), o_mean.unsqueeze(0)).item()
            count += 1

    for h in hooks:
        h.remove()
    cos_sims /= max(count, 1)

    # Higher cosine similarity = input ≈ output = layer is redundant
    # Rank by highest similarity first (least important)
    bi_scores = cos_sims  # higher = more redundant
    ranked = sorted(range(num_layers), key=lambda i: -bi_scores[i].item())

    print(f"[BI] Layer importance (most redundant first):")
    for i, idx in enumerate(ranked):
        marker = " ← REMOVE" if i < (num_layers - 22) else ""
        print(f"  Layer {idx:2d}: BI={bi_scores[idx]:.4f}{marker}")

    return ranked


# ════════════════════════════════════════════════════════════
#  Step 2: Layer Pruning (Depth)
# ════════════════════════════════════════════════════════════
def prune_layers(model, layers_to_remove):
    """Remove transformer layers and re-index."""
    keep = [i for i in range(model.config.num_hidden_layers) if i not in layers_to_remove]
    print(f"[PRUNE] Keeping layers: {keep}")
    print(f"[PRUNE] Removing layers: {sorted(layers_to_remove)}")

    model.model.layers = nn.ModuleList([model.model.layers[i] for i in keep])
    model.config.num_hidden_layers = len(keep)

    if hasattr(model.config, "layer_types") and model.config.layer_types:
        model.config.layer_types = [model.config.layer_types[i] for i in keep]

    for new_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn"):
            layer.self_attn.layer_idx = new_idx

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[PRUNE] After layer pruning: {model.config.num_hidden_layers}L, {total_params/1e6:.0f}M params")
    return model


# ════════════════════════════════════════════════════════════
#  Step 3: FFN Width Pruning
# ════════════════════════════════════════════════════════════
def prune_ffn_width(model, target_intermediate_size, device):
    """Prune FFN intermediate dimension by neuron importance."""
    current_size = model.config.intermediate_size
    if target_intermediate_size >= current_size:
        print(f"[FFN] No pruning needed (target={target_intermediate_size} >= current={current_size})")
        return model

    print(f"[FFN] Pruning FFN: {current_size} → {target_intermediate_size}")
    keep_n = target_intermediate_size

    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp

        # Get weight tensors
        gate_w = mlp.gate_proj.weight.data  # [intermediate, hidden]
        up_w = mlp.up_proj.weight.data      # [intermediate, hidden]
        down_w = mlp.down_proj.weight.data  # [hidden, intermediate]

        # Compute neuron importance: L2 norm across all connected weights
        importance = (
            gate_w.float().norm(dim=1) +
            up_w.float().norm(dim=1) +
            down_w.float().norm(dim=0)
        )

        # Select top-k neurons
        _, top_indices = importance.topk(keep_n)
        top_indices = top_indices.sort().values

        # Prune weights
        mlp.gate_proj.weight = nn.Parameter(gate_w[top_indices, :])
        mlp.up_proj.weight = nn.Parameter(up_w[top_indices, :])
        mlp.down_proj.weight = nn.Parameter(down_w[:, top_indices])

        # Update dimensions
        mlp.gate_proj.out_features = keep_n
        mlp.up_proj.out_features = keep_n
        mlp.down_proj.in_features = keep_n

        if layer_idx == 0:
            print(f"  Layer 0: gate_proj {list(gate_w.shape)} → {list(mlp.gate_proj.weight.shape)}")

    model.config.intermediate_size = target_intermediate_size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[FFN] After FFN pruning: intermediate={target_intermediate_size}, {total_params/1e6:.0f}M params")
    return model


# ════════════════════════════════════════════════════════════
#  Step 4: Knowledge Distillation
# ════════════════════════════════════════════════════════════
def load_training_data(tokenizer, num_samples, max_seq_length, seed=42):
    """Load MANTA-1M training data."""
    print(f"[DATA] Loading {num_samples} samples from MANTA-1M...")
    # Try 50K file first, then 5K, then download
    data_50k = os.path.join(ROOT, "data", "manta", "train_50k.json")
    data_5k = os.path.join(ROOT, "data", "manta", "train.json")
    if os.path.exists(data_50k):
        samples = json.load(open(data_50k))[:num_samples]
        print(f"[DATA] Loaded from {data_50k} ({len(samples)} samples)")
    elif os.path.exists(data_5k):
        samples = json.load(open(data_5k))[:num_samples]
        print(f"[DATA] Loaded from {data_5k} ({len(samples)} samples)")
    else:
        ds = load_dataset("LGAI-EXAONE/MANTA-1M", split=f"train[:{num_samples}]")
        samples = [{"conversations": row["conversations"]} for row in ds]

    texts = []
    for s in samples:
        text = tokenizer.apply_chat_template(
            s["conversations"], add_generation_prompt=False, tokenize=False
        )
        texts.append(text)

    print(f"[DATA] Tokenizing with max_seq_length={max_seq_length}...")
    input_ids_list, mask_list = [], []
    for text in texts:
        tok = tokenizer(text, truncation=True, max_length=max_seq_length,
                       padding="max_length", return_tensors="pt")
        input_ids_list.append(tok["input_ids"].squeeze(0))
        mask_list.append(tok["attention_mask"].squeeze(0))

    return TensorDataset(torch.stack(input_ids_list), torch.stack(mask_list))


def run_kd(teacher, student, tokenizer, train_dataset, output_dir, args):
    """Run knowledge distillation training."""
    device = next(student.parameters()).device
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
    print(f"\n[KD] Training Configuration:")
    print(f"  Student: {student.config.num_hidden_layers}L, {total/1e6:.0f}M params (all trainable)")
    print(f"  Data: {len(train_dataset)} samples, batch={args.batch_size}, grad_accum={args.grad_accum}")
    print(f"  Effective batch: {args.batch_size * args.grad_accum}")
    print(f"  Epochs: {args.epochs}, Total steps: {total_steps}, Warmup: {warmup}")
    print(f"  LR: {args.lr}, Temperature: {T}, Alpha: {alpha}")
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

            with torch.no_grad():
                t_logits = teacher(input_ids=ids, attention_mask=mask).logits.float()

            s_out = student(input_ids=ids, attention_mask=mask, labels=ids)
            s_logits = s_out.logits.float()

            # KD loss: KL divergence between teacher and student distributions
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
            print(f"  ✓ Saved best model (loss={best_loss:.4f})")
        print()

    student.gradient_checkpointing_disable()
    total_time = time.time() - start_time
    print(f"[KD] Training complete: {total_time/60:.1f} minutes, best_loss={best_loss:.4f}")
    return output_dir


# ════════════════════════════════════════════════════════════
#  Step 5: FP8 Static Quantization
# ════════════════════════════════════════════════════════════
def apply_fp8_static(model_path, output_path, tokenizer, num_calib=512, max_len=1024):
    """Apply FP8 static quantization using llmcompressor."""
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor import oneshot

    print(f"\n[FP8] Applying FP8 Static quantization...")
    print(f"  Input: {model_path}")
    print(f"  Output: {output_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )

    # Calibration data
    ds = load_dataset("LGAI-EXAONE/MANTA-1M", split=f"train[:{num_calib}]")
    texts = []
    for row in ds:
        text = tokenizer.apply_chat_template(
            row["conversations"], add_generation_prompt=False, tokenize=False
        )
        texts.append(text)

    def preprocess(example):
        return tokenizer(example, truncation=True, max_length=max_len,
                        padding=False, return_tensors="pt")

    calib_data = [preprocess(t) for t in texts[:num_calib]]

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8",
        ignore=["lm_head"],
    )

    oneshot(
        model=model,
        dataset=calib_data,
        recipe=recipe,
        output_dir=output_path,
    )

    # Copy tokenizer files
    tokenizer.save_pretrained(output_path)
    print(f"[FP8] ✓ Saved FP8 Static model to {output_path}")

    del model
    cleanup_gpu()
    return output_path


# ════════════════════════════════════════════════════════════
#  Main Pipeline
# ════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Aggressive Pruning + KD + FP8 Pipeline")

    # Pruning options
    parser.add_argument("--keep-layers", type=int, default=22,
                       help="Number of layers to keep (default: 22 from 30)")
    parser.add_argument("--ffn-size", type=int, default=None,
                       help="Target FFN intermediate size (default: no FFN pruning)")

    # KD training options
    parser.add_argument("--num-samples", type=int, default=20000,
                       help="Number of training samples from MANTA-1M")
    parser.add_argument("--max-seq-length", type=int, default=1024,
                       help="Max sequence length for training")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size per device")
    parser.add_argument("--grad-accum", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=2.0,
                       help="KD temperature")
    parser.add_argument("--alpha", type=float, default=0.7,
                       help="KD loss weight (higher = more KD, less CE)")

    # FP8 options
    parser.add_argument("--apply-fp8", action="store_true",
                       help="Apply FP8 Static quantization after KD")

    # Output
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory for the model")
    parser.add_argument("--config", default="configs/lanes.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-kd", action="store_true",
                       help="Skip KD, only do pruning + FP8")

    args = parser.parse_args()
    cfg = load_config(args.config)
    base_path = get_base_model_path(cfg)
    seed = cfg["project"]["seed"]

    output_dir = args.output if os.path.isabs(args.output) else os.path.join(ROOT, args.output)
    fp8_output = output_dir + "_fp8static"

    # Check existing
    final_output = fp8_output if args.apply_fp8 else output_dir
    if not args.force and os.path.exists(os.path.join(final_output, "model.safetensors")):
        print(f"[SKIP] Model already exists: {final_output}")
        return

    torch.manual_seed(seed)

    print("=" * 70)
    print(f"  Aggressive Pruning + KD Pipeline")
    print(f"  Base: {base_path}")
    print(f"  Target: {args.keep_layers}L" +
          (f" + FFN {args.ffn_size}" if args.ffn_size else "") +
          (" + FP8 Static" if args.apply_fp8 else ""))
    print("=" * 70)

    # ── Load tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # ── Step 1: BI Analysis ──
    print("\n[STEP 1] Layer Importance Analysis (BI Scoring)")
    model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    device = next(model.parameters()).device
    num_total = model.config.num_hidden_layers
    num_remove = num_total - args.keep_layers

    if num_remove > 0:
        ranked = compute_bi_scores(model, tokenizer, device, num_samples=200)
        layers_to_remove = set(ranked[:num_remove])
    else:
        layers_to_remove = set()

    # ── Step 2: Prune Layers ──
    if layers_to_remove:
        print(f"\n[STEP 2] Layer Pruning: {num_total}L → {args.keep_layers}L")
        model = prune_layers(model, layers_to_remove)

    # ── Step 3: FFN Width Pruning ──
    if args.ffn_size:
        print(f"\n[STEP 3] FFN Width Pruning: {model.config.intermediate_size} → {args.ffn_size}")
        model = prune_ffn_width(model, args.ffn_size, device)

    # Save pruned model temporarily
    pruned_dir = output_dir + "_pruned"
    os.makedirs(pruned_dir, exist_ok=True)
    model.save_pretrained(pruned_dir)
    tokenizer.save_pretrained(pruned_dir)
    print(f"[SAVE] Pruned model saved to {pruned_dir}")

    del model
    cleanup_gpu()

    # ── Step 4: Knowledge Distillation ──
    if not args.skip_kd:
        print(f"\n[STEP 4] Knowledge Distillation")

        # Load teacher
        print("[KD] Loading teacher model (original, GPU)...")
        teacher = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        )
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        # Load student (pruned model)
        print("[KD] Loading student model (pruned, GPU)...")
        student = AutoModelForCausalLM.from_pretrained(
            pruned_dir, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        )

        # Load training data
        train_dataset = load_training_data(
            tokenizer, args.num_samples, args.max_seq_length, seed
        )

        # Run KD
        run_kd(teacher, student, tokenizer, train_dataset, output_dir, args)

        del teacher, student
        cleanup_gpu()
    else:
        # No KD, just use pruned model as output
        print("[SKIP] Skipping KD training")
        if pruned_dir != output_dir:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            shutil.copytree(pruned_dir, output_dir)

    # Clean up pruned dir
    if os.path.exists(pruned_dir) and pruned_dir != output_dir:
        shutil.rmtree(pruned_dir)

    # ── Step 5: FP8 Static Quantization ──
    if args.apply_fp8:
        print(f"\n[STEP 5] FP8 Static Quantization")
        apply_fp8_static(output_dir, fp8_output, tokenizer)
        print(f"\n✓ Final model: {fp8_output}")
    else:
        print(f"\n✓ Final model: {output_dir}")

    print("\n[DONE] Pipeline complete!")
    print(f"  Next: python scripts/04_eval_vllm.py --model {final_output} --lane-id <name>")


if __name__ == "__main__":
    main()
