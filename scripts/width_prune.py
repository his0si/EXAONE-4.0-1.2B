#!/usr/bin/env python3
"""
Width Pruning: FFN intermediate dimension + optional attention head pruning.

Pipeline: Importance Analysis → Prune → KD → W8A8 Quantization → Evaluate

Methods:
  - FFN: GLU-aware neuron pair importance (gate+up weight magnitude + activation)
  - Head: Activation-weighted head importance (AMP-style)

Usage:
  python scripts/width_prune.py --ffn-ratio 0.25              # 25% FFN pruning only
  python scripts/width_prune.py --ffn-ratio 0.25 --head-ratio 0.125  # + head pruning
  python scripts/width_prune.py --ffn-ratio 0.25 --skip-kd    # skip KD
  python scripts/width_prune.py --ffn-ratio 0.25 --skip-quant # skip quantization
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


def load_calibration_data(tokenizer, num_samples=256, max_seq=512):
    """Load calibration data for importance estimation."""
    data_path = os.path.join(ROOT, "data", "manta", "train.json")
    samples = json.load(open(data_path))[:num_samples]
    texts = [tokenizer.apply_chat_template(
        s["conversations"], add_generation_prompt=True, tokenize=False
    ) for s in samples]

    encodings = []
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=max_seq)["input_ids"]
        encodings.append(ids)
    return encodings


# ════════════════════════════════════════════════════════════
#  Importance Analysis
# ════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_importance(model, calibration_data, device="cuda"):
    """
    Compute importance scores for FFN neurons and attention heads.

    FFN: For each neuron pair i in [0, intermediate_size):
      score = mean_over_samples(|gate_output[i]| * |up_output[i]|)
      This captures how much each neuron pair actually contributes.

    Heads: For each head h:
      score = mean_over_samples(||head_output_h||_1)
      via the output projection weights.
    """
    model.eval()
    model.to(device)

    num_layers = model.config.num_hidden_layers
    intermediate_size = model.config.intermediate_size
    num_heads = model.config.num_attention_heads
    head_dim = model.config.head_dim

    # Accumulators
    ffn_importance = torch.zeros(num_layers, intermediate_size, device=device)
    head_importance = torch.zeros(num_layers, num_heads, device=device)
    num_tokens_total = 0

    # Register hooks to capture activations
    gate_acts = {}
    up_acts = {}
    attn_acts = {}

    def make_ffn_hook(layer_idx, proj_type):
        def hook(module, input, output):
            if proj_type == "gate":
                gate_acts[layer_idx] = output.detach()
            else:
                up_acts[layer_idx] = output.detach()
        return hook

    def make_attn_hook(layer_idx):
        def hook(module, input, output):
            attn_acts[layer_idx] = input[0].detach()  # input to o_proj
        return hook

    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.gate_proj.register_forward_hook(make_ffn_hook(i, "gate")))
        hooks.append(layer.mlp.up_proj.register_forward_hook(make_ffn_hook(i, "up")))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(make_attn_hook(i)))

    print(f"[IMPORTANCE] Running {len(calibration_data)} calibration samples...")
    for idx, input_ids in enumerate(calibration_data):
        input_ids = input_ids.to(device)
        seq_len = input_ids.shape[1]
        num_tokens_total += seq_len

        model(input_ids)

        # Accumulate FFN importance: |SiLU(gate) * up| per neuron
        for i in range(num_layers):
            if i in gate_acts and i in up_acts:
                gate_out = gate_acts[i]  # [1, seq, intermediate]
                up_out = up_acts[i]
                # SwiGLU: output = SiLU(gate) * up
                neuron_contrib = torch.abs(F.silu(gate_out) * up_out)
                # Sum over batch and sequence dims
                ffn_importance[i] += neuron_contrib.sum(dim=(0, 1))

        # Accumulate head importance: ||head_output||_1
        for i in range(num_layers):
            if i in attn_acts:
                # attn input to o_proj: [1, seq, num_heads * head_dim]
                x = attn_acts[i]
                x = x.view(x.shape[0], x.shape[1], num_heads, head_dim)
                # L1 norm per head
                head_norms = x.abs().sum(dim=(0, 1, 3))  # [num_heads]
                head_importance[i] += head_norms

        gate_acts.clear()
        up_acts.clear()
        attn_acts.clear()

        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{len(calibration_data)} samples processed")

    for h in hooks:
        h.remove()

    # Normalize by total tokens
    ffn_importance /= num_tokens_total
    head_importance /= num_tokens_total

    return ffn_importance, head_importance


# ════════════════════════════════════════════════════════════
#  FFN Pruning
# ════════════════════════════════════════════════════════════

def prune_ffn(model, ffn_importance, target_intermediate, protect_layers=2):
    """
    Prune FFN intermediate dimension uniformly across all layers.

    For each layer:
    1. Rank neuron pairs by importance
    2. Keep top-k neurons (k = target_intermediate)
    3. Slice gate_proj, up_proj (rows) and down_proj (columns)
    """
    num_layers = model.config.num_hidden_layers
    orig_size = model.config.intermediate_size

    print(f"\n[FFN PRUNE] {orig_size} → {target_intermediate} "
          f"({100*(1-target_intermediate/orig_size):.1f}% reduction)")

    with torch.no_grad():
        for i in range(num_layers):
            layer = model.model.layers[i]
            mlp = layer.mlp

            # Get importance scores for this layer
            scores = ffn_importance[i]  # [intermediate_size]

            # Protect first/last layers
            if i < protect_layers or i >= num_layers - protect_layers:
                # Still prune but use global average importance to protect
                pass

            # Top-k indices (most important neurons to keep)
            _, keep_idx = torch.topk(scores, target_intermediate, sorted=True)
            keep_idx = keep_idx.sort().values  # sorted order for cleaner weights

            # Slice weights
            # gate_proj: [intermediate, hidden] → keep rows
            mlp.gate_proj.weight = nn.Parameter(
                mlp.gate_proj.weight.data[keep_idx, :].contiguous())
            # up_proj: [intermediate, hidden] → keep rows
            mlp.up_proj.weight = nn.Parameter(
                mlp.up_proj.weight.data[keep_idx, :].contiguous())
            # down_proj: [hidden, intermediate] → keep columns
            mlp.down_proj.weight = nn.Parameter(
                mlp.down_proj.weight.data[:, keep_idx].contiguous())

            # Update in_features/out_features
            mlp.gate_proj.out_features = target_intermediate
            mlp.up_proj.out_features = target_intermediate
            mlp.down_proj.in_features = target_intermediate

    # Update config
    model.config.intermediate_size = target_intermediate

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[FFN PRUNE] Done. Total params: {total_params/1e6:.0f}M")
    return model


# ════════════════════════════════════════════════════════════
#  Attention Head Pruning
# ════════════════════════════════════════════════════════════

def prune_heads(model, head_importance, num_heads_to_remove):
    """
    Prune attention heads uniformly across all layers.

    With GQA (4 query heads per KV head), must remove in groups of 4.
    """
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.head_dim
    heads_per_kv = num_heads // num_kv_heads  # 4 for EXAONE

    # Must remove in multiples of heads_per_kv
    groups_to_remove = num_heads_to_remove // heads_per_kv
    if groups_to_remove == 0:
        print("[HEAD PRUNE] Not enough heads to remove (need multiple of "
              f"{heads_per_kv}). Skipping.")
        return model

    actual_heads_removed = groups_to_remove * heads_per_kv
    new_num_heads = num_heads - actual_heads_removed
    new_num_kv_heads = num_kv_heads - groups_to_remove

    print(f"\n[HEAD PRUNE] heads: {num_heads}→{new_num_heads}, "
          f"kv_heads: {num_kv_heads}→{new_num_kv_heads}")

    with torch.no_grad():
        for i in range(num_layers):
            layer = model.model.layers[i]
            attn = layer.self_attn

            # Group importance by KV head groups
            scores = head_importance[i]  # [num_heads]
            group_scores = scores.view(num_kv_heads, heads_per_kv).sum(dim=1)

            # Keep top groups
            _, keep_groups = torch.topk(group_scores, new_num_kv_heads, sorted=True)
            keep_groups = keep_groups.sort().values

            # Expand to individual head indices
            keep_q_heads = []
            for g in keep_groups:
                for h in range(heads_per_kv):
                    keep_q_heads.append(g * heads_per_kv + h)
            keep_q_heads = torch.tensor(keep_q_heads, device=scores.device)
            keep_kv_heads = keep_groups

            # q_proj: [num_heads * head_dim, hidden] → keep rows
            q_indices = torch.cat([
                torch.arange(h * head_dim, (h + 1) * head_dim, device=scores.device)
                for h in keep_q_heads
            ])
            attn.q_proj.weight = nn.Parameter(
                attn.q_proj.weight.data[q_indices, :].contiguous())
            attn.q_proj.out_features = new_num_heads * head_dim

            # k_proj: [num_kv_heads * head_dim, hidden] → keep rows
            k_indices = torch.cat([
                torch.arange(h * head_dim, (h + 1) * head_dim, device=scores.device)
                for h in keep_kv_heads
            ])
            attn.k_proj.weight = nn.Parameter(
                attn.k_proj.weight.data[k_indices, :].contiguous())
            attn.k_proj.out_features = new_num_kv_heads * head_dim

            # v_proj: same as k_proj
            attn.v_proj.weight = nn.Parameter(
                attn.v_proj.weight.data[k_indices, :].contiguous())
            attn.v_proj.out_features = new_num_kv_heads * head_dim

            # o_proj: [hidden, num_heads * head_dim] → keep columns
            attn.o_proj.weight = nn.Parameter(
                attn.o_proj.weight.data[:, q_indices].contiguous())
            attn.o_proj.in_features = new_num_heads * head_dim

            # Update attention module attributes
            attn.num_heads = new_num_heads
            attn.num_key_value_heads = new_num_kv_heads

    # Update config
    model.config.num_attention_heads = new_num_heads
    model.config.num_key_value_heads = new_num_kv_heads

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[HEAD PRUNE] Done. Total params: {total_params/1e6:.0f}M")
    return model


# ════════════════════════════════════════════════════════════
#  Knowledge Distillation
# ════════════════════════════════════════════════════════════

def run_kd(teacher, student, tokenizer, output_dir,
           num_samples=5000, max_seq=512, batch_size=2,
           epochs=3, lr=2e-5, temperature=2.0, alpha=0.7):
    """KD: All parameters trainable (pruned model in FP16)."""
    device = next(teacher.parameters()).device

    if os.path.isdir(output_dir) and os.path.exists(
            os.path.join(output_dir, "config.json")):
        print(f"[KD SKIP] {output_dir} exists, loading...")
        student = AutoModelForCausalLM.from_pretrained(
            output_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)
        return student

    # Load data
    print("[KD] Loading training data...")
    data_path = os.path.join(ROOT, "data", "manta", "train.json")
    samples = json.load(open(data_path))[:num_samples]
    texts = [tokenizer.apply_chat_template(
        s["conversations"], add_generation_prompt=True, tokenize=False
    ) for s in samples]

    student.to(device)
    student.train()
    student.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[KD] Student: {student.config.num_hidden_layers}L, "
          f"inter={student.config.intermediate_size}, "
          f"heads={student.config.num_attention_heads}, "
          f"{trainable/1e6:.0f}M trainable params")

    # Tokenize
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
            padded = torch.full((len(batch_ids), max_len), tokenizer.pad_token_id or 0,
                               dtype=torch.long, device=device)
            mask = torch.zeros(len(batch_ids), max_len, dtype=torch.long, device=device)
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
    parser.add_argument("--ffn-ratio", type=float, default=0.25,
                       help="Fraction of FFN neurons to remove (0.25 = 25%%)")
    parser.add_argument("--ffn-target", type=int, default=None,
                       help="Explicit target intermediate_size (overrides --ffn-ratio)")
    parser.add_argument("--head-ratio", type=float, default=0.0,
                       help="Fraction of heads to remove (0 = no head pruning)")
    parser.add_argument("--cal-samples", type=int, default=256,
                       help="Calibration samples for importance estimation")
    parser.add_argument("--cal-seq", type=int, default=512,
                       help="Max sequence length for calibration")
    parser.add_argument("--kd-samples", type=int, default=5000)
    parser.add_argument("--kd-epochs", type=int, default=3)
    parser.add_argument("--kd-seq", type=int, default=512)
    parser.add_argument("--kd-batch", type=int, default=2)
    parser.add_argument("--kd-lr", type=float, default=2e-5)
    parser.add_argument("--skip-kd", action="store_true")
    parser.add_argument("--skip-quant", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Naming
    orig_intermediate = 4096
    if args.ffn_target:
        target_intermediate = args.ffn_target
    else:
        target_intermediate = int(orig_intermediate * (1 - args.ffn_ratio))
        # Round to multiple of 128 for GPU efficiency
        target_intermediate = (target_intermediate // 128) * 128

    name_parts = [f"wp_ffn{target_intermediate}"]
    if args.head_ratio > 0:
        num_heads_remove = int(32 * args.head_ratio)
        # Round to multiple of 4 (GQA group size)
        num_heads_remove = (num_heads_remove // 4) * 4
        new_heads = 32 - num_heads_remove
        name_parts.append(f"h{new_heads}")
    else:
        num_heads_remove = 0

    method_name = "_".join(name_parts)

    print("=" * 60)
    print(f"  Width Pruning: {method_name}")
    print(f"  FFN: {orig_intermediate} → {target_intermediate}")
    if num_heads_remove > 0:
        print(f"  Heads: 32 → {32 - num_heads_remove}")
    print("=" * 60)

    # Step 1: Load model and compute importance
    print("\n" + "=" * 60)
    print("Phase 1: Importance Analysis")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)

    cal_data = load_calibration_data(tokenizer, args.cal_samples, args.cal_seq)
    ffn_imp, head_imp = compute_importance(model, cal_data, device)

    # Print summary
    print("\n[IMPORTANCE SUMMARY]")
    print(f"  FFN neuron importance range: "
          f"[{ffn_imp.min():.6f}, {ffn_imp.max():.6f}]")
    print(f"  Head importance range: "
          f"[{head_imp.min():.6f}, {head_imp.max():.6f}]")

    # Step 2: Prune
    print("\n" + "=" * 60)
    print("Phase 2: Pruning")
    print("=" * 60)

    model = prune_ffn(model, ffn_imp, target_intermediate)

    if num_heads_remove > 0:
        model = prune_heads(model, head_imp, num_heads_remove)

    # Save pruned model
    pruned_dir = os.path.join(ROOT, "checkpoints", method_name)
    os.makedirs(pruned_dir, exist_ok=True)
    model.save_pretrained(pruned_dir)
    tokenizer.save_pretrained(pruned_dir)
    print(f"\n[SAVE] Pruned model: {pruned_dir} ({get_dir_size_mb(pruned_dir):.1f} MB)")

    # Step 3: KD
    if not args.skip_kd:
        print("\n" + "=" * 60)
        print("Phase 3: Knowledge Distillation")
        print("=" * 60)

        # Reload teacher
        cleanup_gpu()
        teacher = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        kd_dir = os.path.join(ROOT, "checkpoints", f"{method_name}_kd")
        model = run_kd(teacher, model, tokenizer, kd_dir,
                       num_samples=args.kd_samples, max_seq=args.kd_seq,
                       batch_size=args.kd_batch, epochs=args.kd_epochs,
                       lr=args.kd_lr)

        del teacher
        cleanup_gpu()
        source_for_quant = kd_dir
    else:
        source_for_quant = pruned_dir

    del model
    cleanup_gpu()

    # Step 4: W8A8 Quantization
    if not args.skip_quant:
        print("\n" + "=" * 60)
        print("Phase 4: W8A8 Quantization")
        print("=" * 60)

        quant_name = f"{method_name}_{'kd_' if not args.skip_kd else ''}w8a8"
        quant_dir = os.path.join(ROOT, "models", quant_name)
        quantize_w8a8(source_for_quant, quant_dir, tokenizer)
        eval_dir = quant_dir
        eval_name = quant_name
    else:
        eval_dir = source_for_quant
        eval_name = method_name + ("_kd" if not args.skip_kd else "")

    # Step 5: Evaluation
    if not args.skip_eval:
        print("\n" + "=" * 60)
        print("Phase 5: Evaluation")
        print("=" * 60)
        evaluate(eval_name, eval_dir)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
