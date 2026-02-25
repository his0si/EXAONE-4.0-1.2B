#!/usr/bin/env python3
"""Smart Pruning: LaCo (Layer Collapse) + BlockPruner + KD + W8A16 quantization.

LaCo: Instead of deleting layers, MERGE adjacent layers to preserve information.
BlockPruner: Selectively remove MHA or MLP blocks (finer than full layer removal).

Pipeline: Analyze → Prune → KD → W8A16 Quantization → Evaluate

Usage:
  # LaCo: merge 2 most redundant layers (30→28)
  conda run -n quant python scripts/smart_prune.py --method laco --remove 2

  # BlockPruner: remove 4 least important blocks (MHA or MLP individually)
  conda run -n quant python scripts/smart_prune.py --method blockpruner --remove 4

  # Skip KD (prune + quantize only)
  conda run -n quant python scripts/smart_prune.py --method laco --remove 2 --skip-kd

  # Skip quantization (prune + KD only)
  conda run -n quant python scripts/smart_prune.py --method laco --remove 2 --skip-quant
"""
import os, sys, gc, json, argparse, time, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_MODEL = os.path.join(ROOT, "base_model")
BASE_PERF = 0.325
BASE_SPT = 0.000552


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_dir_size_mb(path):
    total = 0
    for f in Path(path).rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


# ════════════════════════════════════════════════════════════
#  Block Influence Analysis
# ════════════════════════════════════════════════════════════

def compute_block_scores(model, tokenizer, device, num_samples=200, max_len=512):
    """Compute importance scores for each MHA and MLP block separately.
    Returns dict with 'layer_N_mha' and 'layer_N_mlp' scores.
    Higher cosine similarity = more redundant = safer to remove/merge."""
    print(f"[ANALYSIS] Computing block-level importance ({num_samples} samples)...")

    data_5k = os.path.join(ROOT, "data", "manta", "train.json")
    if os.path.exists(data_5k):
        samples = json.load(open(data_5k))[:num_samples]
        texts = [tokenizer.apply_chat_template(
            s["conversations"], add_generation_prompt=False, tokenize=False) for s in samples]
    else:
        ds = load_dataset("LGAI-EXAONE/MANTA-1M", split=f"train[:{num_samples}]")
        texts = [tokenizer.apply_chat_template(
            r["conversations"], add_generation_prompt=False, tokenize=False) for r in ds]

    num_layers = model.config.num_hidden_layers

    # We need to hook into each layer's forward to capture:
    # - Input to MHA, output of MHA (after post_attention_layernorm + residual)
    # - Input to MLP, output of MLP (after post_feedforward_layernorm + residual)
    # EXAONE post-norm architecture:
    #   residual = hidden_states
    #   hidden_states = self_attn(hidden_states)
    #   hidden_states = post_attention_layernorm(hidden_states)
    #   hidden_states = residual + hidden_states  <-- MHA block output
    #   residual = hidden_states
    #   hidden_states = mlp(hidden_states)
    #   hidden_states = post_feedforward_layernorm(hidden_states)
    #   hidden_states = residual + hidden_states  <-- MLP block output

    layer_inputs = {}   # input to each full layer
    layer_mid = {}      # after MHA+residual (= input to MLP)
    layer_outputs = {}  # after MLP+residual

    def make_layer_hook(idx):
        def fn(module, inp, out):
            layer_inputs[idx] = inp[0].detach().float()
            layer_outputs[idx] = (out[0] if isinstance(out, tuple) else out).detach().float()
        return fn

    # Hook each MLP to capture mid-point
    def make_mlp_hook(idx):
        def fn(module, inp, out):
            # Input to MLP = hidden_states after MHA block
            layer_mid[idx] = inp[0].detach().float()
        return fn

    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(make_layer_hook(i)))
        hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(i)))

    mha_sims = torch.zeros(num_layers)
    mlp_sims = torch.zeros(num_layers)
    full_sims = torch.zeros(num_layers)
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
                sl = int(seq_len)
                inp = layer_inputs[li][0, :sl].mean(0)
                mid = layer_mid[li][0, :sl].mean(0)
                out = layer_outputs[li][0, :sl].mean(0)

                # MHA block: how much does input change after attention?
                mha_sims[li] += F.cosine_similarity(inp.unsqueeze(0), mid.unsqueeze(0)).item()
                # MLP block: how much does input change after MLP?
                mlp_sims[li] += F.cosine_similarity(mid.unsqueeze(0), out.unsqueeze(0)).item()
                # Full layer
                full_sims[li] += F.cosine_similarity(inp.unsqueeze(0), out.unsqueeze(0)).item()

            count += 1

    for h in hooks:
        h.remove()

    mha_sims /= max(count, 1)
    mlp_sims /= max(count, 1)
    full_sims /= max(count, 1)

    print(f"\n{'Layer':>6} {'MHA cos':>9} {'MLP cos':>9} {'Full cos':>9}")
    print("-" * 38)
    for i in range(num_layers):
        print(f"  {i:>3d}   {mha_sims[i]:.5f}   {mlp_sims[i]:.5f}   {full_sims[i]:.5f}")

    return mha_sims, mlp_sims, full_sims


# ════════════════════════════════════════════════════════════
#  LaCo: Layer Collapse (Merge instead of Delete)
# ════════════════════════════════════════════════════════════

def laco_merge(model, layers_to_merge):
    """LaCo: Merge redundant layers into adjacent layers instead of deleting.

    For each layer to remove, we merge its weight differentials into the next layer.
    This preserves the information that would be lost by simple deletion.

    layers_to_merge: list of layer indices to collapse into their neighbors.
    """
    print(f"\n[LaCo] Merging layers: {sorted(layers_to_merge)}")
    num_layers = model.config.num_hidden_layers

    with torch.no_grad():
        for rm_idx in sorted(layers_to_merge, reverse=True):
            # Find merge target: prefer next layer, fallback to previous
            if rm_idx < num_layers - 1:
                target_idx = rm_idx + 1
            else:
                target_idx = rm_idx - 1

            rm_layer = model.model.layers[rm_idx]
            tgt_layer = model.model.layers[target_idx]

            print(f"  Merging layer {rm_idx} → layer {target_idx}")

            # Merge strategy: average weights of removed layer into target
            # For each submodule, target_new = 0.5 * (target_old + removed)
            for (name_rm, param_rm), (name_tgt, param_tgt) in zip(
                rm_layer.named_parameters(), tgt_layer.named_parameters()
            ):
                param_tgt.data = 0.5 * (param_tgt.data + param_rm.data)

    # Now remove the merged layers (they've been folded into neighbors)
    keep = [i for i in range(num_layers) if i not in layers_to_merge]
    model.model.layers = nn.ModuleList([model.model.layers[i] for i in keep])
    model.config.num_hidden_layers = len(keep)

    # Fix layer_types to match new num_hidden_layers
    if hasattr(model.config, "layer_types") and model.config.layer_types is not None:
        model.config.layer_types = [model.config.layer_types[i] for i in keep]

    # Fix layer indices
    for new_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn"):
            layer.self_attn.layer_idx = new_idx

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[LaCo] Result: {model.config.num_hidden_layers}L, {total_params/1e6:.0f}M params")
    return model


# ════════════════════════════════════════════════════════════
#  BlockPruner: Selective MHA/MLP Removal
# ════════════════════════════════════════════════════════════

def block_prune(model, blocks_to_remove):
    """Remove specific MHA or MLP blocks by replacing with identity.

    blocks_to_remove: list of (layer_idx, 'mha'|'mlp')

    For EXAONE post-norm:
      MHA removal: skip self_attn + post_attention_layernorm (residual passes through)
      MLP removal: skip mlp + post_feedforward_layernorm (residual passes through)
    """
    print(f"\n[BlockPruner] Removing {len(blocks_to_remove)} blocks:")

    for layer_idx, block_type in blocks_to_remove:
        layer = model.model.layers[layer_idx]
        print(f"  Layer {layer_idx}: removing {block_type.upper()}")

        if block_type == 'mha':
            # Replace self_attn with identity (zero out all weights)
            for name, param in layer.self_attn.named_parameters():
                param.data.zero_()
            # Replace post_attention_layernorm with identity (weight=1)
            layer.post_attention_layernorm.weight.data.fill_(1.0)
        elif block_type == 'mlp':
            # Replace mlp with identity (zero out all weights)
            for name, param in layer.mlp.named_parameters():
                param.data.zero_()
            # Replace post_feedforward_layernorm with identity (weight=1)
            layer.post_feedforward_layernorm.weight.data.fill_(1.0)

    # Check if any layer has BOTH mha and mlp removed → remove entire layer
    removed_per_layer = {}
    for layer_idx, block_type in blocks_to_remove:
        removed_per_layer.setdefault(layer_idx, set()).add(block_type)

    full_removes = [idx for idx, types in removed_per_layer.items()
                    if types == {'mha', 'mlp'}]

    if full_removes:
        print(f"  Layers {full_removes} have both MHA+MLP removed → deleting entirely")
        keep = [i for i in range(model.config.num_hidden_layers) if i not in full_removes]
        model.model.layers = nn.ModuleList([model.model.layers[i] for i in keep])
        model.config.num_hidden_layers = len(keep)
        if hasattr(model.config, "layer_types") and model.config.layer_types is not None:
            model.config.layer_types = [model.config.layer_types[i] for i in keep]
        for new_idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "self_attn"):
                layer.self_attn.layer_idx = new_idx

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[BlockPruner] Result: {model.config.num_hidden_layers}L, {total_params/1e6:.0f}M params")
    return model


# ════════════════════════════════════════════════════════════
#  Knowledge Distillation
# ════════════════════════════════════════════════════════════

def load_training_data(tokenizer, num_samples, max_seq_length):
    data_50k = os.path.join(ROOT, "data", "manta", "train_50k.json")
    data_5k = os.path.join(ROOT, "data", "manta", "train.json")
    if os.path.exists(data_50k):
        samples = json.load(open(data_50k))[:num_samples]
    elif os.path.exists(data_5k):
        samples = json.load(open(data_5k))[:num_samples]
    else:
        ds = load_dataset("LGAI-EXAONE/MANTA-1M", split=f"train[:{num_samples}]")
        samples = [{"conversations": r["conversations"]} for r in ds]

    texts = [tokenizer.apply_chat_template(
        s["conversations"], add_generation_prompt=False, tokenize=False) for s in samples]

    input_ids_list, mask_list = [], []
    for text in texts:
        tok = tokenizer(text, truncation=True, max_length=max_seq_length,
                       padding="max_length", return_tensors="pt")
        input_ids_list.append(tok["input_ids"].squeeze(0))
        mask_list.append(tok["attention_mask"].squeeze(0))

    return TensorDataset(torch.stack(input_ids_list), torch.stack(mask_list))


def run_kd(teacher, student, tokenizer, output_dir,
           num_samples=5000, max_seq=512, batch_size=4, grad_accum=4,
           epochs=3, lr=2e-5, temperature=2.0, alpha=0.7):
    """Knowledge distillation from teacher to student."""
    device = next(student.parameters()).device

    print(f"\n[KD] Loading training data...")
    train_data = load_training_data(tokenizer, num_samples, max_seq)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    total_steps = (len(loader) // grad_accum) * epochs
    warmup = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    student.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[KD] Student: {student.config.num_hidden_layers}L, {trainable/1e6:.0f}M trainable params")
    print(f"[KD] Data: {len(train_data)} samples, batch={batch_size}x{grad_accum}, epochs={epochs}")
    print(f"[KD] LR={lr}, T={temperature}, alpha={alpha}, total_steps={total_steps}")

    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(epochs):
        epoch_loss = 0
        optimizer.zero_grad()

        for step, (ids, mask) in enumerate(loader):
            ids, mask = ids.to(device), mask.to(device)

            with torch.no_grad():
                t_logits = teacher(input_ids=ids, attention_mask=mask).logits.float()

            s_out = student(input_ids=ids, attention_mask=mask, labels=ids)
            s_logits = s_out.logits.float()

            kd_loss = F.kl_div(
                F.log_softmax(s_logits / temperature, dim=-1),
                F.softmax(t_logits / temperature, dim=-1),
                reduction="batchmean") * (temperature ** 2)

            loss = (alpha * kd_loss + (1 - alpha) * s_out.loss) / grad_accum
            loss.backward()
            epoch_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % (grad_accum * 50) == 0:
                avg = epoch_loss / (step + 1)
                elapsed = time.time() - t0
                print(f"  E{epoch+1} step {step+1}/{len(loader)} | loss={avg:.4f} | {elapsed/60:.1f}min")

        avg = epoch_loss / len(loader)
        print(f"  Epoch {epoch+1}/{epochs} avg_loss={avg:.4f} [{(time.time()-t0)/60:.1f}min]")

        if avg < best_loss:
            best_loss = avg
            student.gradient_checkpointing_disable()
            os.makedirs(output_dir, exist_ok=True)
            student.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            student.gradient_checkpointing_enable()
            print(f"  Saved best (loss={best_loss:.4f})")

    student.gradient_checkpointing_disable()
    return student


# ════════════════════════════════════════════════════════════
#  Quantization + Evaluation
# ════════════════════════════════════════════════════════════

def quantize_model(model_path, output_dir, tokenizer, scheme="W8A8"):
    """GPTQ quantization (W8A8 or W8A16)."""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from datasets import Dataset

    if os.path.isdir(output_dir) and os.path.exists(os.path.join(output_dir, "config.json")):
        print(f"[SKIP] {output_dir} exists")
        return output_dir

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)

    data_5k = os.path.join(ROOT, "data", "manta", "train.json")
    samples = json.load(open(data_5k))[:2048]
    texts = [tokenizer.apply_chat_template(
        s["conversations"], add_generation_prompt=True, tokenize=False) for s in samples]
    cal = Dataset.from_dict({"text": texts})

    recipe = [GPTQModifier(
        scheme=scheme, targets=["Linear"], ignore=["lm_head"], dampening_frac=0.02)]

    t0 = time.time()
    oneshot(model=model, dataset=cal, recipe=recipe,
            max_seq_length=1024, num_calibration_samples=2048)
    print(f"  {scheme} quantization done in {time.time()-t0:.1f}s")

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
        d = json.load(open(result_path))
        perf = d.get("perf_aggregate", 0) or 0
        spt = d["speed"]["sec_per_token"]
        pn = perf / BASE_PERF
        sn = 1 - (spt / BASE_SPT)
        sc = max(0.5 * pn + 0.5 * sn, 0)
        print(f"\n[RESULT] {name}: perf={perf:.4f}, PN={pn:.3f}, SN={sn:+.3f}, Score={sc:.4f}")
        for k, v in d["benchmarks"].items():
            print(f"  {k}: {v['score']}")
        return d
    else:
        print(f"[FAIL] {name}")
        return None


# ════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["laco", "blockpruner"],
                        help="Pruning method")
    parser.add_argument("--remove", type=int, default=2,
                        help="Number of layers (laco) or blocks (blockpruner) to remove")
    parser.add_argument("--skip-kd", action="store_true", help="Skip knowledge distillation")
    parser.add_argument("--skip-quant", action="store_true", help="Skip quantization")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--kd-samples", type=int, default=5000)
    parser.add_argument("--kd-epochs", type=int, default=3)
    parser.add_argument("--kd-lr", type=float, default=2e-5)
    parser.add_argument("--kd-batch", type=int, default=4)
    parser.add_argument("--kd-seq", type=int, default=512)
    parser.add_argument("--analysis-only", action="store_true", help="Only run block analysis")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Step 1: Analysis
    print("="*60)
    print(f"Smart Pruning: {args.method} (remove {args.remove})")
    print("="*60)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)

    mha_sims, mlp_sims, full_sims = compute_block_scores(
        model, tokenizer, device, num_samples=200, max_len=512)

    if args.analysis_only:
        del model
        return

    # Step 2: Pruning
    num_layers = model.config.num_hidden_layers

    if args.method == "laco":
        # Find most redundant layers (highest full cosine similarity)
        ranked = sorted(range(num_layers), key=lambda i: -full_sims[i].item())
        # Don't remove first or last 2 layers (critical)
        safe_ranked = [i for i in ranked if 2 <= i <= num_layers - 3]
        to_merge = safe_ranked[:args.remove]
        print(f"\n[LaCo] Most redundant layers: {safe_ranked[:8]}")
        print(f"[LaCo] Will merge: {sorted(to_merge)}")

        model = laco_merge(model, to_merge)
        method_name = f"laco_{num_layers - args.remove}L"

    elif args.method == "blockpruner":
        # Build list of all blocks with their redundancy scores
        blocks = []
        for i in range(num_layers):
            if 2 <= i <= num_layers - 3:  # protect first/last 2 layers
                blocks.append((mha_sims[i].item(), i, 'mha'))
                blocks.append((mlp_sims[i].item(), i, 'mlp'))

        # Sort by redundancy (highest cosine similarity = most redundant)
        blocks.sort(key=lambda x: -x[0])

        # Select blocks to remove, but don't remove both MHA+MLP from same layer
        to_remove = []
        removed_layers = {}
        for sim, layer_idx, block_type in blocks:
            if len(to_remove) >= args.remove:
                break
            # Don't remove both from same layer
            if layer_idx in removed_layers:
                continue
            to_remove.append((layer_idx, block_type))
            removed_layers[layer_idx] = block_type

        print(f"\n[BlockPruner] Most redundant blocks:")
        for sim, idx, bt in blocks[:10]:
            print(f"  Layer {idx} {bt.upper()}: cos_sim={sim:.5f}")
        print(f"\n[BlockPruner] Will remove: {to_remove}")

        model = block_prune(model, to_remove)
        method_name = f"blockpruner_{args.remove}blk"

    # Save pruned model
    pruned_dir = os.path.join(ROOT, "checkpoints", method_name)
    os.makedirs(pruned_dir, exist_ok=True)
    model.save_pretrained(pruned_dir)
    tokenizer.save_pretrained(pruned_dir)
    print(f"\n[SAVE] Pruned model: {pruned_dir} ({get_dir_size_mb(pruned_dir):.1f} MB)")

    # Step 3: Knowledge Distillation
    if not args.skip_kd:
        print("\n" + "="*60)
        print("Phase 2: Knowledge Distillation")
        print("="*60)

        # Load teacher
        teacher = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        kd_dir = os.path.join(ROOT, "checkpoints", f"{method_name}_kd")
        model = run_kd(teacher, model, tokenizer, kd_dir,
                       num_samples=args.kd_samples, max_seq=args.kd_seq,
                       batch_size=args.kd_batch, epochs=args.kd_epochs, lr=args.kd_lr)

        del teacher
        cleanup_gpu()
        source_for_quant = kd_dir
    else:
        source_for_quant = pruned_dir

    del model
    cleanup_gpu()

    # Step 4: W8A16 Quantization
    if not args.skip_quant:
        print("\n" + "="*60)
        print("Phase 3: W8A8 Quantization")
        print("="*60)

        quant_name = f"{method_name}_{'kd_' if not args.skip_kd else ''}w8a8"
        quant_dir = os.path.join(ROOT, "models", quant_name)
        quantize_model(source_for_quant, quant_dir, tokenizer, scheme="W8A8")
        eval_dir = quant_dir
        eval_name = quant_name
    else:
        eval_dir = source_for_quant
        eval_name = method_name + ("_kd" if not args.skip_kd else "")

    # Step 5: Evaluation
    if not args.skip_eval:
        print("\n" + "="*60)
        print("Phase 4: Evaluation")
        print("="*60)
        evaluate(eval_name, eval_dir)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
