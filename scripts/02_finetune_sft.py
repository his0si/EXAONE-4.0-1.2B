#!/usr/bin/env python3
"""
02_finetune_sft.py — SFT (LoRA), Post-quant SFT (norms-only),
                     and Knowledge Distillation for pruned models.

Usage:
  python scripts/02_finetune_sft.py --mode sft           # LoRA SFT on base FP16
  python scripts/02_finetune_sft.py --mode post_sft --input-model models/lane01_gptq_w4a16 --output checkpoints/lane08_postsft
  python scripts/02_finetune_sft.py --mode kd --keep-layers 28 --output checkpoints/lane10_pruned_kd
"""
import os, sys, gc, json, argparse, time
import yaml, torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_config(path="configs/lanes.yaml"):
    with open(os.path.join(ROOT, path)) as f:
        return yaml.safe_load(f)


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_base_model_path(cfg):
    return os.path.normpath(os.path.join(ROOT, cfg["project"]["base_model"]))


def load_manta_texts(tokenizer, n_samples, seed=42):
    """Load MANTA and return list of formatted texts."""
    data_path = os.path.join(ROOT, "data", "manta", "train.json")
    if os.path.exists(data_path):
        samples = json.load(open(data_path))[:n_samples]
    else:
        ds = load_dataset("LGAI-EXAONE/MANTA-1M", split=f"train[:{n_samples}]")
        samples = [{"conversations": row["conversations"]} for row in ds]

    texts = []
    for s in samples:
        text = tokenizer.apply_chat_template(
            s["conversations"], add_generation_prompt=False, tokenize=False
        )
        texts.append(text)
    return texts


# ════════════════════════════════════════════════════════════
#  Mode 1: LoRA SFT (pre-quantization, all model weights frozen except LoRA)
# ════════════════════════════════════════════════════════════
def run_lora_sft(cfg, output_dir, force=False):
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig

    base_path = get_base_model_path(cfg)
    sft_cfg = cfg["training"]["sft"]
    seed = cfg["project"]["seed"]

    merged_dir = os.path.join(ROOT, "checkpoints", "sft_merged")
    if not force and os.path.exists(os.path.join(merged_dir, "model.safetensors")):
        print(f"[SFT] Merged model already exists at {merged_dir}, skipping. (use --force to re-run)")
        return merged_dir

    print(f"[SFT] Loading base model from {base_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )

    # LoRA config
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=sft_cfg["lora_rank"],
        lora_alpha=sft_cfg["lora_alpha"],
        target_modules=sft_cfg["lora_target_modules"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Load data
    print("[SFT] Loading training data...")
    texts = load_manta_texts(tokenizer, sft_cfg["num_samples"], seed)
    ds = Dataset.from_dict({"text": texts})

    # Training
    adapter_dir = os.path.join(ROOT, "checkpoints", "sft_adapter")
    training_args = SFTConfig(
        output_dir=adapter_dir,
        num_train_epochs=sft_cfg["epochs"],
        per_device_train_batch_size=sft_cfg["batch_size"],
        gradient_accumulation_steps=sft_cfg["grad_accum"],
        learning_rate=sft_cfg["lr"],
        bf16=True,
        logging_steps=20,
        save_strategy="no",
        max_length=sft_cfg["max_seq_length"],
        dataset_text_field="text",
        warmup_ratio=sft_cfg["warmup_ratio"],
        weight_decay=sft_cfg["weight_decay"],
        lr_scheduler_type="cosine",
        seed=seed,
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print("[SFT] Starting LoRA training...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"[SFT] Training done in {elapsed:.0f}s")

    # Save adapter
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Merge LoRA → full model
    print("[SFT] Merging LoRA into base model...")
    model = model.merge_and_unload()
    os.makedirs(merged_dir, exist_ok=True)
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"[SFT] Merged model saved to {merged_dir}")

    del model, trainer
    cleanup_gpu()
    return merged_dir


# ════════════════════════════════════════════════════════════
#  Mode 2: Post-quantization SFT (train norms + embeddings only)
# ════════════════════════════════════════════════════════════
def run_post_sft(cfg, input_model_dir, output_dir, force=False):
    from trl import SFTTrainer, SFTConfig

    base_path = get_base_model_path(cfg)
    ps_cfg = cfg["training"]["post_sft"]
    seed = cfg["project"]["seed"]

    if not force and os.path.exists(os.path.join(output_dir, "model.safetensors")):
        print(f"[Post-SFT] Model already exists at {output_dir}, skipping. (use --force to re-run)")
        return output_dir

    print(f"[Post-SFT] Loading quantized model from {input_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        input_model_dir, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )

    # Freeze all, then unfreeze norms + embeddings
    for param in model.parameters():
        param.requires_grad = False
    trainable_count = 0
    for name, param in model.named_parameters():
        if any(k in name for k in ["layernorm", "norm", "embed_tokens", "lm_head"]):
            param.requires_grad = True
            trainable_count += param.numel()
    total_count = sum(p.numel() for p in model.parameters())
    print(f"[Post-SFT] Trainable: {trainable_count:,} / {total_count:,} "
          f"({trainable_count/total_count*100:.1f}%)")

    # Load data
    texts = load_manta_texts(tokenizer, ps_cfg["num_samples"], seed)
    ds = Dataset.from_dict({"text": texts})

    training_args = SFTConfig(
        output_dir=output_dir + "_ckpt",
        num_train_epochs=ps_cfg["epochs"],
        per_device_train_batch_size=ps_cfg["batch_size"],
        gradient_accumulation_steps=ps_cfg["grad_accum"],
        learning_rate=ps_cfg["lr"],
        bf16=True,
        logging_steps=20,
        save_strategy="no",
        max_length=ps_cfg["max_seq_length"],
        dataset_text_field="text",
        warmup_ratio=ps_cfg["warmup_ratio"],
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print("[Post-SFT] Starting training (norms + embeddings)...")
    t0 = time.time()
    trainer.train()
    print(f"[Post-SFT] Done in {time.time()-t0:.0f}s")

    # Save temporary SFT output
    sft_temp = output_dir + "_sft_temp"
    os.makedirs(sft_temp, exist_ok=True)
    model.save_pretrained(sft_temp, save_compressed=True)

    # Fix: merge trained weights back into original compressed format
    # (SFT trainer may save both packed + unpacked tensors; this fix
    #  starts from the original compressed model and only replaces
    #  the trained norm/embedding tensors.)
    print("[Post-SFT] Fixing compressed format...")
    input_abs = os.path.join(ROOT, input_model_dir) if not os.path.isabs(input_model_dir) else input_model_dir
    _fix_post_sft_model(input_abs, sft_temp, output_dir)
    tokenizer.save_pretrained(output_dir)

    # Cleanup temp
    import shutil
    shutil.rmtree(sft_temp, ignore_errors=True)
    shutil.rmtree(output_dir + "_ckpt", ignore_errors=True)

    print(f"[Post-SFT] Saved to {output_dir}")

    del model, trainer
    cleanup_gpu()
    return output_dir


def _fix_post_sft_model(original_dir, sft_dir, output_dir):
    """Copy original compressed model, then replace only trained tensors
    (norms, embeddings) from the SFT output. This preserves the exact
    quantized/packed tensor format."""
    import shutil
    from safetensors.torch import load_file, save_file

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    shutil.copytree(original_dir, output_dir)

    orig_tensors = load_file(os.path.join(original_dir, "model.safetensors"))
    sft_tensors = load_file(os.path.join(sft_dir, "model.safetensors"))

    updated = 0
    for k in sft_tensors:
        if any(pat in k for pat in ["layernorm", "norm", "embed_tokens", "lm_head"]):
            if not any(suf in k for suf in ["_packed", "_scale", "_shape", "_zero_point"]):
                if k in orig_tensors:
                    orig_tensors[k] = sft_tensors[k]
                    updated += 1

    save_file(orig_tensors, os.path.join(output_dir, "model.safetensors"))
    print(f"[Fix] Updated {updated} trained tensors in compressed model")


# ════════════════════════════════════════════════════════════
#  Mode 3: Layer Pruning + Knowledge Distillation (pre-quant)
# ════════════════════════════════════════════════════════════
def compute_bi_scores(model, tokenizer, device, num_samples=64):
    """Return layer indices sorted by importance (least important first)."""
    ds = load_dataset("LGAI-EXAONE/MANTA-1M", split=f"train[:{num_samples}]")
    def preprocess(ex):
        text = tokenizer.apply_chat_template(
            ex["conversations"], add_generation_prompt=True, tokenize=False)
        tokens = tokenizer(text, truncation=True, max_length=256,
                           padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in tokens.items()}
    ds = ds.map(preprocess, remove_columns=ds.column_names)
    ds.set_format("torch")

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
        for idx in range(len(ds)):
            sample = ds[idx]
            ids = sample["input_ids"].unsqueeze(0).to(device)
            mask = sample["attention_mask"].unsqueeze(0).to(device)
            seq_len = mask.sum().item()
            if seq_len < 2:
                continue
            model(input_ids=ids, attention_mask=mask)
            for li in range(num_layers):
                i_mean = layer_inputs[li][0, :int(seq_len)].mean(0)
                o_mean = layer_outputs[li][0, :int(seq_len)].mean(0)
                cos_sims[li] += F.cosine_similarity(
                    i_mean.unsqueeze(0), o_mean.unsqueeze(0)).item()
            count += 1

    for h in hooks:
        h.remove()
    cos_sims /= count
    ranked = sorted(range(num_layers), key=lambda i: (1 - cos_sims[i]).item())
    return ranked


def prune_model(model, layers_to_remove):
    """Remove layers and re-index attention."""
    keep = [i for i in range(model.config.num_hidden_layers) if i not in layers_to_remove]
    model.model.layers = nn.ModuleList([model.model.layers[i] for i in keep])
    model.config.num_hidden_layers = len(keep)
    if hasattr(model.config, "layer_types") and model.config.layer_types:
        model.config.layer_types = [model.config.layer_types[i] for i in keep]
    for new_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn"):
            layer.self_attn.layer_idx = new_idx
    return model


def run_kd(cfg, keep_layers, output_dir, force=False):
    """Full pipeline: BI analysis → prune → KD → save FP16."""
    base_path = get_base_model_path(cfg)
    kd_cfg = cfg["training"]["kd"]
    seed = cfg["project"]["seed"]

    if not force and os.path.exists(os.path.join(output_dir, "model.safetensors")):
        print(f"[KD] Model already exists at {output_dir}, skipping. (use --force to re-run)")
        return output_dir

    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model for BI analysis
    print("[KD] Loading base model for BI analysis...")
    model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    device = next(model.parameters()).device
    num_total = model.config.num_hidden_layers
    num_remove = num_total - keep_layers

    ranked = compute_bi_scores(model, tokenizer, device)
    layers_to_remove = set(ranked[:num_remove])
    print(f"[KD] Removing layers: {sorted(layers_to_remove)}")

    model = prune_model(model, layers_to_remove)

    pruned_dir = output_dir + "_pruned"
    os.makedirs(pruned_dir, exist_ok=True)
    model.save_pretrained(pruned_dir)
    tokenizer.save_pretrained(pruned_dir)

    del model
    cleanup_gpu()

    # Load teacher and student — both on GPU (24 GiB RTX 4090 충분)
    print("[KD] Loading teacher (GPU) and student (GPU)...")
    teacher = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = AutoModelForCausalLM.from_pretrained(
        pruned_dir, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    device = next(student.parameters()).device

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    print(f"[KD] Student: {student.config.num_hidden_layers}L, "
          f"trainable {trainable:,}/{total:,}")

    # Prepare data
    print(f"[KD] Tokenizing {kd_cfg['num_samples']} samples...")
    texts = load_manta_texts(tokenizer, kd_cfg["num_samples"], seed)
    input_ids_list, mask_list = [], []
    for text in texts:
        tok = tokenizer(text, truncation=True, max_length=kd_cfg["max_seq_length"],
                        padding="max_length", return_tensors="pt")
        input_ids_list.append(tok["input_ids"].squeeze(0))
        mask_list.append(tok["attention_mask"].squeeze(0))

    ds = TensorDataset(torch.stack(input_ids_list), torch.stack(mask_list))
    loader = DataLoader(ds, batch_size=kd_cfg["batch_size"], shuffle=True)

    optimizer = torch.optim.AdamW(student.parameters(), lr=kd_cfg["lr"], weight_decay=0.01)
    total_steps = (len(loader) // kd_cfg["grad_accum"]) * kd_cfg["epochs"]
    warmup = int(total_steps * kd_cfg["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    student.gradient_checkpointing_enable()
    T = kd_cfg["temperature"]
    alpha = kd_cfg["alpha"]

    print(f"[KD] Training {kd_cfg['epochs']} epochs, {total_steps} steps...")
    best_loss = float("inf")
    for epoch in range(kd_cfg["epochs"]):
        epoch_loss = 0
        optimizer.zero_grad()
        for step, (ids, mask) in enumerate(loader):
            ids, mask = ids.to(device), mask.to(device)
            with torch.no_grad():
                t_logits = teacher(input_ids=ids,
                                   attention_mask=mask).logits.float()
            s_out = student(input_ids=ids, attention_mask=mask, labels=ids)
            s_logits = s_out.logits.float()

            kd_loss = F.kl_div(
                F.log_softmax(s_logits / T, dim=-1),
                F.softmax(t_logits / T, dim=-1),
                reduction="batchmean") * (T ** 2)
            loss = (alpha * kd_loss + (1 - alpha) * s_out.loss) / kd_cfg["grad_accum"]
            loss.backward()
            epoch_loss += loss.item() * kd_cfg["grad_accum"]

            if (step + 1) % kd_cfg["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % 200 == 0:
                print(f"  Epoch {epoch+1} step {step+1}/{len(loader)} "
                      f"loss={epoch_loss/(step+1):.4f}")

        avg = epoch_loss / len(loader)
        print(f"  Epoch {epoch+1} avg_loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            student.gradient_checkpointing_disable()
            os.makedirs(output_dir, exist_ok=True)
            student.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            student.gradient_checkpointing_enable()
            print(f"  Saved best model (loss={best_loss:.4f})")

    student.gradient_checkpointing_disable()
    del teacher, student
    cleanup_gpu()
    return output_dir


# ════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["sft", "post_sft", "kd"])
    parser.add_argument("--config", default="configs/lanes.yaml")
    parser.add_argument("--input-model", type=str, default=None,
                        help="Path to input model (for post_sft)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--keep-layers", type=int, default=28,
                        help="Layers to keep after pruning (for kd mode)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing outputs")
    args = parser.parse_args()

    cfg = load_config(os.path.join(ROOT, args.config))

    if args.mode == "sft":
        out = args.output or os.path.join(ROOT, "checkpoints", "sft_merged")
        run_lora_sft(cfg, out, force=args.force)

    elif args.mode == "post_sft":
        if not args.input_model:
            print("ERROR: --input-model required for post_sft mode")
            sys.exit(1)
        inp = os.path.join(ROOT, args.input_model) if not os.path.isabs(args.input_model) else args.input_model
        out = args.output or inp + "_postsft"
        if not os.path.isabs(out):
            out = os.path.join(ROOT, out)
        run_post_sft(cfg, inp, out, force=args.force)

    elif args.mode == "kd":
        out = args.output or os.path.join(ROOT, "checkpoints",
                                           f"pruned{args.keep_layers}_kd")
        if not os.path.isabs(out):
            out = os.path.join(ROOT, out)
        run_kd(cfg, args.keep_layers, out, force=args.force)


if __name__ == "__main__":
    main()
