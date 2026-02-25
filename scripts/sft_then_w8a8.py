#!/usr/bin/env python3
"""
sft_then_w8a8.py — Two experiments to push W8A8 PerfNorm above 1.0:

  Experiment 1 (pre-SFT):  LoRA SFT on base FP16 → merge → W8A8 quantize
  Experiment 2 (post-SFT): Train norms/embeddings on existing W8A8 model

Usage:
  python scripts/sft_then_w8a8.py                    # Run all stages
  python scripts/sft_then_w8a8.py --stage sft         # Only LoRA SFT
  python scripts/sft_then_w8a8.py --stage quant       # Only W8A8 quantize (needs sft_merged)
  python scripts/sft_then_w8a8.py --stage post_sft    # Only post-SFT on existing W8A8
  python scripts/sft_then_w8a8.py --stage eval        # Only evaluate both models
"""
import os, sys, gc, json, argparse, time, shutil
import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Paths
SFT_MERGED_DIR = os.path.join(ROOT, "checkpoints", "sft_merged")
SFT_W8A8_DIR = os.path.join(ROOT, "models", "sft_w8a8")
EXISTING_W8A8_DIR = os.path.join(ROOT, "models", "w8a8_cal2048_d02")
POSTSFT_W8A8_DIR = os.path.join(ROOT, "models", "w8a8_postsft")
BASE_MODEL = os.path.join(ROOT, "base_model")

LOG = os.path.join(ROOT, "logs", "sft_then_w8a8.log")
os.makedirs(os.path.dirname(LOG), exist_ok=True)
log_f = open(LOG, "w", buffering=1)


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


sys.stdout = Tee(sys.stdout, log_f)
sys.stderr = Tee(sys.stderr, log_f)


def load_config():
    with open(os.path.join(ROOT, "configs", "lanes.yaml")) as f:
        return yaml.safe_load(f)


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_manta_texts(tokenizer, n_samples, seed=42):
    data_path = os.path.join(ROOT, "data", "manta", "train.json")
    if os.path.exists(data_path):
        samples = json.load(open(data_path))[:n_samples]
    else:
        from datasets import load_dataset
        ds = load_dataset("LGAI-EXAONE/MANTA-1M", split=f"train[:{n_samples}]")
        samples = [{"conversations": row["conversations"]} for row in ds]
    texts = []
    for s in samples:
        text = tokenizer.apply_chat_template(
            s["conversations"], add_generation_prompt=False, tokenize=False)
        texts.append(text)
    return texts


# ════════════════════════════════════════════════════════════
#  Stage 1: LoRA SFT on base FP16 model → merge
# ════════════════════════════════════════════════════════════
def run_lora_sft(cfg, force=False):
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import Dataset

    sft_cfg = cfg["training"]["sft"]
    seed = cfg["project"]["seed"]

    if not force and os.path.exists(os.path.join(SFT_MERGED_DIR, "model.safetensors")):
        print(f"[SFT] Merged model already exists at {SFT_MERGED_DIR}, skipping. (use --force)")
        return SFT_MERGED_DIR

    print(f"[SFT] Loading base model from {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)

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

    print("[SFT] Loading training data...")
    texts = load_manta_texts(tokenizer, sft_cfg["num_samples"], seed)
    ds = Dataset.from_dict({"text": texts})

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

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print("[SFT] Merging LoRA into base model...")
    model = model.merge_and_unload()
    os.makedirs(SFT_MERGED_DIR, exist_ok=True)
    model.save_pretrained(SFT_MERGED_DIR)
    tokenizer.save_pretrained(SFT_MERGED_DIR)
    print(f"[SFT] Merged model saved to {SFT_MERGED_DIR}")

    del model, trainer
    cleanup_gpu()
    return SFT_MERGED_DIR


# ════════════════════════════════════════════════════════════
#  Stage 2: W8A8 quantize the merged SFT model
# ════════════════════════════════════════════════════════════
def run_w8a8_quantize(input_dir, output_dir, force=False):
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import Dataset

    if not force and os.path.isdir(output_dir) and os.path.exists(
            os.path.join(output_dir, "config.json")):
        print(f"[QUANT] Model already exists at {output_dir}, skipping. (use --force)")
        return output_dir

    if not os.path.isdir(input_dir):
        print(f"[QUANT] ERROR: Input model not found at {input_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"[QUANT] W8A8 quantization: {input_dir} → {output_dir}")
    print(f"  damp=0.02, cal=2048, seq=1024")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(input_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        input_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)

    # Load calibration data
    data_50k = os.path.join(ROOT, "data", "manta", "train_50k.json")
    data_5k = os.path.join(ROOT, "data", "manta", "train.json")
    cal_samples = 2048
    if os.path.exists(data_50k):
        samples = json.load(open(data_50k))[:cal_samples]
    elif os.path.exists(data_5k):
        samples = json.load(open(data_5k))[:cal_samples]
    else:
        from datasets import load_dataset as ld
        ds = ld("LGAI-EXAONE/MANTA-1M", split=f"train[:{cal_samples}]")
        samples = [{"conversations": row["conversations"]} for row in ds]

    texts = []
    for s in samples:
        text = tokenizer.apply_chat_template(
            s["conversations"], add_generation_prompt=True, tokenize=False)
        texts.append(text)
    cal_ds = Dataset.from_dict({"text": texts})
    print(f"[QUANT] Loaded {len(cal_ds)} calibration samples")

    recipe = [
        GPTQModifier(
            scheme="W8A8",
            targets=["Linear"],
            ignore=["embed_tokens", "lm_head"],
            dampening_frac=0.02,
        ),
    ]

    t0 = time.time()
    oneshot(
        model=model,
        dataset=cal_ds,
        recipe=recipe,
        max_seq_length=1024,
        num_calibration_samples=cal_samples,
    )
    elapsed = time.time() - t0
    print(f"[QUANT] Quantization done in {elapsed:.1f}s")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)

    from pathlib import Path
    size_mb = sum(f.stat().st_size for f in Path(output_dir).rglob("*") if f.is_file()) / (1024*1024)
    print(f"[QUANT] Saved to {output_dir} ({size_mb:.1f} MB)")

    del model, cal_ds
    cleanup_gpu()
    return output_dir


# ════════════════════════════════════════════════════════════
#  Stage 3: Post-SFT on existing W8A8 (train norms + embeddings)
# ════════════════════════════════════════════════════════════
def run_post_sft(cfg, input_dir, output_dir, force=False):
    from trl import SFTTrainer, SFTConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import Dataset

    ps_cfg = cfg["training"]["post_sft"]
    seed = cfg["project"]["seed"]

    if not force and os.path.exists(os.path.join(output_dir, "model.safetensors")):
        print(f"[Post-SFT] Model already exists at {output_dir}, skipping. (use --force)")
        return output_dir

    if not os.path.isdir(input_dir):
        print(f"[Post-SFT] ERROR: Input model not found at {input_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"[Post-SFT] Training norms/embeddings on {input_dir}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        input_dir, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)

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

    texts = load_manta_texts(tokenizer, ps_cfg["num_samples"], seed)
    ds = Dataset.from_dict({"text": texts})

    ckpt_dir = output_dir + "_ckpt"
    training_args = SFTConfig(
        output_dir=ckpt_dir,
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
    print("[Post-SFT] Fixing compressed format...")
    _fix_post_sft_model(input_dir, sft_temp, output_dir)
    tokenizer.save_pretrained(output_dir)

    # Cleanup temp
    shutil.rmtree(sft_temp, ignore_errors=True)
    shutil.rmtree(ckpt_dir, ignore_errors=True)

    print(f"[Post-SFT] Saved to {output_dir}")

    del model, trainer
    cleanup_gpu()
    return output_dir


def _fix_post_sft_model(original_dir, sft_dir, output_dir):
    """Copy original compressed model, then replace only trained tensors."""
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
#  Stage 4: Evaluate
# ════════════════════════════════════════════════════════════
def run_eval(model_dir, lane_id, force=False):
    import subprocess

    result_path = os.path.join(ROOT, "results", lane_id, "metrics.json")
    if not force and os.path.exists(result_path):
        print(f"[EVAL] Results already exist at {result_path}, skipping. (use --force)")
        with open(result_path) as f:
            return json.load(f)

    if not os.path.isdir(model_dir):
        print(f"[EVAL] ERROR: Model not found at {model_dir}")
        return None

    print(f"\n{'='*60}")
    print(f"[EVAL] Evaluating {model_dir} (lane={lane_id})")
    print(f"{'='*60}")

    eval_script = os.path.join(ROOT, "scripts", "04_eval_vllm.py")
    cmd = [sys.executable, eval_script,
           "--model", model_dir, "--lane-id", lane_id, "--force"]

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[EVAL] FAILED for {lane_id}")
        return None

    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)
    return None


def print_comparison(results, baseline_path):
    """Print comparison table of all results."""
    with open(baseline_path) as f:
        baseline = json.load(f)
    base_perf = baseline["perf_aggregate"]
    base_bench_spt = baseline["speed"]["bench_sec_per_token"]

    print(f"\n{'#'*60}")
    print(f"# Results Comparison")
    print(f"{'#'*60}")
    print(f"  Baseline: perf_aggregate={base_perf}, bench_spt={base_bench_spt:.6f}")
    print(f"  Current best: w8a8_cal2048_d02 → Competition=0.6208\n")

    print(f"  {'Model':<25} {'PerfAgg':>8} {'PerfNorm':>9} {'SpeedNorm':>10} {'Score':>7}")
    print(f"  {'-'*25} {'-'*8} {'-'*9} {'-'*10} {'-'*7}")

    # Reference
    ref_perf = 0.3233
    ref_pn = ref_perf / base_perf
    ref_sn = 1 - (0.000237 / base_bench_spt)
    ref_sc = max(0.5 * ref_pn + 0.5 * ref_sn, 0)
    print(f"  {'w8a8_cal2048_d02 (ref)':<25} {ref_perf:>8.4f} {ref_pn:>9.4f} {ref_sn:>10.4f} {ref_sc:>7.4f}")

    for name, r in results.items():
        if r is None:
            continue
        perf = r.get("perf_aggregate", 0)
        bench_spt = r.get("speed", {}).get("bench_sec_per_token", 0)
        if not perf or not bench_spt:
            print(f"  {name:<25} {'INCOMPLETE':>8}")
            continue
        perf_norm = perf / base_perf
        speed_norm = 1 - (bench_spt / base_bench_spt)
        score = max(0.5 * perf_norm + 0.5 * speed_norm, 0)
        marker = " ★" if score > ref_sc else ""
        print(f"  {name:<25} {perf:>8.4f} {perf_norm:>9.4f} {speed_norm:>10.4f} {score:>7.4f}{marker}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "sft", "quant", "post_sft", "eval"],
                        help="Which stage to run")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing outputs")
    args = parser.parse_args()

    cfg = load_config()
    stages = args.stage

    print(f"{'='*60}")
    print(f"SFT → W8A8 Pipeline")
    print(f"  Stage: {stages}")
    print(f"  Force: {args.force}")
    print(f"  Log: {LOG}")
    print(f"{'='*60}")

    # Stage 1: LoRA SFT
    if stages in ("all", "sft"):
        print(f"\n{'#'*60}")
        print(f"# Stage 1: LoRA SFT on base model")
        print(f"{'#'*60}")
        run_lora_sft(cfg, force=args.force)

    # Stage 2: W8A8 quantize the SFT-merged model
    if stages in ("all", "quant"):
        print(f"\n{'#'*60}")
        print(f"# Stage 2: W8A8 quantize SFT-merged model")
        print(f"{'#'*60}")
        run_w8a8_quantize(SFT_MERGED_DIR, SFT_W8A8_DIR, force=args.force)

    # Stage 3: Post-SFT on existing W8A8
    if stages in ("all", "post_sft"):
        print(f"\n{'#'*60}")
        print(f"# Stage 3: Post-SFT on existing W8A8")
        print(f"{'#'*60}")
        run_post_sft(cfg, EXISTING_W8A8_DIR, POSTSFT_W8A8_DIR, force=args.force)

    # Stage 4: Evaluate both
    if stages in ("all", "eval"):
        print(f"\n{'#'*60}")
        print(f"# Stage 4: Evaluation")
        print(f"{'#'*60}")

        results = {}

        if os.path.isdir(SFT_W8A8_DIR):
            r = run_eval(SFT_W8A8_DIR, "sft_w8a8", force=args.force)
            results["sft_w8a8"] = r
            cleanup_gpu()

        if os.path.isdir(POSTSFT_W8A8_DIR):
            r = run_eval(POSTSFT_W8A8_DIR, "w8a8_postsft", force=args.force)
            results["w8a8_postsft"] = r
            cleanup_gpu()

        # Print comparison
        baseline_path = os.path.join(ROOT, "results", "baseline.json")
        if os.path.exists(baseline_path):
            print_comparison(results, baseline_path)

    print(f"\n[DONE] Pipeline complete. Log: {LOG}")


if __name__ == "__main__":
    main()
