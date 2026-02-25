#!/usr/bin/env python3
"""W8A8 score improvement experiments.

Exp1: Calibration data optimization (KMMLU/benchmark data instead of MANTA)
Exp2: 29L pruning + W8A8 (remove 1 most redundant layer)
Exp3: Quantize embed_tokens/lm_head too (ignore=[])
Exp4: More calibration samples (c4096, c8192)

Usage:
  conda run -n quant python scripts/w8a8_boost.py
  conda run -n quant python scripts/w8a8_boost.py --exp 1
  conda run -n quant python scripts/w8a8_boost.py --exp 2
  conda run -n quant python scripts/w8a8_boost.py --eval-only
"""
import os, sys, gc, time, json, subprocess, argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_MODEL = os.path.join(ROOT, "base_model")
LOG = os.path.join(ROOT, "logs", "w8a8_boost.log")

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

DAMPENING = 0.02
MAX_SEQ = 1024


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
#  Calibration Data Loaders
# ════════════════════════════════════════════════════════════

def load_manta_cal(tokenizer, num_samples, max_seq):
    """Standard MANTA calibration data."""
    data_50k = os.path.join(ROOT, "data", "manta", "train_50k.json")
    data_5k = os.path.join(ROOT, "data", "manta", "train.json")
    if os.path.exists(data_50k) and num_samples > 5000:
        samples = json.load(open(data_50k))[:num_samples]
    elif os.path.exists(data_5k):
        samples = json.load(open(data_5k))[:num_samples]
    else:
        ds = load_dataset("LGAI-EXAONE/MANTA-1M", split=f"train[:{num_samples}]")
        samples = [{"conversations": row["conversations"]} for row in ds]
    texts = []
    for s in samples:
        text = tokenizer.apply_chat_template(
            s["conversations"], add_generation_prompt=True, tokenize=False)
        texts.append(text)
    return Dataset.from_dict({"text": texts})


def load_kmmlu_cal(tokenizer, num_samples):
    """KMMLU-Pro + KMMLU-Redux as calibration data (MCQA format)."""
    texts = []

    # KMMLU-Pro
    ds_pro = load_dataset("LGAI-EXAONE/KMMLU-Pro", split="test")
    for row in ds_pro:
        q = row["question"]
        opts = row["options"]
        opts_text = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(opts))
        # Format as chat
        msg = [{"role": "user", "content": f"다음 문제의 정답을 고르세요.\n\n{q}\n\n{opts_text}"}]
        text = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        texts.append(text)
        if len(texts) >= num_samples // 2:
            break

    # KMMLU-Redux
    ds_redux = load_dataset("LGAI-EXAONE/KMMLU-Redux", split="test")
    for row in ds_redux:
        q = row.get("question", "")
        opts = row.get("options", [])
        if not q or not opts:
            continue
        opts_text = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(opts))
        msg = [{"role": "user", "content": f"다음 문제의 정답을 고르세요.\n\n{q}\n\n{opts_text}"}]
        text = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        texts.append(text)
        if len(texts) >= num_samples:
            break

    print(f"[CAL] Loaded {len(texts)} KMMLU calibration samples")
    return Dataset.from_dict({"text": texts})


def load_mixed_cal(tokenizer, num_samples):
    """Mixed: KMMLU + Ko-LongRAG + MANTA for diverse calibration."""
    texts = []
    per_source = num_samples // 3

    # KMMLU-Pro
    ds_pro = load_dataset("LGAI-EXAONE/KMMLU-Pro", split="test")
    for row in ds_pro:
        q = row["question"]
        opts = row["options"]
        opts_text = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(opts))
        msg = [{"role": "user", "content": f"다음 문제의 정답을 고르세요.\n\n{q}\n\n{opts_text}"}]
        text = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        texts.append(text)
        if len(texts) >= per_source:
            break

    # Ko-LongRAG (truncate context to max_seq-friendly length)
    ds_lr = load_dataset("LGAI-EXAONE/Ko-LongRAG", split="test")
    for row in ds_lr:
        ctx = row.get("context", "")[:2000]  # truncate for calibration
        q = row.get("question", "")
        msg = [{"role": "user", "content": f"다음 문서를 읽고 질문에 답하세요.\n\n{ctx}\n\n질문: {q}"}]
        text = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        texts.append(text)
        if len(texts) >= per_source * 2:
            break

    # MANTA (fill remaining)
    remaining = num_samples - len(texts)
    if remaining > 0:
        data_5k = os.path.join(ROOT, "data", "manta", "train.json")
        if os.path.exists(data_5k):
            manta = json.load(open(data_5k))[:remaining]
        else:
            ds_m = load_dataset("LGAI-EXAONE/MANTA-1M", split=f"train[:{remaining}]")
            manta = [{"conversations": r["conversations"]} for r in ds_m]
        for s in manta:
            text = tokenizer.apply_chat_template(
                s["conversations"], add_generation_prompt=True, tokenize=False)
            texts.append(text)

    print(f"[CAL] Loaded {len(texts)} mixed calibration samples")
    return Dataset.from_dict({"text": texts})


# ════════════════════════════════════════════════════════════
#  Quantization
# ════════════════════════════════════════════════════════════

def quantize_w8a8(model_path, output_dir, tokenizer, cal_dataset, cal_samples,
                  max_seq=1024, dampening=0.02, ignore=None):
    """Generic W8A8 quantization."""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    if os.path.isdir(output_dir) and os.path.exists(os.path.join(output_dir, "config.json")):
        print(f"  [SKIP] Already exists: {output_dir}")
        return output_dir

    if ignore is None:
        ignore = ["embed_tokens", "lm_head"]

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    print(f"  Model: {model.config.num_hidden_layers} layers")

    recipe = [
        GPTQModifier(
            scheme="W8A8",
            targets=["Linear"],
            ignore=ignore,
            dampening_frac=dampening,
        ),
    ]

    t0 = time.time()
    oneshot(
        model=model,
        dataset=cal_dataset,
        recipe=recipe,
        max_seq_length=max_seq,
        num_calibration_samples=cal_samples,
    )
    elapsed = time.time() - t0
    print(f"  Quantization done in {elapsed:.1f}s")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)
    size_mb = get_dir_size_mb(output_dir)
    print(f"  Saved: {output_dir} ({size_mb:.1f} MB)")

    del model
    cleanup_gpu()
    return output_dir


# ════════════════════════════════════════════════════════════
#  Exp2: 29L Pruning
# ════════════════════════════════════════════════════════════

def compute_bi_scores(model, tokenizer, device, num_samples=64):
    """Return layer indices sorted by redundancy (most redundant first)."""
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

    # Most redundant = highest cosine similarity (input ≈ output)
    ranked = sorted(range(num_layers), key=lambda i: -cos_sims[i].item())
    print(f"  BI scores (top-5 most redundant):")
    for i in ranked[:5]:
        print(f"    Layer {i}: cos_sim={cos_sims[i].item():.4f}")
    return ranked


def prune_and_save(keep_layers, output_dir):
    """Prune model to keep_layers and save FP16."""
    if os.path.isdir(output_dir) and os.path.exists(os.path.join(output_dir, "config.json")):
        print(f"  [SKIP] Pruned model exists: {output_dir}")
        return output_dir

    import torch.nn as nn
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    device = next(model.parameters()).device

    num_total = model.config.num_hidden_layers
    num_remove = num_total - keep_layers
    print(f"  Pruning {num_total} → {keep_layers} layers (removing {num_remove})")

    ranked = compute_bi_scores(model, tokenizer, device)
    layers_to_remove = set(ranked[:num_remove])
    print(f"  Removing layers: {sorted(layers_to_remove)}")

    # Prune
    keep = [i for i in range(num_total) if i not in layers_to_remove]
    model.model.layers = nn.ModuleList([model.model.layers[i] for i in keep])
    model.config.num_hidden_layers = len(keep)
    if hasattr(model.config, "layer_types") and model.config.layer_types:
        model.config.layer_types = [model.config.layer_types[i] for i in keep]
    for new_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn"):
            layer.self_attn.layer_idx = new_idx

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  Saved pruned model: {output_dir}")

    del model
    cleanup_gpu()
    return output_dir


# ════════════════════════════════════════════════════════════
#  Evaluation
# ════════════════════════════════════════════════════════════

def evaluate(name, model_dir):
    result_path = os.path.join(ROOT, "results", name, "metrics.json")
    if os.path.exists(result_path):
        print(f"  [SKIP] {name} already evaluated")
        with open(result_path) as f:
            return json.load(f)

    print(f"  [EVAL] {name}...")
    eval_script = os.path.join(ROOT, "scripts", "04_eval_vllm.py")
    cmd = [sys.executable, eval_script,
           "--model", model_dir, "--lane-id", name, "--force"]
    r = subprocess.run(cmd, capture_output=False, text=True)
    if r.returncode != 0:
        print(f"  [FAIL] {name}")
        return None
    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)
    return None


# ════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, nargs="+", help="Run specific experiments (1-4)")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    run_exps = args.exp or [1, 2, 3, 4]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Define all experiments
    experiments = []  # (name, exp_id, model_path, cal_loader, cal_samples, max_seq, ignore)

    if 1 in run_exps:
        experiments.append(("boost_kmmlu_cal", 1, BASE_MODEL, "kmmlu", 2048, 1024, ["embed_tokens", "lm_head"]))
        experiments.append(("boost_mixed_cal", 1, BASE_MODEL, "mixed", 2048, 1024, ["embed_tokens", "lm_head"]))

    if 3 in run_exps:
        experiments.append(("boost_embed_quant", 3, BASE_MODEL, "manta", 2048, 1024, []))

    if 4 in run_exps:
        experiments.append(("boost_cal4096", 4, BASE_MODEL, "manta", 4096, 1024, ["embed_tokens", "lm_head"]))
        experiments.append(("boost_cal8192", 4, BASE_MODEL, "manta", 8192, 1024, ["embed_tokens", "lm_head"]))

    # Exp2: pruning (separate flow)
    if 2 in run_exps:
        experiments.append(("boost_29L_w8a8", 2, None, "manta", 2048, 1024, ["embed_tokens", "lm_head"]))

    print(f"{'='*60}")
    print(f"W8A8 Boost Experiments")
    print(f"  Experiments: {[n for n,_,_,_,_,_,_ in experiments]}")
    print(f"{'='*60}")

    # Phase 1: Quantize
    if not args.eval_only:
        print(f"\n{'#'*60}")
        print(f"# Phase 1: Quantization")
        print(f"{'#'*60}")

        for name, exp_id, model_path, cal_type, cal_samples, max_seq, ignore in experiments:
            output_dir = os.path.join(ROOT, "models", name)
            print(f"\n{'='*60}")
            print(f"[{name}] exp={exp_id}, cal={cal_type}, n={cal_samples}, ignore={ignore}")
            print(f"{'='*60}")

            try:
                if exp_id == 2:
                    # Pruning flow: prune first, then quantize
                    pruned_dir = os.path.join(ROOT, "checkpoints", "pruned29")
                    prune_and_save(29, pruned_dir)
                    cal_ds = load_manta_cal(tokenizer, cal_samples, max_seq)
                    quantize_w8a8(pruned_dir, output_dir, tokenizer, cal_ds,
                                  cal_samples, max_seq, DAMPENING, ignore)
                else:
                    # Standard quantize flow
                    if cal_type == "kmmlu":
                        cal_ds = load_kmmlu_cal(tokenizer, cal_samples)
                    elif cal_type == "mixed":
                        cal_ds = load_mixed_cal(tokenizer, cal_samples)
                    else:
                        cal_ds = load_manta_cal(tokenizer, cal_samples, max_seq)
                    quantize_w8a8(model_path, output_dir, tokenizer, cal_ds,
                                  cal_samples, max_seq, DAMPENING, ignore)
            except Exception as e:
                print(f"  [ERROR] {name}: {e}")
                import traceback
                traceback.print_exc()
                cleanup_gpu()

    # Phase 2: Evaluate
    print(f"\n{'#'*60}")
    print(f"# Phase 2: Evaluation")
    print(f"{'#'*60}")

    results = {}
    for name, exp_id, _, _, _, _, _ in experiments:
        model_dir = os.path.join(ROOT, "models", name)
        if not os.path.isdir(model_dir):
            print(f"  [SKIP] {name} not found")
            continue
        try:
            r = evaluate(name, model_dir)
            if r:
                results[name] = r
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    # Phase 3: Results
    print(f"\n{'#'*60}")
    print(f"# Phase 3: Results")
    print(f"{'#'*60}")

    bl = json.load(open(os.path.join(ROOT, "results", "baseline.json")))
    base_perf = bl["perf_aggregate"]
    base_spt = bl["speed"]["sec_per_token"]
    base_bspt = bl["speed"]["bench_sec_per_token"]

    print(f"\nBaseline: perf={base_perf}, spt={base_spt:.6f}, bench_spt={base_bspt:.6f}")
    print(f"Reference: w8a8_cal2048_d02 → Server 0.6208\n")

    header = f"{'Model':<22} {'Perf':>6} {'PN':>5} │ {'SN_spd':>7} {'Sc_spd':>7} │ {'SN_ben':>7} {'Sc_ben':>7}"
    print(header)
    print("-" * len(header))

    # Reference
    ref_path = os.path.join(ROOT, "results", "w8a8_cal2048_d02", "metrics.json")
    if os.path.exists(ref_path):
        r = json.load(open(ref_path))
        perf = r["perf_aggregate"]
        spt = r["speed"]["sec_per_token"]
        bspt = r["speed"]["bench_sec_per_token"]
        pn = perf / base_perf
        sn1 = 1 - spt/base_spt
        sn2 = 1 - bspt/base_bspt
        print(f"{'w8a8_cal2048_d02 (ref)':<22} {perf:>6.4f} {pn:>5.3f} │ {sn1:>+7.3f} {max(0.5*pn+0.5*sn1,0):>7.4f} │ {sn2:>+7.3f} {max(0.5*pn+0.5*sn2,0):>7.4f}")

    print()
    for name, exp_id, _, _, _, _, _ in experiments:
        if name not in results:
            continue
        r = results[name]
        perf = r.get("perf_aggregate", 0) or 0
        spt = r["speed"]["sec_per_token"]
        bspt = r["speed"]["bench_sec_per_token"]
        pn = perf / base_perf
        sn1 = 1 - spt/base_spt
        sn2 = 1 - bspt/base_bspt
        sc1 = max(0.5*pn + 0.5*sn1, 0)
        sc2 = max(0.5*pn + 0.5*sn2, 0)
        star = " ★" if sc1 > 0.626 else ""
        print(f"{name:<22} {perf:>6.4f} {pn:>5.3f} │ {sn1:>+7.3f} {sc1:>7.4f} │ {sn2:>+7.3f} {sc2:>7.4f}{star}")

    print(f"\n[DONE]")


if __name__ == "__main__":
    main()
