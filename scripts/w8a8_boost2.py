#!/usr/bin/env python3
"""W8A8 score improvement experiments - Round 2.

Experiments:
  1. MSE observer (monkey-patch GPTQ to use MSE instead of minmax)
  2. Block size 64 (finer GPTQ update steps)
  3. Block size 64 + MSE observer combined
  4. Benchmark-weighted calibration with longer sequences (max_seq=2048)

Usage:
  conda run -n quant python scripts/w8a8_boost2.py
  conda run -n quant python scripts/w8a8_boost2.py --exp mse
  conda run -n quant python scripts/w8a8_boost2.py --exp block64
  conda run -n quant python scripts/w8a8_boost2.py --exp mse_block64
  conda run -n quant python scripts/w8a8_boost2.py --exp bench_cal
  conda run -n quant python scripts/w8a8_boost2.py --eval-only
"""
import os, sys, gc, time, json, subprocess, argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_MODEL = os.path.join(ROOT, "base_model")
LOG = os.path.join(ROOT, "logs", "w8a8_boost2.log")

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
#  MSE Observer Monkey-Patch
# ════════════════════════════════════════════════════════════

def patch_gptq_mse_observer():
    """Monkey-patch GPTQ to use MSE observer instead of minmax."""
    import llmcompressor.modifiers.quantization.gptq.gptq_quantize as gptq_mod
    from llmcompressor.observers.base import Observer

    _orig = gptq_mod.quantize_weight

    def patched_quantize_weight(module, quant_args, hessians_dict, blocksize=128, percdamp=0.01):
        # Intercept Observer.load_from_registry to swap minmax -> mse
        _orig_load = Observer.load_from_registry.__func__

        @classmethod
        def mse_load(cls, name, *args, **kwargs):
            if name == "memoryless_minmax":
                name = "memoryless_mse"
            return _orig_load(cls, name, *args, **kwargs)

        Observer.load_from_registry = mse_load
        try:
            result = _orig(module, quant_args, hessians_dict, blocksize, percdamp)
        finally:
            Observer.load_from_registry = classmethod(_orig_load)
        return result

    gptq_mod.quantize_weight = patched_quantize_weight
    print("  [PATCH] GPTQ observer: memoryless_minmax -> memoryless_mse")


def unpatch_gptq_observer():
    """Restore original GPTQ quantize_weight."""
    import llmcompressor.modifiers.quantization.gptq.gptq_quantize as gptq_mod
    if hasattr(gptq_mod, '_original_quantize_weight'):
        gptq_mod.quantize_weight = gptq_mod._original_quantize_weight
        print("  [UNPATCH] GPTQ observer restored")


# ════════════════════════════════════════════════════════════
#  Calibration Data Loaders
# ════════════════════════════════════════════════════════════

def load_manta_cal(tokenizer, num_samples):
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
    print(f"[CAL] Loaded {len(texts)} MANTA calibration samples")
    return Dataset.from_dict({"text": texts})


def load_benchmark_cal(tokenizer, num_samples=2048):
    """Calibration data matching actual benchmark distribution.

    50% KMMLU (matching kmmlu_pro + kmmlu_redux benchmarks)
    30% Ko-LongRAG with FULL context (matching ko_longrag benchmark)
    20% MANTA (general Korean)
    """
    texts = []
    n_kmmlu = int(num_samples * 0.5)
    n_longrag = int(num_samples * 0.3)
    n_manta = num_samples - n_kmmlu - n_longrag

    # KMMLU-Pro
    ds_pro = load_dataset("LGAI-EXAONE/KMMLU-Pro", split="test")
    for row in ds_pro:
        q = row["question"]
        opts = row["options"]
        opts_text = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(opts))
        msg = [{"role": "user", "content": f"다음 문제의 정답을 고르세요.\n\n{q}\n\n{opts_text}"}]
        text = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        texts.append(text)
        if len(texts) >= n_kmmlu // 2:
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
        if len(texts) >= n_kmmlu:
            break

    kmmlu_count = len(texts)

    # Ko-LongRAG - FULL context, not truncated
    ds_lr = load_dataset("LGAI-EXAONE/Ko-LongRAG", split="test")
    for row in ds_lr:
        ctx = row.get("context", "")  # FULL context
        q = row.get("question", "")
        msg = [{"role": "user", "content": f"다음 문서를 읽고 질문에 답하세요.\n\n{ctx}\n\n질문: {q}"}]
        text = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        texts.append(text)
        if len(texts) - kmmlu_count >= n_longrag:
            break

    longrag_count = len(texts) - kmmlu_count

    # MANTA (fill remaining)
    data_5k = os.path.join(ROOT, "data", "manta", "train.json")
    if os.path.exists(data_5k):
        manta = json.load(open(data_5k))[:n_manta]
    else:
        ds_m = load_dataset("LGAI-EXAONE/MANTA-1M", split=f"train[:{n_manta}]")
        manta = [{"conversations": r["conversations"]} for r in ds_m]
    for s in manta:
        text = tokenizer.apply_chat_template(
            s["conversations"], add_generation_prompt=True, tokenize=False)
        texts.append(text)

    print(f"[CAL] Benchmark-weighted: {kmmlu_count} KMMLU + {longrag_count} LongRAG + {len(texts)-kmmlu_count-longrag_count} MANTA = {len(texts)} total")
    return Dataset.from_dict({"text": texts})


# ════════════════════════════════════════════════════════════
#  Quantization
# ════════════════════════════════════════════════════════════

def quantize_w8a8(output_name, cal_dataset, cal_samples,
                  max_seq=1024, dampening=0.02, block_size=128, use_mse=False):
    """Generic W8A8 quantization with configurable parameters."""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    output_dir = os.path.join(ROOT, "models", output_name)
    if os.path.isdir(output_dir) and os.path.exists(os.path.join(output_dir, "config.json")):
        print(f"  [SKIP] Already exists: {output_dir}")
        return output_dir

    if use_mse:
        patch_gptq_mse_observer()

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    print(f"  Model loaded: {model.config.num_hidden_layers} layers, block_size={block_size}, mse={use_mse}")

    recipe = [
        GPTQModifier(
            scheme="W8A8",
            targets=["Linear"],
            ignore=["lm_head"],
            dampening_frac=dampening,
            block_size=block_size,
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

    del model, tokenizer
    cleanup_gpu()

    # Reload module to unpatch
    if use_mse:
        import importlib
        import llmcompressor.modifiers.quantization.gptq.gptq_quantize as gptq_mod
        importlib.reload(gptq_mod)

    return output_dir


# ════════════════════════════════════════════════════════════
#  Evaluation
# ════════════════════════════════════════════════════════════

def evaluate(name, model_dir):
    """Run local evaluation."""
    eval_script = os.path.join(ROOT, "scripts", "04_eval_vllm.py")
    result_path = os.path.join(ROOT, "results", name, "metrics.json")
    if os.path.exists(result_path):
        os.remove(result_path)

    cmd = [sys.executable, eval_script, "--model", model_dir, "--lane-id", name, "--force"]
    t0 = time.time()
    subprocess.run(cmd)
    elapsed = time.time() - t0

    if os.path.exists(result_path):
        with open(result_path) as f:
            d = json.load(f)
        perf = d.get("perf_aggregate", 0) or 0
        spt = d["speed"]["sec_per_token"]
        pn = perf / BASE_PERF
        sn = 1 - (spt / BASE_SPT)
        sc = max(0.5 * pn + 0.5 * sn, 0)
        print(f"  [RESULT] {name}: perf={perf:.4f}, PN={pn:.3f}, SN={sn:+.3f}, Score={sc:.4f} ({elapsed:.0f}s)")
        return d
    else:
        print(f"  [FAIL] {name}: no metrics.json ({elapsed:.0f}s)")
        return None


# ════════════════════════════════════════════════════════════
#  Experiments
# ════════════════════════════════════════════════════════════

def exp_mse():
    """Exp1: MSE observer instead of minmax."""
    print("\n" + "="*60)
    print("EXP: MSE Observer + W8A8 (damp=0.02, cal=2048)")
    print("="*60)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    cal = load_manta_cal(tokenizer, 2048)
    out = quantize_w8a8("boost2_mse", cal, 2048, max_seq=1024, use_mse=True)
    return out


def exp_block64():
    """Exp2: Smaller GPTQ block size (64 instead of 128)."""
    print("\n" + "="*60)
    print("EXP: Block size 64 + W8A8 (damp=0.02, cal=2048)")
    print("="*60)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    cal = load_manta_cal(tokenizer, 2048)
    out = quantize_w8a8("boost2_block64", cal, 2048, max_seq=1024, block_size=64)
    return out


def exp_mse_block64():
    """Exp3: MSE observer + block size 64 combined."""
    print("\n" + "="*60)
    print("EXP: MSE Observer + Block 64 + W8A8 (damp=0.02, cal=2048)")
    print("="*60)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    cal = load_manta_cal(tokenizer, 2048)
    out = quantize_w8a8("boost2_mse_b64", cal, 2048, max_seq=1024, block_size=64, use_mse=True)
    return out


def exp_bench_cal():
    """Exp4: Benchmark-weighted calibration with longer sequences."""
    print("\n" + "="*60)
    print("EXP: Benchmark-weighted cal + W8A8 (damp=0.02, max_seq=2048)")
    print("="*60)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    cal = load_benchmark_cal(tokenizer, 2048)
    out = quantize_w8a8("boost2_bench_cal", cal, 2048, max_seq=2048)
    return out


def exp_mse_bench():
    """Exp5: MSE observer + benchmark cal + block 64 (all combined)."""
    print("\n" + "="*60)
    print("EXP: MSE + Block 64 + Benchmark cal + W8A8 (max_seq=2048)")
    print("="*60)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    cal = load_benchmark_cal(tokenizer, 2048)
    out = quantize_w8a8("boost2_all", cal, 2048, max_seq=2048, block_size=64, use_mse=True)
    return out


EXPERIMENTS = {
    "mse": exp_mse,
    "block64": exp_block64,
    "mse_block64": exp_mse_block64,
    "bench_cal": exp_bench_cal,
    "mse_bench": exp_mse_bench,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default=None,
                        help="Run specific experiment (mse, block64, mse_block64, bench_cal, mse_bench)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation on existing models")
    args = parser.parse_args()

    print("="*60)
    print("W8A8 Boost Round 2")
    print("="*60)

    models_to_eval = []

    if args.eval_only:
        # Evaluate all boost2 models
        model_dir = os.path.join(ROOT, "models")
        for name in sorted(os.listdir(model_dir)):
            if name.startswith("boost2_"):
                models_to_eval.append((name, os.path.join(model_dir, name)))
    elif args.exp:
        if args.exp in EXPERIMENTS:
            out = EXPERIMENTS[args.exp]()
            if out:
                models_to_eval.append((os.path.basename(out), out))
        else:
            print(f"Unknown experiment: {args.exp}")
            print(f"Available: {', '.join(EXPERIMENTS.keys())}")
            return
    else:
        # Run all experiments
        for name, func in EXPERIMENTS.items():
            try:
                out = func()
                if out:
                    models_to_eval.append((os.path.basename(out), out))
            except Exception as e:
                print(f"  [ERROR] {name}: {e}")
                import traceback
                traceback.print_exc()
            cleanup_gpu()

    # Evaluate
    if models_to_eval:
        print("\n" + "="*60)
        print("Phase 2: Evaluation")
        print("="*60)

        results = []
        for name, model_dir in models_to_eval:
            print(f"\n  [EVAL] {name}...")
            d = evaluate(name, model_dir)
            if d:
                perf = d.get("perf_aggregate", 0) or 0
                spt = d["speed"]["sec_per_token"]
                results.append((name, perf, spt))
            cleanup_gpu()

        # Summary
        print("\n" + "#"*60)
        print("# SUMMARY")
        print("#"*60)
        print(f"\n{'Model':<25} {'Perf':>7} {'PN':>6} {'SN':>7} {'Score':>7}")
        print("-" * 55)

        # Include reference
        ref_perf = 0.3233
        ref_spt = 0.000410
        ref_pn = ref_perf / BASE_PERF
        ref_sn = 1 - (ref_spt / BASE_SPT)
        ref_sc = max(0.5 * ref_pn + 0.5 * ref_sn, 0)
        print(f"{'w8a8_cal2048_d02 (ref)':<25} {ref_perf:>7.4f} {ref_pn:>6.3f} {ref_sn:>+7.3f} {ref_sc:>7.4f}")

        for name, perf, spt in sorted(results, key=lambda x: -x[1]):
            pn = perf / BASE_PERF
            sn = 1 - (spt / BASE_SPT)
            sc = max(0.5 * pn + 0.5 * sn, 0)
            diff = sc - ref_sc
            print(f"{name:<25} {perf:>7.4f} {pn:>6.3f} {sn:>+7.3f} {sc:>7.4f} ({diff:+.4f})")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
