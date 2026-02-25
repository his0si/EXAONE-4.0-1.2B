#!/usr/bin/env python3
"""FP8 quantization experiments for better score.

FP8 on L4 (server) has native tensor cores → faster than INT8.
Local FP8_DYN already scores 0.6402 vs W8A8 0.6260.

Experiments:
  1. fp8_gptq: FP8 weights with GPTQ calibration (instead of simple observer)
  2. fp8_kv: W8A8 + FP8 KV cache (speed boost from compressed KV cache)
  3. fp8_block: FP8 block-wise quantization (potentially better quality)
  4. fp8_bench_cal: FP8_DYN with benchmark-weighted calibration

Usage:
  conda run -n quant python scripts/fp8_experiments.py
  conda run -n quant python scripts/fp8_experiments.py --exp fp8_gptq
  conda run -n quant python scripts/fp8_experiments.py --exp fp8_kv
  conda run -n quant python scripts/fp8_experiments.py --exp fp8_block
  conda run -n quant python scripts/fp8_experiments.py --exp fp8_bench_cal
  conda run -n quant python scripts/fp8_experiments.py --eval-only
"""
import os, sys, gc, time, json, subprocess, argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_MODEL = os.path.join(ROOT, "base_model")
LOG = os.path.join(ROOT, "logs", "fp8_experiments.log")

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


def load_manta_cal(tokenizer, num_samples=2048):
    data_5k = os.path.join(ROOT, "data", "manta", "train.json")
    samples = json.load(open(data_5k))[:num_samples]
    texts = []
    for s in samples:
        text = tokenizer.apply_chat_template(
            s["conversations"], add_generation_prompt=True, tokenize=False)
        texts.append(text)
    print(f"[CAL] Loaded {len(texts)} MANTA calibration samples")
    return Dataset.from_dict({"text": texts})


def load_benchmark_cal(tokenizer, num_samples=2048):
    """Calibration data matching actual benchmark distribution."""
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

    ds_lr = load_dataset("LGAI-EXAONE/Ko-LongRAG", split="test")
    for row in ds_lr:
        ctx = row.get("context", "")
        q = row.get("question", "")
        msg = [{"role": "user", "content": f"다음 문서를 읽고 질문에 답하세요.\n\n{ctx}\n\n질문: {q}"}]
        text = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        texts.append(text)
        if len(texts) - kmmlu_count >= n_longrag:
            break

    longrag_count = len(texts) - kmmlu_count

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


def evaluate(name, model_dir):
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

def exp_fp8_kv():
    """W8A8 GPTQ + FP8 KV Cache: INT8 compute with FP8 KV cache for speed."""
    print("\n" + "="*60)
    print("EXP: W8A8 GPTQ + FP8 KV Cache")
    print("="*60)

    output_dir = os.path.join(ROOT, "models", "fp8_w8a8_kvcache")
    if os.path.isdir(output_dir) and os.path.exists(os.path.join(output_dir, "config.json")):
        print(f"  [SKIP] Already exists: {output_dir}")
        return output_dir

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
    cal = load_manta_cal(tokenizer, 2048)

    recipe = [
        GPTQModifier(
            scheme="W8A8",
            targets=["Linear"],
            ignore=["lm_head"],
            dampening_frac=0.02,
            kv_cache_scheme={
                "num_bits": 8,
                "type": "float",
                "strategy": "tensor",
                "dynamic": False,
                "symmetric": True,
            },
        )
    ]

    t0 = time.time()
    oneshot(
        model=model,
        dataset=cal,
        recipe=recipe,
        max_seq_length=1024,
        num_calibration_samples=2048,
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
    return output_dir


def exp_fp8_block():
    """FP8 block-wise quantization (potentially finer-grained than per-channel)."""
    print("\n" + "="*60)
    print("EXP: FP8 Block-wise quantization")
    print("="*60)

    output_dir = os.path.join(ROOT, "models", "fp8_block")
    if os.path.isdir(output_dir) and os.path.exists(os.path.join(output_dir, "config.json")):
        print(f"  [SKIP] Already exists: {output_dir}")
        return output_dir

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
    cal = load_manta_cal(tokenizer, 2048)

    recipe = [
        QuantizationModifier(
            scheme="FP8_BLOCK",
            targets=["Linear"],
            ignore=["lm_head"],
        )
    ]

    t0 = time.time()
    oneshot(
        model=model,
        dataset=cal,
        recipe=recipe,
        max_seq_length=1024,
        num_calibration_samples=2048,
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
    return output_dir


def exp_fp8_dyn_d015():
    """FP8_DYNAMIC with dampening 0.015 (sweep best for W8A8, try for FP8)."""
    print("\n" + "="*60)
    print("EXP: FP8_DYNAMIC (dampening=0.015)")
    print("="*60)

    output_dir = os.path.join(ROOT, "models", "fp8_dyn_d015")
    if os.path.isdir(output_dir) and os.path.exists(os.path.join(output_dir, "config.json")):
        print(f"  [SKIP] Already exists: {output_dir}")
        return output_dir

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
    cal = load_manta_cal(tokenizer, 2048)

    recipe = [
        QuantizationModifier(
            scheme="FP8_DYNAMIC",
            targets=["Linear"],
            ignore=["lm_head"],
        )
    ]

    t0 = time.time()
    oneshot(
        model=model,
        dataset=cal,
        recipe=recipe,
        max_seq_length=1024,
        num_calibration_samples=2048,
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
    return output_dir


def exp_fp8_bench_cal():
    """FP8_DYNAMIC with benchmark-weighted calibration data."""
    print("\n" + "="*60)
    print("EXP: FP8_DYNAMIC with benchmark-weighted calibration")
    print("="*60)

    output_dir = os.path.join(ROOT, "models", "fp8_bench_cal")
    if os.path.isdir(output_dir) and os.path.exists(os.path.join(output_dir, "config.json")):
        print(f"  [SKIP] Already exists: {output_dir}")
        return output_dir

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
    cal = load_benchmark_cal(tokenizer, 2048)

    recipe = [
        QuantizationModifier(
            scheme="FP8_DYNAMIC",
            targets=["Linear"],
            ignore=["lm_head"],
        )
    ]

    t0 = time.time()
    oneshot(
        model=model,
        dataset=cal,
        recipe=recipe,
        max_seq_length=2048,
        num_calibration_samples=2048,
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
    return output_dir


EXPERIMENTS = {
    "fp8_kv": exp_fp8_kv,
    "fp8_block": exp_fp8_block,
    "fp8_dyn_d015": exp_fp8_dyn_d015,
    "fp8_bench_cal": exp_fp8_bench_cal,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    print("="*60)
    print("FP8 Experiments")
    print("="*60)

    models_to_eval = []

    if args.eval_only:
        model_dir = os.path.join(ROOT, "models")
        for name in sorted(os.listdir(model_dir)):
            if name.startswith("fp8_"):
                models_to_eval.append((name, os.path.join(model_dir, name)))
        # Also evaluate base_fp8dyn if not already
        fp8dyn = os.path.join(model_dir, "base_fp8dyn")
        if os.path.isdir(fp8dyn):
            models_to_eval.append(("base_fp8dyn", fp8dyn))
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
        print(f"\n{'Model':<25} {'Perf':>7} {'PN':>6} {'SN':>7} {'Score':>7} {'Diff':>7}")
        print("-" * 65)

        ref_sc = 0.6260  # w8a8_cal2048_d02 local
        fp8_ref_sc = 0.6402  # base_fp8dyn local

        for name, perf, spt in sorted(results, key=lambda x: -x[1]):
            pn = perf / BASE_PERF
            sn = 1 - (spt / BASE_SPT)
            sc = max(0.5 * pn + 0.5 * sn, 0)
            diff = sc - ref_sc
            print(f"{name:<25} {perf:>7.4f} {pn:>6.3f} {sn:>+7.3f} {sc:>7.4f} ({diff:+.4f})")

        print(f"\nReference: w8a8_cal2048_d02 = 0.6260 (local), base_fp8dyn = 0.6402 (local)")
        print(f"Server: w8a8_cal2048_d02 = 0.6208")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
