#!/usr/bin/env python3
"""W8A16 GPTQ quantization experiments."""
import os, sys, gc, time, json, subprocess
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

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


def load_manta_cal(tokenizer, num_samples=2048):
    data_5k = os.path.join(ROOT, "data", "manta", "train.json")
    samples = json.load(open(data_5k))[:num_samples]
    texts = []
    for s in samples:
        text = tokenizer.apply_chat_template(
            s["conversations"], add_generation_prompt=True, tokenize=False)
        texts.append(text)
    return Dataset.from_dict({"text": texts})


def quantize_w8a16(name, dampening=0.02, block_size=128):
    output_dir = os.path.join(ROOT, "models", name)
    if os.path.isdir(output_dir) and os.path.exists(os.path.join(output_dir, "config.json")):
        print(f"[SKIP] {name} already exists")
        return output_dir

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
    cal = load_manta_cal(tokenizer, 2048)

    recipe = [
        GPTQModifier(
            scheme="W8A16",
            targets=["Linear"],
            ignore=["lm_head"],
            dampening_frac=dampening,
            block_size=block_size,
        ),
    ]

    print(f"[QUANT] {name}: W8A16, damp={dampening}, block_size={block_size}")
    t0 = time.time()
    oneshot(
        model=model,
        dataset=cal,
        recipe=recipe,
        max_seq_length=1024,
        num_calibration_samples=2048,
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)
    print(f"  Saved: {output_dir} ({get_dir_size_mb(output_dir):.1f} MB)")
    del model, tokenizer
    cleanup_gpu()
    return output_dir


def evaluate(name, model_dir):
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
        print(f"[RESULT] {name}: perf={perf:.4f}, PN={pn:.3f}, SN={sn:+.3f}, Score={sc:.4f}")
        for k, v in d["benchmarks"].items():
            print(f"  {k}: {v['score']}")
        return d
    else:
        print(f"[FAIL] {name}")
        return None


def main():
    configs = [
        ("w8a16_d02", 0.02, 128),
        ("w8a16_d02_b64", 0.02, 64),
        ("w8a16_d015", 0.015, 128),
        ("w8a16_d015_b64", 0.015, 64),
    ]

    results = []
    for name, damp, bs in configs:
        out = quantize_w8a16(name, dampening=damp, block_size=bs)
        print(f"\n[EVAL] {name}...")
        d = evaluate(name, out)
        if d:
            perf = d.get("perf_aggregate", 0) or 0
            spt = d["speed"]["sec_per_token"]
            results.append((name, perf, spt))
        cleanup_gpu()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<22} {'Perf':>7} {'PN':>6} {'SN':>7} {'Score':>7}")
    print("-" * 55)
    for name, perf, spt in sorted(results, key=lambda x: -(x[1]/0.325*0.5 + (1-x[2]/0.000552)*0.5)):
        pn = perf / BASE_PERF
        sn = 1 - (spt / BASE_SPT)
        sc = max(0.5 * pn + 0.5 * sn, 0)
        print(f"{name:<22} {perf:>7.4f} {pn:>6.3f} {sn:>+7.3f} {sc:>7.4f}")
    print(f"\nRef: lane03_gptq_w8a16 = 0.6575 (local)")


if __name__ == "__main__":
    main()
