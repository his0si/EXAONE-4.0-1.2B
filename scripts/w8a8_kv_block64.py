#!/usr/bin/env python3
"""W8A8 GPTQ (block_size=64) + FP8 KV Cache."""
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


def main():
    output_dir = os.path.join(ROOT, "models", "w8a8_b64_kvfp8")

    if os.path.isdir(output_dir) and os.path.exists(os.path.join(output_dir, "config.json")):
        print(f"[SKIP] Already exists: {output_dir}")
    else:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import GPTQModifier

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
        cal = load_manta_cal(tokenizer, 2048)
        print(f"[CAL] {len(cal)} samples")

        recipe = [
            GPTQModifier(
                scheme="W8A8",
                targets=["Linear"],
                ignore=["lm_head"],
                dampening_frac=0.02,
                block_size=64,
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
        print(f"Quantization done in {time.time()-t0:.1f}s")

        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir, save_compressed=True)
        tokenizer.save_pretrained(output_dir)
        print(f"Saved: {output_dir} ({get_dir_size_mb(output_dir):.1f} MB)")
        del model, tokenizer
        cleanup_gpu()

    # Evaluate
    print(f"\n[EVAL] w8a8_b64_kvfp8...")
    eval_script = os.path.join(ROOT, "scripts", "04_eval_vllm.py")
    result_path = os.path.join(ROOT, "results", "w8a8_b64_kvfp8", "metrics.json")
    if os.path.exists(result_path):
        os.remove(result_path)

    subprocess.run([sys.executable, eval_script,
                    "--model", output_dir,
                    "--lane-id", "w8a8_b64_kvfp8", "--force"])

    if os.path.exists(result_path):
        d = json.load(open(result_path))
        perf = d.get("perf_aggregate", 0) or 0
        spt = d["speed"]["sec_per_token"]
        pn = perf / BASE_PERF
        sn = 1 - (spt / BASE_SPT)
        sc = max(0.5 * pn + 0.5 * sn, 0)
        print(f"\n[RESULT] w8a8_b64_kvfp8: perf={perf:.4f}, PN={pn:.3f}, SN={sn:+.3f}, Score={sc:.4f}")
        for k, v in d["benchmarks"].items():
            print(f"  {k}: {v['score']}")
    else:
        print("[FAIL] No metrics.json")


if __name__ == "__main__":
    main()
