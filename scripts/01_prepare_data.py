#!/usr/bin/env python3
"""
01_prepare_data.py — Download & preprocess MANTA-1M (train/val split)
                     and probe benchmark datasets.

Usage:  python scripts/01_prepare_data.py [--config configs/lanes.yaml]
"""
import os, sys, json, argparse, random
import yaml
from datasets import load_dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ── MANTA-1M train/val ─────────────────────────────────────
def prepare_manta(cfg):
    ds_cfg = cfg["datasets"]["train"]
    hf_id = ds_cfg["hf_id"]
    n = ds_cfg["num_samples"]
    val_ratio = ds_cfg["val_ratio"]
    seed = cfg["project"]["seed"]

    out_dir = os.path.join(DATA_DIR, "manta")
    train_path = os.path.join(out_dir, "train.json")
    val_path = os.path.join(out_dir, "val.json")
    if os.path.exists(train_path) and os.path.exists(val_path):
        train = json.load(open(train_path))
        val = json.load(open(val_path))
        print(f"[MANTA] Already prepared: {len(train)} train, {len(val)} val")
        return

    print(f"[MANTA] Downloading {hf_id} (first {n} samples)...")
    ds = load_dataset(hf_id, split=f"train[:{n}]")

    indices = list(range(len(ds)))
    random.seed(seed)
    random.shuffle(indices)
    n_val = max(1, int(len(indices) * val_ratio))
    val_idx = set(indices[:n_val])

    train_samples, val_samples = [], []
    for i, row in enumerate(ds):
        sample = {"conversations": row["conversations"]}
        if i in val_idx:
            val_samples.append(sample)
        else:
            train_samples.append(sample)

    os.makedirs(out_dir, exist_ok=True)
    with open(train_path, "w") as f:
        json.dump(train_samples, f, ensure_ascii=False)
    with open(val_path, "w") as f:
        json.dump(val_samples, f, ensure_ascii=False)
    print(f"[MANTA] Saved {len(train_samples)} train, {len(val_samples)} val → {out_dir}")


# ── Benchmark datasets (probe structure) ───────────────────
def probe_benchmark(name, bench_cfg):
    hf_id = bench_cfg["hf_id"]
    out_dir = os.path.join(DATA_DIR, "benchmarks", name)
    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        print(f"[{name}] Already probed: {meta.get('num_samples', '?')} samples, "
              f"columns={meta.get('columns', '?')}")
        return meta

    print(f"[{name}] Loading {hf_id}...")
    try:
        # Try different splits
        ds = None
        for split_name in ["test", "validation", "train"]:
            try:
                ds = load_dataset(hf_id, split=split_name)
                used_split = split_name
                break
            except (ValueError, KeyError):
                continue

        if ds is None:
            # Try without specifying split (takes default)
            info = load_dataset(hf_id)
            available = list(info.keys())
            if available:
                used_split = available[0]
                ds = info[used_split]
            else:
                raise RuntimeError("No splits found")

        meta = {
            "hf_id": hf_id,
            "type": bench_cfg["type"],
            "split": used_split,
            "num_samples": len(ds),
            "columns": ds.column_names,
            "sample": {k: str(v)[:200] for k, v in ds[0].items()} if len(ds) > 0 else {},
        }
        os.makedirs(out_dir, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[{name}] {len(ds)} samples, split={used_split}, "
              f"columns={ds.column_names}")
        return meta

    except Exception as e:
        print(f"[{name}] FAILED to load: {e}")
        meta = {"hf_id": hf_id, "type": bench_cfg["type"], "error": str(e)}
        os.makedirs(out_dir, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lanes.yaml")
    args = parser.parse_args()
    cfg = load_config(os.path.join(ROOT, args.config))

    print("=" * 60)
    print("  Data Preparation")
    print("=" * 60)

    # 1. Training data
    print("\n[1] Training data (MANTA-1M)")
    prepare_manta(cfg)

    # 2. Benchmarks
    print("\n[2] Benchmark datasets (probe)")
    benchmarks = cfg["datasets"]["benchmarks"]
    for name, bench_cfg in benchmarks.items():
        probe_benchmark(name, bench_cfg)

    print("\n" + "=" * 60)
    print("  Data preparation complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
