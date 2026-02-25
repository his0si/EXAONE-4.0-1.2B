#!/usr/bin/env python3
"""
05_rank_and_package.py — Compute Score_proxy, rank lanes, export top-3.

Usage:  python scripts/05_rank_and_package.py [--config configs/lanes.yaml]
"""
import os, sys, json, csv, shutil, argparse
from pathlib import Path
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_config(path="configs/lanes.yaml"):
    with open(os.path.join(ROOT, path)) as f:
        return yaml.safe_load(f)


def get_dir_size_mb(path):
    total = 0
    for f in Path(path).rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lanes.yaml")
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(os.path.join(ROOT, args.config))
    results_dir = os.path.join(ROOT, "results")
    submissions_dir = os.path.join(ROOT, "submissions")

    # ── Load baseline ──
    baseline_path = os.path.join(results_dir, "baseline.json")
    if not os.path.exists(baseline_path):
        print("[ERROR] baseline.json not found. Run baseline eval first.")
        sys.exit(1)

    with open(baseline_path) as f:
        baseline = json.load(f)

    base_perf = baseline.get("perf_aggregate")
    # 대회와 동일: 벤치마크 전체의 sec_per_token 사용 (prefill + decode 포함)
    base_spt = baseline["speed"].get("bench_sec_per_token",
               baseline["speed"].get("sec_per_token", 0))
    base_tps = 1.0 / base_spt if base_spt > 0 else 0
    # 전용 속도 벤치마크도 참고용으로 표시
    base_decode_tps = baseline["speed"].get("tokens_per_sec", 0)

    print("=" * 80)
    print("  Score Ranking & Packaging")
    print("=" * 80)
    print(f"\n  Baseline: perf={base_perf}, bench_sec/tok={base_spt:.6f}, "
          f"bench_tok/s={base_tps:.1f}, decode_tok/s={base_decode_tps:.1f}")

    # ── Load all lane results ──
    lanes_cfg = cfg.get("lanes", {})
    rows = []

    for lane_id in lanes_cfg:
        metrics_path = os.path.join(results_dir, lane_id, "metrics.json")
        if not os.path.exists(metrics_path):
            print(f"  [{lane_id}] No metrics found, skipping")
            continue

        with open(metrics_path) as f:
            m = json.load(f)

        if m.get("status") != "OK":
            print(f"  [{lane_id}] Status={m.get('status')}, skipping")
            continue

        lane_perf = m.get("perf_aggregate")
        # 대회와 동일: 벤치마크 전체의 sec_per_token 사용
        lane_spt = m["speed"].get("bench_sec_per_token",
                   m["speed"].get("sec_per_token", 0))
        lane_tps = 1.0 / lane_spt if lane_spt > 0 else 0
        lane_decode_tps = m["speed"].get("tokens_per_sec", 0)

        # Model directory
        model_dir = os.path.join(ROOT, "models", lane_id)
        if not os.path.isdir(model_dir):
            # Some lanes (post_sft) store in checkpoints
            model_dir = m.get("model_path", "")
        size_mb = get_dir_size_mb(model_dir) if os.path.isdir(model_dir) else 0

        # Compute scores
        perf_norm = (lane_perf / base_perf) if (base_perf and lane_perf) else None
        speed_norm = (1 - lane_spt / base_spt) if base_spt > 0 else 0

        if perf_norm is not None:
            score_proxy = max(0.5 * perf_norm + 0.5 * speed_norm, 0)
        else:
            # If perf is unknown, use speed only (penalized)
            score_proxy = max(0.5 * speed_norm, 0)

        # Per-benchmark scores
        bench_scores = {}
        for bname, bdata in m.get("benchmarks", {}).items():
            bench_scores[bname] = bdata.get("score")

        row = {
            "lane_id": lane_id,
            "desc": lanes_cfg[lane_id].get("desc", ""),
            "group": lanes_cfg[lane_id].get("group", ""),
            "perf_aggregate": lane_perf,
            "perf_norm": round(perf_norm, 4) if perf_norm else None,
            "sec_per_token": lane_spt,
            "tokens_per_sec": lane_tps,
            "speed_norm": round(speed_norm, 4),
            "score_proxy": round(score_proxy, 4),
            "size_mb": round(size_mb, 1),
            "model_dir": model_dir,
            **{f"bench_{k}": v for k, v in bench_scores.items()},
        }
        rows.append(row)

    if not rows:
        print("\n  [ERROR] No lane results found!")
        sys.exit(1)

    # Sort by score
    rows.sort(key=lambda r: r["score_proxy"], reverse=True)

    # ── Print summary table ──
    print(f"\n{'─'*80}")
    print(f"{'Rank':>4} {'Lane':<30} {'PerfNorm':>9} {'SpeedNorm':>10} "
          f"{'Score':>8} {'tok/s':>8} {'Size(MB)':>9}")
    print(f"{'─'*80}")

    for i, r in enumerate(rows, 1):
        pn = f"{r['perf_norm']:.4f}" if r['perf_norm'] else "  N/A  "
        marker = " ★" if i <= args.top_n else ""
        print(f"{i:>4}  {r['lane_id']:<30} {pn:>9} {r['speed_norm']:>+10.4f} "
              f"{r['score_proxy']:>8.4f} {r['tokens_per_sec']:>8.1f} "
              f"{r['size_mb']:>9.1f}{marker}")

    # ── Save summary.csv ──
    csv_path = os.path.join(results_dir, "summary.csv")
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  summary.csv → {csv_path}")

    # ── Save summary.md ──
    md_path = os.path.join(results_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("# EXAONE-4.0-1.2B Compression — Results Summary\n\n")
        f.write(f"**Baseline**: perf={base_perf}, sec/tok={base_spt:.6f}, "
                f"tok/s={base_tps:.1f}\n\n")
        f.write("## Ranking\n\n")
        f.write("| Rank | Lane | PerfNorm | SpeedNorm | Score | tok/s | Size(MB) |\n")
        f.write("|------|------|----------|-----------|-------|-------|----------|\n")
        for i, r in enumerate(rows, 1):
            pn = f"{r['perf_norm']:.4f}" if r['perf_norm'] else "N/A"
            star = " ★" if i <= args.top_n else ""
            f.write(f"| {i} | {r['lane_id']} | {pn} | "
                    f"{r['speed_norm']:+.4f} | {r['score_proxy']:.4f} | "
                    f"{r['tokens_per_sec']:.1f} | {r['size_mb']:.1f}{star} |\n")

        # Top-N details
        f.write(f"\n## Top-{args.top_n} Details\n\n")
        for i, r in enumerate(rows[:args.top_n], 1):
            f.write(f"### #{i}: {r['lane_id']}\n")
            f.write(f"- **Description**: {r['desc']}\n")
            f.write(f"- **Group**: {r['group']}\n")
            f.write(f"- **Score**: {r['score_proxy']:.4f}\n")
            pn = f"{r['perf_norm']:.4f}" if r['perf_norm'] else "N/A"
            f.write(f"- **PerfNorm**: {pn}\n")
            f.write(f"- **SpeedNorm**: {r['speed_norm']:+.4f}\n")
            f.write(f"- **Speed**: {r['tokens_per_sec']:.1f} tok/s "
                    f"({r['sec_per_token']*1000:.2f} ms/tok)\n")
            f.write(f"- **Size**: {r['size_mb']:.1f} MB\n")
            # Benchmark breakdown
            bench_keys = [k for k in r if k.startswith("bench_")]
            if bench_keys:
                f.write("- **Benchmarks**:\n")
                for bk in bench_keys:
                    bname = bk.replace("bench_", "")
                    val = r[bk]
                    f.write(f"  - {bname}: {val if val is not None else 'N/A'}\n")
            f.write("\n")

    print(f"  summary.md → {md_path}")

    # ── Package top-N into submissions/ ──
    print(f"\n[PACKAGE] Copying top-{args.top_n} to submissions/")
    for i, r in enumerate(rows[:args.top_n], 1):
        src = r["model_dir"]
        dst = os.path.join(submissions_dir, f"top{i}")

        if not os.path.isdir(src):
            print(f"  top{i}: {r['lane_id']} — model dir not found at {src}, skipping")
            continue

        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.makedirs(dst, exist_ok=True)

        # Copy only HF model files (no logs, etc.)
        hf_files = [
            "config.json", "generation_config.json",
            "model.safetensors", "model.safetensors.index.json",
            "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "vocab.json", "merges.txt",
            "chat_template.jinja",
        ]
        copied = 0
        for fname in os.listdir(src):
            fpath = os.path.join(src, fname)
            if os.path.isfile(fpath):
                # Copy known HF files or any .safetensors / .json / .jinja
                if fname in hf_files or fname.endswith((".safetensors", ".json", ".jinja", ".txt")):
                    shutil.copy2(fpath, os.path.join(dst, fname))
                    copied += 1

        size = get_dir_size_mb(dst)
        print(f"  top{i}: {r['lane_id']} ({r['score_proxy']:.4f}) → "
              f"{dst} ({copied} files, {size:.1f} MB)")

        # Also create submit zip
        zip_path = os.path.join(submissions_dir, f"submit_top{i}")
        # We need model/ subfolder in the zip
        temp_dir = os.path.join(submissions_dir, f"_temp_top{i}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        model_subdir = os.path.join(temp_dir, "model")
        shutil.copytree(dst, model_subdir)
        shutil.make_archive(zip_path, "zip", temp_dir, "model")
        shutil.rmtree(temp_dir)
        zip_size = os.path.getsize(zip_path + ".zip") / (1024**2)
        print(f"       → {zip_path}.zip ({zip_size:.1f} MB)")

    print(f"\n{'='*80}")
    print(f"  Done! Top-{args.top_n} packaged in {submissions_dir}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
