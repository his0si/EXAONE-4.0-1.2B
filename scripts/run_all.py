#!/usr/bin/env python3
"""
run_all.py — End-to-end pipeline orchestrator.

Ensures baseline exists → builds each lane → evaluates → ranks → packages.

Usage:
  python scripts/run_all.py --lanes configs/lanes.yaml
  python scripts/run_all.py --lanes configs/lanes.yaml --max-lanes 5
  python scripts/run_all.py --only lane01_gptq_w4a16,lane02_gptq_w4a16_damp
  python scripts/run_all.py --skip-train   # only quantize + eval
  python scripts/run_all.py --eval-only    # only evaluate existing models
"""
import os, sys, json, time, subprocess, argparse
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = sys.executable  # current python interpreter


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def run_cmd(cmd, desc=""):
    """Run a command as subprocess, streaming output."""
    print(f"\n{'─'*60}")
    print(f"  [{desc}] {' '.join(cmd)}")
    print(f"{'─'*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"  [{desc}] {status} ({elapsed:.0f}s)")
    return result.returncode == 0


def lane_model_exists(lane_id):
    """Check if a lane's model already exists."""
    model_dir = os.path.join(ROOT, "models", lane_id)
    return os.path.exists(os.path.join(model_dir, "model.safetensors"))


def lane_result_exists(lane_id):
    """Check if a lane's evaluation result already exists."""
    return os.path.exists(os.path.join(ROOT, "results", lane_id, "metrics.json"))


def resolve_dependencies(lane_id, lane_cfg, all_lanes):
    """Return ordered list of lane_ids that must be built first."""
    deps = []
    dep = lane_cfg.get("depends_on")
    if dep and dep in all_lanes:
        # Recursively resolve
        deps.extend(resolve_dependencies(dep, all_lanes[dep], all_lanes))
        deps.append(dep)
    return deps


def build_lane(lane_id, lane_cfg, cfg, force=False):
    """Build a single lane (train if needed, then quantize).
    If force=True, rebuild even if outputs already exist."""
    group = lane_cfg.get("group", "")
    train_mode = lane_cfg.get("train_mode", "none")
    quant_mode = lane_cfg.get("quant_mode", "none")
    force_args = ["--force"] if force else []

    print(f"\n{'='*60}")
    print(f"  Building lane: {lane_id}")
    print(f"  Desc: {lane_cfg.get('desc', '')}")
    print(f"  Group: {group}, Train: {train_mode}, Quant: {quant_mode}")
    if force:
        print(f"  ⚡ Force mode: overwrite existing outputs")
    print(f"{'='*60}")

    # ── Step 1: Training (if needed) ──
    if train_mode == "sft":
        sft_merged = os.path.join(ROOT, "checkpoints", "sft_merged")
        if force or not os.path.exists(os.path.join(sft_merged, "model.safetensors")):
            ok = run_cmd(
                [PYTHON, "scripts/02_finetune_sft.py", "--mode", "sft"] + force_args,
                desc=f"{lane_id}/sft"
            )
            if not ok:
                return False

    elif train_mode == "post_sft":
        input_model = lane_cfg.get("input_model", "")
        if input_model:
            abs_input = os.path.join(ROOT, input_model)
            output = os.path.join(ROOT, "models", lane_id)
            if force or not os.path.exists(os.path.join(output, "model.safetensors")):
                ok = run_cmd(
                    [PYTHON, "scripts/02_finetune_sft.py",
                     "--mode", "post_sft",
                     "--input-model", abs_input,
                     "--output", output] + force_args,
                    desc=f"{lane_id}/post_sft"
                )
                if not ok:
                    return False
            return True  # post_sft produces the final model directly

    elif train_mode == "kd":
        keep = lane_cfg.get("prune_params", {}).get("keep_layers", 28)
        kd_output = os.path.join(ROOT, "checkpoints", f"pruned{keep}_kd")
        if force or not os.path.exists(os.path.join(kd_output, "model.safetensors")):
            ok = run_cmd(
                [PYTHON, "scripts/02_finetune_sft.py",
                 "--mode", "kd",
                 "--keep-layers", str(keep),
                 "--output", kd_output] + force_args,
                desc=f"{lane_id}/kd"
            )
            if not ok:
                return False

    # ── Step 2: Quantization (if needed) ──
    if quant_mode != "none":
        if force or not lane_model_exists(lane_id):
            ok = run_cmd(
                [PYTHON, "scripts/03_quantize.py", "--lane", lane_id] + force_args,
                desc=f"{lane_id}/quant"
            )
            if not ok:
                return False
    elif train_mode == "sft" and quant_mode == "none":
        # lane05: just copy SFT model to models/
        src = os.path.join(ROOT, "checkpoints", "sft_merged")
        dst = os.path.join(ROOT, "models", lane_id)
        if force or not os.path.exists(os.path.join(dst, "model.safetensors")):
            import shutil
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  Copied SFT model to {dst}")

    return True


def eval_lane(lane_id, lane_cfg, cfg):
    """Evaluate a lane with vLLM."""
    model_dir = os.path.join(ROOT, "models", lane_id)
    if not os.path.isdir(model_dir):
        print(f"  [{lane_id}] Model dir not found: {model_dir}")
        return False

    return run_cmd(
        [PYTHON, "scripts/04_eval_vllm.py",
         "--model", model_dir,
         "--lane-id", lane_id],
        desc=f"{lane_id}/eval"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lanes", default="configs/lanes.yaml")
    parser.add_argument("--max-lanes", type=int, default=None)
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated lane IDs to run")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-rank", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Force overwrite: re-build models and re-evaluate even if results exist")
    args = parser.parse_args()

    cfg_path = os.path.join(ROOT, args.lanes)
    cfg = load_config(cfg_path)
    all_lanes = cfg.get("lanes", {})
    base_path = os.path.normpath(os.path.join(ROOT, cfg["project"]["base_model"]))

    # Determine which lanes to run
    if args.only:
        lane_ids = [l.strip() for l in args.only.split(",")]
    else:
        lane_ids = list(all_lanes.keys())
    if args.max_lanes:
        lane_ids = lane_ids[:args.max_lanes]

    print("=" * 60)
    print("  EXAONE-4.0-1.2B Compression — Full Pipeline")
    print("=" * 60)
    print(f"  Lanes to process: {len(lane_ids)}")
    print(f"  Base model: {base_path}")
    print(f"  Eval-only: {args.eval_only}")

    # ── Step 0: Setup check ──
    run_cmd([PYTHON, "scripts/00_setup_check.py"], desc="setup_check")

    # ── Step 1: Prepare data ──
    if not args.eval_only:
        run_cmd([PYTHON, "scripts/01_prepare_data.py"], desc="prepare_data")

    # ── Step 2: Baseline evaluation ──
    if not args.skip_baseline:
        baseline_path = os.path.join(ROOT, "results", "baseline.json")
        if args.force or not os.path.exists(baseline_path):
            print("\n[BASELINE] Running baseline evaluation...")
            run_cmd(
                [PYTHON, "scripts/04_eval_vllm.py",
                 "--model", base_path, "--baseline"],
                desc="baseline"
            )
        else:
            print("\n[BASELINE] Already exists, skipping. (use --force to re-run)")

    # ── Step 3: Build & evaluate each lane ──
    lane_status = {}
    for i, lane_id in enumerate(lane_ids, 1):
        lane_cfg = all_lanes.get(lane_id)
        if not lane_cfg:
            print(f"\n[{lane_id}] Not found in config, skipping")
            lane_status[lane_id] = "NOT_FOUND"
            continue

        print(f"\n\n{'▓'*60}")
        print(f"  Lane {i}/{len(lane_ids)}: {lane_id}")
        print(f"{'▓'*60}")

        # Resolve and build dependencies first
        deps = resolve_dependencies(lane_id, lane_cfg, all_lanes)
        for dep_id in deps:
            if (args.force or not lane_model_exists(dep_id)) and not args.eval_only:
                print(f"  Building dependency: {dep_id}")
                dep_cfg = all_lanes[dep_id]
                build_lane(dep_id, dep_cfg, cfg, force=args.force)

        # Build
        if not args.eval_only and not args.skip_train:
            ok = build_lane(lane_id, lane_cfg, cfg, force=args.force)
            if not ok:
                lane_status[lane_id] = "BUILD_FAILED"
                continue

        # Evaluate (always re-run with --force)
        if args.force or not lane_result_exists(lane_id):
            ok = eval_lane(lane_id, lane_cfg, cfg)
            lane_status[lane_id] = "OK" if ok else "EVAL_FAILED"
        else:
            print(f"  [{lane_id}] Evaluation already exists, skipping (use --force to re-run)")
            lane_status[lane_id] = "OK (cached)"

    # ── Step 4: Rank and package ──
    if not args.skip_rank:
        print("\n\n[RANK] Computing scores and packaging top-3...")
        run_cmd(
            [PYTHON, "scripts/05_rank_and_package.py"],
            desc="rank_and_package"
        )

    # ── Final summary ──
    print(f"\n\n{'='*60}")
    print("  Pipeline Complete!")
    print(f"{'='*60}")
    for lid, status in lane_status.items():
        mark = "✓" if "OK" in status else "✗"
        print(f"  {mark} {lid}: {status}")

    print(f"\n  Results: {os.path.join(ROOT, 'results')}")
    print(f"  Submissions: {os.path.join(ROOT, 'submissions')}")


if __name__ == "__main__":
    main()
