#!/usr/bin/env python3
"""
00_setup_check.py — Environment & GPU / vLLM sanity check.

Usage:  python scripts/00_setup_check.py
"""
import sys, os, json

REQUIRED = {
    "torch":              "2.9",
    "transformers":       "4.57",
    "vllm":               "0.14",
    "datasets":           None,
    "peft":               None,
    "trl":                None,
    "yaml":               None,       # PyYAML
    "safetensors":        None,
    "compressed_tensors": None,
}


def check_packages():
    ok = True
    results = {}
    for pkg, min_ver in REQUIRED.items():
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            match = True
            if min_ver and not ver.startswith(min_ver):
                match = False
            status = "OK" if match else f"WARN (expected {min_ver}.x)"
            results[pkg] = {"version": ver, "status": status}
            if not match:
                ok = False
        except ImportError:
            results[pkg] = {"version": None, "status": "MISSING"}
            ok = False
    return results, ok


def check_gpu():
    import torch
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    return info


def check_vllm():
    try:
        from vllm import LLM
        return {"importable": True}
    except Exception as e:
        return {"importable": False, "error": str(e)}


def check_base_model():
    base = os.path.join(os.path.dirname(__file__), "..", "..", "base_model")
    base = os.path.abspath(base)
    exists = os.path.isdir(base)
    has_weights = os.path.exists(os.path.join(base, "model.safetensors")) if exists else False
    return {"path": base, "exists": exists, "has_weights": has_weights}


def main():
    print("=" * 60)
    print("  EXAONE-4.0-1.2B Compression — Setup Check")
    print("=" * 60)

    # 1. Packages
    print("\n[1] Python packages")
    pkgs, pkgs_ok = check_packages()
    for name, info in pkgs.items():
        mark = "✓" if info["status"] == "OK" else "✗"
        ver = info["version"] or "—"
        print(f"  {mark} {name:<22} {ver:<15} {info['status']}")

    # 2. GPU
    print("\n[2] GPU")
    gpu = check_gpu()
    if gpu["cuda_available"]:
        print(f"  ✓ {gpu['device_name']} ({gpu['vram_gb']} GB)")
    else:
        print("  ✗ CUDA not available")

    # 3. vLLM
    print("\n[3] vLLM import")
    vllm_info = check_vllm()
    if vllm_info["importable"]:
        print("  ✓ vLLM importable")
    else:
        print(f"  ✗ vLLM import failed: {vllm_info.get('error','')}")

    # 4. Base model
    print("\n[4] Base model")
    bm = check_base_model()
    if bm["exists"] and bm["has_weights"]:
        print(f"  ✓ Found at {bm['path']}")
    else:
        print(f"  ✗ Not found at {bm['path']}")

    # Summary
    all_ok = pkgs_ok and gpu["cuda_available"] and vllm_info["importable"] and bm["has_weights"]
    print("\n" + "=" * 60)
    if all_ok:
        print("  ✓ All checks passed — ready to run pipeline")
    else:
        print("  ✗ Some checks failed — fix issues above before proceeding")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
