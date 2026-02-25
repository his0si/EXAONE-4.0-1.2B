#!/usr/bin/env python3
"""
package_v2.py — 지정된 lane들을 제출용 zip으로 패키징

전략: 원본 모델과 최대한 같은 결과 + 속도 향상
  - SFT 없이 순수 양자화만 적용된 모델
  - submit.zip 내부 구조: model/*

Usage:
  python scripts/package_v2.py
  python scripts/package_v2.py --lanes lane01_gptq_w4a16 lane03_gptq_w8a16 lane04_fp8_dynamic lane_eg_w4a16
"""
import os, sys, shutil, argparse, json
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_dir_size_mb(path):
    total = 0
    for f in Path(path).rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


def package_model(lane_id, model_dir, output_name):
    """모델을 submit zip으로 패키징"""
    submissions_dir = os.path.join(ROOT, "submissions")
    os.makedirs(submissions_dir, exist_ok=True)

    # 1) HF 표준 파일만 복사
    dst = os.path.join(submissions_dir, output_name)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)

    hf_extensions = (".safetensors", ".json", ".jinja", ".txt", ".model")
    copied = 0
    for fname in os.listdir(model_dir):
        fpath = os.path.join(model_dir, fname)
        if os.path.isfile(fpath):
            _, ext = os.path.splitext(fname)
            if ext in hf_extensions or fname in ("merges.txt", "vocab.json"):
                # recipe.yaml, quant_meta.json 등은 제출 불필요하지만 포함해도 무방
                # 다만 quant_meta.json은 우리 로컬용이므로 제외
                if fname == "quant_meta.json":
                    continue
                shutil.copy2(fpath, os.path.join(dst, fname))
                copied += 1

    size_mb = get_dir_size_mb(dst)
    print(f"  [{output_name}] {lane_id} → {copied} files, {size_mb:.1f} MB")

    # 2) submit zip 생성 (model/ 서브폴더 포함)
    zip_base = os.path.join(submissions_dir, f"submit_{output_name}")
    temp_dir = os.path.join(submissions_dir, f"_temp_{output_name}")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    model_subdir = os.path.join(temp_dir, "model")
    shutil.copytree(dst, model_subdir)
    shutil.make_archive(zip_base, "zip", temp_dir, "model")
    shutil.rmtree(temp_dir)

    zip_path = zip_base + ".zip"
    zip_size = os.path.getsize(zip_path) / (1024**2)
    print(f"       → {zip_path} ({zip_size:.1f} MB)")
    return zip_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lanes", nargs="+", default=None)
    args = parser.parse_args()

    # 기본 패키징 대상: SFT 없는 순수 양자화 모델들
    default_lanes = [
        ("lane_eg_w4a16",       "models/lane_eg_w4a16",       "eg_w4a16"),
        ("lane01_gptq_w4a16",   "models/lane01_gptq_w4a16",   "lane01_w4a16"),
        ("lane03_gptq_w8a16",   "models/lane03_gptq_w8a16",   "lane03_w8a16"),
        ("lane04_fp8_dynamic",  "models/lane04_fp8_dynamic",   "lane04_fp8"),
    ]

    if args.lanes:
        # 사용자 지정 lane
        targets = []
        for lane_id in args.lanes:
            model_dir = os.path.join(ROOT, "models", lane_id)
            output_name = lane_id
            targets.append((lane_id, f"models/{lane_id}", output_name))
    else:
        targets = default_lanes

    print("=" * 70)
    print("  V2 Packaging — 원본 성능 보존 전략")
    print("=" * 70)
    print()

    results = []
    for lane_id, model_rel, output_name in targets:
        model_dir = os.path.join(ROOT, model_rel)
        if not os.path.isdir(model_dir):
            print(f"  [{output_name}] 모델 없음: {model_dir}, 건너뜀")
            continue
        if not os.path.exists(os.path.join(model_dir, "model.safetensors")):
            # safetensors가 여러 shard일 수도 있으므로 glob 확인
            safetensors = list(Path(model_dir).glob("*.safetensors"))
            if not safetensors:
                print(f"  [{output_name}] safetensors 없음: {model_dir}, 건너뜀")
                continue

        zip_path = package_model(lane_id, model_dir, output_name)
        results.append((output_name, lane_id, zip_path))

    print()
    print("=" * 70)
    print(f"  완료! {len(results)}개 제출 zip 생성")
    print("=" * 70)
    print()
    print("  제출 우선순위 (추천):")
    print("  1. eg_w4a16     — 대회 예시 코드 동일 설정 (가장 안전)")
    print("  2. lane01_w4a16 — W4A16 (512 cal, 1024 seq)")
    print("  3. lane03_w8a16 — W8A16 (성능 최대 보존, 속도 32%↑)")
    print("  4. lane04_fp8   — FP8 Dynamic (성능 100% 보존, 속도 27%↑)")
    print()
    for name, lane_id, zpath in results:
        print(f"  {name}: {zpath}")


if __name__ == "__main__":
    main()
