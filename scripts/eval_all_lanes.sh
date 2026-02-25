#!/bin/bash
# 모든 lane 재평가 (바뀐 기준: 전체 샘플, 개선된 프롬프트)
cd /home/sunwoo/quant/0208

LANES=(
  "lane02_gptq_w4a16_damp"
  "lane05_sft_fp16"
  "lane06_sft_gptq_w4a16"
  "lane07_sft_gptq_w4a16_damp"
  "lane08_w4a16_postsft"
  "lane09_w4a16_damp_postsft"
  "lane10_prune28_kd_w4a16"
  "lane11_prune26_kd_w4a16"
  "lane12_fp8_static"
)

echo "=========================================="
echo " 전체 Lane 재평가 시작 ($(date))"
echo " 대상: ${#LANES[@]}개 lane"
echo "=========================================="

for lane in "${LANES[@]}"; do
  model_dir="models/${lane}"
  if [ ! -d "$model_dir" ]; then
    echo "[SKIP] $lane — 모델 없음"
    continue
  fi
  echo ""
  echo "────────────────────────────────────────"
  echo " [$lane] 평가 시작 ($(date '+%H:%M:%S'))"
  echo "────────────────────────────────────────"
  python scripts/04_eval_vllm.py \
    --model "$model_dir" \
    --lane-id "$lane" \
    --force 2>&1 | grep -E "(BENCH|Accuracy|F1:|SPEED|Time:|Score:|tok/s|perf_aggregate|Results saved)"
  echo " [$lane] 완료 ($(date '+%H:%M:%S'))"
done

echo ""
echo "=========================================="
echo " 전체 완료 ($(date))"
echo "=========================================="
