#!/bin/bash
# 1. FP8 KV Cache experiment (damp=0.02)
# 2. Evaluate all new models
cd /home/sunwoo/quant/0208

echo "=== FP8 KV Cache (damp=0.02) ==="
conda run -n quant python -u scripts/w8a8_kvcache_fp8.py --dampening 0.02 --output models/w8a8_kv_fp8 2>&1
echo "  Done: w8a8_kv_fp8 ($(date '+%H:%M:%S'))"

echo ""
echo "=== FP8 KV Cache (damp=0.01) ==="
conda run -n quant python -u scripts/w8a8_kvcache_fp8.py --dampening 0.01 --output models/w8a8_kv_fp8_nodamp 2>&1
echo "  Done: w8a8_kv_fp8_nodamp ($(date '+%H:%M:%S'))"

echo ""
echo "=== Evaluating all new models ==="
MODELS=(
  "w8a8_sparse"
  "w8a8_sparse_damp"
  "w8a8_gptq_adv"
  "w8a8_gptq_adv_damp"
  "w8a8_kv_fp8"
  "w8a8_kv_fp8_nodamp"
)

for m in "${MODELS[@]}"; do
  if [ ! -d "models/$m" ]; then
    echo "  [SKIP] $m — not found"
    continue
  fi
  echo ""
  echo "  [EVAL] $m ($(date '+%H:%M:%S'))"
  conda run -n quant python scripts/04_eval_vllm.py \
    --model "models/$m" --lane-id "$m" --force 2>&1 | \
    grep -E "(perf_aggregate|tok/s|Results saved|Error|error)"
  echo "  [DONE] $m ($(date '+%H:%M:%S'))"
done

echo ""
echo "=== ALL COMPLETE ($(date)) ==="
