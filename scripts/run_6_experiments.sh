#!/bin/bash
# Run 6 W8A8 advanced experiments sequentially
cd /home/sunwoo/quant/0208

EXPERIMENTS=(
  "smoothquant"
  "smoothquant_damp"
  "sparse"
  "sparse_damp"
  "gptq_adv"
  "gptq_adv_damp"
)

for exp in "${EXPERIMENTS[@]}"; do
  echo ""
  echo "============================================"
  echo "  Starting: $exp  ($(date '+%H:%M:%S'))"
  echo "============================================"
  conda run -n quant python -u scripts/w8a8_advanced.py --experiment "$exp" 2>&1
  echo "  Finished: $exp  ($(date '+%H:%M:%S'))"
  echo ""
done

echo ""
echo "=========================================="
echo " All 6 experiments complete ($(date))"
echo "=========================================="
