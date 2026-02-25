#!/bin/bash
# Fair speed comparison: run each model independently with fresh GPU state
cd /home/sunwoo/quant/0208
OUT=logs/speed_fair_results.jsonl
> "$OUT"

MODELS=(
    "models/baseline_bf16"
    "models/w8a8_cal2048_d02"
    "models/sweep_d030_c2048"
    "models/sweep_d015_c2048"
    "models/sweep_d02_c2048_s2048"
    "models/sweep_d025_c3072"
    "models/sweep_d02_c1536"
    "models/sweep_d02_c3072"
)

for m in "${MODELS[@]}"; do
    name=$(basename "$m")
    echo "[$(date)] Testing: $name"
    result=$(conda run -n quant python scripts/speed_single.py "$m" --name "$name" --runs 5 2>/dev/null | tail -1)
    echo "$result" >> "$OUT"
    echo "[$(date)] Result: $result"
    sleep 5
done

echo ""
echo "=== All Results ==="
cat "$OUT"
echo ""
echo "=== Summary ==="
python3 -c "
import json
results = []
for line in open('$OUT'):
    line = line.strip()
    if line and line.startswith('{'):
        results.append(json.loads(line))

if not results:
    print('No results')
    exit()

base_spt = None
for r in results:
    if r['name'] == 'baseline_bf16':
        base_spt = r['spt']

print(f'{\"Model\":<28} {\"tok/s\":>8} {\"spt\":>10} {\"SpeedNorm\":>10}')
print('-'*60)
for r in results:
    sn = 1 - (r['spt'] / base_spt) if base_spt else 0
    print(f'{r[\"name\"]:<28} {r[\"tok_s\"]:>8.1f} {r[\"spt\"]:>10.6f} {sn:>10.4f}')
"
