#!/bin/bash
# Test speed impact of different max_position_embeddings values
cd /home/sunwoo/quant/0208
OUT=logs/maxpos_speed.jsonl
> "$OUT"

MODELS=(
    "models/w8a8_cal2048_d02"
    "models/w8a8_d02_ml32768"
    "models/w8a8_d02_ml20480"
    "models/w8a8_d02_ml16384"
)

for m in "${MODELS[@]}"; do
    name=$(basename "$m")
    echo "[$(date)] Testing: $name"
    conda run -n quant python scripts/speed_single.py "$m" --name "$name" --runs 5 2>/dev/null
    echo "[$(date)] Done: $name"
    sleep 3
done

echo ""
echo "=== Results ==="
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

print(f'{\"Model\":<28} {\"tok/s\":>8} {\"spt\":>10}')
print('-'*50)
for r in results:
    print(f'{r[\"name\"]:<28} {r[\"tok_s\"]:>8.1f} {r[\"spt\"]:>10.6f}')
"
