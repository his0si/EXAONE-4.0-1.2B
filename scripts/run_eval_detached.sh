#!/usr/bin/env bash
# 평가를 nohup으로 실행해 세션 끊겨도 계속 돌아가게 함.
# 사용: ./scripts/run_eval_detached.sh [모델경로] [lane_id]
# 예:   ./scripts/run_eval_detached.sh models/pruned22_kd_fp8static pruned22_kd_fp8static

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"

MODEL="${1:-models/pruned22_kd_fp8static}"
LANE="${2:-$(basename "$MODEL")}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="${LOG_DIR}/eval_${LANE}_${TIMESTAMP}.log"

echo "[run_eval_detached] Model=$MODEL Lane=$LANE"
echo "[run_eval_detached] Log: $LOGFILE"
echo "[run_eval_detached] nohup 시작 (세션 끊어도 계속 실행됨)"

nohup env PYTHONUNBUFFERED=1 python scripts/04_eval_vllm.py \
  --model "$MODEL" \
  --lane-id "$LANE" \
  --force \
  >> "$LOGFILE" 2>&1 &

PID=$!
echo $PID > "${LOG_DIR}/eval_${LANE}.pid"
echo "[run_eval_detached] PID=$PID (종료: kill $PID)"
echo "[run_eval_detached] 로그 보기: tail -f $LOGFILE"
