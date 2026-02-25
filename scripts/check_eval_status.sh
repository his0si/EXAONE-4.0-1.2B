#!/usr/bin/env bash
# 실행 중인 평가/빌드 프로세스와 최근 로그 확인
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${ROOT}/logs"

echo "=== 실행 중인 평가/파이프라인 (04_eval_vllm, aggressive_prune_kd, run_all) ==="
pgrep -af "04_eval_vllm|aggressive_prune_kd|run_all" || echo "(없음)"

echo ""
echo "=== logs/ 최근 PID 파일 ==="
shopt -s nullglob 2>/dev/null || true
for f in "${LOG_DIR}"/eval_*.pid; do
  [ -f "$f" ] || continue
  pid=$(cat "$f")
  if kill -0 "$pid" 2>/dev/null; then
    echo "  RUNNING: $f -> PID $pid"
  else
    echo "  done:    $f -> PID $pid"
  fi
done

echo ""
echo "=== logs/ 최근 로그 (마지막 5줄) ==="
for f in "${LOG_DIR}"/eval_*.log; do
  [ -f "$f" ] || continue
  echo "--- $(basename "$f") ---"
  tail -5 "$f" 2>/dev/null | sed 's/^/  /'
  echo ""
done

echo "=== results/ 최근 metrics ==="
for d in "${ROOT}"/results/*/; do
  [ -d "$d" ] || continue
  m="$d/metrics.json"
  [ -f "$m" ] || continue
  if command -v jq >/dev/null 2>&1; then
    echo "  $(basename "$d"): perf=$(jq -r '.perf_aggregate // empty' "$m") status=$(jq -r '.status // empty' "$m")"
  else
    echo "  $(basename "$d"): $(head -6 "$m" | tr '\n' ' ')"
  fi
done
