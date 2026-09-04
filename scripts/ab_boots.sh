#!/bin/bash
# Alternating A/B boots for the deepgemm tuned-config fix.
#
# Boot-to-boot variance (0.91%) turned out to be as large as the effect being
# measured (~1.3%), so a single boot per arm proves nothing. This runs a given
# sequence of arms, one full boot + case1 benchmark each, appending to a summary.
#
# Usage: ab_boots.sh <tag> <arm> [<arm> ...]      arm = on | off
# Each arm: boot on cards 8-15, wait for /health, run bench_case1.sh, shut down
# by exact PID (never pattern-kill: another container shares this host).
set -u
TAG=${1:?usage: ab_boots.sh <tag> <arm>...}
shift
SUM=/workspace/pytorch/ab_summary_${TAG}.txt
: > "$SUM"

boot_and_bench() {
  local arm=$1 idx=$2
  local slog=/workspace/pytorch/ab_${TAG}_${idx}_${arm}_serve.log
  local blog=/workspace/pytorch/ab_${TAG}_${idx}_${arm}_bench.log
  local cfg=1
  [ "$arm" = "off" ] && cfg=0

  echo "=== [$idx] arm=$arm (VLLM_FL_DEEPGEMM_CONFIGS=$cfg) ===" | tee -a "$SUM"
  VLLM_FL_DEEPGEMM_CONFIGS=$cfg LOG="$slog" PROF_DIR=/workspace/pytorch/prof_ab \
    nohup bash /workspace/pytorch/serve_opt_cards8_15.sh > /tmp/ab_boot_${idx}.log 2>&1 &

  local ready=0
  for i in $(seq 1 60); do
    if curl -s -o /dev/null --max-time 3 http://localhost:8011/health 2>/dev/null; then
      ready=1; break
    fi
    if grep -q "died unexpectedly\|Application startup failed\|EngineDeadError" "$slog" 2>/dev/null; then
      echo "  BOOT FAILED" | tee -a "$SUM"; break
    fi
    sleep 10
  done

  if [ "$ready" = "1" ]; then
    local n
    n=$(grep -c "deepgemm loading device_name_signature" "$slog" 2>/dev/null || echo 0)
    echo "  config-file load lines: $n (expect ~70 for on, 0 for off)" | tee -a "$SUM"
    bash /workspace/pytorch/bench_case1.sh "$blog" > /dev/null 2>&1
    grep -E "Output token throughput" "$blog" | awk '{print "  tok/s: "$5}' | tee -a "$SUM"
  fi

  # Shut down by exact PID only.
  local api
  api=$(grep -oE "APIServer pid=[0-9]+" "$slog" 2>/dev/null | head -1 | cut -d= -f2)
  [ -n "$api" ] && kill -TERM "$api" 2>/dev/null
  sleep 12
  for p in $(grep -oE "(EngineCore|Worker_TP[0-9]+) pid=[0-9]+" "$slog" 2>/dev/null | cut -d= -f2 | sort -u) $api; do
    kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null
  done
  sleep 10
  echo "  other-container util at end: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -2 | tr '\n' '/')" | tee -a "$SUM"
}

i=1
for arm in "$@"; do
  boot_and_bench "$arm" "$i"
  i=$((i+1))
done
echo "ALL DONE" | tee -a "$SUM"
