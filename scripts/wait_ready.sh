#!/bin/bash
# Wait until the server is GENUINELY serving, then print the boot facts.
#
# Do not poll /health: the APIServer answers 200 while the engine is still
# initialising, so a readiness loop on it returns instantly and a failed boot
# looks successful. "Application startup complete" in the serve log is the real
# signal; a boot failure is detected separately so we don't wait the full timeout.
#
# Usage: wait_ready.sh <serve_log> [max_seconds]
set -u
LOG=${1:?usage: wait_ready.sh <serve_log> [max_seconds]}
MAX=${2:-900}

for i in $(seq 1 $((MAX / 10))); do
  if grep -q "Application startup complete" "$LOG" 2>/dev/null; then
    echo "[ready] after ~$((i * 10))s"
    break
  fi
  if grep -qE "died unexpectedly|Application startup failed|EngineDeadError|Free memory on device" "$LOG" 2>/dev/null; then
    echo "[ready] BOOT FAILED:"
    grep -oE "Free memory on device \([^)]*\)|illegal memory access|EngineDeadError" "$LOG" 2>/dev/null | sort -u | head -3
    exit 1
  fi
  sleep 10
done

grep -q "Application startup complete" "$LOG" 2>/dev/null || { echo "[ready] TIMEOUT after ${MAX}s"; exit 1; }

echo "[ready] KV cache: $(grep -oE 'GPU KV cache size: [0-9,]+ tokens' "$LOG" | tail -1)"
echo "[ready] q head padding: $(grep -q 'Q head padding disabled' "$LOG" && echo 'disabled (patch active)' || echo 'UPSTREAM PADDING STILL ON')"
echo "[ready] deepgemm config files loaded: $(grep -c 'deepgemm loading device_name_signature' "$LOG")"
echo "[ready] fatal errors: $(grep -cE 'illegal memory|died unexpectedly|Free memory on device' "$LOG")"
