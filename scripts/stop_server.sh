#!/bin/bash
# Stop the DSv4 server on cards 8-15 and WAIT for the driver to release memory.
#
# Written because two ad-hoc shutdowns leaked the whole 90 GB/card:
#   1. grepping the serve log for "Worker_TP<n> pid=" missed the 8 workers, which
#      this build also logs as plain "Worker pid=". They were orphaned, kept the
#      memory, and the next boot died with "Free memory on device (6.64/95.62)".
#   2. An orphan from an earlier boot was still answering /health on 8011, so the
#      readiness poll returned 200 instantly and the failure looked like success.
#
# So: enumerate by PROCESS TREE (never by log grep), then poll the driver until
# the memory is actually back. Never pattern-kill across the host — another
# tenant holds cards 0-3 and must not be touched.
set -u
PORT=${PORT:-8011}

# Roots: any `vllm` process owning a VLLM::EngineCore child, plus stray
# VLLM::* processes. All within this container's PID namespace only.
roots=$(ps -eo pid,comm 2>/dev/null | awk '$2=="vllm"{print $1}')
if [ -z "$roots" ]; then
  echo "[stop] no 'vllm' root process in this namespace"
fi

pids=""
for r in $roots; do
  kids=$(ps -eo pid,ppid 2>/dev/null | awk -v r="$r" '$2==r{print $1}')
  gkids=""
  for k in $kids; do
    gkids="$gkids $(ps -eo pid,ppid 2>/dev/null | awk -v k="$k" '$2==k{print $1}')"
  done
  pids="$pids $r $kids $gkids"
done
# Catch VLLM::* whose parent already died.
pids="$pids $(ps -eo pid,comm 2>/dev/null | awk '$2 ~ /^VLLM::/ {print $1}')"
pids=$(echo "$pids" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u | tr '\n' ' ')

if [ -z "$(echo "$pids" | tr -d ' ')" ]; then
  echo "[stop] nothing to stop"
else
  echo "[stop] tree: $pids"
  for p in $pids; do kill -TERM "$p" 2>/dev/null; done
  sleep 15
  for p in $pids; do
    kill -0 "$p" 2>/dev/null && { echo "[stop] force-kill $p"; kill -9 "$p" 2>/dev/null; }
  done
fi

# Poll the driver: booting before memory is released is what caused failure (1).
echo "[stop] waiting for cards 8-15 to drain"
for i in $(seq 1 30); do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
         | awk -F', ' '$1>=8 && $2>2000 {c++} END{print c+0}')
  [ "$busy" = "0" ] && { echo "[stop] drained after ${i}0s"; break; }
  sleep 10
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | sed -n '9,16p' | tr '\n' ' '
echo
# Confirm nothing still serves the port (failure mode 2).
code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "http://localhost:$PORT/health" 2>/dev/null)
[ "$code" = "200" ] && echo "[stop] WARNING: something still answers :$PORT" || echo "[stop] port $PORT clear"
