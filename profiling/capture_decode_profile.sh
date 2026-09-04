#!/bin/bash
# Capture a torch profile during STEADY-STATE DECODE only.
#
# 64 prompts @ c64: all 64 requests prefill together, then decode together for
# ~1024 steps (~39 s). We arm the profiler ~WAIT s in, well past prefill.
# The server is started with --profiler-config.max_iterations 12, so the capture
# covers exactly 12 engine steps (~0.45 s of decode) and auto-stops — small
# trace, no prefill contamination.
#
# Per the porting report's lesson: light-load/warmup windows give misleading
# communication ratios, so this window is deliberately mid-decode at full batch.
set -u
M=/mnt/cpfs/models/DeepSeek-V4-Flash-0731-Quant-W-INT8-PerChannel-A-INT8-PerToken
WAIT=${WAIT:-20}
NPROMPTS=${NPROMPTS:-64}
CONC=${CONC:-64}
vllm bench serve --backend vllm --model "$M" \
  --endpoint /v1/completions --host localhost --port 8011 \
  --dataset-name random --ignore-eos \
  --random-input-len 1024 --random-output-len 1024 \
  --max-concurrency "$CONC" --num-prompts "$NPROMPTS" \
  > /tmp/profile_driver_bench.log 2>&1 &
BENCH=$!
sleep "$WAIT"
echo "[profile] arming profiler at t=${WAIT}s (expect mid-decode, batch=${CONC})"
curl -s -X POST http://localhost:8011/start_profile -o /dev/null -w "  start:%{http_code}\n" --max-time 15

# DO NOT also call /stop_profile here. The server runs with
# --profiler-config.max_iterations 12, so the profiler stops and flushes ITSELF
# after 12 engine steps. Calling /stop_profile afterwards issued a second
# stop, which wrote a duplicate trace set (two dirs 38 s apart from one capture)
# and then tore the profiler subsystem down for the rest of the server's life —
# the next /start_profile could not even connect (curl 000) and produced no
# trace at all. Just wait for the flush.
echo "[profile] waiting for self-stop after max_iterations + flush"
sleep 25
wait $BENCH
grep -E "Output token throughput|Mean TPOT" /tmp/profile_driver_bench.log
