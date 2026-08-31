#!/bin/bash
# Rerun only the 16384 case (4 runs) after the sqlite lock fix.
# Mirrors benchmark_throughput_serve.py's exact command for case (16384,1024,64,256).
set -u
LOG=/workspace/pytorch/bench_16384_rerun.log
: > "$LOG"
for i in 1 2 3 4; do
  echo "===== Run $i/4 =====" | tee -a "$LOG"
  vllm bench serve \
    --backend vllm \
    --model /mnt/cpfs/models/DeepSeek-V4-Flash-0731-Quant-W-INT8-PerChannel-A-INT8-PerToken \
    --endpoint /v1/completions \
    --host localhost --port 8011 \
    --dataset-name random --ignore-eos \
    --random-input-len 16384 --random-output-len 1024 \
    --max-concurrency 64 --num-prompts 256 \
    2>&1 | tee -a "$LOG"
done
echo "ALL RUNS DONE" | tee -a "$LOG"