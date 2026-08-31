#!/bin/bash
# Rerun cases 2 and 3 (4096 and 16384 input) after util adjustment.
set -u
M=/mnt/cpfs/models/DeepSeek-V4-Flash-0731-Quant-W-INT8-PerChannel-A-INT8-PerToken
LOG=/workspace/pytorch/bench_cases23_final.log
: > "$LOG"
for CASE in "4096 1024" "16384 1024"; do
  set -- $CASE
  IN=$1; OUT=$2
  for i in 1 2 3 4; do
    echo "===== ${IN}/${OUT} Run $i/4 =====" | tee -a "$LOG"
    vllm bench serve --backend vllm --model "$M" \
      --endpoint /v1/completions --host localhost --port 8011 \
      --dataset-name random --ignore-eos \
      --random-input-len "$IN" --random-output-len "$OUT" \
      --max-concurrency 64 --num-prompts 256 \
      2>&1 | grep -E "Successful requests|Benchmark duration|Output token throughput|Total token throughput|Request throughput|Mean TTFT|P99 TTFT|Mean TPOT|P99 TPOT" | tee -a "$LOG"
  done
done
echo "ALL DONE" | tee -a "$LOG"