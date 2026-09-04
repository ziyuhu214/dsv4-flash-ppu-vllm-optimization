#!/bin/bash
# Case1 (1024/1024, c64, 256 prompts) x4 runs — same params as
# benchmarks/benchmark_throughput_serve.py's first case. Arg1 = output log.
set -u
M=/mnt/cpfs/models/DeepSeek-V4-Flash-0731-Quant-W-INT8-PerChannel-A-INT8-PerToken
LOG=${1:?usage: bench_case1.sh <logfile>}
: > "$LOG"
for i in 1 2 3 4; do
  echo "===== Run $i/4 =====" | tee -a "$LOG"
  vllm bench serve --backend vllm --model "$M" \
    --endpoint /v1/completions --host localhost --port 8011 \
    --dataset-name random --ignore-eos \
    --random-input-len 1024 --random-output-len 1024 \
    --max-concurrency 64 --num-prompts 256 \
    2>&1 | grep -E "Successful requests|Output token throughput|Total token throughput|Mean TTFT|Mean TPOT" | tee -a "$LOG"
done
echo "DONE" | tee -a "$LOG"
