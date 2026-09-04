#!/bin/bash
# All three reference cases, 4 runs each, incremental logging.
#
# Params taken from the project's own scripts (bench_case1.sh, rerun_cases23.sh)
# so the numbers are comparable to the archived reference results:
#   case1  1024/1024   c64  256 prompts
#   case2  4096/1024   c64  256 prompts
#   case3 16384/1024   c64  256 prompts
#
# Logs after every run, so a failure in case 3 (the ~36 min one) does not lose
# cases 1 and 2. Also records other-card utilisation per run: another tenant
# shares this host's CPU and DSv4 decode spends real time in host-side
# prepare_inputs, so contention shows up in the numbers.
#
# Arg1 = output log.
set -u
M=/mnt/cpfs/models/DeepSeek-V4-Flash-0731-Quant-W-INT8-PerChannel-A-INT8-PerToken
LOG=${1:?usage: bench_all_cases.sh <logfile>}
RUNS=${RUNS:-4}
: > "$LOG"

for CASE in "1024 1024" "4096 1024" "16384 1024"; do
  set -- $CASE
  IN=$1; OUT=$2
  for i in $(seq 1 "$RUNS"); do
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null \
           | sed -n '9,16p' | tr -d ' %' | sort -u | paste -sd,)
    echo "===== ${IN}/${OUT} run ${i}/${RUNS}  (other-cards util: ${util}) =====" | tee -a "$LOG"
    vllm bench serve --backend vllm --model "$M" \
      --endpoint /v1/completions --host localhost --port 8011 \
      --dataset-name random --ignore-eos \
      --random-input-len "$IN" --random-output-len "$OUT" \
      --max-concurrency 64 --num-prompts 256 \
      2>&1 | grep -E "Successful requests|Benchmark duration|Output token throughput|Total token throughput|Mean TTFT|Mean TPOT" \
      | tee -a "$LOG"
  done
  echo "----- ${IN}/${OUT} complete -----" | tee -a "$LOG"
done
echo "ALL CASES DONE" | tee -a "$LOG"
