#!/bin/bash
# Ladder experiment: 20 capture sizes on TP=4 (cards 2-5; 0/1 + 8-15 are occupied).
# Goal: compare profiling ESTIMATE vs ACTUAL post-capture memory to determine
# whether the 51-size OOM was mis-estimation or real pool-reuse failure.

export CUDA_VISIBLE_DEVICES=2,3,4,5

MODEL=/mnt/cpfs/models/DeepSeek-V4-Flash-0731-Quant-W-INT8-PerChannel-A-INT8-PerToken
LOG=/workspace/pytorch/vllm_ladder_tp4.log

vllm serve "$MODEL" \
  --tensor-parallel-size 4 \
  --max-model-len 16384 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.92 \
  --max-num-batched-tokens 16384 \
  --chat-template /workspace/pytorch/chat_template/chat_template_deepseekv4.jinja \
  --trust-remote-code \
  --no-enable-log-requests \
  --no-enable-prefix-caching \
  --compilation-config.cudagraph_mode FULL_AND_PIECEWISE \
  --compilation-config.cudagraph_capture_sizes [1,2,4,8,12,16,24,32,40,48,56,64,96,128,192,256,320,384,448,512] \
  --port 8011 \
  2>&1 | tee "$LOG"