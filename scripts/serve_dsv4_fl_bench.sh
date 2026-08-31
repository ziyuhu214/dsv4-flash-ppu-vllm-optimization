#!/bin/bash
# Benchmark serve: DeepSeek-V4-Flash INT8 on official vLLM 0.24.0+empty + plugin-FL + FlagGems.
# Flags mirror the T-Head-stack baseline run (serve_dsv4_flash_int8.sh) for comparison:
# TP=8, max-model-len 65536, fp8 KV, util 0.85, batch budget 32768,
# no prefix caching / no request logs.

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL=/mnt/cpfs/models/DeepSeek-V4-Flash-0731-Quant-W-INT8-PerChannel-A-INT8-PerToken
LOG=/workspace/pytorch/vllm_dsv4_fl_bench_serve.log

vllm serve "$MODEL" \
  --tensor-parallel-size 8 \
  --max-model-len 65536 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 32768 \
  --chat-template /workspace/pytorch/chat_template/chat_template_deepseekv4.jinja \
  --trust-remote-code \
  --no-enable-log-requests \
  --no-enable-prefix-caching \
  --compilation-config.cudagraph_mode FULL_AND_PIECEWISE \
  --compilation-config.cudagraph_capture_sizes [1,2,4,8,12,16,24,32,40,48,56,64,96,128,192,256,320,384,448,512] --compilation-config.compile_sizes [1,2,4] \
  --port 8011 \
  2>&1 | tee "$LOG"