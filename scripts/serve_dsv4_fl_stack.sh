#!/bin/bash
# DeepSeek-V4-Flash INT8 on official vLLM 0.24.0+empty + vllm-plugin-FL(main) + FlagGems v5.3.4
# Cards 0-7 fully free -> gpu_mem_util 0.85.

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_LOGGING_LEVEL=DEBUG

MODEL=/mnt/cpfs/models/DeepSeek-V4-Flash-0731-Quant-W-INT8-PerChannel-A-INT8-PerToken
LOG=/workspace/pytorch/vllm_dsv4_fl_serve.log

vllm serve "$MODEL" \
  --tensor-parallel-size 8 \
  --served-model-name deepseek-v4-flash-int8 \
  --max-model-len 8192 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 8192 \
  --chat-template /workspace/pytorch/chat_template/chat_template_deepseekv4.jinja \
  --trust-remote-code \
  --enforce-eager \
  --port 8011 \
  2>&1 | tee "$LOG"
