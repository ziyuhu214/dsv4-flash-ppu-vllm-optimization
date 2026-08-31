#!/bin/bash
# DeepSeek-V4-Flash-0731 W8A8-INT8 (per-channel weight / per-token activation) local vLLM serve
# Cards: 0-7 (8-15 are occupied by another container's job as of 2026-08-26; 0-7 are free).
# Notes:
#  - USE_FLAGGEMS not set: flag_gems is not installed in this container; DeepSeek-V4
#    uses PPU kernels (PPUInt8ScaledMMLinearKernel + FlashMLA sparse) selected automatically.
#  - --kv-cache-dtype fp8 is mandatory: vLLM DeepseekV4MLAAttention asserts fp8 KV cache
#    (auto-converts to fp8_ds_mla internally).
#  - chat template is custom-built, verified char-identical to the official
#    encoding/encoding_dsv4.py on chat+thinking multi-turn cases.

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL=/mnt/cpfs/models/DeepSeek-V4-Flash-0731-Quant-W-INT8-PerChannel-A-INT8-PerToken
LOG=/workspace/pytorch/vllm_dsv4_flash_int8.log

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
  --port 8011 \
  2>&1 | tee "$LOG"
