#!/bin/bash
# SUPERSEDED 2026-09-03 by serve_opt_cards8_15.sh. Two things here are wrong:
#
#  1. VLLM_TORCH_PROFILER_DIR does nothing in vLLM 0.24.0 (it only logs "Unknown
#     vLLM environment variable"). The profile routes are never registered, so
#     /start_profile returns 404 and no trace is written — which is why
#     profiles_new/ is empty. Use --profiler-config.profiler torch and
#     --profiler-config.torch_profiler_dir instead.
#  2. VLLM_FL_EMPTY_INT_ZERO=0 does NOT reproduce the 1486.9 tok/s run. It does
#     not run at all: it faults with an illegal memory access during graph
#     capture, in compress_norm_rope_store_triton_ppu
#     (vllm_fl/ops/deepseek_v4_compress_cache.py), which reads int32 block_table /
#     int64 slot_mapping / token_to_req_indices. The fast run used the default =1,
#     which costs nothing measurable.
#
# Kept only so the two wrong claims above are not repeated. Use
# serve_opt_cards8_15.sh.
export CUDA_VISIBLE_DEVICES=4,5,6,7,12,13,14,15
export VLLM_FL_DSV4_TORCH_COMPILE=1
export VLLM_FL_EMPTY_INT_ZERO=1
export VLLM_TORCH_PROFILER_DIR=/workspace/pytorch/profiles_new
mkdir -p "$VLLM_TORCH_PROFILER_DIR"
MODEL=/mnt/cpfs/models/DeepSeek-V4-Flash-0731-Quant-W-INT8-PerChannel-A-INT8-PerToken
LOG=/workspace/pytorch/serve_profile.log
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
  --compilation-config.cudagraph_capture_sizes [1,2,4,8,12,16,24,32,40,48,56,64,96,128,192,256,320,384,448,512] \
  --compilation-config.compile_sizes [1,2,4] \
  --port 8011 \
  > "$LOG" 2>&1
