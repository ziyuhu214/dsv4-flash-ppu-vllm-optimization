#!/bin/bash
# T-Head baseline stack (vLLM 0.20.1+ppu) on cards 8-15, with profiler enabled.
#
# Runs the archived T-Head vLLM WITHOUT touching the installed 0.24.0+empty:
#   /workspace/vllm-thead-ref is the vllm/ package dir itself, so a symlink
#   /tmp/thead_stack/vllm -> it, plus PYTHONPATH=/tmp/thead_stack, makes `import
#   vllm` resolve to 0.20.1. The console script `vllm` belongs to 0.24.0, so we
#   invoke T-Head's own CLI module directly.
#
# VLLM_PLUGINS= (empty) is essential: vllm-plugin-fl is registered as an
# entrypoint and would otherwise inject itself into 0.20.1.
#
# Flags mirror serve_dsv4_flash_int8.sh, the script that produced the archived
# baseline (1719.78 tok/s, TPOT 33.36 ms) — notably NO cudagraph/compile flags,
# i.e. the fork's own defaults. Only --profiler-config.* is added.
set -u
export CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export PYTHONPATH=/tmp/thead_stack
export VLLM_PLUGINS=
mkdir -p /tmp/thead_stack
ln -sfn /workspace/vllm-thead-ref /tmp/thead_stack/vllm

PROF_DIR=${PROF_DIR:-/workspace/pytorch/prof_thead}
mkdir -p "$PROF_DIR"
MODEL=/mnt/cpfs/models/DeepSeek-V4-Flash-0731-Quant-W-INT8-PerChannel-A-INT8-PerToken
LOG=${LOG:-/workspace/pytorch/serve_thead.log}
echo "[serve-thead] cards=$CUDA_VISIBLE_DEVICES prof=$PROF_DIR log=$LOG"
python3 -m vllm.entrypoints.cli.main serve "$MODEL" \
  --tensor-parallel-size 8 \
  --max-model-len 65536 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 32768 \
  --chat-template /workspace/pytorch/chat_template/chat_template_deepseekv4.jinja \
  --trust-remote-code \
  --no-enable-log-requests \
  --no-enable-prefix-caching \
  --profiler-config.profiler torch \
  --profiler-config.torch_profiler_dir "$PROF_DIR" \
  --profiler-config.torch_profiler_with_stack false \
  --profiler-config.ignore_frontend true \
  --profiler-config.max_iterations 12 \
  --port 8011 \
  > "$LOG" 2>&1
