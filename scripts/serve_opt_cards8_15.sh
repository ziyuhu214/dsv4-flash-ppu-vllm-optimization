#!/bin/bash
# DSv4-Flash INT8 TP=8 on cards 8-15 (cards 0-1 are in use by another container).
# Baseline == the config that measured 1486.9 tok/s on 09-02:
#   VLLM_FL_EMPTY_INT_ZERO=0 (global int zero-fill override off)
# Profiler dir is set so /start_profile works.
# Env overrides accepted from caller: VLLM_FL_EMPTY_INT_ZERO, VLLM_FL_EMPTY_AUDIT,
# VLLM_TORCH_PROFILER_DIR, LOG.
# Card group is overridable: CARDS=0,1,2,3,4,5,6,7 bash serve_opt_cards8_15.sh
# NOTE for comparisons: every baseline in this project's notes (baseline 1481.4,
# T-Head 1726.5) was measured on cards 8-15. Numbers from a different card group
# are not strictly apples-to-apples with those.
export CUDA_VISIBLE_DEVICES=${CARDS:-8,9,10,11,12,13,14,15}
export VLLM_FL_DSV4_TORCH_COMPILE=1
# Route cheap memory-movement / elementwise ops back to native ATen. FlagGems
# implements these as triton kernels, so each call pays Python dispatch + JIT
# cache lookup + launch where native ATen issues a memcpy/memset. Measured
# host-side cost per call on this host (isolated, one op per process, median of
# 7 batches of 200):
#     copy_      58.44 us -> 4.24  (13.8x)      clamp_    55.06 -> 4.19 (13.1x)
#     clamp_min  69.43    -> 6.17  (11.3x)      cat       41.11 -> 6.66 ( 6.2x)
#     arange     25.10    -> 9.07  ( 2.8x)
# This is what the decode profile saw as GPU idle: FL spent 18.4 ms/step of CPU
# inside aten calls vs the T-Head baseline's 4.7 ms at matching call counts, and
# that baseline container had FlagGems absent entirely.
# DELIBERATELY EXCLUDED: zero_ measured 15.42 us on FlagGems vs 55.97 native
# (native is 3.6x SLOWER — the PPU torch backend's zero_ path is poor), and
# zeros / zeros_like / _to_copy came out neutral. So this is not "turn FlagGems
# off"; it is only the ops where native measurably wins.
#
# The keys are FlagGems FUNCTION names, not aten op names: ("arange.start",
# arange_start) means the key is `arange_start`. A first attempt used dotted op
# names and was silently half-effective — arange.start / clamp.Tensor / cat.out
# kept running on FlagGems (measured 0.99x, i.e. no change) until the function
# names were used, at which point they became 2.8x / 15.6x / 6.6x.
# Numerical equivalence verified by verify_native_ops.py: 28/28, covering
# non-contiguous sources, dtype-converting copies, empty tensors, tensor-valued
# clamp bounds, in-place variants, cat.out and strided arange.
# The list itself lives in the platform config, vllm_fl/dispatch/config/thead.yaml
# (flagos_blacklist), so it applies on every launch with no env var — one source
# of truth. Export VLLM_FL_FLAGOS_BLACKLIST here only to override for an A/B;
# note that the env var takes PRECEDENCE over the yaml, so a stale env list
# silently masks the correct default (that happened: an env list spelled with
# dotted overload names left arange_start running on FlagGems).
# =1 (default) is REQUIRED: =0 faults with an illegal memory access in the KV
# compressor triton kernel (deepseek_v4_compress_cache.py) reading garbage index
# buffers. =1 costs nothing measurable (verified 09-03: 1481 tok/s).
export VLLM_FL_EMPTY_INT_ZERO=${VLLM_FL_EMPTY_INT_ZERO:-1}
# vLLM 0.24.0 dropped VLLM_TORCH_PROFILER_DIR in favour of --profiler-config.
PROF_DIR=${PROF_DIR:-/workspace/pytorch/profiles_c815}
mkdir -p "$PROF_DIR"
MODEL=/mnt/cpfs/models/DeepSeek-V4-Flash-0731-Quant-W-INT8-PerChannel-A-INT8-PerToken
LOG=${LOG:-/workspace/pytorch/serve_c815.log}
echo "[serve] cards=$CUDA_VISIBLE_DEVICES EMPTY_INT_ZERO=$VLLM_FL_EMPTY_INT_ZERO prof=$PROF_DIR log=$LOG"
echo "[serve] flaggems blacklist=${VLLM_FL_FLAGOS_BLACKLIST:-from thead.yaml (13 keys)}"
echo "[serve] q head padding=${VLLM_FL_Q_HEAD_PADDING:-disabled (padded_heads=n_local_heads)}"
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
  --profiler-config.profiler torch \
  --profiler-config.torch_profiler_dir "$PROF_DIR" \
  --profiler-config.torch_profiler_with_stack false \
  --profiler-config.ignore_frontend true \
  --profiler-config.max_iterations 12 \
  --port 8011 \
  > "$LOG" 2>&1
