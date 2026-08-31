#!/usr/bin/env python3
"""Autotune PPU deep_gemm int8 configs for DeepSeek-V4-Flash @ TP8.

Uses deep_gemm's bundled tuner (DEBUG single-GPU mode, no ray) on the exact
GEMM shapes this deployment runs:

MoE grouped GEMMs (num_groups = local experts = 256/8 = 32, and 33 for the
+shared-expert fusion case, matching the vendor script's convention):
  up/gate: K=hidden(4096)      -> N=2*moe_inter(4096)
  down:    K=moe_inter(2048)   -> N=hidden(4096)

Dense int8 GEMMs (num_groups=1) for attention/linear layers @ TP8:
  wq_a+wkv fused: K=4096 -> N=q_lora(1024)+head(512)+rope(64)=1600 (approx,
      actual fused width read from the checkpoint would be better, but the
      dominant dense GEMMs below matter most)
  wq_b:  K=1024 -> N=heads_local(8)*head_dim(512)=4096
  wo_b:  K=o_lora(1024)*o_groups... -> use N=4096 K=1024 (per-group)
  shared expert up/gate: K=4096 -> N=4096 ; down: K=2048 -> N=4096

M sweep: tuner's CANDIDATE_Ms for decode region; prefill handled by
M > MAX_DECODE_BS fallback (no config -> heuristic), so we also tune a few
large-M points the closest-match logic can use.
"""
import json
import os
import sys
import torch

from deep_gemm.deep_gemm_tuner.autotune_deepgemm import (
    tune_gemm_config,
    load_tuned_configs,
)
from deep_gemm.deep_gemm_tuner.utils import CANDIDATE_Ms

OUT = "/workspace/vllm-plugin-FL/vllm_fl/ops/ppu_deepgemm_configs/DeepSeek-V4-Flash-0731-w8a8-int8-tp8,device_name=PPU-ZW810E-deepgemm_configs.json"

HIDDEN = 4096
MOE_INTER = 2048
LOCAL_EXPERTS = 256 // 8  # TP8/EP8 -> 32

# (K, N, num_groups) triples
SHAPES = [
    # MoE grouped (nopad path used by PPUDeepGemmExperts)
    (HIDDEN, 2 * MOE_INTER, LOCAL_EXPERTS),      # up/gate  4096 -> 4096, g=32
    (MOE_INTER, HIDDEN, LOCAL_EXPERTS),          # down     2048 -> 4096, g=32
    (HIDDEN, 2 * MOE_INTER, LOCAL_EXPERTS + 1),  # +shared fusion, g=33
    (MOE_INTER, HIDDEN, LOCAL_EXPERTS + 1),
    # Dense (acext handles most of these now, but deepgemm dense may be used;
    # cheap to tune, keep the two dominant ones)
    (HIDDEN, 2 * MOE_INTER, 1),
    (MOE_INTER, HIDDEN, 1),
]

# decode-region Ms (the runtime lookup snaps to closest CANDIDATE_M);
# thin the sweep to keep wall time sane: powers of two + 48/96/... up to 1024
MS = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
# a few prefill sizes (M > MAX_DECODE_BS uses exact match only)
MS_PREFILL = [2048, 4096, 8192]

def main():
    assert all(m in CANDIDATE_Ms or m > 1024 for m in MS + MS_PREFILL), \
        "decode Ms must be in CANDIDATE_Ms for closest-match to hit"
    tuned = load_tuned_configs(OUT) if os.path.exists(OUT) else {}
    results = [v for v in tuned.values()]
    cases = [(m, k, n, g) for (k, n, g) in SHAPES for m in MS + MS_PREFILL]
    print(f"{len(cases)} cases, {len(tuned)} already tuned")
    for i, (m, k, n, g) in enumerate(cases):
        nopad = g > 1  # grouped path is nopad; dense is regular
        try:
            best = tune_gemm_config(m, k, n, g, nopad, "int8", tuned)
        except Exception as e:
            print(f"[{i+1}/{len(cases)}] M={m} K={k} N={n} g={g} ERROR: {e}")
            continue
        if best is not None:
            key = (m, k, n, g, nopad)
            if key not in tuned:
                tuned[key] = best
                results.append(best)
            print(f"[{i+1}/{len(cases)}] M={m} K={k} N={n} g={g}: "
                  f"acc={best.get('acc', 0):.2%}")
        else:
            print(f"[{i+1}/{len(cases)}] M={m} K={k} N={n} g={g}: baseline best")
        # incremental save
        with open(OUT, "w") as f:
            json.dump(results, f)
    print(f"saved {len(results)} configs -> {OUT}")

if __name__ == "__main__":
    main()
