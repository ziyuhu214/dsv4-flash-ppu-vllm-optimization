#!/usr/bin/env python3
"""One shard of the DSv4 deep_gemm tuning sweep. Args: shard_id num_shards"""
import json
import os
import sys

from tune_dsv4_deepgemm import SHAPES, MS, MS_PREFILL, tune_gemm_config

shard, nshards = int(sys.argv[1]), int(sys.argv[2])
OUT = f"/workspace/pytorch/tune_out/shard_{shard}.json"
os.makedirs("/workspace/pytorch/tune_out", exist_ok=True)

cases = [(m, k, n, g) for (k, n, g) in SHAPES for m in MS + MS_PREFILL]
mine = [c for i, c in enumerate(cases) if i % nshards == shard]
tuned, results = {}, []
print(f"shard {shard}: {len(mine)} cases", flush=True)
for i, (m, k, n, g) in enumerate(mine):
    nopad = g > 1
    try:
        best = tune_gemm_config(m, k, n, g, nopad, "int8", tuned)
    except Exception as e:
        print(f"[{i+1}/{len(mine)}] M={m} K={k} N={n} g={g} ERROR: {str(e)[:80]}",
              flush=True)
        continue
    if best is not None:
        results.append(best)
        print(f"[{i+1}/{len(mine)}] M={m} K={k} N={n} g={g}: "
              f"acc={best.get('acc', 0):.2%}", flush=True)
    else:
        print(f"[{i+1}/{len(mine)}] M={m} K={k} N={n} g={g}: baseline best",
              flush=True)
    with open(OUT, "w") as f:
        json.dump(results, f)
print("done", flush=True)
