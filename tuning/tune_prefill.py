#!/usr/bin/env python3
"""Prefill-M tuning shard for DSv4 shapes. Args: shard nshards"""
import json, os, sys
from tune_dsv4_deepgemm import tune_gemm_config

shard, nshards = int(sys.argv[1]), int(sys.argv[2])
OUT = f"/workspace/pytorch/tune_out/prefill_shard_{shard}.json"
SHAPES = [
    (4096, 4096, 32), (2048, 4096, 32),
    (4096, 4096, 33), (2048, 4096, 33),
    (4096, 4096, 1), (2048, 4096, 1),
]
MS = [1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768]
cases = [(m, k, n, g) for (k, n, g) in SHAPES for m in MS]
mine = [c for i, c in enumerate(cases) if i % nshards == shard]
tuned, results = {}, []
print(f"shard {shard}: {len(mine)} cases", flush=True)
for i, (m, k, n, g) in enumerate(mine):
    nopad = g > 1
    try:
        best = tune_gemm_config(m, k, n, g, nopad, "int8", tuned)
    except Exception as e:
        print(f"[{i+1}] M={m} g={g} ERR {str(e)[:60]}", flush=True); continue
    if best is not None:
        results.append(best)
        print(f"[{i+1}/{len(mine)}] M={m} K={k} N={n} g={g}: acc={best.get('acc',0):.2%}", flush=True)
    else:
        print(f"[{i+1}/{len(mine)}] M={m} K={k} N={n} g={g}: baseline", flush=True)
    json.dump(results, open(OUT, "w"))
print("done", flush=True)
