#!/usr/bin/env python3
"""Final T-Head vs FL decode attribution, with the traps corrected.

Corrections applied vs a naive name-based diff:
  * T-Head routes dense int8 linears through the SAME deep_gemm GemmKernel as
    the MoE grouped GEMMs, so split them by per-kernel median duration: the MoE
    up/gate GEMM is ~123 us at 43/step, dense ones are <35 us.
  * comm is EXCLUDED from the gap. Medians are identical (~22 us) on all 8 ranks
    of both stacks; the totals differ only through multi-ms tail outliers where
    allreduce waits on a straggler. T-Head's outliers are in fact far larger
    (27.5 ms vs 5.2 ms) while T-Head is faster overall, so counting them inverts
    the conclusion.
  * Uses GPU busy union (not summed kernel time) for the headline, since decode
    runs on 3-4 concurrent streams.

Usage: final_attribution.py <thead.gz> <fl.gz> [--iters N]
"""
import gzip
import json
import sys
from collections import defaultdict
import statistics as st

a_path, b_path = sys.argv[1], sys.argv[2]
iters = 12
for i, x in enumerate(sys.argv):
    if x == "--iters":
        iters = int(sys.argv[i + 1])

KCATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def load(p):
    with gzip.open(p, "rt") as f:
        tr = json.load(f)
    per = defaultdict(list)
    iv = []
    for e in tr["traceEvents"]:
        if e.get("cat") in KCATS and "dur" in e:
            per[e["name"]].append(e["dur"])
            iv.append((e["ts"], e["ts"] + e["dur"]))
    return per, iv


def union(iv):
    iv = sorted(iv)
    tot = 0.0
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce:
            tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + ce - cs


def role(name, med):
    s = name.lower()
    if "batched_gemvt_kernel_small_k" in s:
        return "MoE grouped GEMM down"
    if "deep_gemm" in s and "gemmkernel" in s:
        return "MoE grouped GEMM up/gate" if med > 100 else "dense GEMM (deep_gemm)"
    if "deep_gemm" in s and "prenorm" in s:
        return "mHC prenorm GEMM"
    if "per_token_group_quant_int8" in s:
        return "quant (per-token-group)"
    if "per_token_quant_int8" in s or "dynamic_scaled_int8_quant" in s:
        return "quant (per-token)"
    if "ep_gather" in s or "ep_scatter" in s:
        return "MoE gather/scatter"
    if "moe_align" in s or "expert_num_tokens" in s or "count_expert" in s:
        return "MoE align/count"
    if "silu" in s:
        return "MoE activation"
    if "softplus" in s:
        return "MoE routing"
    if "flash_sparse_decode" in s:
        return "ATTN sparse decode  <- head padding"
    if "splitkv_mla_combine" in s:
        return "ATTN splitkv combine <- head padding"
    if "mqa_logits" in s or "indexer" in s:
        return "indexer"
    if "topk" in s:
        return "indexer topk"
    if "qnorm" in s or "kv_insert" in s or "kv_compress" in s or "compress" in s:
        return "KV insert/compress"
    if "gather_k" in s or "dequant" in s:
        return "KV dequant/gather"
    if "mhc" in s or "hc_prenorm" in s:
        return "mHC (tilelang)"
    if "pccl" in s or "nccl" in s:
        return "__COMM (excluded)"
    if "gemmwithepiloguevisitor" in s or "gemm_ktype" in s or "linear_kernel" in s \
            or "mm_kernel" in s or ("cutlass" in s and "device_kernel" in s):
        return "dense GEMM (vendor/acext)"
    if "zeros_kernel" in s or "fill" in s or "empty" in s or "memset" in s:
        return "alloc zero/fill"
    if "copy" in s or "memcpy" in s:
        return "copy"
    if "rope" in s:
        return "rope"
    if "norm" in s:
        return "norm"
    if "slot_mapping" in s or "swa_indices" in s or "block_info" in s or "metadata" in s:
        return "metadata build"
    return "other small"


A, Aiv = load(a_path)
B, Biv = load(b_path)
ra, rb = defaultdict(lambda: [0.0, 0]), defaultdict(lambda: [0.0, 0])
for per, acc in ((A, ra), (B, rb)):
    for k, ds in per.items():
        r = role(k, st.median(ds))
        acc[r][0] += sum(ds)
        acc[r][1] += len(ds)

ua, ub = union(Aiv) / 1000 / iters, union(Biv) / 1000 / iters
print(f"GPU busy union:  T-Head {ua:.3f}  FL {ub:.3f}  ->  FL is {ub-ua:+.3f} ms/step slower")
print(f"(measured TPOT:  T-Head 33.2 ms, FL 38.1 ms -> {38.1-33.2:+.1f} ms)")
print()

rows = []
for r in set(ra) | set(rb):
    at, ac = ra[r]
    bt, bc = rb[r]
    rows.append((bt / 1000 / iters - at / 1000 / iters, r,
                 at / 1000 / iters, ac / iters, bt / 1000 / iters, bc / iters))
rows.sort(key=lambda x: -x[0])
print("delta = FL - T-Head, ms/step  (positive = FL slower)")
print(f"{'delta':>7} {'role':<38} {'T-Head':>8} {'n':>6} {'FL':>8} {'n':>6}")
print("-" * 78)
for d, r, at, ac, bt, bc in rows:
    if abs(d) < 0.03:
        continue
    print(f"{d:>+7.3f} {r:<38} {at:>8.3f} {ac:>6.1f} {bt:>8.3f} {bc:>6.1f}")
print("-" * 78)
excl = sum(d for d, r, *_ in rows if not r.startswith("__COMM"))
print(f"{excl:>+7.3f} {'SUM excluding comm':<38}")

GROUP = {
    "MoE core": ["MoE grouped GEMM up/gate", "MoE grouped GEMM down"],
    "MoE surround": ["MoE gather/scatter", "MoE align/count", "MoE activation", "MoE routing"],
    "ATTENTION (head padding)": ["ATTN sparse decode  <- head padding",
                                 "ATTN splitkv combine <- head padding"],
    "dense GEMM": ["dense GEMM (deep_gemm)", "dense GEMM (vendor/acext)"],
    "mHC": ["mHC (tilelang)", "mHC prenorm GEMM"],
    "quant": ["quant (per-token)", "quant (per-token-group)"],
    "KV": ["KV insert/compress", "KV dequant/gather"],
    "indexer": ["indexer", "indexer topk"],
    "overhead": ["alloc zero/fill", "copy", "metadata build", "norm", "rope", "other small"],
}
print()
print(f"{'group':<26} {'T-Head':>8} {'FL':>8} {'delta':>8}")
print("-" * 54)
gr = []
for g, ks in GROUP.items():
    at = sum(ra[k][0] for k in ks) / 1000 / iters
    bt = sum(rb[k][0] for k in ks) / 1000 / iters
    gr.append((bt - at, g, at, bt))
for d, g, at, bt in sorted(gr, key=lambda x: -x[0]):
    print(f"{g:<26} {at:>8.3f} {bt:>8.3f} {d:>+8.3f}")
print("-" * 54)
print(f"{'TOTAL (excl. comm)':<26} {sum(x[2] for x in gr):>8.3f} "
      f"{sum(x[3] for x in gr):>8.3f} {sum(x[0] for x in gr):>+8.3f}")
