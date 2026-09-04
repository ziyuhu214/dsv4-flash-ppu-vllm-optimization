#!/usr/bin/env python3
"""Locate GPU idle bubbles in a decode trace and attribute them.

For each gap in the union-of-busy timeline, records the kernel that ended just
before it and the one that starts just after, then aggregates gaps by that
(predecessor -> successor) pair. The pairs with the largest total idle time are
where the decode step is losing wall-clock to stalls rather than to compute.

Usage: analyze_gaps.py <trace.json.gz> [--iters N] [--min-us F] [--top N]
"""
import gzip
import json
import sys
from collections import defaultdict

path = sys.argv[1]
iters, min_us, top = 12, 2.0, 25
for i, a in enumerate(sys.argv):
    if a == "--iters":
        iters = int(sys.argv[i + 1])
    if a == "--min-us":
        min_us = float(sys.argv[i + 1])
    if a == "--top":
        top = int(sys.argv[i + 1])

with gzip.open(path, "rt") as f:
    trace = json.load(f)

KERNEL_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
evs = []
for e in trace["traceEvents"]:
    if e.get("cat") in KERNEL_CATS and "ts" in e and "dur" in e:
        evs.append((e["ts"], e["ts"] + e["dur"], e["name"]))
evs.sort()

# Sweep the union timeline, recording gaps.
gaps = []
cur_e = evs[0][1]
last_name = evs[0][2]
for s, e, n in evs[1:]:
    if s > cur_e:
        gaps.append((s - cur_e, last_name, n, cur_e))
    if e > cur_e:
        cur_e = e
        last_name = n

total_gap = sum(g[0] for g in gaps)
print(f"trace: {path}")
print(f"total idle: {total_gap/1000/iters:.3f} ms/step across {len(gaps)/iters:.0f} gaps/step\n")

agg = defaultdict(lambda: [0.0, 0])
for dur, prev, nxt, _ in gaps:
    if dur < min_us:
        continue
    k = (prev[:52], nxt[:52])
    agg[k][0] += dur
    agg[k][1] += 1

shown = sum(v[0] for v in agg.values())
print(f"gaps >= {min_us}us account for {shown/1000/iters:.3f} ms/step "
      f"({100*shown/total_gap:.0f}% of idle)\n")
print(f"{'ms/step':>8} {'n/step':>7} {'us/gap':>8}  after -> before")
for (prev, nxt), (t, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:top]:
    print(f"{t/1000/iters:>8.3f} {c/iters:>7.1f} {t/c:>8.1f}  {prev}\n{'':>26}-> {nxt}")

# Also: distribution of gap sizes
print("\n=== gap size histogram ===")
buckets = [(0, 2), (2, 5), (5, 10), (10, 25), (25, 50), (50, 100), (100, 1e9)]
for lo, hi in buckets:
    sel = [g[0] for g in gaps if lo <= g[0] < hi]
    if sel:
        label = f"{lo}-{hi}us" if hi < 1e9 else f">{lo}us"
        print(f"{label:>12}: {len(sel)/iters:>7.1f} gaps/step  "
              f"{sum(sel)/1000/iters:>7.3f} ms/step")
