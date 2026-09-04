#!/usr/bin/env python3
"""Critical-path analysis of a vLLM decode trace.

Summed kernel time overcounts when kernels run concurrently on multiple streams,
so it is the wrong basis for "where can I save time". This computes, per GPU
stream: busy time, and across all streams: the UNION of busy intervals (real GPU
wall time) and the idle gaps. Savings on a kernel only help if that kernel is on
the busy-union critical path and its stream is the bottleneck.

Usage: analyze_timeline.py <trace.json.gz> [--iters N]
"""
import gzip
import json
import sys
from collections import defaultdict

path = sys.argv[1]
iters = 12
for i, a in enumerate(sys.argv):
    if a == "--iters":
        iters = int(sys.argv[i + 1])

with gzip.open(path, "rt") as f:
    trace = json.load(f)
events = trace["traceEvents"]

KERNEL_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
per_stream = defaultdict(list)  # stream -> [(start, end, name)]
for e in events:
    if e.get("cat") in KERNEL_CATS and "ts" in e and "dur" in e:
        st = e.get("args", {}).get("stream", e.get("tid"))
        per_stream[st].append((e["ts"], e["ts"] + e["dur"], e["name"]))

if not per_stream:
    print("no GPU kernel events found")
    sys.exit(1)


def union_busy(intervals):
    """Total length of the union of [start,end) intervals, plus span."""
    iv = sorted(intervals)
    total = 0.0
    cur_s, cur_e = iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    total += cur_e - cur_s
    return total, iv[0][0], max(e for _, e in iv)


print(f"trace: {path}")
print(f"iterations: {iters}\n")

print("=== per-stream busy time ===")
print(f"{'stream':>10} {'busy ms/step':>13} {'kernels/step':>13} {'top kernel':<50}")
all_iv = []
for st, evs in sorted(per_stream.items(), key=lambda kv: -sum(e - s for s, e, _ in kv[1])):
    iv = [(s, e) for s, e, _ in evs]
    all_iv.extend(iv)
    busy, _, _ = union_busy(iv)
    by_name = defaultdict(float)
    for s, e, n in evs:
        by_name[n] += e - s
    topn = max(by_name.items(), key=lambda kv: kv[1])[0][:48]
    print(f"{str(st):>10} {busy/1000/iters:>13.3f} {len(evs)/iters:>13.1f} {topn:<50}")

busy_all, t0, t1 = union_busy(all_iv)
span = t1 - t0
summed = sum(e - s for s, e in all_iv)
print()
print(f"summed kernel time : {summed/1000/iters:>8.3f} ms/step  (overcounts concurrency)")
print(f"GPU busy (union)   : {busy_all/1000/iters:>8.3f} ms/step  <- real GPU wall time")
print(f"trace span         : {span/1000/iters:>8.3f} ms/step")
print(f"GPU idle in span   : {(span-busy_all)/1000/iters:>8.3f} ms/step "
      f"({100*(span-busy_all)/span:.1f}% of span)")
print(f"concurrency factor : {summed/busy_all:>8.2f}x")
