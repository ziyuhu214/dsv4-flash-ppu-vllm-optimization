#!/usr/bin/env python3
"""Verify the native-ATen fallbacks are numerically identical to FlagGems.

Only the ops we intend to blacklist are checked, against CPU-computed references,
including the awkward cases: non-contiguous sources, dtype-converting copies,
empty tensors, negative/zero clamp bounds, multi-dim cat, strided arange.

The T-Head baseline ran with FlagGems entirely absent and was numerically correct
on PPU, so these paths are already proven in aggregate; this pins it per-op.

Usage: MODE=on|off python3 verify_native_ops.py
"""
import os

import torch

MODE = os.environ.get("MODE", "off")
import flag_gems  # noqa: E402

# NOTE: flag_gems.enable(unused=...) matches FUNCTION names, not aten op names.
# ("arange.start", arange_start) means the key is `arange_start`; passing
# "arange.start" silently matches nothing. A first pass used dotted op names and
# was half-effective — arange.start/clamp.Tensor/cat.out kept running on FlagGems.
BL = [
    "copy_",
    "clamp", "clamp_tensor", "clamp_", "clamp_tensor_",
    "clamp_max", "clamp_max_", "clamp_min", "clamp_min_",
    "cat", "cat_out",
    "arange", "arange_start",
]
if MODE == "off":
    flag_gems.enable(unused=BL)
else:
    flag_gems.enable()

dev = "cuda"
fails = []


def chk(name, got, ref, exact=True):
    if isinstance(ref, torch.Tensor):
        ref = ref.to(got.device)
        if got.shape != ref.shape:
            ok = False
        elif got.numel() == 0:
            # torch.equal on a 0-element tensor dies inside FlagGems
            # (ops/all.py: triton.cdiv(0, 0) -> ZeroDivisionError), which is a
            # FlagGems bug unrelated to the ops under test. Shape+dtype+emptiness
            # is the whole content of an empty tensor, so compare that instead.
            ok = got.dtype == ref.dtype
        else:
            ok = torch.equal(got, ref) if exact else torch.allclose(got, ref, atol=1e-6)
    else:
        ok = bool(got == ref)
    print(f"  {'OK  ' if ok else 'FAIL'} {name}")
    if not ok:
        fails.append(name)
        if isinstance(ref, torch.Tensor):
            d = (got.float() - ref.float()).abs()
            print(f"       max|diff|={d.max().item():.3e} at {d.argmax().item()}")


print(f"=== MODE={MODE} (blacklist={BL if MODE=='off' else 'none'}) ===")

print("copy_:")
for shape, dt in [((64,), torch.int32), ((64, 1024), torch.int32),
                  ((97, 3), torch.int64), ((128, 64), torch.bfloat16),
                  ((0,), torch.int32)]:
    src_c = torch.arange(int(torch.tensor(shape).prod()) or 0, dtype=torch.int64)
    src_c = src_c.reshape(shape).to(dt)
    src = src_c.to(dev)
    dst = torch.empty(shape, dtype=dt, device=dev)
    dst.copy_(src)
    chk(f"contig {tuple(shape)} {dt}", dst, src_c)

# non-contiguous source (slice with stride)
base = torch.arange(128, dtype=torch.int32).reshape(8, 16)
nc = base.to(dev)[:, ::2]
dst = torch.empty_like(nc)
dst.copy_(nc)
chk("non-contiguous src (stride 2)", dst, base[:, ::2])

# dtype-converting copy_
si = torch.arange(64, dtype=torch.int32)
d64 = torch.empty(64, dtype=torch.int64, device=dev)
d64.copy_(si.to(dev))
chk("dtype-converting i32->i64", d64, si.to(torch.int64))

print("clamp_ / clamp_min:")
c_cpu = torch.randint(-500, 500, (64, 1024), dtype=torch.int32)
g = c_cpu.to(dev).clone()
g.clamp_(min=0)
chk("clamp_(min=0) int32", g, c_cpu.clamp(min=0))
g2 = c_cpu.to(dev).clone()
g2.clamp_(min=-1)
chk("clamp_(min=-1) int32", g2, c_cpu.clamp(min=-1))
chk("clamp_min(0) int32", torch.clamp_min(c_cpu.to(dev), 0), c_cpu.clamp(min=0))
f_cpu = torch.randn(33, 7)
chk("clamp_min(0.0) float", torch.clamp_min(f_cpu.to(dev), 0.0),
    f_cpu.clamp(min=0.0), exact=False)
bt_cpu = torch.randint(-1, 500, (64, 1024), dtype=torch.int32)  # block_table shape
chk("block_table clamp_(min=0)", bt_cpu.to(dev).clone().clamp_(min=0),
    bt_cpu.clamp(min=0))

print("clamp variants (tensor bounds / max side):")
lo_cpu = torch.zeros_like(c_cpu)
chk("clamp.Tensor(min=tensor)", torch.clamp(c_cpu.to(dev), min=lo_cpu.to(dev)),
    torch.clamp(c_cpu, min=lo_cpu))
gt = c_cpu.to(dev).clone()
gt.clamp_(min=lo_cpu.to(dev))
chk("clamp_.Tensor in-place", gt, c_cpu.clamp(min=lo_cpu))
chk("clamp_max(100)", torch.clamp_max(c_cpu.to(dev), 100), c_cpu.clamp(max=100))
gm = c_cpu.to(dev).clone()
gm.clamp_max_(100)
chk("clamp_max_(100) in-place", gm, c_cpu.clamp(max=100))
gn = c_cpu.to(dev).clone()
gn.clamp_min_(0)
chk("clamp_min_(0) in-place", gn, c_cpu.clamp(min=0))
chk("clamp(min,max) both", torch.clamp(c_cpu.to(dev), -10, 10),
    c_cpu.clamp(-10, 10))

print("cat:")
x = torch.randn(12, 5)
chk("cat dim0", torch.cat([x.to(dev), x.to(dev)], 0), torch.cat([x, x], 0), exact=False)
chk("cat dim1", torch.cat([x.to(dev), x.to(dev)], 1), torch.cat([x, x], 1), exact=False)
_o = torch.empty(24, 5, device=dev)
torch.cat([x.to(dev), x.to(dev)], 0, out=_o)
chk("cat.out", _o, torch.cat([x, x], 0), exact=False)
i1 = torch.arange(6, dtype=torch.int32).reshape(2, 3)
i2 = torch.arange(6, 12, dtype=torch.int32).reshape(2, 3)
chk("cat int32", torch.cat([i1.to(dev), i2.to(dev)], 0), torch.cat([i1, i2], 0))
e = torch.empty(0, 3)
chk("cat with empty", torch.cat([x[:, :3].to(dev), e.to(dev)], 0),
    torch.cat([x[:, :3], e], 0), exact=False)

print("arange:")
chk("arange(64) i32", torch.arange(64, dtype=torch.int32, device=dev),
    torch.arange(64, dtype=torch.int32))
chk("arange(0) i32", torch.arange(0, dtype=torch.int32, device=dev),
    torch.arange(0, dtype=torch.int32))
chk("arange(3,17) i64", torch.arange(3, 17, dtype=torch.int64, device=dev),
    torch.arange(3, 17, dtype=torch.int64))
chk("arange(0,20,3) i32", torch.arange(0, 20, 3, dtype=torch.int32, device=dev),
    torch.arange(0, 20, 3, dtype=torch.int32))
chk("arange(1025) i32", torch.arange(1025, dtype=torch.int32, device=dev),
    torch.arange(1025, dtype=torch.int32))

print()
print(f"RESULT: {'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
