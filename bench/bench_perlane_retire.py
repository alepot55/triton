"""Standalone benchmark + oracle check for the per-lane retirement PLUGIN.

Compiles the canonical lock-step kernel twice -- baseline and with the
plugin pass enabled -- checks both against an exact CPU oracle, and reports
timings and the PTX-level evidence (redux.sync count).

Usage:
  TRITON_PLUGIN_PATHS=<build>/libPerLaneLoopRetirement.so \
  PYTHONPATH=<this repo>/python \
    python bench/bench_perlane_retire.py
"""

from __future__ import annotations

import statistics

import numpy as np
import torch
import triton
import triton.language as tl

import perlane_retire

N, BLOCK = 1 << 20, 32


@triton.jit
def perlane_while(inp, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < n
    trip = tl.load(inp + i, mask=valid, other=0)
    acc = tl.zeros((BLOCK,), tl.int32)
    j = tl.zeros((BLOCK,), tl.int32)
    while tl.max((j < trip).to(tl.int32)) > 0:
        active = j < trip
        acc = acc + tl.where(active, j, 0)
        j = j + 1
    tl.store(out + i, acc, mask=valid)


def run(d, ref, retire: bool):
    o = torch.zeros(N, dtype=torch.int32, device="cuda")
    grid = (triton.cdiv(N, BLOCK),)
    kw = dict(num_warps=1)
    h = perlane_while[grid](d, o, N, BLOCK=BLOCK, **kw)
    ok = np.array_equal(o.cpu().numpy(), ref)
    for _ in range(3):
        perlane_while[grid](d, o, N, BLOCK=BLOCK, **kw)
    ts = []
    for _ in range(9):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        perlane_while[grid](d, o, N, BLOCK=BLOCK, **kw)
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    redux = h.asm["ptx"].count("redux.sync")
    warpsync = h.asm["ptx"].count("bar.warp.sync")
    return ok, statistics.median(ts) * 1e3, redux, warpsync


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    rng = np.random.default_rng(0)
    trip = np.clip(rng.geometric(1 / 16, size=N), 1, 256).astype(np.int32)
    d = torch.as_tensor(trip, device="cuda")
    ref = (trip.astype(np.int64) * (trip.astype(np.int64) - 1) // 2).astype(np.int32)

    ok0, t0, rdx0, ws0 = run(d, ref, retire=False)
    perlane_retire.enable()
    ok1, t1, rdx1, ws1 = run(d, ref, retire=True)
    print(f"baseline: oracle={'OK' if ok0 else 'FAIL'} time={t0:7.1f}us ptx redux.sync={rdx0} bar.warp.sync={ws0}")
    print(f"retire  : oracle={'OK' if ok1 else 'FAIL'} time={t1:7.1f}us ptx redux.sync={rdx1} bar.warp.sync={ws1}")
    print(f"speedup : {t0 / t1:.2f}x")
    assert ok0 and ok1, "oracle mismatch"
    assert rdx0 >= 1 and rdx1 == 0, "pass did not fire"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
