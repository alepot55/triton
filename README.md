# triton-perlane-retire

An out-of-tree [Triton](https://github.com/triton-lang/triton) plugin pass
that lets the lanes of a warp **retire independently from divergent
`while` loops**, instead of iterating in lock-step to the warp's maximum
trip count.

Proposed upstream and moved out-of-tree following maintainer guidance:
RFC [triton-lang/triton#10773](https://github.com/triton-lang/triton/issues/10773),
draft PR [triton-lang/triton#10774](https://github.com/triton-lang/triton/pull/10774).

## What it does

Triton lowers a data-dependent loop such as

```python
while tl.max((j < trip).to(tl.int32)) > 0:
    active = j < trip
    acc += tl.where(active, j, 0)
    j += 1
```

to a warp-lock-step loop: the latch warp-reduces the per-lane predicate
(`nvvm.redux.sync max`) and branches on the uniform result, so every lane
iterates until the slowest lane in its warp finishes, and pays a
warp-collective reduction each iteration. On sm_70+ (independent thread
scheduling) that schedule is a codegen choice, not a hardware requirement.

The pass (`nv-per-lane-loop-retirement`, an LLVM-dialect module pass)
redirects such latches to the per-lane predicate — each lane exits as soon
as its own condition fails — deletes the dead cross-lane reduction, and
reconverges the entering lanes (`activemask` captured at the preheader)
with `bar.warp.sync` at the loop exit.

**A static verifier proves observational equivalence before rewriting**
(loops it cannot prove safe are left untouched):

1. body free of NVVM ops (collectives, barriers, nested redux latches),
   calls, atomics, unpredicated `llvm.store`, and inline asm that stores or
   synchronizes — predicated gathers (masked `tl.load`) are safe;
2. single loop exit; unconditionally-entered preheader;
3. every live-out loop-carried value **frozen** on lane-inactive iterations
   (`select(pred, x, old)` / masked-identity updates, followed through the
   struct-typed block-argument projections of the real lowering);
4. the per-lane predicate is **monotone**: once false it cannot re-arm
   under continued lock-step execution. (Live-out freezing alone is *not*
   sufficient — an unmasked induction on a lock-step-inactive lane could
   re-arm a non-monotone predicate; the tests include the counterexample.)

## Measured (RTX 4070, sm_89; every run oracle-checked bit-exact)

| | |
|---|---|
| canonical divergent-trip loop | **2.5–4.2×** end-to-end |
| issued instructions (Nsight) | **39× fewer** (work = `sum(trip)` not `32×warp-max`) |
| SASS | `REDUX.MAX.S32` eliminated, 48→40 static instructions |
| masked-cost law | `t = 50.3 + 1.08·E[warp-max trip]` µs, R²=0.998, holds out-of-sample |
| gather-bound kernels with the same latch (CSR SpMV, MoE top-k) | 1.14–1.25× |

## Build

Needs a Triton source/build tree compiled with `TRITON_EXT_ENABLED=1` and
its LLVM/MLIR distribution:

```bash
cmake -B build -G Ninja \
  -DTRITON_SRC_DIR=/path/to/triton \
  -DTRITON_LLVM_DIR=$HOME/.triton/llvm/<dist> \
  -DCMAKE_BUILD_TYPE=Release
ninja -C build
```

## Use

```bash
export TRITON_PLUGIN_PATHS=/path/to/build/libPerLaneLoopRetirement.so
export PYTHONPATH=/path/to/triton-perlane-retire/python:$PYTHONPATH
```

```python
import perlane_retire
perlane_retire.enable()   # installs the (cache-key aware) pipeline hook
# ... define and launch Triton kernels as usual
```

`bench/bench_perlane_retire.py` reproduces the headline result end-to-end
(bit-exact oracle + PTX `redux.sync` 1→0).

## Pinned version

Developed and tested against Triton main @ `81a46fa` (3.8.0-dev). The
pipeline hook interposes on `add_warp_specialize_to_llvm` inside
`make_llir` (the first point where the warp-uniform latch exists); if a
future Triton renames that anchor, `enable()` fails loudly.

## License

MIT.
