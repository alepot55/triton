"""Enable the per-lane loop retirement plugin pass in Triton's NVIDIA pipeline.

Usage (TRITON_PLUGIN_PATHS must point at libPerLaneLoopRetirement.so BEFORE
`import triton`):

    import perlane_retire
    perlane_retire.enable()   # installs the pipeline hook (cache-key aware)

The pass runs inside `make_llir`, right after the post-conversion
canonicalize/CSE (immediately before `add_warp_specialize_to_llvm`) -- the
first point where the warp-uniform latch (`nvvm.redux.sync` feeding the
loop branch) exists, and before NVVM ops are lowered further.  Insertion is
done by wrapping the backend's `make_llir` and momentarily interposing on
`add_warp_specialize_to_llvm`, so the upstream pipeline is not duplicated
here; if a future Triton renames that pass, `enable()` fails loudly.
"""

import hashlib
import pathlib

from triton import knobs
from triton._C.libtriton import passes

_ANCHOR = "add_warp_specialize_to_llvm"

# Computed once: the hook is re-evaluated on every kernel launch to build
# the compilation cache key, so it must be cheap.
_KEY = "perlane-retire-v0.1.0:" + pathlib.Path(__file__).read_text()
_HASH = hashlib.sha256(_KEY.encode()).hexdigest()


def enable() -> None:
    if not hasattr(passes.plugin, "add_nv_per_lane_loop_retirement"):
        raise RuntimeError(
            "plugin pass not loaded; set TRITON_PLUGIN_PATHS to "
            "libPerLaneLoopRetirement.so before importing triton")

    def hook(self=None, stages=None, options=None, language=None, capability=None):
        if all(a is None for a in (stages, options, language, capability)):
            return _KEY, _HASH

        make_llir = self.make_llir

        def make_llir_with_retire(src, metadata, opt, cap):
            from triton._C.libtriton import nvidia
            sub = nvidia.passes.ttnvgpuir
            if not hasattr(sub, _ANCHOR):
                raise RuntimeError(
                    f"pipeline anchor '{_ANCHOR}' not found; this Triton "
                    "version needs the insertion point updated")
            real = getattr(sub, _ANCHOR)

            def patched(pm):
                passes.plugin.add_nv_per_lane_loop_retirement(pm)
                real(pm)

            setattr(sub, _ANCHOR, patched)
            try:
                return make_llir(src, metadata, opt, cap)
            finally:
                setattr(sub, _ANCHOR, real)

        stages["llir"] = lambda src, metadata: make_llir_with_retire(
            src, metadata, options, capability)
        return _KEY, _HASH

    knobs.runtime.add_stages_inspection_hook = hook
