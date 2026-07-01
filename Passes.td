#ifndef PERLANE_RETIRE_PASSES
#define PERLANE_RETIRE_PASSES

include "mlir/Pass/PassBase.td"

def NVPerLaneLoopRetirement : Pass<"nv-per-lane-loop-retirement", "mlir::ModuleOp"> {
  let summary = "Let lanes retire independently from divergent while-loops";
  let description = [{
    Rewrites warp-uniform (lock-step) while-loop latches -- an
    `nvvm.redux.sync max` over a per-lane predicate feeding the conditional
    branch -- so the branch uses the per-lane predicate directly. Each lane
    then exits the loop as soon as its own condition fails (independent
    thread scheduling, sm_70+); the entering lanes reconverge via
    `nvvm.bar.warp.sync` on the `activemask` captured at the preheader.
    Guarded by a static verifier that proves observational equivalence with
    the lock-step schedule (body safety, single exit, live-out freezing,
    predicate monotonicity) and refuses the loop otherwise.
  }];
  let dependentDialects = ["mlir::LLVM::LLVMDialect", "mlir::NVVM::NVVMDialect"];
}

#endif
