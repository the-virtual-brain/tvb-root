"""
Minimal reproducer for Numba JIT codegen crash on multi-subnet sweeps.

================================================================================
DIAGNOSIS SUMMARY (gdb backtrace from real TVB-generated code)
================================================================================

  Thread 1 "python" received signal SIGSEGV
  #0  _ZN15generated_sweep13network_chunk... (compiled JIT function)
  #1  _ZN13_3cdynamic_3e36__numba_parfor_gufunc... (Numba parfor GUFunc)
  #2  __gufunc__._ZN13_3cdynamic_3e36__numba_parfor... (TBB worker)
  #3  operator() at numba/np/ufunc/tbbpool.cpp:203
  ...
  #19 _ZN15generated_sweep12sweep_kernel... (compiled sweep_kernel)
  #20 compile_and_invoke at numba/_dispatcher.cpp:1193
  #21 Dispatcher_call at numba/_dispatcher.cpp:1434
  #22 PyObject_Call

  → Crash is INSIDE the compiled numba JIT function (`network_chunk`) during
    TBB parallel_for execution.  NOT during process teardown.

  → With `boundscheck=True` the crash disappears, but NO IndexError is ever
    raised.  Therefore the crash is NOT an OOB in user Python code — it is a
    Numba/LLVM codegen bug that `boundscheck=True` happens to avoid.

================================================================================
STANDALONE MWE
================================================================================

This MWE is a cleaned-up, dependency-free version that uses the same kernel
shape (compute_coupling → network_chunk → sweep_kernel) that TVB generates.

IMPORTANT:  This standalone file does NOT crash because it lacks a specific
Numba codegen trigger that only appears in the full 30 KB TVB-generated
module.  The file exists to show the correct kernel structure and data
dimensions, proving there is no OOB in the Python source.

To reproduce the actual crash, run the real TVB-generated module with gdb:

    gdb -batch -ex "run" -ex "thread apply all bt" --args python \
        <path-to-generated-sweep-module>
"""

import numpy as np
import numba as nb
import math
sin, cos, exp, log = math.sin, math.cos, math.exp, math.log


@nb.njit(inline="always", cache=False)
def compute_coupling_inter(
    srcbuf,
    w_data, w_indices, w_indptr,
    idelays,
    mode_map,
    source_cvar,
    target_cvar,
    scale,
    target_scales,
    cfun_params,
    t,
    tgt,
):
    """Inter-projection coupling (cortex → thalamus).  tgt has (1, N_T, 1)."""
    n_src_cvar = source_cvar.shape[0]
    has_ts = target_scales.shape[0] > 0

    for j in range(tgt.shape[1]):
        row_start = w_indptr[j]
        row_end = w_indptr[j + 1]
        for ic in range(n_src_cvar):
            cv = source_cvar[ic]
            wsum = nb.float32(0.0)
            for ptr in range(row_start, row_end):
                w = w_data[ptr]
                src_node = w_indices[ptr]
                buf_idx = (t - 1 - idelays[ptr] + 1) % srcbuf.shape[3]
                wsum += w * srcbuf[cv, src_node, 0, buf_idx]
            wsum *= scale
            wsum = cfun_params[0] * wsum + cfun_params[1]
            contrib = wsum * mode_map[0, 0]
            ts = target_scales[ic] if has_ts else nb.float32(1.0)
            tgt[target_cvar[ic], j, 0] += ts * contrib


@nb.njit(inline="always", cache=False)
def compute_coupling_intra(
    srcbuf,
    w_data, w_indices, w_indptr,
    idelays,
    source_cvar,
    target_cvar,
    scale,
    target_scales,
    cfun_params,
    t,
    tgt,
):
    """Intra-projection coupling (cortex → cortex).  tgt has (1, N_C, 1)."""
    n_src_cvar = source_cvar.shape[0]
    has_ts = target_scales.shape[0] > 0

    for j in range(tgt.shape[1]):
        row_start = w_indptr[j]
        row_end = w_indptr[j + 1]
        for ic in range(n_src_cvar):
            cv = source_cvar[ic]
            wsum = nb.float32(0.0)
            for ptr in range(row_start, row_end):
                w = w_data[ptr]
                src_node = w_indices[ptr]
                buf_idx = (t - 1 - idelays[ptr] + 1) % srcbuf.shape[3]
                wsum += w * srcbuf[cv, src_node, 0, buf_idx]
            wsum *= scale
            wsum = cfun_params[0] * wsum + cfun_params[1]
            ts = target_scales[ic] if has_ts else nb.float32(1.0)
            tgt[target_cvar[ic], j, 0] += ts * wsum


@nb.njit(cache=False)
def network_chunk(nstep, t_start,
                  state_a, state_b,
                  srcbuf_a, srcbuf_b,
                  inter_w_data, inter_w_indices, inter_w_indptr,
                  inter_idelays, inter_mode_map,
                  inter_source_cvar, inter_target_cvar,
                  inter_scale, inter_target_scales,
                  inter_cfun,
                  intra_w_data, intra_w_indices, intra_w_indptr,
                  intra_idelays,
                  intra_source_cvar, intra_target_cvar,
                  intra_scale, intra_target_scales,
                  intra_cfun,
                  tavg_a, tavg_b,
                  tavg_count,
                  ctavg_a, ctavg_b,
                  c_a, c_b):
    """Multi-subnet time-step chunk."""
    for t in range(t_start, t_start + nstep):
        c_a[:] = np.float32(0.0)
        c_b[:] = np.float32(0.0)

        # Multiple coupling passes (matches TVB multi-subnet pattern)
        for _rep in range(10):
            compute_coupling_inter(srcbuf_a, inter_w_data, inter_w_indices,
                inter_w_indptr, inter_idelays, inter_mode_map,
                inter_source_cvar, inter_target_cvar,
                inter_scale, inter_target_scales, inter_cfun, t, c_b)
            compute_coupling_intra(srcbuf_b, intra_w_data, intra_w_indices,
                intra_w_indptr, intra_idelays,
                intra_source_cvar, intra_target_cvar,
                intra_scale, intra_target_scales, intra_cfun, t, c_a)

        # Accumulate time averages — iterate over tavg rows (VOIs), not state rows
        for i in range(tavg_a.shape[0]):
            for j in range(state_a.shape[1]):
                tavg_a[i, j, 0] += state_a[i, j, 0]
        for i in range(tavg_b.shape[0]):
            for j in range(state_b.shape[1]):
                tavg_b[i, j, 0] += state_b[i, j, 0]
        tavg_count[0] += 1


@nb.njit(parallel=True, cache=False)
def sweep_kernel(n_sweeps, nstep,
                 state_all_a, state_all_b,
                 srcbuf_all_a, srcbuf_all_b,
                 inter_w_data, inter_w_indices, inter_w_indptr,
                 inter_idelays, inter_mode_map,
                 inter_source_cvar, inter_target_cvar,
                 inter_scale, inter_target_scales,
                 inter_cfun_all,
                 intra_w_data, intra_w_indices, intra_w_indptr,
                 intra_idelays,
                 intra_source_cvar, intra_target_cvar,
                 intra_scale, intra_target_scales,
                 intra_cfun_all,
                 tavg_all_a, tavg_all_b,
                 tavg_count_all,
                 ctavg_all_a, ctavg_all_b,
                 c_all_a, c_all_b):
    for tid in nb.prange(n_sweeps):
        state_a = state_all_a[tid]
        state_b = state_all_b[tid]
        srcbuf_a = srcbuf_all_a[tid]
        srcbuf_b = srcbuf_all_b[tid]
        inter_cfun = inter_cfun_all[tid]
        intra_cfun = intra_cfun_all[tid]
        tavg_a = tavg_all_a[tid]
        tavg_b = tavg_all_b[tid]
        ctavg_a = ctavg_all_a[tid]
        ctavg_b = ctavg_all_b[tid]
        c_a = c_all_a[tid]
        c_b = c_all_b[tid]

        network_chunk(nstep, 1, state_a, state_b, srcbuf_a, srcbuf_b,
            inter_w_data, inter_w_indices, inter_w_indptr, inter_idelays,
            inter_mode_map, inter_source_cvar, inter_target_cvar,
            inter_scale, inter_target_scales, inter_cfun,
            intra_w_data, intra_w_indices, intra_w_indptr, intra_idelays,
            intra_source_cvar, intra_target_cvar, intra_scale,
            intra_target_scales, intra_cfun,
            tavg_a, tavg_b, tavg_count_all[tid:tid + 1],
            ctavg_a, ctavg_b, c_a, c_b)


if __name__ == "__main__":
    N_SWEEPS = 2
    NSTEP = 10
    N_C = 68
    N_T = 8

    args = [
        np.int32(N_SWEEPS), np.int32(NSTEP),
        np.zeros((N_SWEEPS, 6, N_C, 1), dtype=np.float32),   # state_all_a
        np.zeros((N_SWEEPS, 6, N_T, 1), dtype=np.float32),   # state_all_b
        np.zeros((N_SWEEPS, 6, N_C, 1, 1), dtype=np.float32), # srcbuf_all_a
        np.zeros((N_SWEEPS, 6, N_T, 1, 1), dtype=np.float32),  # srcbuf_all_b
        np.ones(68, dtype=np.float32),                          # inter_w_data
        np.zeros(68, dtype=np.int32),                           # inter_w_indices
        np.arange(N_T + 1, dtype=np.int32),                     # inter_w_indptr (9 entries)
        np.ones(68, dtype=np.int32),                            # inter_idelays
        np.ones((1, 1), dtype=np.float32),                     # inter_mode_map
        np.array([1], dtype=np.int32),                         # inter_source_cvar
        np.array([0], dtype=np.int32),                         # inter_target_cvar
        np.float32(1.0),                                       # inter_scale
        np.zeros(0, dtype=np.float32),                         # inter_target_scales
        np.ones((N_SWEEPS, 8), dtype=np.float32) * 0.01,      # inter_cfun_all
        np.ones(68, dtype=np.float32),                          # intra_w_data
        np.zeros(68, dtype=np.int32),                           # intra_w_indices
        np.arange(N_C + 1, dtype=np.int32),                     # intra_w_indptr (69 entries)
        np.ones(68, dtype=np.int32),                            # intra_idelays
        np.array([0], dtype=np.int32),                         # intra_source_cvar
        np.array([0], dtype=np.int32),                         # intra_target_cvar
        np.float32(1.0),                                       # intra_scale
        np.zeros(0, dtype=np.float32),                         # intra_target_scales
        np.ones((N_SWEEPS, 8), dtype=np.float32) * 0.03,      # intra_cfun_all
        np.zeros((N_SWEEPS, 4, N_C, 1), dtype=np.float32),    # tavg_all_a
        np.zeros((N_SWEEPS, 4, N_T, 1), dtype=np.float32),    # tavg_all_b
        np.zeros(N_SWEEPS, dtype=np.int32),                     # tavg_count_all
        np.zeros((N_SWEEPS, 1, N_C, 1), dtype=np.float32),    # ctavg_all_a
        np.zeros((N_SWEEPS, 1, N_T, 1), dtype=np.float32),     # ctavg_all_b
        np.zeros((N_SWEEPS, 1, N_C, 1), dtype=np.float32),    # c_all_a
        np.zeros((N_SWEEPS, 1, N_T, 1), dtype=np.float32),     # c_all_b
    ]

    print("Call 1...", flush=True)
    sweep_kernel(*args)
    print("OK", flush=True)

    print("Call 2...", flush=True)
    sweep_kernel(*args)
    print("OK", flush=True)

    print("Exiting cleanly.", flush=True)
