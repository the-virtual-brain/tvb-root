"""
Minimal reproducer for Numba JIT compilation segfault.

Context: TVB hybrid-numba backends generate @nb.njit kernels for neural
mass simulations.  When the generated code contains a large outer loop
that repeatedly inlines a moderately-complex inner function (e.g. sparse
CSR coupling), Numba/LLVM segfaults during first-time compilation.

The crash does NOT happen when:
  - the same function is loaded from Numba's disk cache, OR
  - the inner function is removed (all loops inline directly into outer).

The crash DOES happen when:
  - @nb.njit(inline="always") is used on the inner function, AND
  - outer() calls it ~80 times inside a loop.

No TVB dependencies — only numpy + numba.
"""

import numpy as np
import numba as nb


@nb.njit(cache=False)
def inner_sparse(srcbuf, w_data, w_indices, w_indptr, idelays,
                 cfun_params, t, tgt):
    """Simulates a sparse coupling computation (TVB-style)."""
    for j in range(tgt.shape[1]):
        row_start = w_indptr[j]
        row_end = w_indptr[j + 1]
        for ptr in range(row_start, row_end):
            w = w_data[ptr]
            src_node = w_indices[ptr]
            buf_idx = (t - 1 - idelays[ptr]) % srcbuf.shape[3]
            tgt[0, j, 0] += cfun_params[0] * w * srcbuf[0, src_node, 0, buf_idx]


@nb.njit(cache=False)
def outer(nstep, srcbuf, w_data, w_indices, w_indptr, idelays,
          cfun_params, tgt):
    """Outer time-step loop — inlines inner_sparse many times."""
    for t in range(1, nstep + 1):
        for _rep in range(20):
            inner_sparse(srcbuf, w_data, w_indices, w_indptr, idelays,
                         cfun_params, t, tgt)
            inner_sparse(srcbuf, w_data, w_indices, w_indptr, idelays,
                         cfun_params, t, tgt)
            inner_sparse(srcbuf, w_data, w_indices, w_indptr, idelays,
                         cfun_params, t, tgt)
            inner_sparse(srcbuf, w_data, w_indices, w_indptr, idelays,
                         cfun_params, t, tgt)


if __name__ == "__main__":
    N_C = 68
    NSPARSE = 20
    srcbuf = np.zeros((1, N_C, 1, 1), dtype=np.float32)
    w_data = np.ones(NSPARSE, dtype=np.float32)
    w_indices = np.zeros(NSPARSE, dtype=np.int32)
    w_indptr = np.arange(N_C + 1, dtype=np.int32)
    idelays = np.zeros(NSPARSE, dtype=np.int32)
    cfun_params = np.ones(8, dtype=np.float32) * 0.01
    tgt = np.zeros((1, N_C, 1), dtype=np.float32)

    print("Calling outer(10, ...)...", flush=True)
    outer(10, srcbuf, w_data, w_indices, w_indptr, idelays,
          cfun_params, tgt)
    print("  OK", flush=True)
    print("Exiting cleanly.", flush=True)
