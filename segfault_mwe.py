"""
Minimal reproducer for Numba JIT compilation segfault.

Now with boundscheck=True to locate the exact OOB index.
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
    NSPARSE = 68
    srcbuf = np.zeros((1, N_C, 1, 1), dtype=np.float32)
    w_data = np.ones(NSPARSE, dtype=np.float32)
    w_indices = np.zeros(NSPARSE, dtype=np.int32)
    w_indptr = np.arange(N_C + 1, dtype=np.int32)
    idelays = np.zeros(NSPARSE, dtype=np.int32)
    cfun_params = np.ones(8, dtype=np.float32) * 0.01
    tgt = np.zeros((1, N_C, 1), dtype=np.float32)

    # Now remove boundscheck to see if the crash was purely from OOB
    print("Calling outer(10, ...) without boundscheck...", flush=True)
    try:
        outer(10, srcbuf, w_data, w_indices, w_indptr, idelays,
              cfun_params, tgt)
        print("  OK", flush=True)
    except IndexError as e:
        print("  IndexError:", e, flush=True)
        print("  w_data shape:", w_data.shape, flush=True)
        print("  w_indptr:", w_indptr[:10], "...", w_indptr[-5:], flush=True)
        print("  NSPARSE:", NSPARSE, "N_C:", N_C, flush=True)
        print("  w_indptr says", w_indptr[-1], "nonzeros but w_data has", len(w_data), flush=True)
    print("Exiting.", flush=True)
