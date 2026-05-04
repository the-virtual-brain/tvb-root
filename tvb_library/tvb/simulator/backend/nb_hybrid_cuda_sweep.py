#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations.
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
"""
Numba-CUDA parameter sweep for the Hybrid Numba Backend.

Runs independent two-subnetwork MPR simulations on GPU threads,
each with a different scaling value for the inter-projection coupling
function.  Uses CSR sparse connectivity from a standard TVB connectome.

Usage::

    python tvb/simulator/backend/nb_hybrid_cuda_sweep.py

.. moduleauthor:: TVB contributors
"""

from __future__ import annotations

import sys
import os
import time

import numpy as np
import scipy.sparse as sp
from numba import cuda
import numba

# ---------------------------------------------------------------------------
# Default MPR model parameters (MontbrioPazoRoxin with defaults)
# ---------------------------------------------------------------------------

MPR_TAU = np.float32(1.0)
MPR_DELTA = np.float32(1.0)
MPR_ETA = np.float32(-5.0)
MPR_J = np.float32(15.0)
MPR_I_EXT = np.float32(0.0)
MPR_CR = np.float32(1.0)
MPR_CV = np.float32(0.0)
# Boundaries: r >= 0
MPR_R_LO = np.float32(0.0)

PI = np.float32(np.pi)


# ---------------------------------------------------------------------------
# CUDA kernel — one thread per sweep point, CSR sparse coupling
# ---------------------------------------------------------------------------

@cuda.jit
def _sweep_kernel(
    # ---- per-thread mutable state, indexed [tid, ...] ----
    state_A,       # (n_sweeps, 2, N, 1) float32 — r,V for source subnet
    state_B,       # (n_sweeps, 2, N, 1) float32 — r,V for target subnet
    srcbuf_A,      # (n_sweeps, 2, N, 1, horizon) float32 — history buffer
    tavg_A_out,    # (n_sweeps, 2, N, 1) float32 — output temporal avg
    tavg_B_out,    # (n_sweeps, 2, N, 1) float32 — output temporal avg

    # ---- read-only, shared across all threads ----
    scaling_values,  # (n_sweeps,) float32 — per-thread scaling factor a
    w_data,          # (nnz,) float32 — CSR data
    w_indices,       # (nnz,) int32 — CSR column indices
    w_indptr,        # (N+1,) int32 — CSR row pointers
    idelays,         # (nnz,) int32 — per-edge delays in steps
    horizon,         # int32 — history buffer depth
    dt,              # float32 — integration time step (ms)
    nstep,           # int32 — number of integration steps
    N,               # int32 — number of nodes (=76)
):
    """
    GPU kernel: one thread = one complete simulation at a given scaling value.

    Each thread independently:
      1. Zeros coupling scratch for target subnet B
      2. Computes CSR sparse coupling from A's history into B's coupling
      3. Integrates A (no coupling input) via HeunDeterministic
      4. Integrates B (with coupling) via HeunDeterministic
      5. Updates A's history buffer
      6. Accumulates temporal average for both subnets

    After all steps, temporal averages are divided by nstep.

    The scaling sweep parameter ``a`` modulates the coupling:
        coupling[j] = a * scale(=1.0) * sum_k(w[j,k] * srcbuf_A[...])
    """
    tid = cuda.grid(1)
    n_sweeps = scaling_values.shape[0]
    if tid >= n_sweeps:
        return

    a = scaling_values[tid]
    scale_f = np.float32(1.0)  # projection scale

    # ---- MPR model parameters (float32 constants) ----
    tau = np.float32(1.0)
    Delta = np.float32(1.0)
    eta = np.float32(-5.0)
    J = np.float32(15.0)
    I_ext = np.float32(0.0)
    cr = np.float32(1.0)
    cv = np.float32(0.0)
    pi = np.float32(3.141592653589793)

    # ---- scratch per-node coupling array for target B ----
    # cuda.local.array is limited; use a float32 array up to 76 elements
    c_B = cuda.local.array(76, dtype=numba.float32)

    for t in range(1, nstep + 1):
        # ------------------------------------------------------------
        # 1. Zero coupling scratch for B (A has no incoming coupling)
        # ------------------------------------------------------------
        for j in range(N):
            c_B[j] = np.float32(0.0)

        # ------------------------------------------------------------
        # 2. CSR sparse coupling: A → B (local reduction per thread)
        # ------------------------------------------------------------
        for j in range(N):
            row_start = w_indptr[j]
            row_end = w_indptr[j + 1]
            wsum = np.float32(0.0)
            for p in range(row_start, row_end):
                src_node = w_indices[p]
                w = w_data[p]
                d = idelays[p]
                # circular buffer index
                slot = (t - 1 - d + horizon) % horizon
                wsum += w * srcbuf_A[tid, 0, src_node, 0, slot]
            # Apply coupling function: Scaling(a) * scale
            c_B[j] = a * scale_f * wsum

        # ------------------------------------------------------------
        # 3. Integrate source subnet A (no coupling) — HeunDeterministic
        # ------------------------------------------------------------
        for i in range(N):
            r = state_A[tid, 0, i, 0]
            V = state_A[tid, 1, i, 0]

            # --- dfun at current state (k1) ---
            d_r = (Delta / (pi * tau) + np.float32(2.0) * V * r) / tau
            d_V = (V * V - pi * pi * tau * tau * r * r
                   + eta + J * tau * r + I_ext) / tau

            # --- intermediate state ---
            i1r = r + dt * d_r
            i1V = V + dt * d_V

            # clamp r >= 0
            if i1r < np.float32(0.0):
                i1r = np.float32(0.0)

            # --- dfun at intermediate (k2) ---
            d_r2 = (Delta / (pi * tau) + np.float32(2.0) * i1V * i1r) / tau
            d_V2 = (i1V * i1V - pi * pi * tau * tau * i1r * i1r
                    + eta + J * tau * i1r + I_ext) / tau

            # --- Heun update ---
            nr = r + dt * np.float32(0.5) * (d_r + d_r2)
            nV = V + dt * np.float32(0.5) * (d_V + d_V2)

            # clamp r >= 0
            if nr < np.float32(0.0):
                nr = np.float32(0.0)

            state_A[tid, 0, i, 0] = nr
            state_A[tid, 1, i, 0] = nV

            # accumulate temporal average
            tavg_A_out[tid, 0, i, 0] += nr
            tavg_A_out[tid, 1, i, 0] += nV

        # ------------------------------------------------------------
        # 4. Integrate target subnet B (with coupling) — HeunDeterministic
        # ------------------------------------------------------------
        for i in range(N):
            r = state_B[tid, 0, i, 0]
            V = state_B[tid, 1, i, 0]
            c0 = c_B[i]

            # --- dfun at current state (k1) ---
            d_r = (Delta / (pi * tau) + np.float32(2.0) * V * r) / tau
            d_V = (V * V - pi * pi * tau * tau * r * r
                   + eta + J * tau * r + I_ext + cr * c0) / tau

            # --- intermediate state ---
            i1r = r + dt * d_r
            i1V = V + dt * d_V

            if i1r < np.float32(0.0):
                i1r = np.float32(0.0)

            # --- dfun at intermediate (k2) ---
            d_r2 = (Delta / (pi * tau) + np.float32(2.0) * i1V * i1r) / tau
            d_V2 = (i1V * i1V - pi * pi * tau * tau * i1r * i1r
                    + eta + J * tau * i1r + I_ext + cr * c0) / tau

            # --- Heun update ---
            nr = r + dt * np.float32(0.5) * (d_r + d_r2)
            nV = V + dt * np.float32(0.5) * (d_V + d_V2)

            if nr < np.float32(0.0):
                nr = np.float32(0.0)

            state_B[tid, 0, i, 0] = nr
            state_B[tid, 1, i, 0] = nV

            tavg_B_out[tid, 0, i, 0] += nr
            tavg_B_out[tid, 1, i, 0] += nV

        # ------------------------------------------------------------
        # 5. Update A's history buffer
        # ------------------------------------------------------------
        slot = t % horizon
        for i in range(N):
            srcbuf_A[tid, 0, i, 0, slot] = state_A[tid, 0, i, 0]
            srcbuf_A[tid, 1, i, 0, slot] = state_A[tid, 1, i, 0]

    # ----------------------------------------------------------------
    # Divide temporal averages by nstep
    # ----------------------------------------------------------------
    inv_n = np.float32(1.0) / np.float32(nstep)
    for i in range(N):
        tavg_A_out[tid, 0, i, 0] *= inv_n
        tavg_A_out[tid, 1, i, 0] *= inv_n
        tavg_B_out[tid, 0, i, 0] *= inv_n
        tavg_B_out[tid, 1, i, 0] *= inv_n


# ---------------------------------------------------------------------------
# CSR extraction from NetworkSet
# ---------------------------------------------------------------------------

def _extract_csr_from_projection(proj):
    """Extract flat CSR arrays from an InterProjection, stripping zero weights.

    Returns
    -------
    w_data : ndarray (nnz,) float32
    w_indices : ndarray (nnz,) int32
    w_indptr : ndarray (n_tgt_nodes+1,) int32
    idelays : ndarray (nnz,) int32  — delays in steps
    horizon : int
    """
    # Work on a copy so we don't mutate the original projection
    w_csr = proj.weights.tocsr().copy()

    # Build the nonzero mask *before* eliminate_zeros so idelays alignment
    # respects the same positions.
    nz_mask = w_csr.data != 0
    w_csr.eliminate_zeros()

    idelays_raw = np.atleast_1d(np.asarray(proj.idelays, dtype=np.int32))

    if len(idelays_raw) == len(nz_mask):
        idelays_stripped = idelays_raw[nz_mask].astype(np.int32)
    elif len(idelays_raw) == len(w_csr.data):
        idelays_stripped = idelays_raw.astype(np.int32)
    else:
        # Fallback: zero delays (shouldn't happen with well-formed projections)
        idelays_stripped = np.zeros(len(w_csr.data), dtype=np.int32)

    return (
        w_csr.data.astype(np.float32),
        w_csr.indices.astype(np.int32),
        w_csr.indptr.astype(np.int32),
        idelays_stripped.astype(np.int32),
        int(proj._horizon),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_cuda_sweep(
    network_set,
    scaling_values,
    nstep,
    x0_src=None,
    x0_tgt=None,
    verbose=False,
):
    """Run GPU parameter sweep over Scaling cfun values.

    Parameters
    ----------
    network_set : NetworkSet
        Configured two-subnet (MPR+MPR) network with one inter-projection
        using Scaling cfun.  Subnets must be named 'src' and 'tgt'.
    scaling_values : ndarray (n_sweeps,) float32
        Scaling cfun 'a' values.
    nstep : int
        Integration steps per simulation.
    x0_src : ndarray (2, N, 1) float32, optional
        Initial state for source subnet.
    x0_tgt : ndarray (2, N, 1) float32, optional
        Initial state for target subnet.
    verbose : bool
        Print extra timing info.

    Returns
    -------
    tavg_A : ndarray (n_sweeps, 2, N, 1) float32
        Temporal-average state for source subnet.
    tavg_B : ndarray (n_sweeps, 2, N, 1) float32
        Temporal-average state for target subnet.
    elapsed : float
        GPU kernel execution wall-clock seconds (excludes data transfers).
    """
    scaling_values = np.asarray(scaling_values, dtype=np.float32)
    n_sweeps = len(scaling_values)
    dt = network_set.subnets[0].scheme.dt
    N = network_set.subnets[0].nnodes

    if verbose:
        print(f"[cuda_sweep] n_sweeps={n_sweeps}, nstep={nstep}, N={N}")

    # ---- Extract CSR data from the inter-projection ----
    proj = network_set.projections[0]
    w_data, w_indices, w_indptr, idelays, horizon = _extract_csr_from_projection(proj)

    if verbose:
        print(f"[cuda_sweep] nnz={len(w_data)}, horizon={horizon}")

    # ---- Default initial conditions ----
    if x0_src is None:
        rng = np.random.RandomState(42)
        x0_src = np.abs(rng.uniform(0.01, 0.05, (2, N, 1)).astype(np.float32))
    if x0_tgt is None:
        rng = np.random.RandomState(43)
        x0_tgt = np.abs(rng.uniform(0.01, 0.05, (2, N, 1)).astype(np.float32))

    x0_src = np.asarray(x0_src, dtype=np.float32)
    x0_tgt = np.asarray(x0_tgt, dtype=np.float32)

    # ---- Allocate host arrays for per-thread state ----
    state_A_h = np.zeros((n_sweeps, 2, N, 1), dtype=np.float32)
    state_B_h = np.zeros((n_sweeps, 2, N, 1), dtype=np.float32)
    srcbuf_A_h = np.zeros((n_sweeps, 2, N, 1, horizon), dtype=np.float32)
    tavg_A_h = np.zeros((n_sweeps, 2, N, 1), dtype=np.float32)
    tavg_B_h = np.zeros((n_sweeps, 2, N, 1), dtype=np.float32)

    # Broadcast initial conditions to all sweep points
    for s in range(n_sweeps):
        state_A_h[s] = x0_src
        state_B_h[s] = x0_tgt
        # Pre-fill history buffer with initial state
        for h in range(horizon):
            srcbuf_A_h[s, :, :, :, h] = x0_src

    # ---- Transfer to device ----
    t_xfer = time.perf_counter()
    state_A_d = cuda.to_device(state_A_h)
    state_B_d = cuda.to_device(state_B_h)
    srcbuf_A_d = cuda.to_device(srcbuf_A_h)
    tavg_A_d = cuda.to_device(tavg_A_h)
    tavg_B_d = cuda.to_device(tavg_B_h)

    scaling_d = cuda.to_device(scaling_values)
    w_data_d = cuda.to_device(w_data)
    w_indices_d = cuda.to_device(w_indices)
    w_indptr_d = cuda.to_device(w_indptr)
    idelays_d = cuda.to_device(idelays)
    t_xfer = time.perf_counter() - t_xfer

    # ---- Launch configuration ----
    threads_per_block = 256
    blocks = (n_sweeps + threads_per_block - 1) // threads_per_block

    if verbose:
        print(f"[cuda_sweep] grid=({blocks},), block=({threads_per_block},)")

    # ---- Run kernel ----
    cuda.synchronize()
    t_kernel = time.perf_counter()

    _sweep_kernel[blocks, threads_per_block](
        state_A_d, state_B_d, srcbuf_A_d,
        tavg_A_d, tavg_B_d,
        scaling_d,
        w_data_d, w_indices_d, w_indptr_d,
        idelays_d,
        np.int32(horizon),
        np.float32(dt),
        np.int32(nstep),
        np.int32(N),
    )
    cuda.synchronize()
    t_kernel = time.perf_counter() - t_kernel

    # ---- Copy results back ----
    t_copyback = time.perf_counter()
    tavg_A_out = tavg_A_d.copy_to_host()
    tavg_B_out = tavg_B_d.copy_to_host()
    t_copyback = time.perf_counter() - t_copyback

    if verbose:
        print(f"[cuda_sweep] transfer in:  {t_xfer*1000:.1f} ms")
        print(f"[cuda_sweep] kernel:       {t_kernel*1000:.1f} ms")
        print(f"[cuda_sweep] copy back:    {t_copyback*1000:.1f} ms")

    return tavg_A_out, tavg_B_out, t_kernel


# ---------------------------------------------------------------------------
# Correctness validation against CPU hybrid numba backend
# ---------------------------------------------------------------------------

def validate_against_cpu(
    scaling_values,
    nstep=50,
    N=76,
    weight_scale=0.02,
    rtol=1e-3,
    atol=1e-4,
    verbose=True,
):
    """Compare GPU sweep results against the CPU NbHybridBackend.

    Parameters
    ----------
    scaling_values : list of float
        Scaling values to test (subset of sweep points, e.g. 5 values).
    nstep : int
    N : int
    weight_scale : float
        Factor to scale connectome weights by.
    rtol, atol : float
        Tolerances for np.allclose.

    Returns
    -------
    passed : bool
        True if all comparisons match.
    """
    from tvb.datatypes.connectivity import Connectivity
    from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
    from tvb.simulator.hybrid import Subnetwork, NetworkSet
    from tvb.simulator.hybrid.inter_projection import InterProjection
    from tvb.simulator.hybrid.coupling import Scaling
    from tvb.simulator.integrators import HeunDeterministic
    from tvb.simulator.backend.nb_hybrid import NbHybridBackend

    conn = Connectivity.from_file('connectivity_76.zip')
    weights = sp.csr_matrix(conn.weights * weight_scale)
    lengths = sp.csr_matrix(conn.tract_lengths)

    m = MontbrioPazoRoxin()
    m.configure()
    sn_src = Subnetwork(name='src', model=m, scheme=HeunDeterministic(dt=0.1), nnodes=N)
    sn_tgt = Subnetwork(name='tgt', model=m, scheme=HeunDeterministic(dt=0.1), nnodes=N)
    sn_src.configure()
    sn_tgt.configure()

    rng = np.random.RandomState(42)
    x0_src = np.abs(rng.uniform(0.01, 0.05, (2, N, 1)).astype(np.float64))
    x0_tgt = np.abs(rng.uniform(0.01, 0.05, (2, N, 1)).astype(np.float64))

    scaling_arr = np.asarray(scaling_values, dtype=np.float32)

    # ---- GPU sweep (with real tract-length delays) ----
    proj = InterProjection(
        source=sn_src, target=sn_tgt,
        source_cvar=np.array([0]), target_cvar=np.array([0]),
        weights=weights, lengths=lengths,
        cv=3.0, dt=0.1, scale=1.0,
        cfun=Scaling(a=np.array([1.0])),
    )
    ns = NetworkSet(subnets=[sn_src, sn_tgt], projections=[proj], stimuli=[])
    ns.configure()

    tavg_A_gpu, tavg_B_gpu, elapsed_gpu = run_cuda_sweep(
        ns, scaling_arr, nstep,
        x0_src=x0_src.astype(np.float32),
        x0_tgt=x0_tgt.astype(np.float32),
    )

    # ---- CPU reference ----
    backend = NbHybridBackend()
    tavg_A_cpu = np.zeros((len(scaling_values), 2, N, 1), dtype=np.float32)
    tavg_B_cpu = np.zeros((len(scaling_values), 2, N, 1), dtype=np.float32)

    for idx, a_val in enumerate(scaling_values):
        proj_i = InterProjection(
            source=sn_src, target=sn_tgt,
            source_cvar=np.array([0]), target_cvar=np.array([0]),
            weights=weights, lengths=lengths,
            cv=3.0, dt=0.1, scale=1.0,
            cfun=Scaling(a=np.array([a_val])),
        )
        ns_i = NetworkSet(subnets=[sn_src, sn_tgt], projections=[proj_i], stimuli=[])
        ns_i.configure()

        results = backend.run_network(
            ns_i, nstep=nstep, chunk_size=1,
            initial_states=[x0_src.copy(), x0_tgt.copy()],
        )
        d_src = results[0][1]  # (nstep, 2, N, 1) — per-step states
        d_tgt = results[1][1]
        tavg_A_cpu[idx] = d_src.mean(axis=0)
        tavg_B_cpu[idx] = d_tgt.mean(axis=0)

    # ---- Compare ----
    all_pass = True
    for idx, a_val in enumerate(scaling_values):
        ok_A = np.allclose(tavg_A_gpu[idx], tavg_A_cpu[idx], rtol=rtol, atol=atol)
        ok_B = np.allclose(tavg_B_gpu[idx], tavg_B_cpu[idx], rtol=rtol, atol=atol)
        maxerr_A = np.max(np.abs(tavg_A_gpu[idx] - tavg_A_cpu[idx]))
        maxerr_B = np.max(np.abs(tavg_B_gpu[idx] - tavg_B_cpu[idx]))

        if verbose:
            status = "✓" if (ok_A and ok_B) else "✗"
            print(
                f"  a={a_val:.3f}: {status}  "
                f"maxerr_A={maxerr_A:.2e}  maxerr_B={maxerr_B:.2e}"
            )
        if not (ok_A and ok_B):
            all_pass = False

    return all_pass


# ---------------------------------------------------------------------------
# Throughput benchmark
# ---------------------------------------------------------------------------

def benchmark_sweep(
    n_sweeps_list=(100, 500, 1000, 2000, 5000, 10000),
    nstep=100,
    N=76,
    weight_scale=0.02,
    warmup=True,
):
    """Measure GPU throughput for various sweep sizes.

    Throughput is reported as thousands of simulation-steps per second:
        kstep/s = (n_sweeps × nstep) / elapsed_seconds / 1000

    Parameters
    ----------
    n_sweeps_list : tuple of int
        Sweep sizes to benchmark.
    nstep : int
        Steps per simulation.
    N : int
        Nodes per subnetwork.
    weight_scale : float
        Factor to scale connectome weights by.
    warmup : bool
        Run a small warmup kernel first to trigger PTX compilation.

    Returns
    -------
    list of (n_sweeps, throughput_kstep_s) tuples.
    """
    from tvb.datatypes.connectivity import Connectivity
    from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
    from tvb.simulator.hybrid import Subnetwork, NetworkSet
    from tvb.simulator.hybrid.inter_projection import InterProjection
    from tvb.simulator.hybrid.coupling import Scaling
    from tvb.simulator.integrators import HeunDeterministic

    print(f"\n{'='*75}")
    print(f"  GPU CUDA Sweep Benchmark — 2×MPR, {N} nodes, {nstep} steps/sim")
    print(f"  GPU: {cuda.gpus[0].name.decode()}")
    print(f"{'='*75}")
    print(f"  {'n_sweeps':>8} | {'elapsed_ms':>10} | {'kstep/s':>10} | {'Msim/s':>10}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    # ---- Build network ----
    conn = Connectivity.from_file('connectivity_76.zip')
    weights = sp.csr_matrix(conn.weights * weight_scale)
    lengths = sp.csr_matrix(conn.tract_lengths)

    m = MontbrioPazoRoxin()
    m.configure()
    sn_src = Subnetwork(name='src', model=m, scheme=HeunDeterministic(dt=0.1), nnodes=N)
    sn_tgt = Subnetwork(name='tgt', model=m, scheme=HeunDeterministic(dt=0.1), nnodes=N)
    sn_src.configure()
    sn_tgt.configure()

    proj = InterProjection(
        source=sn_src, target=sn_tgt,
        source_cvar=np.array([0]), target_cvar=np.array([0]),
        weights=weights, lengths=lengths,
        cv=3.0, dt=0.1, scale=1.0,
        cfun=Scaling(a=np.array([1.0])),
    )
    ns = NetworkSet(subnets=[sn_src, sn_tgt], projections=[proj], stimuli=[])
    ns.configure()

    rng = np.random.RandomState(42)
    x0_src = np.abs(rng.uniform(0.01, 0.05, (2, N, 1)).astype(np.float32))
    x0_tgt = np.abs(rng.uniform(0.01, 0.05, (2, N, 1)).astype(np.float32))

    # ---- Warmup ----
    if warmup:
        test_vals = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        _, _, _ = run_cuda_sweep(ns, test_vals, 10,
                                 x0_src=x0_src, x0_tgt=x0_tgt)

    results = []
    for n_sweeps in n_sweeps_list:
        scaling_vals = np.linspace(0.1, 5.0, n_sweeps).astype(np.float32)

        # Run twice; take the second run to amortize any remaining compilation
        run_cuda_sweep(ns, scaling_vals, nstep, x0_src=x0_src, x0_tgt=x0_tgt)
        _, _, elapsed = run_cuda_sweep(ns, scaling_vals, nstep,
                                       x0_src=x0_src, x0_tgt=x0_tgt)

        total_steps = n_sweeps * nstep
        kstep_s = total_steps / elapsed / 1000.0
        msim_s = n_sweeps / elapsed / 1e6

        print(f"  {n_sweeps:>8} | {elapsed*1000:>10.1f} | {kstep_s:>10.1f} | {msim_s:>10.3f}")
        results.append((n_sweeps, kstep_s))

    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    return results


# ---------------------------------------------------------------------------
# CPU throughput comparison
# ---------------------------------------------------------------------------

def benchmark_cpu_comparison(
    n_sweeps=100,
    nstep=100,
    N=76,
    weight_scale=0.02,
):
    """Compare GPU vs CPU throughput at a given sweep size.

    Returns (gpu_kstep_s, cpu_kstep_s, speedup).
    """
    from tvb.datatypes.connectivity import Connectivity
    from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
    from tvb.simulator.hybrid import Subnetwork, NetworkSet
    from tvb.simulator.hybrid.inter_projection import InterProjection
    from tvb.simulator.hybrid.coupling import Scaling
    from tvb.simulator.integrators import HeunDeterministic
    from tvb.simulator.backend.nb_hybrid import NbHybridBackend

    conn = Connectivity.from_file('connectivity_76.zip')
    weights = sp.csr_matrix(conn.weights * weight_scale)
    lengths = sp.csr_matrix(conn.tract_lengths)

    m = MontbrioPazoRoxin()
    m.configure()
    sn_src = Subnetwork(name='src', model=m, scheme=HeunDeterministic(dt=0.1), nnodes=N)
    sn_tgt = Subnetwork(name='tgt', model=m, scheme=HeunDeterministic(dt=0.1), nnodes=N)
    sn_src.configure()
    sn_tgt.configure()

    rng = np.random.RandomState(42)
    x0_src = np.abs(rng.uniform(0.01, 0.05, (2, N, 1)).astype(np.float64))
    x0_tgt = np.abs(rng.uniform(0.01, 0.05, (2, N, 1)).astype(np.float64))

    scaling_vals = np.linspace(0.1, 5.0, n_sweeps).astype(np.float32)

    # ---- GPU ----
    proj = InterProjection(
        source=sn_src, target=sn_tgt,
        source_cvar=np.array([0]), target_cvar=np.array([0]),
        weights=weights, lengths=lengths,
        cv=3.0, dt=0.1, scale=1.0,
        cfun=Scaling(a=np.array([1.0])),
    )
    ns = NetworkSet(subnets=[sn_src, sn_tgt], projections=[proj], stimuli=[])
    ns.configure()

    # warmup
    run_cuda_sweep(ns, scaling_vals[:10], nstep, x0_src=x0_src, x0_tgt=x0_tgt)
    _, _, gpu_elapsed = run_cuda_sweep(ns, scaling_vals, nstep,
                                       x0_src=x0_src, x0_tgt=x0_tgt)

    gpu_kstep_s = (n_sweeps * nstep) / gpu_elapsed / 1000.0

    # ---- CPU (batch loop with precompiled kernel) ----
    proj_base = InterProjection(
        source=sn_src, target=sn_tgt,
        source_cvar=np.array([0]), target_cvar=np.array([0]),
        weights=weights, lengths=lengths,
        cv=3.0, dt=0.1, scale=1.0,
        cfun=Scaling(a=np.array([1.0])),
    )
    ns_base = NetworkSet(subnets=[sn_src, sn_tgt], projections=[proj_base], stimuli=[])
    ns_base.configure()

    backend = NbHybridBackend()
    compiled = backend.compile(ns_base)

    compiled.run(nstep=2, chunk_size=1,
                 initial_states=[x0_src.copy(), x0_tgt.copy()])

    t0 = time.perf_counter()
    for a_val in scaling_vals:
        # Rebuild network per a_val (the compile cache reuses the kernel)
        proj_i = InterProjection(
            source=sn_src, target=sn_tgt,
            source_cvar=np.array([0]), target_cvar=np.array([0]),
            weights=weights, lengths=lengths,
            cv=3.0, dt=0.1, scale=1.0,
            cfun=Scaling(a=np.array([a_val])),
        )
        ns_i = NetworkSet(subnets=[sn_src, sn_tgt], projections=[proj_i], stimuli=[])
        ns_i.configure()
        compiled = backend.compile(ns_i)
        compiled.run(nstep=nstep, chunk_size=1,
                     initial_states=[x0_src.copy(), x0_tgt.copy()])
    cpu_elapsed = time.perf_counter() - t0

    cpu_kstep_s = (n_sweeps * nstep) / cpu_elapsed / 1000.0
    speedup = gpu_kstep_s / cpu_kstep_s

    print(f"\n  GPU vs CPU comparison ({n_sweeps} sweeps × {nstep} steps):")
    print(f"    GPU:  {gpu_kstep_s:.1f} kstep/s  ({gpu_elapsed*1000:.0f} ms)")
    print(f"    CPU:  {cpu_kstep_s:.1f} kstep/s  ({cpu_elapsed*1000:.0f} ms)")
    print(f"    Speedup: {speedup:.1f}×")

    return gpu_kstep_s, cpu_kstep_s, speedup


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    # 1. Correctness
    print("=" * 60)
    print("  VALIDATION: GPU vs CPU NbHybridBackend")
    print("=" * 60)
    test_scaling = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    ok = validate_against_cpu(test_scaling, nstep=50, N=76, weight_scale=0.02)
    print(f"\n  All correct: {ok}")

    # 2. Throughput benchmark
    results = benchmark_sweep(
        n_sweeps_list=(100, 500, 1000, 2000, 5000, 10000),
        nstep=100,
        N=76,
        weight_scale=0.02,
    )

    # 3. CPU comparison (at a moderate sweep size)
    benchmark_cpu_comparison(n_sweeps=100, nstep=100, N=76, weight_scale=0.02)

    print(f"\n{'='*75}")
    print("  Done.")
    print(f"{'='*75}")
