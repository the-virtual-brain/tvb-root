#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Performance benchmark: Python vs Numba backend for TVB hybrid simulator.

Benchmarks the Python (NetworkSet.step loop) path against the Numba JIT
compiled path (NbHybridBackend.run_network) for several configurations:

  1. Single MPR subnet (4 nodes, 1000 steps)
  2. Node scaling (4, 16, 68 nodes)
  3. Coupled subnets (2 × 34-node MPR with inter-projection)
  4. Different models (MPR, JansenRit, Epileptor) at 34 nodes

Usage::

    python tvb/simulator/backend/benchmark_nb_hybrid.py
"""

import sys
import os
import time

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

import numpy as np
import scipy.sparse as sp

from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.models.jansen_rit import JansenRit
from tvb.simulator.models.epileptor import Epileptor
from tvb.simulator.hybrid import Subnetwork, NetworkSet
from tvb.simulator.hybrid.inter_projection import InterProjection
from tvb.simulator.hybrid.coupling import Linear
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.backend.nb_hybrid import NbHybridBackend

DT = 0.1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_subnet(name, model_cls, n_nodes):
    """Create and configure a subnetwork with the given model and size."""
    model = model_cls()
    model.configure()
    sn = Subnetwork(
        name=name, model=model, scheme=HeunDeterministic(dt=DT), nnodes=n_nodes
    )
    sn.configure()
    return sn


def make_ic(model, n_nodes):
    """Create mid-range initial conditions for a single subnetwork."""
    sv_range = model.state_variable_range
    svars = list(model.state_variables)
    x0 = np.zeros((len(svars), n_nodes, 1), dtype=np.float64)
    for i, sv in enumerate(svars):
        if sv in sv_range:
            lo, hi = float(sv_range[sv][0]), float(sv_range[sv][1])
            x0[i, :, 0] = (lo + hi) / 2.0
    return [x0]


def run_python(ns, x0, nstep):
    """Run the pure-Python integration loop and return elapsed seconds."""
    x = ns.States(*[arr.copy() for arr in x0])
    ns.init_projection_buffers(x)
    t0 = time.perf_counter()
    for step in range(1, nstep + 1):
        x = ns.step(step, x)
    return time.perf_counter() - t0


def run_numba(ns, x0, nstep):
    """Run the Numba-compiled path (with warmup) and return elapsed seconds."""
    backend = NbHybridBackend()
    # Warmup to trigger JIT compilation (excluded from timing)
    backend.run_network(
        ns, nstep=2, chunk_size=1, initial_states=[arr.copy() for arr in x0]
    )
    t0 = time.perf_counter()
    backend.run_network(
        ns, nstep=nstep, chunk_size=1, initial_states=[arr.copy() for arr in x0]
    )
    return time.perf_counter() - t0


def make_coupled_network(n_per_subnet):
    """Create two coupled MPR subnets of *n_per_subnet* nodes each."""
    sn1 = make_subnet("sn1", MontbrioPazoRoxin, n_per_subnet)
    sn2 = make_subnet("sn2", MontbrioPazoRoxin, n_per_subnet)
    # Inter-projection sn1 → sn2
    w = sp.eye(n_per_subnet, format="csr", dtype=np.float64) * 0.1
    lengths = sp.csr_matrix(w.shape, dtype=np.float64)  # zero delays
    proj = InterProjection(
        source=sn1,
        target=sn2,
        source_cvar=np.array([0]),
        target_cvar=np.array([0]),
        weights=w,
        lengths=lengths,
        cv=3.0,
        dt=DT,
        scale=1.0,
        cfun=Linear(),
    )
    ns = NetworkSet(subnets=[sn1, sn2], projections=[proj], stimuli=[])
    ns.configure()
    return ns


# ---------------------------------------------------------------------------
# Benchmark routines
# ---------------------------------------------------------------------------


def bench_single_mpr():
    """Benchmark 1: single MPR subnet, 4 nodes, 1000 steps."""
    print("--- Single MPR subnet (4 nodes, 1000 steps) ---")
    sn = make_subnet("mpr", MontbrioPazoRoxin, 4)
    ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
    ns.configure()
    x0 = make_ic(sn.model, sn.nnodes)
    py_t = run_python(ns, x0, 1000)
    nb_t = run_numba(ns, x0, 1000)
    print(f"Python:  {py_t:.3f}s")
    print(f"Numba:   {nb_t:.3f}s (excluding compile)")
    print(f"Speedup: {py_t / nb_t:.1f}x")
    print()
    return py_t, nb_t


def bench_node_scaling():
    """Benchmark 2: MPR at 4, 16, 68 nodes."""
    print("--- Node scaling (MPR, 1000 steps) ---")
    print(f" {'Nodes':>5} | {'Python (s)':>10} | {'Numba (s)':>10} | {'Speedup':>8}")
    print(f" {'-' * 5} | {'-' * 10} | {'-' * 10} | {'-' * 8}")
    for n in (4, 16, 68):
        sn = make_subnet("mpr", MontbrioPazoRoxin, n)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        ns.configure()
        x0 = make_ic(sn.model, sn.nnodes)
        py_t = run_python(ns, x0, 1000)
        nb_t = run_numba(ns, x0, 1000)
        print(f" {n:>5} | {py_t:>10.3f} | {nb_t:>10.3f} | {py_t / nb_t:>7.1f}x")
    print()


def bench_coupled():
    """Benchmark 3: two coupled MPR subnets (34 + 34 nodes)."""
    print("--- Coupled subnets (2 × 34-node MPR, 1000 steps) ---")
    ns = make_coupled_network(34)
    x0 = []
    for sn in ns.subnets:
        x0.extend(make_ic(sn.model, sn.nnodes))
    py_t = run_python(ns, x0, 1000)
    nb_t = run_numba(ns, x0, 1000)
    print(f"Python:  {py_t:.3f}s")
    print(f"Numba:   {nb_t:.3f}s (excluding compile)")
    print(f"Speedup: {py_t / nb_t:.1f}x")
    print()


def bench_models():
    """Benchmark 4: different models at 34 nodes, 1000 steps."""
    print("--- Different models (34 nodes, 1000 steps) ---")
    print(f" {'Model':>22} | {'Python (s)':>10} | {'Numba (s)':>10} | {'Speedup':>8}")
    print(f" {'-' * 22} | {'-' * 10} | {'-' * 10} | {'-' * 8}")
    for label, cls in [
        ("MontbrioPazoRoxin", MontbrioPazoRoxin),
        ("JansenRit", JansenRit),
        ("Epileptor", Epileptor),
    ]:
        sn = make_subnet("sn", cls, 34)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        ns.configure()
        x0 = make_ic(sn.model, sn.nnodes)
        py_t = run_python(ns, x0, 1000)
        nb_t = run_numba(ns, x0, 1000)
        print(
            f" {label:>22} | {py_t:>10.3f} | {nb_t:>10.3f} | {py_t / nb_t:>7.1f}x"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== TVB Hybrid Numba Backend Benchmark ===")
    print()
    bench_single_mpr()
    bench_node_scaling()
    bench_coupled()
    bench_models()
    print("=== Done ===")
