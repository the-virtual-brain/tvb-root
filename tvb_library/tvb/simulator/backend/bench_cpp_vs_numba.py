# -*- coding: utf-8 -*-
"""Task-7 benchmark: C++ hybrid backend vs NbHybridBackend (numba).

Compares measured wall-time per step on representative workloads:
  1. single simulation   (coupled two-subnet MPR network)
  2. parameter sweep     (my SIMD lane-sweep vs per-point numba runs)
"""
import time
import numpy as np
from scipy import sparse

from tvb.simulator.backend.cpp_hybrid.backend import CppHybridBackend
from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.hybrid.coupling import Linear
from tvb.simulator.hybrid.inter_projection import InterProjection


def build_net(n, coupling_scale=2.0):
    dt = 0.1
    A = _subnet("A", n, dt)
    B = _subnet("B", n, dt)
    w = sparse.csr_matrix(np.full((n, n), 0.01, np.float64) -
                          np.eye(n) * 0.01)
    proj = InterProjection(
        source=A, target=B,
        source_cvar=np.array([0], np.int32),
        target_cvar=np.array([0], np.int32),
        weights=w, lengths=sparse.csr_matrix(np.zeros((n, n))),
        cv=1.0, dt=dt, scale=coupling_scale, cfun=Linear(),
    )
    ns = NetworkSet(subnets=[A, B], projections=[proj])
    ns.configure()
    return ns


def _subnet(name, n, dt):
    m = MontbrioPazoRoxin(); m.configure()
    return Subnetwork(name=name, model=m,
                      scheme=HeunDeterministic(dt=dt), nnodes=n)


def _init_states(ns, n):
    rng = np.random.RandomState(42)
    return [np.abs(rng.uniform(0.0, 0.2, (2, n, 1))).astype(np.float64)
            for _ in ns.subnets]


def time_run(fn, warmup=3, reps=5):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def bench_single(n, nstep):
    ns = build_net(n)
    x0 = _init_states(ns, n)
    cpp = CppHybridBackend()
    nb = NbHybridBackend()
    # warm both (JIT for numba)
    time_run(lambda: cpp.run_network(ns, nstep=10, initial_states=x0), 1, 1)
    time_run(lambda: nb.run_network(ns, nstep=10, initial_states=x0), 1, 1)
    tc = time_run(lambda: cpp.run_network(ns, nstep=nstep, initial_states=x0),
                  warmup=2, reps=5)
    tn = time_run(lambda: nb.run_network(ns, nstep=nstep, initial_states=x0),
                  warmup=2, reps=5)
    return tc, tn


def bench_sweep(n, nstep, values):
    ns0 = build_net(n)
    x0 = _init_states(ns0, n)
    cpp = CppHybridBackend()
    nb = NbHybridBackend()
    # my SIMD lane sweep
    time_run(lambda: cpp.sweep(ns0, {"coupling_scale": values}, nstep=nstep),
             warmup=1, reps=1)
    t0 = time.perf_counter()
    cpp.sweep(ns0, {"coupling_scale": values}, nstep=nstep)
    t_sweep = time.perf_counter() - t0
    # per-point numba runs (one per value)
    def nb_points():
        for v in values:
            ns = build_net(n, coupling_scale=float(v))
            nb.run_network(ns, nstep=nstep, initial_states=x0)
    time_run(nb_points, warmup=1, reps=1)
    t0 = time.perf_counter()
    nb_points()
    t_points = time.perf_counter() - t0
    return t_sweep, t_points


def main():
    print("=" * 70)
    print("Single simulation: coupled 2-subnet MPR (heun), N nodes, nstep steps")
    for n, nstep in [(200, 200), (500, 500)]:
        tc, tn = bench_single(n, nstep)
        print(f"  N={n} nstep={nstep}: cpp={tc*1e3:.1f}ms ({tc/nstep*1e6:.1f}us/step) "
              f"numba={tn*1e3:.1f}ms ({tn/nstep*1e6:.1f}us/step) "
              f"x{numba_speedup(tc, tn):.2f}")
    print("=" * 70)
    print("Parameter sweep: {n_vals} values of coupling_scale")
    n, nstep = 300, 300
    values = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0], np.float32)
    t_sweep, t_points = bench_sweep(n, nstep, values)
    print(f"  cpp SIMD sweep(8-lane)={t_sweep*1e3:.1f}ms  "
          f"numba per-point(8 runs)={t_points*1e3:.1f}ms  "
          f"x{numba_speedup(t_sweep, t_points):.2f}")


def numba_speedup(tc, tn):
    return tn / tc if tc > 0 else float("inf")


if __name__ == "__main__":
    main()
