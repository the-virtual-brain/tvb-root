#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark: classic TVB simulator vs Numba backend for MPR and KIonEx.

Run with:
    .venv/bin/python tvb/tests/library/simulator/backend/benchmark_nb.py

Reports wall-clock time for a 1-second simulation (excluding JIT compile time
for the Numba variants, which is also reported separately).
"""

import time
import numpy as np

from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.integrators import EulerDeterministic
from tvb.simulator.monitors import Raw
from tvb.simulator.simulator import Simulator
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.models.k_ion_exchange import KIonEx
from tvb.simulator.backend.nb import NbBackend

SIM_LENGTH = 1000.0   # ms  (= 1 s of simulated time)
DT         = 0.05     # ms  — dt=0.1 causes KIonEx to diverge in float32

# KIonEx initial condition chosen so log arguments are positive:
#   K_i = 130 + (-2) = 128 > 0, K_o = 4.8 - 3*(-2) + 0 = 10.8 > 0
#   Na_i = 16 + 2 = 18 > 0,  Na_o = 138 - 6 = 132 > 0
_KIONEX_IC = np.array([0.3, -55.0, 0.4, -2.0, 0.5])  # x, V, n, DKi, Kg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_conn():
    conn = Connectivity.from_file()
    conn.speed = np.r_[np.inf]   # no delays for a clean comparison
    return conn


def _make_sim(model, conn):
    sim = Simulator(
        connectivity=conn,
        model=model,
        integrator=EulerDeterministic(dt=DT),
        monitors=[Raw()],
        simulation_length=SIM_LENGTH,
    )
    sim.configure()
    return sim


def _set_mpr_ic(sim):
    """Set MPR IC near the stable attractor so float32 does not overflow."""
    sim.current_state[0, :, :] = 0.1   # r
    sim.current_state[1, :, :] = -2.5  # V
    for k in range(sim.history.buffer.shape[1]):
        cv = sim.model.cvar[k]
        sim.history.buffer[:, k, :, :] = sim.current_state[cv, :, :]


def _set_kionex_ic(sim):
    for k, cv in enumerate(sim.model.cvar):
        sim.history.buffer[:, k, :, :] = _KIONEX_IC[cv]
    sim.current_state[:] = _KIONEX_IC[:, None, None]


def _nb_run_sim(sim, dfun_combined=True, setup_ic=None):
    """Build and return the Numba run_sim function + measured compile time."""
    template = '<%include file="nb-sim.py.mako"/>'
    content = dict(sim=sim, np=np, debug_nojit=False, dfun_combined=dfun_combined)
    t0 = time.perf_counter()
    kernel = NbBackend().build_py_func(template, content, name='run_sim')
    # Trigger JIT compilation with a 1-step warmup
    warmup_sim = _clone_sim(sim, setup_ic=setup_ic)
    kernel(warmup_sim, nstep=1)
    compile_time = time.perf_counter() - t0
    return kernel, compile_time


def _clone_sim(sim, setup_ic=None):
    """Return a fresh, configured copy of sim (reset current_step to 0)."""
    model_cls = type(sim.model)
    conn = Connectivity.from_file()
    conn.speed = np.r_[np.inf]
    new_sim = Simulator(
        connectivity=conn,
        model=model_cls(),
        integrator=EulerDeterministic(dt=DT),
        monitors=[Raw()],
        simulation_length=SIM_LENGTH,
    )
    new_sim.configure()
    if setup_ic is not None:
        setup_ic(new_sim)
    elif isinstance(new_sim.model, KIonEx):
        _set_kionex_ic(new_sim)
    return new_sim


def _time(fn, repeats=3):
    """Return median wall-clock time (seconds) over `repeats` calls."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Benchmark cases
# ---------------------------------------------------------------------------

def bench_classic(model_cls, setup_ic=None, label=""):
    conn = _make_conn()
    sim = _make_sim(model_cls(), conn)
    if setup_ic:
        setup_ic(sim)

    # run once to warm numpy caches, then time
    output = list(sim.run())  # consume generator
    all_data = np.concatenate([d for _, d in output], axis=0)
    assert np.all(np.isfinite(all_data)), f"NaN/Inf found in '{label}' classic output"
    sim2 = _make_sim(model_cls(), _make_conn())
    if setup_ic:
        setup_ic(sim2)
    t = _time(lambda: list(sim2.run()))
    print(f"  {label:45s}  {t*1000:8.1f} ms")
    return t


def bench_numba(model_cls, setup_ic=None, label="", dfun_combined=True):
    conn = _make_conn()
    sim = _make_sim(model_cls(), conn)
    if setup_ic:
        setup_ic(sim)

    kernel, compile_time = _nb_run_sim(sim, dfun_combined=dfun_combined, setup_ic=setup_ic)
    print(f"  {'':>2}compile+warmup{' (combined)' if dfun_combined else ' (separate)':12s}  "
          f"{compile_time*1000:8.1f} ms  [JIT overhead, not included below]")

    # assert finite on one full run before timing
    check_state = kernel(_clone_sim(sim, setup_ic=setup_ic), nstep=int(SIM_LENGTH / DT))
    assert np.all(np.isfinite(check_state)), f"NaN/Inf found in '{label}' numba output"

    # time execution only (fresh sim each repeat)
    def run():
        s = _clone_sim(sim, setup_ic=setup_ic)
        kernel(s, nstep=int(SIM_LENGTH / DT))

    t = _time(run)
    print(f"  {label:45s}  {t*1000:8.1f} ms")
    return t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    nsteps = int(SIM_LENGTH / DT)
    conn = _make_conn()
    nnode = conn.weights.shape[0]
    print(f"\n{'='*65}")
    print(f"  TVB Numba backend benchmark")
    print(f"  sim_length={SIM_LENGTH} ms  dt={DT} ms  steps={nsteps}  nodes={nnode}")
    print(f"{'='*65}\n")

    results = {}

    # --- MPR ---
    print("MontbrioPazoRoxin")
    results['mpr_classic']   = bench_classic(MontbrioPazoRoxin, setup_ic=_set_mpr_ic, label="classic simulator")
    results['mpr_nb_sep']    = bench_numba(MontbrioPazoRoxin,  setup_ic=_set_mpr_ic, label="numba  (strategy A: per-svar)",  dfun_combined=False)
    results['mpr_nb_comb']   = bench_numba(MontbrioPazoRoxin,  setup_ic=_set_mpr_ic, label="numba  (strategy B: combined)",  dfun_combined=True)

    print()

    # --- KIonEx ---
    print("KIonEx")
    results['kionex_classic']  = bench_classic(KIonEx, setup_ic=_set_kionex_ic, label="classic simulator")
    results['kionex_nb_sep']   = bench_numba(KIonEx,  setup_ic=_set_kionex_ic,
                                             label="numba  (strategy A: per-svar)",  dfun_combined=False)
    results['kionex_nb_comb']  = bench_numba(KIonEx,  setup_ic=_set_kionex_ic,
                                             label="numba  (strategy B: combined)",  dfun_combined=True)

    print(f"\n{'='*65}")
    print("  Speedups vs classic simulator (higher is better)")
    print(f"{'='*65}")
    for model, key_classic, key_sep, key_comb in [
        ("MPR",    "mpr_classic",    "mpr_nb_sep",    "mpr_nb_comb"),
        ("KIonEx", "kionex_classic", "kionex_nb_sep", "kionex_nb_comb"),
    ]:
        sep  = results[key_classic] / results[key_sep]
        comb = results[key_classic] / results[key_comb]
        print(f"  {model:8s}  strategy A: {sep:5.1f}x   strategy B: {comb:5.1f}x"
              f"   (B vs A: {results[key_sep]/results[key_comb]:5.1f}x)")
    print()


if __name__ == '__main__':
    main()
