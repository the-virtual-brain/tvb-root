#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark pure Python hybrid stepping against the Numba backend.

This is a documentation-facing benchmark script that compares:

- the pure Python ``NetworkSet.step()`` loop
- the compiled ``NbHybridBackend`` path

for a small set of representative hybrid-network configurations.
"""

from __future__ import annotations

import os
import tempfile
import time
import warnings
from pathlib import Path

os.environ.setdefault("TVB_USER_HOME", str(Path(tempfile.gettempdir()) / "tvb-user"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

warnings.filterwarnings("ignore", message="Hybrid simulation is experimental: .*")

import numpy as np
import scipy.sparse as sp

from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.hybrid import InterProjection, NetworkSet, Subnetwork
from tvb.simulator.hybrid.coupling import Linear
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.models.epileptor import Epileptor
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.models.jansen_rit import JansenRit
from tvb.simulator.models.wilson_cowan import WilsonCowan
from tvb.simulator.models.wong_wang import ReducedWongWang
from tvb.simulator.models.larter_breakspear import LarterBreakspear
from tvb.simulator.models.oscillator import Generic2dOscillator
from tvb.simulator.models.zerlaut import ZerlautAdaptationFirstOrder


DT = 0.1
NSTEP = 1000


def make_subnet(name: str, model_cls, n_nodes: int) -> Subnetwork:
    model = model_cls()
    model.configure()
    return Subnetwork(
        name=name,
        model=model,
        scheme=HeunDeterministic(dt=DT),
        nnodes=n_nodes,
    ).configure()


def make_initial_state(subnetwork: Subnetwork) -> np.ndarray:
    model = subnetwork.model
    x0 = np.zeros(
        (model.nvar, subnetwork.nnodes, model.number_of_modes),
        dtype=np.float64,
    )
    for i, state_var in enumerate(model.state_variables):
        if state_var not in model.state_variable_range:
            continue
        low, high = map(float, model.state_variable_range[state_var])
        x0[i, :, :] = (low + high) / 2.0
    return x0


def run_python(network_set: NetworkSet, initial_states: list[np.ndarray], nstep: int) -> float:
    x = network_set.States(*[arr.copy() for arr in initial_states])
    network_set.init_projection_buffers(x)
    t0 = time.perf_counter()
    for step in range(1, nstep + 1):
        x = network_set.step(step, x)
    return time.perf_counter() - t0


def run_numba(network_set: NetworkSet, initial_states: list[np.ndarray], nstep: int, chunk_size: int = 1) -> tuple[float, float]:
    backend = NbHybridBackend()

    t0 = time.perf_counter()
    compiled = backend.compile(network_set)
    compiled.run(nstep=5, chunk_size=chunk_size, initial_states=[arr.copy() for arr in initial_states])
    compile_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    compiled.run(nstep=nstep, chunk_size=chunk_size, initial_states=[arr.copy() for arr in initial_states])
    run_time = time.perf_counter() - t0
    return compile_time, run_time


def make_single_network(model_cls, n_nodes: int) -> tuple[str, NetworkSet, list[np.ndarray]]:
    sn = make_subnet("sn", model_cls, n_nodes)
    nets = NetworkSet(subnets=[sn], projections=[], stimuli=[])
    nets.configure()
    return model_cls.__name__, nets, [make_initial_state(sn)]


def make_coupled_network(model_cls, n_nodes: int) -> tuple[str, NetworkSet, list[np.ndarray]]:
    sn1 = make_subnet("sn1", model_cls, n_nodes)
    sn2 = make_subnet("sn2", model_cls, n_nodes)
    weights = sp.eye(n_nodes, format="csr", dtype=np.float64) * 0.1
    lengths = sp.csr_matrix((n_nodes, n_nodes), dtype=np.float64)
    projection = InterProjection(
        source=sn1,
        target=sn2,
        source_cvar=np.array([0]),
        target_cvar=np.array([0]),
        weights=weights,
        lengths=lengths,
        cv=3.0,
        dt=DT,
        scale=1.0,
        cfun=Linear(),
    )
    nets = NetworkSet(subnets=[sn1, sn2], projections=[projection], stimuli=[])
    nets.configure()
    return f"Coupled{model_cls.__name__}", nets, [make_initial_state(sn1), make_initial_state(sn2)]


def run_case(label: str, nets: NetworkSet, initial_states: list[np.ndarray], chunk_size: int = 1) -> None:
    py_t = run_python(nets, initial_states, NSTEP)
    compile_t, nb_t = run_numba(nets, initial_states, NSTEP, chunk_size)
    speedup = py_t / nb_t if nb_t > 0 else float("inf")

    print(f"{label:<30} | {py_t:>10.4f} | {compile_t:>11.4f} | {nb_t:>10.4f} | {speedup:>7.2f}x")


def main() -> None:
    print("=== Hybrid Simulator vs Numba Benchmark ===")
    print(f"TVB_USER_HOME = {os.environ['TVB_USER_HOME']}")
    print(f"dt            = {DT} ms")
    print(f"nstep         = {NSTEP}")
    print()

    # Test with chunk_size=1 (raw output)
    print("Chunk Size = 1 (Raw Output)")
    print(f"{'Case':<30} | {'Python(s)':>10} | {'Compile(s)':>11} | {'Numba(s)':>10} | {'Speedup':>8}")
    print(f"{'-' * 30} | {'-' * 10} | {'-' * 11} | {'-' * 10} | {'-' * 8}")

    for model_cls, n_nodes in [
        (MontbrioPazoRoxin, 16),
        (JansenRit, 16),
        (Epileptor, 16),
        (WilsonCowan, 16),
        (ReducedWongWang, 16),
        (LarterBreakspear, 16),
        (Generic2dOscillator, 16),
        (ZerlautAdaptationFirstOrder, 16),
    ]:
        label, nets, initial_states = make_single_network(model_cls, n_nodes)
        run_case(label, nets, initial_states, chunk_size=1)

    label, nets, initial_states = make_coupled_network(MontbrioPazoRoxin, 16)
    run_case(label, nets, initial_states, chunk_size=1)

    print()
    print("Chunk Size = 20 (Temporally Averaged - Single Networks Only)")
    print(f"{'Case':<30} | {'Python(s)':>10} | {'Compile(s)':>11} | {'Numba(s)':>10} | {'Speedup':>8}")
    print(f"{'-' * 30} | {'-' * 10} | {'-' * 11} | {'-' * 10} | {'-' * 8}")

    for model_cls, n_nodes in [
        (MontbrioPazoRoxin, 16),
        (JansenRit, 16),
        (Epileptor, 16),
        (WilsonCowan, 16),
        (ReducedWongWang, 16),
        (LarterBreakspear, 16),
        (Generic2dOscillator, 16),
        (ZerlautAdaptationFirstOrder, 16),
    ]:
        label, nets, initial_states = make_single_network(model_cls, n_nodes)
        run_case(label, nets, initial_states, chunk_size=20)

    print()
    print("NOTE: Coupled networks require chunk_size <= minimum delay (1 step)")
    print("Compile time includes code generation plus first JIT warmup.")
    print("Numba(s) is the cached compiled-kernel run time.")


if __name__ == "__main__":
    main()
