#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark hybrid Python, Numba, Numba-CUDA sweep, and C++ backends.

CUDA is currently exposed as a parameter-sweep backend.  To keep the comparison
explicit, projection scenarios sweep a global coupling value:

* Python, Numba, C++: run ``sweeps`` independent simulations sequentially.
* Numba-CUDA: run ``sweeps`` independent simulations in one GPU batch.

The table reports total run time for the full batch and per-simulation time.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp
from numba.core.errors import NumbaPerformanceWarning


EXAMPLES_DIR = Path(__file__).resolve().parent
BACKEND_CPP_DIR = EXAMPLES_DIR.parent
SIMULATOR_DIR = BACKEND_CPP_DIR.parent
TVB_LIBRARY_ROOT = SIMULATOR_DIR.parent.parent
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))
if str(TVB_LIBRARY_ROOT) not in sys.path:
    sys.path.insert(0, str(TVB_LIBRARY_ROOT))

os.environ.setdefault("TVB_USER_HOME", str(Path(tempfile.gettempdir()) / "tvb-user"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

warnings.filterwarnings("ignore", message="Hybrid simulation is experimental: .*")
from numba.core.errors import NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.backend_cpp import CppHybridBackend
from tvb.simulator.hybrid import InterProjection, NetworkSet, Subnetwork
from tvb.simulator.hybrid.coupling import Linear
from tvb.simulator.integrators import HeunStochastic
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.noise import Additive


DT = 0.1
NOISE_SIGMA = 0.01
DEFAULT_NODES = 16
DEFAULT_NSTEP = 1000
DEFAULT_CHUNK_SIZES = (1, 10)
DEFAULT_REPEATS = 2
DEFAULT_SWEEPS = 8192
DEFAULT_CPU_SAMPLE_SWEEPS = 64
DEFAULT_CPP_PARALLEL_WORKERS = (4, 8)
OUTPUT_DIR = EXAMPLES_DIR / "outputs"


@dataclass(frozen=True)
class Scenario:
    label: str
    builder: Callable[[int, int], tuple[NetworkSet, list[np.ndarray]]]
    sweep_kind: str


def make_subnet(name: str, n_nodes: int, node_offset: int = 0) -> Subnetwork:
    model = MontbrioPazoRoxin(I=np.array([2.0]))
    model.configure()
    noise = Additive(nsig=np.array([NOISE_SIGMA]))
    noise.configure()
    scheme = HeunStochastic(dt=DT, noise=noise)
    scheme.configure()
    subnet = Subnetwork(
        name=name,
        model=model,
        scheme=scheme,
        nnodes=n_nodes,
    ).configure()
    subnet.node_indices = np.arange(node_offset, node_offset + n_nodes)
    return subnet


def make_initial_state(subnet: Subnetwork) -> np.ndarray:
    model = subnet.model
    x0 = np.zeros(
        (model.nvar, subnet.nnodes, model.number_of_modes),
        dtype=np.float64,
    )
    for i, state_var in enumerate(model.state_variables):
        if state_var not in model.state_variable_range:
            continue
        low, high = map(float, model.state_variable_range[state_var])
        x0[i, :, :] = (low + high) / 2.0
    return x0


def make_single_subnet(
    n_nodes: int,
    _chunk_size: int,
) -> tuple[NetworkSet, list[np.ndarray]]:
    sn = make_subnet("sn", n_nodes)
    network = NetworkSet(subnets=[sn], projections=[]).configure()
    return network, [make_initial_state(sn)]


def make_two_subnets_uncoupled(
    n_nodes: int,
    _chunk_size: int,
) -> tuple[NetworkSet, list[np.ndarray]]:
    sn1 = make_subnet("sn1", n_nodes, node_offset=0)
    sn2 = make_subnet("sn2", n_nodes, node_offset=n_nodes)
    network = NetworkSet(subnets=[sn1, sn2], projections=[]).configure()
    return network, [make_initial_state(sn1), make_initial_state(sn2)]


def make_two_subnets_delayed_projection(
    n_nodes: int,
    chunk_size: int,
) -> tuple[NetworkSet, list[np.ndarray]]:
    sn1 = make_subnet("sn1", n_nodes, node_offset=0)
    sn2 = make_subnet("sn2", n_nodes, node_offset=n_nodes)

    weights = sp.eye(n_nodes, format="csr", dtype=np.float64) * 0.1
    cv = 3.0
    delay_steps = max(1, chunk_size)
    delay_ms = delay_steps * DT
    lengths = sp.eye(n_nodes, format="csr", dtype=np.float64) * (delay_ms * cv)
    projection = InterProjection(
        source=sn1,
        target=sn2,
        source_cvar=np.array([0]),
        target_cvar=np.array([0]),
        weights=weights,
        lengths=lengths,
        cv=cv,
        dt=DT,
        scale=1.0,
        cfun=Linear(),
    )

    network = NetworkSet(subnets=[sn1, sn2], projections=[projection]).configure()
    return network, [make_initial_state(sn1), make_initial_state(sn2)]


SCENARIOS = (
    Scenario("single_subnet", make_single_subnet, "none"),
    Scenario("two_subnets_uncoupled", make_two_subnets_uncoupled, "none"),
    Scenario("two_subnets_delayed_projection", make_two_subnets_delayed_projection, "coupling"),
)


@dataclass(frozen=True)
class SweepPlan:
    kind: str
    label: str
    values: np.ndarray
    cuda_descriptor: list[dict[str, Any]] | None


def make_sweep_plan(
    network: NetworkSet,
    scenario: Scenario,
    sweeps: int,
    sweep_start: float,
    sweep_stop: float,
) -> SweepPlan:
    if scenario.sweep_kind == "coupling":
        if not network.projections:
            raise ValueError(f"scenario {scenario.label!r} has no projection to sweep.")
        projection = network.projections[0]
        values = np.linspace(sweep_start, sweep_stop, sweeps, dtype=np.float32)
        return SweepPlan(
            kind="coupling",
            label="global_coupling",
            values=values,
            cuda_descriptor=None,
        )

    return SweepPlan(
        kind="none",
        label="none",
        values=np.ones(sweeps, dtype=np.float32),
        cuda_descriptor=[],
    )


def set_python_sweep_value(network: NetworkSet, plan: SweepPlan, value: float) -> None:
    if plan.kind != "coupling":
        return
    projection = network.projections[0]
    projection.cfun.a = np.array([value], dtype=np.float64)


def set_numba_sweep_value(compiled: Any, plan: SweepPlan, value: float) -> None:
    if plan.kind != "coupling":
        return
    projection = compiled._analysis.all_projections[0]
    projection.cfun.a = np.array([value], dtype=np.float64)


def set_cpp_sweep_value(compiled: Any, plan: SweepPlan, value: float) -> None:
    if plan.kind != "coupling":
        return
    if compiled.spec.inter_projections:
        projections = list(compiled.spec.inter_projections)
        projections[0] = dataclasses.replace(projections[0], scale=float(value))
        object.__setattr__(
            compiled,
            "spec",
            dataclasses.replace(compiled.spec, inter_projections=tuple(projections)),
        )
    elif compiled.spec.intra_projections:
        projections = list(compiled.spec.intra_projections)
        projections[0] = dataclasses.replace(projections[0], scale=float(value))
        object.__setattr__(
            compiled,
            "spec",
            dataclasses.replace(compiled.spec, intra_projections=tuple(projections)),
        )


def cuda_sweep_values(plan: SweepPlan) -> np.ndarray:
    if plan.kind == "coupling":
        return plan.values.reshape(-1, 1).astype(np.float32, copy=False)
    return np.empty((plan.values.size, 0), dtype=np.float32)


def sampled_plan(plan: SweepPlan, sample_sweeps: int) -> SweepPlan:
    return dataclasses.replace(plan, values=plan.values[:sample_sweeps])


def extrapolate_result(
    result: dict[str, Any],
    measured_sweeps: int,
    target_sweeps: int,
) -> dict[str, Any]:
    if result["status"] != "ok":
        return result
    measured_run_s = result["run_s"]
    result["measured_run_s"] = measured_run_s
    result["measured_sweeps"] = measured_sweeps
    result["target_sweeps"] = target_sweeps
    result["extrapolated"] = measured_sweeps != target_sweeps
    result["run_s"] = measured_run_s * (target_sweeps / measured_sweeps)
    result["per_sim_s"] = measured_run_s / measured_sweeps
    return result


def best_of(repeats: int, fn: Callable[[], Any]) -> float:
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - t0)
    return min(timings)


def copies(initial_states: list[np.ndarray]) -> list[np.ndarray]:
    return [arr.copy() for arr in initial_states]


def run_python_once(
    network: NetworkSet,
    initial_states: list[np.ndarray],
    nstep: int,
) -> None:
    x = network.States(*copies(initial_states))
    network.init_projection_buffers(x)
    for step in range(1, nstep + 1):
        x = network.step(step, x)


def run_sweep_values(values: np.ndarray, fn: Callable[[float], Any]) -> None:
    for value in values:
        fn(float(value))


def benchmark_python(
    network: NetworkSet,
    initial_states: list[np.ndarray],
    nstep: int,
    repeats: int,
    plan: SweepPlan,
) -> dict[str, Any]:
    run_time = best_of(
        repeats,
        lambda: run_sweep_values(
            plan.values,
            lambda value: (
                set_python_sweep_value(network, plan, value),
                run_python_once(network, initial_states, nstep),
            ),
        ),
    )
    sweeps = plan.values.size
    return {"status": "ok", "compile_s": 0.0, "run_s": run_time, "per_sim_s": run_time / sweeps}


def benchmark_numba(
    network: NetworkSet,
    initial_states: list[np.ndarray],
    nstep: int,
    chunk_size: int,
    repeats: int,
    plan: SweepPlan,
) -> dict[str, Any]:
    backend = NbHybridBackend()
    t0 = time.perf_counter()
    compiled = backend.compile(network)
    compiled.run(
        nstep=min(5, nstep),
        chunk_size=chunk_size,
        initial_states=copies(initial_states),
    )
    compile_time = time.perf_counter() - t0

    run_time = best_of(
        repeats,
        lambda: run_sweep_values(
            plan.values,
            lambda value: (
                set_numba_sweep_value(compiled, plan, value),
                compiled.run(
                    nstep=nstep,
                    chunk_size=chunk_size,
                    initial_states=copies(initial_states),
                ),
            ),
        ),
    )
    sweeps = plan.values.size
    return {"status": "ok", "compile_s": compile_time, "run_s": run_time, "per_sim_s": run_time / sweeps}


def benchmark_cpp(
    network: NetworkSet,
    initial_states: list[np.ndarray],
    nstep: int,
    chunk_size: int,
    repeats: int,
    plan: SweepPlan,
    source_hint: str,
    allow_scale_proxy: bool,
) -> dict[str, Any]:
    if plan.kind == "coupling" and not allow_scale_proxy:
        return {
            "status": "skipped",
            "reason": (
                "C++ does not currently expose the CUDA cfun.a sweep parameter "
                "at runtime; pass --cpp-scale-proxy to benchmark projection scale "
                "as an approximate proxy."
            ),
        }

    backend = CppHybridBackend()

    t0 = time.perf_counter()
    compiled = backend.compile(
        network,
        monitors=None,
        user_source_hint=source_hint,
    )
    compiled.run(
        nstep=min(5, nstep),
        chunk_size=chunk_size,
        initial_states=copies(initial_states),
    )
    compile_time = time.perf_counter() - t0

    run_time = best_of(
        repeats,
        lambda: run_sweep_values(
            plan.values,
            lambda value: (
                set_cpp_sweep_value(compiled, plan, value),
                compiled.run(
                    nstep=nstep,
                    chunk_size=chunk_size,
                    initial_states=copies(initial_states),
                ),
            ),
        ),
    )
    sweeps = plan.values.size
    result = {"status": "ok", "compile_s": compile_time, "run_s": run_time, "per_sim_s": run_time / sweeps}
    if plan.kind == "coupling" and allow_scale_proxy:
        result["proxy_sweep"] = "projection.scale used as proxy for Linear.a"
    return result


def benchmark_cpp_parallel(
    network: NetworkSet,
    initial_states: list[np.ndarray],
    nstep: int,
    chunk_size: int,
    repeats: int,
    plan: SweepPlan,
    n_workers: int,
) -> dict[str, Any]:
    if plan.kind != "coupling":
        return {
            "status": "skipped",
            "reason": "C++ parallel sweep requires a coupling/cfun parameter sweep.",
        }

    from tvb.simulator.monitors import TemporalAverage

    backend = CppHybridBackend()
    monitor = TemporalAverage(period=chunk_size * DT)
    sweep_values = plan.values.astype(np.float32, copy=False)

    t0 = time.perf_counter()
    backend.sweep(
        network,
        params={"coupling_scale": sweep_values[:1]},
        nstep=min(5, nstep),
        monitors=[monitor],
        initial_states=copies(initial_states),
        n_workers=n_workers,
    )
    compile_time = time.perf_counter() - t0

    def run_once() -> None:
        backend.sweep(
            network,
            params={"coupling_scale": sweep_values},
            nstep=nstep,
            monitors=[monitor],
            initial_states=copies(initial_states),
            n_workers=n_workers,
        )

    run_time = best_of(repeats, run_once)
    sweeps = plan.values.size
    return {
        "status": "ok",
        "backend": f"cpp-par-{n_workers}",
        "workers": n_workers,
        "compile_s": compile_time,
        "run_s": run_time,
        "per_sim_s": run_time / sweeps,
    }


def cuda_available() -> tuple[bool, str | None]:
    try:
        from numba import cuda

        if not cuda.is_available():
            return False, "numba.cuda reports no available CUDA device"
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

def benchmark_cuda(
    network: NetworkSet,
    initial_states: list[np.ndarray],
    nstep: int,
    chunk_size: int,
    repeats: int,
    plan: SweepPlan,
    max_batch_sweeps: int | None,
) -> dict[str, Any]:
    ok, reason = cuda_available()
    if not ok:
        return {"status": "skipped", "reason": reason}

    from tvb.simulator.backend.nb_hybrid_cuda_sweep_backend import (
        NbHybridCUDASweepBackend,
    )

    sweep_values = cuda_sweep_values(plan)
    t0 = time.perf_counter()
    compiled = NbHybridCUDASweepBackend().compile_sweep(
        network,
        sweep_descriptor=plan.cuda_descriptor,
    )
    compiled.run(
        nstep=min(5, nstep),
        sweep_values=sweep_values,
        initial_states=copies(initial_states),
        chunk_size=chunk_size,
        max_batch_sweeps=max_batch_sweeps,
    )
    compile_time = time.perf_counter() - t0

    run_time = best_of(
        repeats,
        lambda: compiled.run(
            nstep=nstep,
            sweep_values=sweep_values,
            initial_states=copies(initial_states),
            chunk_size=chunk_size,
            max_batch_sweeps=max_batch_sweeps,
        ),
    )
    sweeps = plan.values.size
    return {"status": "ok", "compile_s": compile_time, "run_s": run_time, "per_sim_s": run_time / sweeps}


def try_benchmark(fn: Callable[[], dict[str, Any]], strict: bool) -> dict[str, Any]:
    try:
        return fn()
    except Exception as exc:
        if strict:
            raise
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def ok_time(result: dict[str, Any], key: str) -> float | None:
    if result["status"] != "ok":
        return None
    return result[key]


def fmt_result(result: dict[str, Any], key: str) -> str:
    if result["status"] != "ok":
        return "skip"
    return f"{result[key]:9.4f}"


def fmt_ratio(value: float | None) -> str:
    if value is None:
        return "skip"
    return f"{value:7.2f}x"


def benchmark_case(
    scenario: Scenario,
    n_nodes: int,
    nstep: int,
    chunk_size: int,
    repeats: int,
    sweeps: int,
    cpu_sample_sweeps: int | None,
    max_batch_sweeps: int | None,
    cpp_scale_proxy: bool,
    cpp_parallel_workers: tuple[int, ...],
    sweep_start: float,
    sweep_stop: float,
    strict: bool,
) -> dict[str, Any]:
    network, initial_states = scenario.builder(n_nodes, chunk_size)
    plan = make_sweep_plan(network, scenario, sweeps, sweep_start, sweep_stop)
    measured_cpu_sweeps = min(cpu_sample_sweeps or sweeps, sweeps)
    cpu_plan = sampled_plan(plan, measured_cpu_sweeps)

    py = try_benchmark(
        lambda: benchmark_python(network, initial_states, nstep, repeats, cpu_plan),
        strict,
    )
    py = extrapolate_result(py, measured_cpu_sweeps, sweeps)
    nb = try_benchmark(
        lambda: benchmark_numba(network, initial_states, nstep, chunk_size, repeats, cpu_plan),
        strict,
    )
    nb = extrapolate_result(nb, measured_cpu_sweeps, sweeps)
    cuda = try_benchmark(
        lambda: benchmark_cuda(
            network,
            initial_states,
            nstep,
            chunk_size,
            repeats,
            plan,
            max_batch_sweeps,
        ),
        strict,
    )
    cpp = try_benchmark(
        lambda: benchmark_cpp(
            network,
            initial_states,
            nstep,
            chunk_size,
            repeats,
            cpu_plan,
            source_hint=f"benchmark_hybrid_backends_{scenario.label}_chunk_{chunk_size}",
            allow_scale_proxy=cpp_scale_proxy,
        ),
        strict,
    )
    cpp = extrapolate_result(cpp, measured_cpu_sweeps, sweeps)

    cpp_parallel = {
        f"cpp_parallel_{n_workers}": try_benchmark(
            lambda n_workers=n_workers: benchmark_cpp_parallel(
                network,
                initial_states,
                nstep,
                chunk_size,
                repeats,
                plan,
                n_workers,
            ),
            strict,
        )
        for n_workers in cpp_parallel_workers
    }

    py_per = ok_time(py, "per_sim_s")
    py_total = ok_time(py, "run_s")
    cuda_total = ok_time(cuda, "run_s")
    cpp_per = ok_time(cpp, "per_sim_s")
    row = {
        "scenario": scenario.label,
        "nodes_per_subnet": n_nodes,
        "nstep": nstep,
        "chunk_size": chunk_size,
        "repeats": repeats,
        "sweeps": sweeps,
        "cpu_sample_sweeps": measured_cpu_sweeps,
        "cpu_times_extrapolated": measured_cpu_sweeps != sweeps,
        "sweep": {
            "kind": plan.kind,
            "label": plan.label,
            "start": float(plan.values[0]),
            "stop": float(plan.values[-1]),
            "values": plan.values.astype(float).tolist(),
        },
        "python": py,
        "numba": nb,
        "numba_cuda": cuda,
        "cpp": cpp,
        **cpp_parallel,
        "speedups_per_sim": {
            "python_vs_numba": ratio(py_per, ok_time(nb, "per_sim_s")),
            "python_vs_numba_cuda": ratio(py_per, ok_time(cuda, "per_sim_s")),
            "python_vs_cpp": ratio(py_per, cpp_per),
            "numba_vs_cpp": ratio(ok_time(nb, "per_sim_s"), cpp_per),
            "numba_cuda_vs_cpp": ratio(ok_time(cuda, "per_sim_s"), cpp_per),
        },
        "speedups_total_sweep": {
            "python_vs_numba": ratio(py_total, ok_time(nb, "run_s")),
            "python_vs_numba_cuda": ratio(py_total, cuda_total),
            "python_vs_cpp": ratio(py_total, ok_time(cpp, "run_s")),
            "numba_vs_numba_cuda": ratio(ok_time(nb, "run_s"), cuda_total),
            "cpp_vs_numba_cuda": ratio(ok_time(cpp, "run_s"), cuda_total),
        },
    }
    for key, result in cpp_parallel.items():
        row["speedups_per_sim"][f"python_vs_{key}"] = ratio(
            py_per, ok_time(result, "per_sim_s")
        )
        row["speedups_total_sweep"][f"python_vs_{key}"] = ratio(
            py_total, ok_time(result, "run_s")
        )
        row["speedups_total_sweep"][f"{key}_vs_numba_cuda"] = ratio(
            ok_time(result, "run_s"), cuda_total
        )
    return row


def print_header(args: argparse.Namespace) -> None:
    print("=== Hybrid Backend Scenario Benchmark (HeunStochastic) ===")
    print(f"TVB_USER_HOME = {os.environ['TVB_USER_HOME']}")
    print(f"dt            = {DT} ms")
    print(f"noise_sigma   = {NOISE_SIGMA}")
    print(f"nodes/subnet  = {args.nodes}")
    print(f"nstep         = {args.nstep}")
    print(f"chunk_sizes   = {', '.join(str(v) for v in args.chunk_sizes)}")
    print(f"sweeps        = {args.sweeps}")
    print(f"cpu_sample    = {args.cpu_sample_sweeps or args.sweeps}")
    print(
        "cpp workers   = "
        + (
            ", ".join(str(v) for v in args.cpp_parallel_workers)
            if args.cpp_parallel_workers
            else "disabled"
        )
    )
    print(f"sweep_range   = {args.sweep_start:g} .. {args.sweep_stop:g}")
    print(f"repeats       = {args.repeats}")
    print()
    print(
        f"{'Scenario':<32} | {'Sweep':<18} | {'chunk':>5} | "
        f"{'Py/sim':>9} | {'Nb/sim':>9} | "
        f"{'Cu/sim':>9} | {'Cpp/sim':>9} | {'PyΣ(s)':>9} | {'CuΣ(s)':>9} | "
        f"{'PyΣ/NbΣ':>9} | {'PyΣ/CuΣ':>9} | {'PyΣ/CppΣ':>9} | Status"
    )
    print(
        f"{'-' * 32} | {'-' * 18} | {'-' * 5} | {'-' * 9} | {'-' * 9} | {'-' * 9} | "
        f"{'-' * 9} | {'-' * 9} | {'-' * 9} | {'-' * 9} | {'-' * 9} | {'-' * 9} | "
        f"{'-' * 20}"
    )


def print_row(row: dict[str, Any]) -> None:
    reasons = []
    for key, label in (
        ("python", "python"),
        ("numba", "numba"),
        ("numba_cuda", "cuda"),
        ("cpp", "cpp"),
    ):
        if row[key]["status"] != "ok":
            reasons.append(f"{label}: {row[key]['reason']}")
    for key in sorted(k for k in row if k.startswith("cpp_parallel_")):
        if row[key]["status"] != "ok":
            reasons.append(f"{key}: {row[key]['reason']}")
    if row["cpu_times_extrapolated"]:
        reasons.append(
            f"cpu estimated from {row['cpu_sample_sweeps']}/{row['sweeps']} sweep points"
        )
    if row["cpp"].get("proxy_sweep"):
        reasons.append(f"cpp proxy: {row['cpp']['proxy_sweep']}")
    status = "ok" if not reasons else "; ".join(reasons)
    total_speedups = row["speedups_total_sweep"]
    sweep_label = row["sweep"]["label"]
    print(
        f"{row['scenario']:<32} | {sweep_label:<18.18} | {row['chunk_size']:>5} | "
        f"{fmt_result(row['python'], 'per_sim_s')} | "
        f"{fmt_result(row['numba'], 'per_sim_s')} | "
        f"{fmt_result(row['numba_cuda'], 'per_sim_s')} | "
        f"{fmt_result(row['cpp'], 'per_sim_s')} | "
        f"{fmt_result(row['python'], 'run_s')} | "
        f"{fmt_result(row['numba_cuda'], 'run_s')} | "
        f"{fmt_ratio(total_speedups['python_vs_numba']):>9} | "
        f"{fmt_ratio(total_speedups['python_vs_numba_cuda']):>9} | "
        f"{fmt_ratio(total_speedups['python_vs_cpp']):>9} | {status}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Python, Numba, Numba-CUDA sweep, and C++ hybrid backends."
    )
    parser.add_argument("--nodes", type=int, default=DEFAULT_NODES)
    parser.add_argument("--nstep", type=int, default=DEFAULT_NSTEP)
    parser.add_argument(
        "--chunk-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_CHUNK_SIZES),
    )
    parser.add_argument("--sweeps", type=int, default=DEFAULT_SWEEPS)
    parser.add_argument(
        "--cpu-sample-sweeps",
        type=int,
        default=DEFAULT_CPU_SAMPLE_SWEEPS,
        help=(
            "Run Python/Numba/C++ on this many sweep points and linearly "
            "extrapolate their full-sweep times. CUDA still runs all --sweeps points."
        ),
    )
    parser.add_argument(
        "--sweep-start",
        type=float,
        default=0.0,
        help="First global-coupling value for projection sweep scenarios.",
    )
    parser.add_argument(
        "--sweep-stop",
        type=float,
        default=1.0,
        help="Last global-coupling value for projection sweep scenarios.",
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=[scenario.label for scenario in SCENARIOS],
        default=[scenario.label for scenario in SCENARIOS],
    )
    parser.add_argument(
        "--max-batch-sweeps",
        type=int,
        default=None,
        help="Optional CUDA batch cap. Default lets the CUDA backend choose from GPU memory.",
    )
    parser.add_argument(
        "--cpp-parallel-workers",
        type=int,
        nargs="*",
        default=list(DEFAULT_CPP_PARALLEL_WORKERS),
        help=(
            "C++ n_workers values to benchmark through CppHybridBackend.sweep "
            "for coupling sweeps. Use no values to disable."
        ),
    )
    parser.add_argument(
        "--cpp-scale-proxy",
        action="store_true",
        default=True,
        help=(
            "For coupling sweeps, benchmark C++ by changing projection.scale as "
            "a mathematical proxy for CUDA/Numba Linear.a. On by default."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=OUTPUT_DIR / "hybrid_backends_scenario_benchmark.json",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise immediately instead of recording skipped backend cases.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.nodes < 1:
        raise ValueError("nodes must be >= 1.")
    if args.nstep < 1:
        raise ValueError("nstep must be >= 1.")
    if args.repeats < 1:
        raise ValueError("repeats must be >= 1.")
    if args.sweeps < 1:
        raise ValueError("sweeps must be >= 1.")
    if args.cpu_sample_sweeps is not None and args.cpu_sample_sweeps < 1:
        raise ValueError("cpu-sample-sweeps must be >= 1 when provided.")
    if any(chunk_size < 1 for chunk_size in args.chunk_sizes):
        raise ValueError("all chunk sizes must be >= 1.")
    if args.max_batch_sweeps is not None and args.max_batch_sweeps < 1:
        raise ValueError("max-batch-sweeps must be >= 1 when provided.")
    if any(n_workers < 2 for n_workers in args.cpp_parallel_workers):
        raise ValueError("cpp-parallel-workers values must be >= 2.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    selected = [scenario for scenario in SCENARIOS if scenario.label in args.scenarios]

    print_header(args)
    rows = []
    for scenario in selected:
        for chunk_size in args.chunk_sizes:
            row = benchmark_case(
                scenario=scenario,
                n_nodes=args.nodes,
                nstep=args.nstep,
                chunk_size=chunk_size,
                repeats=args.repeats,
                sweeps=args.sweeps,
                cpu_sample_sweeps=args.cpu_sample_sweeps,
                max_batch_sweeps=args.max_batch_sweeps,
                cpp_scale_proxy=args.cpp_scale_proxy,
                cpp_parallel_workers=tuple(dict.fromkeys(args.cpp_parallel_workers)),
                sweep_start=args.sweep_start,
                sweep_stop=args.sweep_stop,
                strict=args.strict,
            )
            rows.append(row)
            print_row(row)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print()
    print(f"JSON summary written to {args.output_json}")


if __name__ == "__main__":
    main()
