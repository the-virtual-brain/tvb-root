#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark Numba and C++ hybrid backends across topology/chunk-size scenarios.

The goal is to expose where native C++ still needs speedup work, especially
for single versus multi-subnetwork runs and raw versus chunked output.
"""

from __future__ import annotations

import argparse
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
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "numba-cache"))
warnings.filterwarnings("ignore", message="Hybrid simulation is experimental: .*")

from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.backend_cpp import CppHybridBackend
from tvb.simulator.hybrid import InterProjection, NetworkSet, Subnetwork
from tvb.simulator.hybrid.coupling import Linear
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.monitors import TemporalAverage


DT = 0.1
DEFAULT_NODES = 16
DEFAULT_NSTEP = 10000
DEFAULT_CHUNK_SIZES = (1, 10)
DEFAULT_REPEATS = 3
OUTPUT_DIR = EXAMPLES_DIR / "outputs"


@dataclass(frozen=True)
class Scenario:
    label: str
    builder: Callable[[int, int], tuple[NetworkSet, list[np.ndarray]]]


def make_subnet(name: str, n_nodes: int, node_offset: int = 0) -> Subnetwork:
    model = MontbrioPazoRoxin(I=np.array([2.0]))
    model.configure()
    subnet = Subnetwork(
        name=name,
        model=model,
        scheme=HeunDeterministic(dt=DT),
        nnodes=n_nodes,
    ).configure()
    subnet.node_indices = np.arange(node_offset, node_offset + n_nodes)
    return subnet


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


def make_single_network(
    n_nodes: int,
    _chunk_size: int,
) -> tuple[NetworkSet, list[np.ndarray]]:
    sn = make_subnet("sn", n_nodes)
    network = NetworkSet(subnets=[sn], projections=[], stimuli=[])
    network.configure()
    return network, [make_initial_state(sn)]


def make_uncoupled_two_subnet_network(
    n_nodes: int,
    _chunk_size: int,
) -> tuple[NetworkSet, list[np.ndarray]]:
    sn1 = make_subnet("sn1", n_nodes, node_offset=0)
    sn2 = make_subnet("sn2", n_nodes, node_offset=n_nodes)
    network = NetworkSet(subnets=[sn1, sn2], projections=[], stimuli=[])
    network.configure()
    return network, [make_initial_state(sn1), make_initial_state(sn2)]


def make_delayed_coupled_two_subnet_network(
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

    network = NetworkSet(subnets=[sn1, sn2], projections=[projection], stimuli=[])
    network.configure()
    return network, [make_initial_state(sn1), make_initial_state(sn2)]


SCENARIOS = (
    Scenario("single_subnet", make_single_network),
    Scenario("two_subnets_uncoupled", make_uncoupled_two_subnet_network),
    Scenario("two_subnets_delayed_projection", make_delayed_coupled_two_subnet_network),
)


def best_of(repeats: int, fn: Callable[[], Any]) -> float:
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - t0)
    return min(timings)


def benchmark_numba(
    network: NetworkSet,
    initial_states: list[np.ndarray],
    nstep: int,
    chunk_size: int,
    repeats: int,
) -> dict[str, Any]:
    backend = NbHybridBackend()
    t0 = time.perf_counter()
    compiled = backend.compile(network)
    compiled.run(
        nstep=min(5, nstep),
        chunk_size=chunk_size,
        initial_states=[arr.copy() for arr in initial_states],
    )
    compile_time = time.perf_counter() - t0

    run_time = best_of(
        repeats,
        lambda: compiled.run(
            nstep=nstep,
            chunk_size=chunk_size,
            initial_states=[arr.copy() for arr in initial_states],
        ),
    )
    return {"status": "ok", "compile_s": compile_time, "run_s": run_time}


def benchmark_cpp(
    network: NetworkSet,
    initial_states: list[np.ndarray],
    nstep: int,
    chunk_size: int,
    repeats: int,
    source_hint: str,
) -> dict[str, Any]:
    backend = CppHybridBackend()
    monitor = TemporalAverage(period=chunk_size * DT)

    t0 = time.perf_counter()
    compiled = backend.compile(
        network,
        monitors=[monitor],
        user_source_hint=source_hint,
    )
    compiled.run(
        nstep=min(5, nstep),
        chunk_size=chunk_size,
        initial_states=[arr.copy() for arr in initial_states],
    )
    compile_time = time.perf_counter() - t0

    run_time = best_of(
        repeats,
        lambda: compiled.run(
            nstep=nstep,
            chunk_size=chunk_size,
            initial_states=[arr.copy() for arr in initial_states],
        ),
    )
    return {"status": "ok", "compile_s": compile_time, "run_s": run_time}


def try_benchmark(fn: Callable[[], dict[str, Any]], strict: bool) -> dict[str, Any]:
    try:
        return fn()
    except Exception as exc:
        if strict:
            raise
        return {
            "status": "skipped",
            "reason": f"{type(exc).__name__}: {exc}",
        }


def speedup(numba_run_s: float | None, cpp_run_s: float | None) -> float | None:
    if numba_run_s is None or cpp_run_s is None or cpp_run_s <= 0:
        return None
    return numba_run_s / cpp_run_s


def fmt_time(result: dict[str, Any], key: str) -> str:
    if result["status"] != "ok":
        return "skip"
    return f"{result[key]:9.4f}"


def fmt_speedup(value: float | None) -> str:
    if value is None:
        return "skip"
    return f"{value:7.2f}x"


def print_header(args: argparse.Namespace) -> None:
    print("=== C++ vs Numba Hybrid Backend Scenario Benchmark ===")
    print(f"TVB_USER_HOME = {os.environ['TVB_USER_HOME']}")
    print(f"dt            = {DT} ms")
    print(f"nodes/subnet  = {args.nodes}")
    print(f"nstep         = {args.nstep}")
    print(f"chunk_sizes   = {', '.join(str(v) for v in args.chunk_sizes)}")
    print(f"repeats       = {args.repeats}")
    print()
    print(
        f"{'Scenario':<32} | {'chunk':>5} | {'NbC(s)':>9} | {'Nb(s)':>9} | "
        f"{'CppC(s)':>9} | {'Cpp(s)':>9} | {'Nb/Cpp':>8} | Status"
    )
    print(
        f"{'-' * 32} | {'-' * 5} | {'-' * 9} | {'-' * 9} | "
        f"{'-' * 9} | {'-' * 9} | {'-' * 8} | {'-' * 20}"
    )


def print_row(row: dict[str, Any]) -> None:
    nb = row["numba"]
    cpp = row["cpp"]
    reasons = []
    if nb["status"] != "ok":
        reasons.append(f"numba: {nb['reason']}")
    if cpp["status"] != "ok":
        reasons.append(f"cpp: {cpp['reason']}")
    status = "ok" if not reasons else "; ".join(reasons)
    print(
        f"{row['scenario']:<32} | {row['chunk_size']:>5} | "
        f"{fmt_time(nb, 'compile_s')} | {fmt_time(nb, 'run_s')} | "
        f"{fmt_time(cpp, 'compile_s')} | {fmt_time(cpp, 'run_s')} | "
        f"{fmt_speedup(row['numba_vs_cpp_speedup']):>8} | {status}"
    )


def benchmark_case(
    scenario: Scenario,
    n_nodes: int,
    nstep: int,
    chunk_size: int,
    repeats: int,
    strict: bool,
) -> dict[str, Any]:
    network, initial_states = scenario.builder(n_nodes, chunk_size)
    numba = try_benchmark(
        lambda: benchmark_numba(network, initial_states, nstep, chunk_size, repeats),
        strict,
    )
    cpp = try_benchmark(
        lambda: benchmark_cpp(
            network,
            initial_states,
            nstep,
            chunk_size,
            repeats,
            source_hint=f"benchmark_cpp_vs_numba_{scenario.label}_chunk_{chunk_size}",
        ),
        strict,
    )

    nb_time = numba.get("run_s") if numba["status"] == "ok" else None
    cpp_time = cpp.get("run_s") if cpp["status"] == "ok" else None
    return {
        "scenario": scenario.label,
        "nodes_per_subnet": n_nodes,
        "nstep": nstep,
        "chunk_size": chunk_size,
        "repeats": repeats,
        "numba": numba,
        "cpp": cpp,
        "numba_vs_cpp_speedup": speedup(nb_time, cpp_time),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Numba vs C++ hybrid backend scenarios."
    )
    parser.add_argument("--nodes", type=int, default=DEFAULT_NODES)
    parser.add_argument("--nstep", type=int, default=DEFAULT_NSTEP)
    parser.add_argument(
        "--chunk-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_CHUNK_SIZES),
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=[scenario.label for scenario in SCENARIOS],
        default=[scenario.label for scenario in SCENARIOS],
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=OUTPUT_DIR / "cpp_vs_numba_scenario_benchmark.json",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise immediately instead of recording skipped backend cases.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.nodes < 1:
        raise ValueError("nodes must be >= 1.")
    if args.nstep < 1:
        raise ValueError("nstep must be >= 1.")
    if args.repeats < 1:
        raise ValueError("repeats must be >= 1.")
    if any(chunk_size < 1 for chunk_size in args.chunk_sizes):
        raise ValueError("all chunk sizes must be >= 1.")

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
