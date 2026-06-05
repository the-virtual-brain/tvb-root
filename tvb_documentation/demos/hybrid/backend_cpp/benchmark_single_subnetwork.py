#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark single-subnetwork hybrid simulations across Python, Numba, and C++.

This script intentionally covers only the current native C++ backend scope:
one subnetwork, no projections, HeunDeterministic integration.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np


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
from tvb.simulator.hybrid import NetworkSet, Subnetwork
from tvb.simulator.integrators import HeunStochastic
from tvb.simulator.monitors import TemporalAverage
from tvb.simulator.noise import Additive


DT = 0.1
NOISE_SIGMA = 0.01
DEFAULT_NODES = 16
DEFAULT_NSTEP = 1000
DEFAULT_CHUNK_SIZE = 1
DEFAULT_REPEATS = 3
OUTPUT_DIR = EXAMPLES_DIR / "outputs"

MODEL_CASES: dict[str, dict[str, Any]] = {
    "JansenRit": {
        "class_path": "tvb.simulator.models.jansen_rit:JansenRit",
        "params": {},
    },
    "Epileptor": {
        "class_path": "tvb.simulator.models.epileptor:Epileptor",
        "params": {},
    },
    "MontbrioPazoRoxin": {
        "class_path": "tvb.simulator.models.infinite_theta:MontbrioPazoRoxin",
        "params": {"I": np.array([2.0])},
    },
    "WilsonCowan": {
        "class_path": "tvb.simulator.models.wilson_cowan:WilsonCowan",
        "params": {},
    },
    "WongWang": {
        "class_path": "tvb.simulator.models.wong_wang:ReducedWongWang",
        "params": {},
    },
    "LarterBreakspear": {
        "class_path": "tvb.simulator.models.larter_breakspear:LarterBreakspear",
        "params": {},
    },
    "Generic2dOscillator": {
        "class_path": "tvb.simulator.models.oscillator:Generic2dOscillator",
        "params": {},
    },
    "ZerlautAdaptationFirstOrder": {
        "class_path": "tvb.simulator.models.zerlaut:ZerlautAdaptationFirstOrder",
        "params": {},
    },
}


def load_model_class(model_name: str):
    module_name, class_name = MODEL_CASES[model_name]["class_path"].split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def make_subnet(name: str, model_name: str, n_nodes: int) -> Subnetwork:
    model_cls = load_model_class(model_name)
    model = model_cls(**MODEL_CASES[model_name]["params"])
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
    )
    subnet.node_indices = np.arange(subnet.nnodes)
    return subnet.configure()


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


def make_single_network(model_name: str, n_nodes: int) -> tuple[NetworkSet, list[np.ndarray]]:
    subnet = make_subnet("sn", model_name, n_nodes)
    nets = NetworkSet(subnets=[subnet], projections=[])
    nets.configure()
    return nets, [make_initial_state(subnet)]


def best_of(repeats: int, fn) -> float:
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - t0)
    return min(timings)


def run_python_once(
    network_set: NetworkSet,
    initial_states: list[np.ndarray],
    nstep: int,
) -> None:
    x = network_set.States(*[arr.copy() for arr in initial_states])
    network_set.init_projection_buffers(x)
    for step in range(1, nstep + 1):
        x = network_set.step(step, x)


def benchmark_python(
    network_set: NetworkSet,
    initial_states: list[np.ndarray],
    nstep: int,
    repeats: int,
) -> float:
    return best_of(
        repeats,
        lambda: run_python_once(network_set, initial_states, nstep),
    )


def benchmark_numba(
    network_set: NetworkSet,
    initial_states: list[np.ndarray],
    nstep: int,
    chunk_size: int,
    repeats: int,
) -> tuple[float, float]:
    backend = NbHybridBackend()

    t0 = time.perf_counter()
    compiled = backend.compile(network_set)
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
    return compile_time, run_time


def benchmark_cpp(
    network_set: NetworkSet,
    initial_states: list[np.ndarray],
    nstep: int,
    chunk_size: int,
    repeats: int,
    model_name: str,
) -> tuple[float, float]:
    backend = CppHybridBackend()
    monitor = TemporalAverage(period=chunk_size * DT)

    t0 = time.perf_counter()
    compiled = backend.compile(
        network_set,
        monitors=[monitor],
        user_source_hint=f"benchmark_single_subnetwork_cpp_{model_name}",
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
    return compile_time, run_time


def speedup(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def format_time(value: float | None) -> str:
    return "skip" if value is None else f"{value:10.4f}"


def format_speedup(value: float | None) -> str:
    return "skip" if value is None else f"{value:7.2f}x"


def benchmark_case(
    model_name: str,
    n_nodes: int,
    nstep: int,
    chunk_size: int,
    repeats: int,
) -> dict[str, Any]:
    network_set, initial_states = make_single_network(model_name, n_nodes)
    py_time = benchmark_python(network_set, initial_states, nstep, repeats)
    nb_compile, nb_time = benchmark_numba(
        network_set,
        initial_states,
        nstep,
        chunk_size,
        repeats,
    )
    cpp_compile, cpp_time = benchmark_cpp(
        network_set,
        initial_states,
        nstep,
        chunk_size,
        repeats,
        model_name,
    )
    return {
        "model": model_name,
        "status": "ok",
        "nodes": n_nodes,
        "nstep": nstep,
        "chunk_size": chunk_size,
        "repeats": repeats,
        "python_run_s": py_time,
        "numba_compile_s": nb_compile,
        "numba_run_s": nb_time,
        "cpp_compile_s": cpp_compile,
        "cpp_run_s": cpp_time,
        "python_vs_numba_speedup": speedup(py_time, nb_time),
        "python_vs_cpp_speedup": speedup(py_time, cpp_time),
        "numba_vs_cpp_speedup": speedup(nb_time, cpp_time),
    }


def print_header(nstep: int, n_nodes: int, chunk_size: int, repeats: int) -> None:
    print("=== Single-Subnetwork Hybrid Backend Benchmark (HeunStochastic) ===")
    print(f"TVB_USER_HOME = {os.environ['TVB_USER_HOME']}")
    print(f"dt            = {DT} ms")
    print(f"noise_sigma   = {NOISE_SIGMA}")
    print(f"nodes         = {n_nodes}")
    print(f"nstep         = {nstep}")
    print(f"chunk_size    = {chunk_size}")
    print(f"repeats       = {repeats}")
    print()
    print(
        f"{'Model':<28} | {'Python(s)':>10} | {'NumbaC(s)':>10} | "
        f"{'Numba(s)':>10} | {'CppC(s)':>10} | {'Cpp(s)':>10} | "
        f"{'Py/Cpp':>8} | {'Nb/Cpp':>8}"
    )
    print(
        f"{'-' * 28} | {'-' * 10} | {'-' * 10} | {'-' * 10} | "
        f"{'-' * 10} | {'-' * 10} | {'-' * 8} | {'-' * 8}"
    )


def print_result(row: dict[str, Any]) -> None:
    if row["status"] != "ok":
        print(f"{row['model']:<28} | skipped: {row['reason']}")
        return
    print(
        f"{row['model']:<28} | "
        f"{format_time(row['python_run_s'])} | "
        f"{format_time(row['numba_compile_s'])} | "
        f"{format_time(row['numba_run_s'])} | "
        f"{format_time(row['cpp_compile_s'])} | "
        f"{format_time(row['cpp_run_s'])} | "
        f"{format_speedup(row['python_vs_cpp_speedup']):>8} | "
        f"{format_speedup(row['numba_vs_cpp_speedup']):>8}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark single-subnetwork Python, Numba, and C++ hybrid backends."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CASES.keys()),
        choices=sorted(MODEL_CASES.keys()),
    )
    parser.add_argument("--nodes", type=int, default=DEFAULT_NODES)
    parser.add_argument("--nstep", type=int, default=DEFAULT_NSTEP)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=OUTPUT_DIR / "single_subnetwork_cpp_benchmark.json",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise immediately instead of recording skipped benchmark cases.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.nstep < 1:
        raise ValueError("nstep must be >= 1.")
    if args.chunk_size < 1:
        raise ValueError("chunk-size must be >= 1.")
    if args.repeats < 1:
        raise ValueError("repeats must be >= 1.")

    print_header(args.nstep, args.nodes, args.chunk_size, args.repeats)
    rows: list[dict[str, Any]] = []
    for model_name in args.models:
        try:
            row = benchmark_case(
                model_name=model_name,
                n_nodes=args.nodes,
                nstep=args.nstep,
                chunk_size=args.chunk_size,
                repeats=args.repeats,
            )
        except Exception as exc:
            if args.strict:
                raise
            row = {
                "model": model_name,
                "status": "skipped",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        print_result(row)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print()
    print(f"JSON summary written to {args.output_json}")


if __name__ == "__main__":
    main()
