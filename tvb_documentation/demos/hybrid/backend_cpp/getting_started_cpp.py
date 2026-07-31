#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.backend_cpp import CppHybridBackend
from tvb.simulator.hybrid import InterProjection, NetworkSet, Simulator, Subnetwork
from tvb.simulator.hybrid.coupling import Linear
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.monitors import TemporalAverage


DT = 0.1
NNODES = 16
SIMULATION_LENGTH = 100.0
TAVG_PERIOD = DT
OUTPUT_DIR = EXAMPLES_DIR / "outputs"


def make_subnet(name: str, n_nodes: int) -> Subnetwork:
    model = MontbrioPazoRoxin()
    model.configure()
    subnet = Subnetwork(
        name=name,
        model=model,
        scheme=HeunDeterministic(dt=DT),
        nnodes=n_nodes,
    ).configure()
    subnet.node_indices = np.arange(n_nodes)
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


def make_coupled_network(n_nodes: int) -> tuple[NetworkSet, Subnetwork, Subnetwork]:
    """Match the documented Numba multi-subnetwork benchmark topology."""
    sn1 = make_subnet("sn1", n_nodes)
    sn2 = make_subnet("sn2", n_nodes)
    sn2.node_indices = np.arange(n_nodes, 2 * n_nodes)

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

    network = NetworkSet(subnets=[sn1, sn2], projections=[projection])
    network.configure()
    return network, sn1, sn2


def make_uncoupled_network(n_nodes: int) -> tuple[NetworkSet, Subnetwork, Subnetwork]:
    sn1 = make_subnet("sn1", n_nodes)
    sn2 = make_subnet("sn2", n_nodes)
    sn2.node_indices = np.arange(n_nodes, 2 * n_nodes)
    network = NetworkSet(subnets=[sn1, sn2], projections=[])
    network.configure()
    return network, sn1, sn2


def make_single_network(name: str, n_nodes: int) -> tuple[NetworkSet, Subnetwork]:
    subnet = make_subnet(name, n_nodes)
    network = NetworkSet(subnets=[subnet], projections=[])
    network.configure()
    return network, subnet


def run_reference_simulator(
    network: NetworkSet,
    initial_states: list[np.ndarray],
    simulation_length: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    sim = Simulator(
        nets=network,
        simulation_length=simulation_length,
        monitors=[TemporalAverage(period=TAVG_PERIOD)],
    )
    sim.configure()
    t0 = time.perf_counter()
    ((times, data),) = sim.run(
        initial_conditions=[arr.copy() for arr in initial_states]
    )
    return np.asarray(times, dtype=np.float64), np.asarray(data, dtype=np.float64), time.perf_counter() - t0


def run_numba(
    network: NetworkSet,
    initial_states: list[np.ndarray],
    simulation_length: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    nstep = int(round(simulation_length / DT))
    chunk_size = int(round(TAVG_PERIOD / DT))
    backend = NbHybridBackend()

    t0 = time.perf_counter()
    compiled = backend.compile(network)
    compiled.run(nstep=2, chunk_size=1, initial_states=[arr.copy() for arr in initial_states])
    compile_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = compiled.run(
        nstep=nstep,
        chunk_size=chunk_size,
        initial_states=[arr.copy() for arr in initial_states],
    )
    run_time = time.perf_counter() - t0

    if isinstance(result, list):
        times = np.asarray(result[0][0], dtype=np.float64)
        subnet_data = [np.asarray(item[1], dtype=np.float64) for item in result]
        n_time, n_voi, n_nodes, n_modes = subnet_data[0].shape
        data = np.zeros(
            (n_time, n_voi, n_nodes * len(subnet_data), n_modes),
            dtype=np.float64,
        )
        for i, y_subnet in enumerate(subnet_data):
            data[:, :, i * n_nodes : (i + 1) * n_nodes, :] = y_subnet
    else:
        times, data = result
        times = np.asarray(times, dtype=np.float64)
        data = np.asarray(data, dtype=np.float64)

    return times, data, {
        "compile_seconds": compile_time,
        "run_seconds": run_time,
    }


def run_cpp_single(
    n_nodes: int,
    simulation_length: float,
    initial_state: np.ndarray,
    user_source_hint: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    network, _subnet = make_single_network(user_source_hint, n_nodes)
    nstep = int(round(simulation_length / DT))
    chunk_size = int(round(TAVG_PERIOD / DT))
    backend = CppHybridBackend()

    t0 = time.perf_counter()
    compiled = backend.compile(
        network,
        monitors=[TemporalAverage(period=TAVG_PERIOD)],
        user_source_hint=user_source_hint,
    )
    compile_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    times, data = compiled.run(
        nstep=nstep,
        chunk_size=chunk_size,
        initial_states=[initial_state.copy()],
    )
    run_time = time.perf_counter() - t0
    summary = compiled.debug_summary()
    summary["compile_seconds"] = compile_time
    summary["run_seconds"] = run_time
    return np.asarray(times, dtype=np.float64), np.asarray(data, dtype=np.float64), summary


def try_cpp_coupled(network: NetworkSet) -> dict:
    backend = CppHybridBackend()
    try:
        lowering = backend.lower(
            network,
            monitors=[TemporalAverage(period=TAVG_PERIOD)],
            user_source_hint="simulate_hybrid_getting_started_cpp_coupled_mpr",
        )
        spec_payload = lowering.spec.payload()
    except Exception as exc:  # pragma: no cover - diagnostic script
        return {
            "lowered": False,
            "compiled": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=5),
        }

    try:
        backend.compile(
            network,
            monitors=[TemporalAverage(period=TAVG_PERIOD)],
            user_source_hint="simulate_hybrid_getting_started_cpp_coupled_mpr",
            build_native=False,
        )
    except Exception as exc:
        return {
            "lowered": True,
            "compiled": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "spec": {
                "n_subnetworks": len(spec_payload["subnetworks"]),
                "n_inter_projections": len(spec_payload["inter_projections"]),
                "n_intra_projections": len(spec_payload["intra_projections"]),
                "subnetworks": [
                    {
                        "name": sn["name"],
                        "model_type": sn["model_type"],
                        "n_nodes": sn["n_nodes"],
                        "n_modes": sn["n_modes"],
                    }
                    for sn in spec_payload["subnetworks"]
                ],
                "inter_projections": spec_payload["inter_projections"],
            },
        }

    return {"lowered": True, "compiled": True}


def split_two_subnet_output(
    data: np.ndarray,
    sn1: Subnetwork,
    sn2: Subnetwork,
) -> tuple[np.ndarray, np.ndarray]:
    y1 = data[:, :, : sn1.nnodes, :]
    y2 = data[:, :, sn1.nnodes : sn1.nnodes + sn2.nnodes, :]
    return y1, y2


def plot_timeseries(
    output_path: Path,
    t_py: np.ndarray,
    y1_py: np.ndarray,
    y2_py: np.ndarray,
    t_nb: np.ndarray,
    y1_nb: np.ndarray,
    y2_nb: np.ndarray,
    t_uncoupled: np.ndarray,
    y1_uncoupled: np.ndarray,
    y2_uncoupled: np.ndarray,
    t_cpp1: np.ndarray,
    y_cpp1: np.ndarray,
    t_cpp2: np.ndarray,
    y_cpp2: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    t_nb_plot = t_nb - 0.5 * DT
    t_cpp1_plot = t_cpp1 - 0.5 * DT
    t_cpp2_plot = t_cpp2 - 0.5 * DT

    axes[0].plot(t_py, y1_py[:, 0, :, 0].mean(axis=1), color="steelblue", linewidth=1.0, label="Python coupled")
    axes[0].plot(t_nb_plot, y1_nb[:, 0, :, 0].mean(axis=1), color="darkorange", linestyle="--", linewidth=1.0, label="Numba coupled")
    axes[0].plot(t_uncoupled, y1_uncoupled[:, 0, :, 0].mean(axis=1), color="gray", linestyle="-.", linewidth=1.0, label="Python uncoupled")
    axes[0].plot(t_cpp1_plot, y_cpp1[:, 0, :, 0].mean(axis=1), color="black", linestyle=":", linewidth=1.0, label="C++ standalone")
    axes[0].set_title("sn1 MontbrioPazoRoxin r")
    axes[0].set_ylabel("r")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(t_py, y2_py[:, 0, :, 0].mean(axis=1), color="firebrick", linewidth=1.0, label="Python coupled")
    axes[1].plot(t_nb_plot, y2_nb[:, 0, :, 0].mean(axis=1), color="darkorange", linestyle="--", linewidth=1.0, label="Numba coupled")
    axes[1].plot(t_uncoupled, y2_uncoupled[:, 0, :, 0].mean(axis=1), color="gray", linestyle="-.", linewidth=1.0, label="Python uncoupled")
    axes[1].plot(t_cpp2_plot, y_cpp2[:, 0, :, 0].mean(axis=1), color="black", linestyle=":", linewidth=1.0, label="C++ standalone")
    axes[1].set_title("sn2 MontbrioPazoRoxin r")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("r")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("Two-subnetwork MPR benchmark topology: sn1 -> sn2", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "C++ backend counterpart for the Numba multi-subnetwork MPR example. "
            "The coupled topology is lowered and diagnosed; standalone native "
            "C++ subnetworks are run because inter-subnet C++ execution is not "
            "implemented yet."
        )
    )
    parser.add_argument("--nodes", type=int, default=NNODES)
    parser.add_argument("--simulation-length", type=float, default=SIMULATION_LENGTH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== C++ backend multi-subnetwork counterpart ===")
    print(f"dt={DT} ms | nodes={args.nodes} per subnetwork | length={args.simulation_length} ms")
    print("Topology: MontbrioPazoRoxin sn1 -> MontbrioPazoRoxin sn2")

    coupled_network, sn1, sn2 = make_coupled_network(args.nodes)
    initial_states = [make_initial_state(sn1), make_initial_state(sn2)]

    t_py, y_py, python_seconds = run_reference_simulator(
        coupled_network,
        initial_states,
        args.simulation_length,
    )
    y1_py, y2_py = split_two_subnet_output(y_py, sn1, sn2)

    uncoupled_network, sn1_uncoupled, sn2_uncoupled = make_uncoupled_network(args.nodes)
    uncoupled_initial_states = [
        make_initial_state(sn1_uncoupled),
        make_initial_state(sn2_uncoupled),
    ]
    t_uncoupled, y_uncoupled, uncoupled_python_seconds = run_reference_simulator(
        uncoupled_network,
        uncoupled_initial_states,
        args.simulation_length,
    )
    y1_uncoupled, y2_uncoupled = split_two_subnet_output(
        y_uncoupled,
        sn1_uncoupled,
        sn2_uncoupled,
    )

    t_nb, y_nb, nb_summary = run_numba(
        coupled_network,
        initial_states,
        args.simulation_length,
    )
    y1_nb, y2_nb = split_two_subnet_output(y_nb, sn1, sn2)

    cpp_coupled_status = try_cpp_coupled(coupled_network)
    t_cpp1, y_cpp1, cpp1_summary = run_cpp_single(
        args.nodes,
        args.simulation_length,
        initial_states[0],
        "simulate_hybrid_getting_started_cpp_sn1_mpr",
    )
    t_cpp2, y_cpp2, cpp2_summary = run_cpp_single(
        args.nodes,
        args.simulation_length,
        initial_states[1],
        "simulate_hybrid_getting_started_cpp_sn2_mpr",
    )

    figure_path = args.output_dir / "cpp_getting_started_coupled_mpr_timeseries.png"
    summary_path = args.output_dir / "cpp_getting_started_summary.json"
    plot_timeseries(
        figure_path,
        t_py,
        y1_py,
        y2_py,
        t_nb,
        y1_nb,
        y2_nb,
        t_uncoupled,
        y1_uncoupled,
        y2_uncoupled,
        t_cpp1,
        y_cpp1,
        t_cpp2,
        y_cpp2,
    )

    summary = {
        "scope_note": (
            "This script mirrors the documented Numba multi-subnetwork benchmark "
            "topology: two MontbrioPazoRoxin subnetworks with an InterProjection "
            "from sn1 to sn2. The current C++ backend can lower that NetworkSet, "
            "but generated native execution is still limited to one subnetwork. "
            "Therefore the full coupled topology is run with Python and Numba, "
            "while C++ is run for the two standalone subnetworks."
        ),
        "config": {
            "dt": DT,
            "nodes_per_subnetwork": args.nodes,
            "simulation_length": args.simulation_length,
            "tavg_period": TAVG_PERIOD,
        },
        "coupled_network": {
            "subnetworks": ["sn1", "sn2"],
            "model": "MontbrioPazoRoxin",
            "projection": "sn1.r -> sn2.coupling[0]",
            "weight": 0.1,
            "delay_steps": 0,
        },
        "python_reference": {
            "seconds": python_seconds,
            "times": list(t_py.shape),
            "data": list(y_py.shape),
            "sn1": list(y1_py.shape),
            "sn2": list(y2_py.shape),
            "has_nan": bool(np.any(np.isnan(y_py))),
        },
        "python_uncoupled_reference": {
            "seconds": uncoupled_python_seconds,
            "times": list(t_uncoupled.shape),
            "data": list(y_uncoupled.shape),
            "sn1": list(y1_uncoupled.shape),
            "sn2": list(y2_uncoupled.shape),
            "has_nan": bool(np.any(np.isnan(y_uncoupled))),
        },
        "numba_coupled": {
            **nb_summary,
            "times": list(t_nb.shape),
            "data": list(y_nb.shape),
            "python_vs_numba_time_max_abs": max_abs_diff(t_py, t_nb),
            "python_vs_numba_data_max_abs": max_abs_diff(y_py, y_nb),
        },
        "cpp_coupled": cpp_coupled_status,
        "cpp_standalone": {
            "sn1": {
                **cpp1_summary,
                "times": list(t_cpp1.shape),
                "data": list(y_cpp1.shape),
                "python_uncoupled_vs_cpp_data_max_abs": max_abs_diff(
                    y1_uncoupled,
                    y_cpp1,
                ),
            },
            "sn2": {
                **cpp2_summary,
                "times": list(t_cpp2.shape),
                "data": list(y_cpp2.shape),
                "python_uncoupled_vs_cpp_data_max_abs": max_abs_diff(
                    y2_uncoupled,
                    y_cpp2,
                ),
            },
        },
        "outputs": {
            "timeseries": str(figure_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print("Python coupled y.shape:", y_py.shape)
    print("Numba coupled y.shape:", y_nb.shape)
    print("C++ coupled compiled:", cpp_coupled_status.get("compiled"))
    if not cpp_coupled_status.get("compiled"):
        print("C++ coupled reason:", cpp_coupled_status.get("error"))
    print("C++ standalone sn1 shape:", y_cpp1.shape)
    print("C++ standalone sn2 shape:", y_cpp2.shape)
    print(f"Wrote {figure_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
