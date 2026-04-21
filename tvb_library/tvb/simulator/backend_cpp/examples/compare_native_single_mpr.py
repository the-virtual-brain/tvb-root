#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np


# Execute like this:
# python3 compare_native_single_mpr.py --nodes 3 --sim-length 10.0 --tavg-period 0.2

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.hybrid import NetworkSet, Simulator, Subnetwork
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.monitors import TemporalAverage

from backend_cpp import CppHybridBackend


DT = 0.1


class ScopedNbHybridBackend(NbHybridBackend):
    def _check_compatibility(self, network_set: NetworkSet) -> None:
        if not network_set.subnets:
            raise ValueError("NetworkSet must contain at least one subnetwork.")
        dt0 = float(network_set.subnets[0].scheme.dt)
        for subnet in network_set.subnets:
            if not isinstance(subnet.model, MontbrioPazoRoxin):
                raise NotImplementedError(
                    "ScopedNbHybridBackend comparison path supports only MontbrioPazoRoxin."
                )
            if not isinstance(subnet.scheme, HeunDeterministic):
                raise NotImplementedError(
                    "ScopedNbHybridBackend comparison path supports only HeunDeterministic."
                )
            if float(subnet.scheme.dt) != dt0:
                raise ValueError("All subnetworks must share the same dt.")


def make_subnet(name: str, n_nodes: int) -> Subnetwork:
    model = MontbrioPazoRoxin(I=np.array([2.0]))
    model.configure()
    subnet = Subnetwork(
        name=name,
        model=model,
        scheme=HeunDeterministic(dt=DT),
        nnodes=n_nodes,
    ).configure()
    subnet.node_indices = np.arange(subnet.nnodes)
    return subnet


def make_network(n_nodes: int) -> tuple[NetworkSet, Subnetwork]:
    subnet = make_subnet("sn", n_nodes)
    network = NetworkSet(subnets=[subnet], projections=[], stimuli=[])
    network.configure()
    return network, subnet


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


def run_python(
    network: NetworkSet,
    initial_state: np.ndarray,
    simulation_length: float,
    tavg_period: float,
) -> tuple[np.ndarray, np.ndarray]:
    sim = Simulator(
        nets=network,
        simulation_length=simulation_length,
        monitors=[TemporalAverage(period=tavg_period)],
    )
    sim.configure()
    ((times, data),) = sim.run(initial_conditions=[initial_state.copy()])
    return np.asarray(times, dtype=np.float64), np.asarray(data, dtype=np.float64)


def run_numba(
    network: NetworkSet,
    initial_state: np.ndarray,
    nstep: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    backend = ScopedNbHybridBackend()
    results = backend.run_network(
        network,
        nstep=nstep,
        chunk_size=chunk_size,
        initial_states=[initial_state.copy()],
    )
    times, data, _ctavg = results[0]
    return np.asarray(times, dtype=np.float64), np.asarray(data, dtype=np.float64)


def run_native(
    network: NetworkSet,
    initial_state: np.ndarray,
    nstep: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    backend = CppHybridBackend()
    compiled = backend.compile(
        network,
        monitors=[TemporalAverage(period=chunk_size * DT)],
        user_source_hint="compare_native_single_mpr",
    )
    times, data = compiled.run(
        initial_states=[initial_state.copy()],
        nstep=nstep,
        chunk_size=chunk_size,
    )
    return np.asarray(times, dtype=np.float64), np.asarray(data, dtype=np.float64)


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def rms_diff(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.mean(diff * diff)))


def plot_timeseries(
    output_path: Path,
    py_times: np.ndarray,
    py_data: np.ndarray,
    nb_times: np.ndarray,
    nb_data: np.ndarray,
    native_times: np.ndarray,
    native_data: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    labels = ("r", "V")
    for voi_idx, ax in enumerate(axes):
        py_mean = py_data[:, voi_idx, :, 0].mean(axis=1)
        nb_mean = nb_data[:, voi_idx, :, 0].mean(axis=1)
        native_mean = native_data[:, voi_idx, :, 0].mean(axis=1)
        ax.plot(py_times, py_mean, label="Python", linewidth=2)
        ax.plot(nb_times, nb_mean, label="Numba", linestyle="--")
        ax.plot(native_times, native_mean, label="Native C++", linestyle=":")
        ax.set_ylabel(labels[voi_idx])
        ax.grid(alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Time (ms)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare native single-network MontbrioPazoRoxin against Python and Numba."
    )
    parser.add_argument("--nodes", type=int, default=3)
    parser.add_argument("--sim-length", type=float, default=10.0)
    parser.add_argument("--tavg-period", type=float, default=0.2)
    parser.add_argument(
        "--plot",
        type=Path,
        default=EXAMPLES_DIR / "native_single_mpr_comparison.png",
    )
    args = parser.parse_args()

    nstep = int(round(args.sim_length / DT))
    chunk_size = int(round(args.tavg_period / DT))
    if chunk_size < 1:
        raise ValueError("tavg-period must be at least dt.")

    network, subnet = make_network(args.nodes)
    initial_state = make_initial_state(subnet)

    py_times, py_data = run_python(
        network=network,
        initial_state=initial_state,
        simulation_length=args.sim_length,
        tavg_period=args.tavg_period,
    )
    nb_times, nb_data = run_numba(
        network=network,
        initial_state=initial_state,
        nstep=nstep,
        chunk_size=chunk_size,
    )
    native_times, native_data = run_native(
        network=network,
        initial_state=initial_state,
        nstep=nstep,
        chunk_size=chunk_size,
    )

    if py_data.shape != nb_data.shape or py_data.shape != native_data.shape:
        raise RuntimeError(
            f"Shape mismatch: python={py_data.shape}, numba={nb_data.shape}, native={native_data.shape}"
        )

    plot_timeseries(
        output_path=args.plot,
        py_times=py_times,
        py_data=py_data,
        nb_times=nb_times,
        nb_data=nb_data,
        native_times=native_times,
        native_data=native_data,
    )

    summary = {
        "config": {
            "nodes": args.nodes,
            "dt": DT,
            "simulation_length": args.sim_length,
            "nstep": nstep,
            "tavg_period": args.tavg_period,
            "chunk_size": chunk_size,
        },
        "shapes": {
            "times": {
                "python": list(py_times.shape),
                "numba": list(nb_times.shape),
                "native": list(native_times.shape),
            },
            "data": {
                "python": list(py_data.shape),
                "numba": list(nb_data.shape),
                "native": list(native_data.shape),
            },
        },
        "time_diffs": {
            "python_vs_numba_max_abs": max_abs_diff(py_times, nb_times),
            "python_vs_native_max_abs": max_abs_diff(py_times, native_times),
            "numba_vs_native_max_abs": max_abs_diff(nb_times, native_times),
        },
        "data_diffs": {
            "python_vs_numba_max_abs": max_abs_diff(py_data, nb_data),
            "python_vs_native_max_abs": max_abs_diff(py_data, native_data),
            "numba_vs_native_max_abs": max_abs_diff(nb_data, native_data),
            "python_vs_numba_rms": rms_diff(py_data, nb_data),
            "python_vs_native_rms": rms_diff(py_data, native_data),
            "numba_vs_native_rms": rms_diff(nb_data, native_data),
        },
        "plot_path": str(args.plot),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
