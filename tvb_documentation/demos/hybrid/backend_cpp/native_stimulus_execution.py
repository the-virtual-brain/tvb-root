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


# Execute from this directory or from the repository root, for example:
# python3 tvb_library/tvb/simulator/backend_cpp/examples/native_stimulus_execution.py

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

from tvb.datatypes import equations, patterns
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.backend_cpp import CppHybridBackend
from tvb.simulator.hybrid import NetworkSet, Simulator, Subnetwork
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.models.oscillator import Generic2dOscillator
from tvb.simulator.monitors import TemporalAverage


DT = 0.1
NNODES = 5
SIMULATION_LENGTH = 200.0
TAVG_PERIOD = 0.5
OUTPUT_DIR = EXAMPLES_DIR / "outputs"


def make_stimulus_connectivity(n_nodes: int) -> Connectivity:
    conn = Connectivity(
        centres=np.ones((n_nodes, 3)),
        weights=np.ones((n_nodes, n_nodes), dtype=np.float64) * 0.1,
        tract_lengths=np.zeros((n_nodes, n_nodes), dtype=np.float64),
        region_labels=np.array([f"region_{i}" for i in range(n_nodes)]),
        speed=np.array([1.0]),
    )
    conn.configure()
    return conn


def make_pulse_stimulus(n_nodes: int) -> patterns.StimuliRegion:
    temporal = equations.PulseTrain()
    temporal.parameters["onset"] = 10.0
    temporal.parameters["T"] = 30.0
    temporal.parameters["tau"] = 8.0
    temporal.parameters["amp"] = 1.0

    weights = np.zeros(n_nodes, dtype=np.float64)
    weights[0] = 1.0
    weights[1] = 0.5

    return patterns.StimuliRegion(
        temporal=temporal,
        connectivity=make_stimulus_connectivity(n_nodes),
        weight=weights,
    )


def make_network(with_stimulus: bool, simulation_length: float) -> tuple[NetworkSet, Subnetwork]:
    model = Generic2dOscillator()
    model.configure()
    subnet = Subnetwork(
        name="g2d",
        model=model,
        scheme=HeunDeterministic(dt=DT),
        nnodes=NNODES,
    ).configure()
    subnet.node_indices = np.arange(NNODES)

    network = NetworkSet(subnets=[subnet], projections=[], stimuli=[])
    if with_stimulus:
        network.add_stimulus(
            target_name="g2d",
            stimulus=make_pulse_stimulus(NNODES),
            stimulus_cvar=np.r_[0],
            projection_scale=2.0,
        )

    network.configure()
    for stim in network.stimuli:
        # The Python Simulator does this in configure(); the native backend needs
        # the same prepared Stim objects before _build_stimulus_arrays() runs.
        stim.configure(simulation_length)
    return network, subnet


def make_initial_state(subnetwork: Subnetwork) -> np.ndarray:
    return np.zeros(
        (
            subnetwork.model.nvar,
            subnetwork.nnodes,
            subnetwork.model.number_of_modes,
        ),
        dtype=np.float64,
    )


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


def run_native(
    network: NetworkSet,
    initial_state: np.ndarray,
    simulation_length: float,
    tavg_period: float,
    source_hint: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    nstep = int(round(simulation_length / DT))
    chunk_size = int(round(tavg_period / DT))
    if chunk_size < 1:
        raise ValueError("tavg-period must be at least dt.")

    backend = CppHybridBackend(build_root=EXAMPLES_DIR / ".build")
    compiled = backend.compile(
        network,
        monitors=[TemporalAverage(period=tavg_period)],
        user_source_hint=source_hint,
    )
    ((times, data),) = compiled.run(
        nstep=nstep,
        chunk_size=chunk_size,
        initial_states=[initial_state.copy()],
    )
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(data, dtype=np.float64),
        compiled.debug_summary(),
    )


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def plot_output(
    output_path: Path,
    times: np.ndarray,
    stimulated_data: np.ndarray,
    control_data: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(
        times,
        stimulated_data[:, 0, 0, 0],
        color="tab:red",
        label="Native C++ with PulseTrain",
    )
    axes[0].plot(
        times,
        control_data[:, 0, 0, 0],
        color="tab:gray",
        linestyle="--",
        label="Native C++ control",
    )
    axes[0].set_title("Node 0 receives full stimulus weight")
    axes[0].set_ylabel("V")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        times,
        stimulated_data[:, 0, 4, 0],
        color="tab:blue",
        label="Native C++ with PulseTrain",
    )
    axes[1].plot(
        times,
        control_data[:, 0, 4, 0],
        color="tab:gray",
        linestyle="--",
        label="Native C++ control",
    )
    axes[1].set_title("Node 4 has zero stimulus weight")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("V")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Native C++ hybrid backend stimulus execution")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a native C++ hybrid simulation with a precomputed "
            "StimuliRegion PulseTrain input."
        )
    )
    parser.add_argument("--sim-length", type=float, default=SIMULATION_LENGTH)
    parser.add_argument("--tavg-period", type=float, default=TAVG_PERIOD)
    parser.add_argument(
        "--plot",
        type=Path,
        default=OUTPUT_DIR / "native_stimulus_execution.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stim_network_py, stim_subnet_py = make_network(
        with_stimulus=True,
        simulation_length=args.sim_length,
    )
    initial_state = make_initial_state(stim_subnet_py)
    py_times, py_data = run_python(
        network=stim_network_py,
        initial_state=initial_state,
        simulation_length=args.sim_length,
        tavg_period=args.tavg_period,
    )

    stim_network_cpp, stim_subnet_cpp = make_network(
        with_stimulus=True,
        simulation_length=args.sim_length,
    )
    cpp_initial_state = make_initial_state(stim_subnet_cpp)
    cpp_times, cpp_data, cpp_summary = run_native(
        network=stim_network_cpp,
        initial_state=cpp_initial_state,
        simulation_length=args.sim_length,
        tavg_period=args.tavg_period,
        source_hint="native_stimulus_execution_pulsetrain",
    )

    control_network_cpp, control_subnet_cpp = make_network(
        with_stimulus=False,
        simulation_length=args.sim_length,
    )
    control_times, control_data, _ = run_native(
        network=control_network_cpp,
        initial_state=make_initial_state(control_subnet_cpp),
        simulation_length=args.sim_length,
        tavg_period=args.tavg_period,
        source_hint="native_stimulus_execution_control",
    )

    if py_data.shape != cpp_data.shape:
        raise RuntimeError(f"Shape mismatch: python={py_data.shape}, native={cpp_data.shape}")

    plot_output(
        output_path=args.plot,
        times=cpp_times,
        stimulated_data=cpp_data,
        control_data=control_data,
    )

    summary = {
        "config": {
            "dt": DT,
            "nodes": NNODES,
            "simulation_length": args.sim_length,
            "tavg_period": args.tavg_period,
            "stimulated_nodes": {"0": 1.0, "1": 0.5},
            "target_cvar": 0,
            "projection_scale": 2.0,
        },
        "shapes": {
            "python_data": list(py_data.shape),
            "native_data": list(cpp_data.shape),
            "control_data": list(control_data.shape),
        },
        "python_vs_native": {
            "time_max_abs": max_abs_diff(py_times, cpp_times),
            "data_max_abs": max_abs_diff(py_data, cpp_data),
            "data_rms": float(np.sqrt(np.mean((py_data - cpp_data) ** 2))),
        },
        "stimulus_effect_native": {
            "node0_stim_vs_control_max_abs": max_abs_diff(
                cpp_data[:, 0, 0, 0],
                control_data[:, 0, 0, 0],
            ),
            "node4_stim_vs_control_max_abs": max_abs_diff(
                cpp_data[:, 0, 4, 0],
                control_data[:, 0, 4, 0],
            ),
        },
        "native_debug": {
            "pipeline_stage": cpp_summary["pipeline_stage"],
            "stimulus_count": len(stim_network_cpp.stimuli),
            "generated_cpp_path": cpp_summary["generated_cpp_path"],
        },
        "plot_path": str(args.plot),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
