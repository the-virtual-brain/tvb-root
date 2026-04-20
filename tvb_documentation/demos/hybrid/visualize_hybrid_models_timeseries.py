#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize time series comparison between Python and Numba backends for hybrid models.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("TVB_USER_HOME", str(Path(tempfile.gettempdir()) / "tvb-user"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

warnings.filterwarnings("ignore", message="Hybrid simulation is experimental: .*")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.hybrid import InterProjection, NetworkSet, Simulator, Subnetwork
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.models.epileptor import Epileptor
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.models.jansen_rit import JansenRit
from tvb.simulator.monitors import TemporalAverage

DT = 0.1
NNODES = 38
SIMULATION_LENGTH = 1000.0
NSTEP = int(round(SIMULATION_LENGTH / DT))
TAVG_PERIOD = 1.0
TAVG_ISTEP = int(round(TAVG_PERIOD / DT))
OUTPUT_DIR = Path(__file__).resolve().parent


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


def run_python_temporal_average(
    nets: NetworkSet,
    initial_states: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    tavg = TemporalAverage(period=TAVG_PERIOD)
    sim = Simulator(
        nets=nets,
        simulation_length=SIMULATION_LENGTH,
        monitors=[tavg],
    )
    sim.configure()
    
    ((t, y),) = sim.run(initial_conditions=[arr.copy() for arr in initial_states])
    
    # Split y into per-subnetwork
    results = []
    offset_voi = 0
    offset_node = 0
    for subnet in nets.subnets:
        n_vois = len(subnet.model.variables_of_interest)
        n_nodes = subnet.nnodes
        y_subnet = y[:, offset_voi:offset_voi + n_vois, offset_node:offset_node + n_nodes, :]
        results.append(y_subnet)
        offset_voi += n_vois
        offset_node += n_nodes
    return t, results


def run_numba_temporal_average(
    nets: NetworkSet,
    initial_states: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    backend = NbHybridBackend()
    results_nb = backend.run_network(
        nets,
        nstep=NSTEP,
        chunk_size=1,
        initial_states=[arr.copy() for arr in initial_states],
    )
    results = []
    for times, data, _ in results_nb:
        tavg_times, tavg_data = temporal_average_raw_output(times, data, TAVG_ISTEP)
        results.append(tavg_data)
    return tavg_times, results


def temporal_average_raw_output(
    times: np.ndarray,
    data: np.ndarray,
    istep: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average raw per-step output into fixed-size temporal-average chunks."""
    n_chunks = data.shape[0] // istep
    if n_chunks == 0:
        raise ValueError("Not enough samples to form one temporal-average chunk.")
    trimmed_steps = n_chunks * istep
    trimmed_data = data[:trimmed_steps]
    chunked_data = trimmed_data.reshape((n_chunks, istep) + data.shape[1:])
    steps = np.arange(istep, trimmed_steps + 1, istep, dtype=np.float64)
    avg_times = (steps - istep / 2.0) * DT
    avg_data = chunked_data.mean(axis=1)
    # Match the pure hybrid Simulator's merged monitor output: collapse modes
    # to a singleton axis so multi-mode subnetworks such as ReducedSetFHN
    # produce the same observable shape/series.
    avg_data = avg_data.sum(axis=-1, keepdims=True)
    return avg_times, avg_data


def plot_comparison(t_py, y_py_list, t_nb, y_nb_list, model_name, output_dir, plots_config=None):
    fig, axes = plt.subplots(len(y_py_list), 1, figsize=(10, 4 * len(y_py_list)))
    if len(y_py_list) == 1:
        axes = [axes]
    for i, (y_py, y_nb) in enumerate(zip(y_py_list, y_nb_list)):
        # Plot first VOI, mean over nodes
        py_mean = y_py[:, 0, :, 0].mean(axis=1)
        nb_mean = y_nb[:, 0, :, 0].mean(axis=1)
        axes[i].plot(t_py, py_mean, label="Python", color="blue")
        axes[i].plot(t_nb, nb_mean, label="Numba", color="red", linestyle="--")
        axes[i].set_title(f"{model_name} - Subnet {i} - VOI 0 mean")
        axes[i].set_xlabel("Time (ms)")
        axes[i].set_ylabel("Value")
        axes[i].legend()
        axes[i].grid(alpha=0.3)
    # Apply custom plot settings
    if plots_config and model_name in plots_config:
        config = plots_config[model_name]
        for ax in axes:
            if 'xlim' in config:
                ax.set_xlim(config['xlim'])
            if 'ylim' in config:
                ax.set_ylim(config['ylim'])
            # Add more options as needed
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_comparison.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    models = {
        "JansenRit": {"class": JansenRit, "params": {}},
        "Epileptor": {"class": Epileptor, "params": {}},
        "MontbrioPazoRoxin": {"class": MontbrioPazoRoxin, "params": {"I": np.array([2])}},
    }
    plots_config = {
        "MontbrioPazoRoxin": {"xlim": (0, 100)},
        # Add more as needed, e.g., "JansenRit": {"ylim": (-1, 1)}
    }
    for model_name, model_info in models.items():
        print(f"Processing {model_name}")
        # Build single network
        model_cls = model_info["class"]
        params = model_info["params"]
        model = model_cls(**params)
        subnet = Subnetwork(
            name="subnet",
            model=model,
            scheme=HeunDeterministic(dt=DT),
            nnodes=NNODES,
        ).configure()
        nets = NetworkSet(subnets=[subnet], projections=[], stimuli=[])
        nets.configure()
        initial_states = [make_initial_state(subnet)]
        # Run Python
        t_py, y_py_list = run_python_temporal_average(nets, initial_states)
        # Run Numba
        t_nb, y_nb_list = run_numba_temporal_average(nets, initial_states)
        # Plot
        plot_comparison(t_py, y_py_list, t_nb, y_nb_list, model_name, OUTPUT_DIR, plots_config)
    # Coupled MontbrioPazoRoxin
    print("Processing Coupled MontbrioPazoRoxin")
    montbrio_params = models["MontbrioPazoRoxin"]["params"]
    sn1 = Subnetwork(
        name="sn1",
        model=MontbrioPazoRoxin(**montbrio_params),
        scheme=HeunDeterministic(dt=DT),
        nnodes=NNODES,
    ).configure()
    sn2 = Subnetwork(
        name="sn2",
        model=MontbrioPazoRoxin(**montbrio_params),
        scheme=HeunDeterministic(dt=DT),
        nnodes=NNODES,
    ).configure()
    rng = np.random.RandomState(42)
    weights = sp.csr_matrix(np.abs(rng.randn(NNODES, NNODES)) * 0.01)
    lengths = sp.csr_matrix((NNODES, NNODES))
    projection = InterProjection(
        source=sn1,
        target=sn2,
        source_cvar="r",
        target_cvar="r",
        weights=weights,
        lengths=lengths,
        cv=1.0,
        dt=DT,
        scale=0.1,
    )
    nets = NetworkSet(subnets=[sn1, sn2], projections=[projection], stimuli=[])
    nets.configure()
    initial_states = [make_initial_state(sn1), make_initial_state(sn2)]
    # Run Python
    t_py, y_py_list = run_python_temporal_average(nets, initial_states)
    # Run Numba
    t_nb, y_nb_list = run_numba_temporal_average(nets, initial_states)
    # Plot
    plot_comparison(t_py, y_py_list, t_nb, y_nb_list, "Coupled_MontbrioPazoRoxin", OUTPUT_DIR, plots_config)
    print("Done")


if __name__ == "__main__":
    main()
