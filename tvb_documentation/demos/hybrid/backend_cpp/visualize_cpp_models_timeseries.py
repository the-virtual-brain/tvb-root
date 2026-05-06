#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize time series reproduced by the native C++ hybrid backend.

The current C++ backend supports single-subnetwork, no-projection simulations
for models exposing expression-based ``state_variable_dfuns``. Unsupported
models are reported and skipped by default.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
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
warnings.filterwarnings("ignore", message="Hybrid simulation is experimental: .*")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tvb.simulator.backend_cpp import CppHybridBackend
from tvb.simulator.hybrid import NetworkSet, Simulator, Subnetwork
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.monitors import TemporalAverage


DT = 0.1
NNODES = 3
TAVG_PERIOD = 1.0
OUTPUT_DIR = EXAMPLES_DIR / "outputs"

MODEL_CASES: dict[str, dict[str, Any]] = {
    "JansenRit": {
        "class_path": "tvb.simulator.models.jansen_rit:JansenRit",
        "params": {},
        "sim_length": 1000.0,
    },
    "Epileptor": {
        "class_path": "tvb.simulator.models.epileptor:Epileptor",
        "params": {},
        "sim_length": 2000.0,
    },
    "MontbrioPazoRoxin": {
        "class_path": "tvb.simulator.models.infinite_theta:MontbrioPazoRoxin",
        "params": {"I": np.array([2.0])},
        "sim_length": 100.0,
    },
    "WilsonCowan": {
        "class_path": "tvb.simulator.models.wilson_cowan:WilsonCowan",
        "params": {},
        "sim_length": 1000.0,
    },
    "WongWang": {
        "class_path": "tvb.simulator.models.wong_wang:ReducedWongWang",
        "params": {},
        "sim_length": 1000.0,
    },
    "LarterBreakspear": {
        "class_path": "tvb.simulator.models.larter_breakspear:LarterBreakspear",
        "params": {},
        "sim_length": 1000.0,
    },
    "Generic2dOscillator": {
        "class_path": "tvb.simulator.models.oscillator:Generic2dOscillator",
        "params": {},
        "sim_length": 1000.0,
    },
    "ZerlautAdaptationFirstOrder": {
        "class_path": "tvb.simulator.models.zerlaut:ZerlautAdaptationFirstOrder",
        "params": {},
        "sim_length": 1000.0,
    },
}


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
    model_name: str,
    n_nodes: int,
) -> tuple[NetworkSet, Subnetwork, np.ndarray]:
    model_info = MODEL_CASES[model_name]
    module_name, class_name = model_info["class_path"].split(":", 1)
    module = importlib.import_module(module_name)
    model_cls = getattr(module, class_name)
    model = model_cls(**model_info["params"])
    model.configure()
    subnet = Subnetwork(
        name="subnet",
        model=model,
        scheme=HeunDeterministic(dt=DT),
        nnodes=n_nodes,
    )
    subnet.node_indices = np.arange(subnet.nnodes)
    subnet.configure()
    nets = NetworkSet(subnets=[subnet], projections=[], stimuli=[])
    nets.configure()
    return nets, subnet, make_initial_state(subnet)


def run_python_temporal_average(
    nets: NetworkSet,
    initial_state: np.ndarray,
    sim_length: float,
    tavg_period: float,
) -> tuple[np.ndarray, np.ndarray]:
    sim = Simulator(
        nets=nets,
        simulation_length=sim_length,
        monitors=[TemporalAverage(period=tavg_period)],
    )
    sim.configure()
    ((times, data),) = sim.run(initial_conditions=[initial_state.copy()])
    return np.asarray(times, dtype=np.float64), np.asarray(data, dtype=np.float64)


def run_cpp_temporal_average(
    nets: NetworkSet,
    initial_state: np.ndarray,
    sim_length: float,
    tavg_period: float,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    nstep = int(round(sim_length / DT))
    chunk_size = int(round(tavg_period / DT))
    if chunk_size < 1:
        raise ValueError("tavg-period must be at least dt.")

    backend = CppHybridBackend()
    compiled = backend.compile(
        nets,
        monitors=[TemporalAverage(period=tavg_period)],
        user_source_hint=f"visualize_cpp_models_timeseries_{model_name}",
    )
    times, data = compiled.run(
        nstep=nstep,
        chunk_size=chunk_size,
        initial_states=[initial_state.copy()],
    )
    return np.asarray(times, dtype=np.float64), np.asarray(data, dtype=np.float64)


def plot_comparison(
    output_path: Path,
    model_name: str,
    t_py: np.ndarray,
    y_py: np.ndarray,
    t_cpp: np.ndarray,
    y_cpp: np.ndarray,
) -> None:
    n_voi = min(y_py.shape[1], y_cpp.shape[1])
    fig, axes = plt.subplots(n_voi, 1, figsize=(10, max(3, 3 * n_voi)), sharex=True)
    if n_voi == 1:
        axes = [axes]

    for voi_idx, ax in enumerate(axes):
        py_mean = y_py[:, voi_idx, :, 0].mean(axis=1)
        cpp_mean = y_cpp[:, voi_idx, :, 0].mean(axis=1)
        ax.plot(t_py, py_mean, label="Python", linewidth=2)
        ax.plot(t_cpp, cpp_mean, label="Native C++", linestyle=":")
        ax.set_title(f"{model_name} - VOI {voi_idx} mean")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Time (ms)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def shape_error(model_name: str, py_data: np.ndarray, cpp_data: np.ndarray) -> str:
    return (
        f"{model_name} shape mismatch: "
        f"python={py_data.shape}, native_cpp={cpp_data.shape}"
    )


def run_case(
    model_name: str,
    n_nodes: int,
    output_dir: Path,
    tavg_period: float,
) -> dict[str, Any]:
    model_info = MODEL_CASES[model_name]
    sim_length = float(model_info["sim_length"])
    nets, _subnet, initial_state = make_single_network(model_name, n_nodes)

    py_times, py_data = run_python_temporal_average(
        nets=nets,
        initial_state=initial_state,
        sim_length=sim_length,
        tavg_period=tavg_period,
    )
    cpp_times, cpp_data = run_cpp_temporal_average(
        nets=nets,
        initial_state=initial_state,
        sim_length=sim_length,
        tavg_period=tavg_period,
        model_name=model_name,
    )
    if py_data.shape != cpp_data.shape:
        raise RuntimeError(shape_error(model_name, py_data, cpp_data))

    output_path = output_dir / f"cpp_comparison_{model_name}.png"
    plot_comparison(
        output_path=output_path,
        model_name=model_name,
        t_py=py_times,
        y_py=py_data,
        t_cpp=cpp_times,
        y_cpp=cpp_data,
    )

    return {
        "model": model_name,
        "status": "ok",
        "plot_path": str(output_path),
        "simulation_length": sim_length,
        "data_shape": list(cpp_data.shape),
        "python_vs_cpp_time_max_abs": float(np.max(np.abs(py_times - cpp_times))),
        "python_vs_cpp_data_max_abs": float(np.max(np.abs(py_data - cpp_data))),
        "python_vs_cpp_data_rms": float(
            np.sqrt(np.mean((py_data - cpp_data) * (py_data - cpp_data)))
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Python reference and native C++ backend time series."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CASES.keys()),
        choices=sorted(MODEL_CASES.keys()),
        help="Model names to run. Defaults to all configured cases.",
    )
    parser.add_argument("--nodes", type=int, default=NNODES)
    parser.add_argument("--tavg-period", type=float, default=TAVG_PERIOD)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise immediately instead of skipping unsupported C++ cases.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for model_name in args.models:
        print(f"Processing {model_name}")
        try:
            summary = run_case(
                model_name=model_name,
                n_nodes=args.nodes,
                output_dir=args.output_dir,
                tavg_period=args.tavg_period,
            )
        except Exception as exc:
            if args.strict:
                raise
            summary = {
                "model": model_name,
                "status": "skipped",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            print(f"  skipped: {summary['reason']}")
        else:
            print(f"  wrote {summary['plot_path']}")
        summaries.append(summary)

    summary_path = args.output_dir / "cpp_models_timeseries_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
