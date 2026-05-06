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

from tvb.simulator.backend_cpp import CppHybridBackend, DelayedSelfFeedbackConfig
from tvb.simulator.hybrid import NetworkSet, Subnetwork
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.monitors import TemporalAverage


DT = 0.1


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


def run_python_reference(
    subnetwork: Subnetwork,
    initial_state: np.ndarray,
    nstep: int,
    chunk_size: int,
    delay_steps: int,
    gain: float,
) -> tuple[np.ndarray, np.ndarray]:
    model = subnetwork.model
    tau = float(np.atleast_1d(model.tau)[0])
    delta = float(np.atleast_1d(model.Delta)[0])
    eta = float(np.atleast_1d(model.eta)[0])
    j = float(np.atleast_1d(model.J)[0])
    current_i = float(np.atleast_1d(model.I)[0])
    cr = float(np.atleast_1d(model.cr)[0])
    cv = float(np.atleast_1d(model.cv)[0])

    def compute_dfun(state: np.ndarray, delayed_r: np.ndarray) -> np.ndarray:
        r = np.maximum(0.0, state[0, :, 0])
        v = state[1, :, 0]
        dx = np.zeros_like(state)
        dx[0, :, 0] = (1.0 / tau) * (delta / (np.pi * tau) + 2.0 * v * r)
        dx[1, :, 0] = (1.0 / tau) * (
            v * v
            - (np.pi * np.pi) * tau * tau * r * r
            + eta
            + j * tau * r
            + current_i
            + cr * 0.0
            + cv * (gain * delayed_r)
        )
        return dx

    history = [initial_state.copy() for _ in range(delay_steps + 1)]
    state = initial_state.copy()
    num_chunks = (nstep + chunk_size - 1) // chunk_size
    times = np.zeros(num_chunks, dtype=np.float64)
    data = np.zeros((num_chunks, model.nvar, subnetwork.nnodes, 1), dtype=np.float64)
    accum = np.zeros((model.nvar, subnetwork.nnodes, 1), dtype=np.float64)
    current_chunk = 0
    steps_in_chunk = 0
    chunk_start_step = 1

    for step in range(1, nstep + 1):
        delayed_state = history[-1 - delay_steps]
        delayed_r = delayed_state[0, :, 0]

        dx0 = compute_dfun(state, delayed_r)
        predictor = state + DT * dx0
        predictor[0, :, 0] = np.maximum(0.0, predictor[0, :, 0])

        dx1 = compute_dfun(predictor, delayed_r)
        state = state + 0.5 * DT * (dx0 + dx1)
        state[0, :, 0] = np.maximum(0.0, state[0, :, 0])

        history.append(state.copy())
        if len(history) > delay_steps + 1:
            history.pop(0)

        accum += state
        steps_in_chunk += 1
        close_chunk = steps_in_chunk == chunk_size or step == nstep
        if not close_chunk:
            continue

        mid_step = chunk_start_step + (steps_in_chunk - 1.0) / 2.0
        times[current_chunk] = mid_step * DT
        data[current_chunk] = accum / float(steps_in_chunk)
        accum.fill(0.0)
        current_chunk += 1
        chunk_start_step = step + 1
        steps_in_chunk = 0

    return times, data


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def rms_diff(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.mean(diff * diff)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare native delayed self-feedback against a pure Python deterministic reference."
    )
    parser.add_argument("--nodes", type=int, default=3)
    parser.add_argument("--nstep", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--delay-steps", type=int, default=3)
    parser.add_argument("--gain", type=float, default=0.25)
    args = parser.parse_args()

    network, subnet = make_network(args.nodes)
    initial_state = make_initial_state(subnet)
    py_times, py_data = run_python_reference(
        subnetwork=subnet,
        initial_state=initial_state,
        nstep=args.nstep,
        chunk_size=args.chunk_size,
        delay_steps=args.delay_steps,
        gain=args.gain,
    )

    backend = CppHybridBackend()
    compiled = backend.compile(
        network,
        monitors=[TemporalAverage(period=args.chunk_size * DT)],
        user_source_hint="compare_native_delayed_self_feedback",
        delayed_self_feedback=DelayedSelfFeedbackConfig(
            delay_steps=args.delay_steps,
            gain=args.gain,
            source_state_var="r",
            target_state_var="V",
        ),
    )
    native_times, native_data = compiled.run(
        initial_states=[initial_state.copy()],
        nstep=args.nstep,
        chunk_size=args.chunk_size,
    )

    payload = {
        "config": {
            "dt": DT,
            "nodes": args.nodes,
            "nstep": args.nstep,
            "chunk_size": args.chunk_size,
            "delay_steps": args.delay_steps,
            "gain": args.gain,
            "deterministic": True,
            "noise_amplitude": 0.0,
        },
        "native": {
            "module_name": compiled.module_name,
            "generated_cpp_path": str(compiled.generated_cpp_path),
            "runtime_header_path": str(compiled.generated_source.runtime_header_path),
        },
        "shapes": {
            "python_times": list(py_times.shape),
            "native_times": list(native_times.shape),
            "python_data": list(py_data.shape),
            "native_data": list(native_data.shape),
        },
        "diffs": {
            "times_max_abs": max_abs_diff(py_times, native_times),
            "data_max_abs": max_abs_diff(py_data, native_data),
            "data_rms": rms_diff(py_data, native_data),
        },
        "preview": {
            "times": native_times.tolist(),
            "python_first_chunk_first_voi": py_data[0, 0, :, 0].tolist(),
            "native_first_chunk_first_voi": native_data[0, 0, :, 0].tolist(),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
