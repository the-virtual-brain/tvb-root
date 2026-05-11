#!/usr/bin/env python3
"""Minimal inter-subnetwork delayed-projection ordering check.

This example exercises the C++ runtime phase ordering:

  1. zero all coupling arrays
  2. accumulate all intra-projections
  3. accumulate all inter-projections from source histories
  4. integrate all subnetworks
  5. push all updated states into history

The important point is step 3: both inter-projections read source history before
any subnet is integrated for the current step.  Therefore A->B and B->A both see
the same logical source time, independent of subnet traversal order.

Run from the repo root:

    python tvb_library/tvb/simulator/backend_cpp/examples/minimal_inter_delayed_projection_order.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
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
warnings.filterwarnings("ignore", message="Hybrid simulation is experimental: .*")

from tvb.simulator.backend_cpp import CppHybridBackend
from tvb.simulator.hybrid import InterProjection, NetworkSet, Simulator, Subnetwork
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.monitors import TemporalAverage


DT = 0.1
CV = 10.0


def make_subnet(name: str, node_index: int) -> Subnetwork:
    model = MontbrioPazoRoxin(I=np.array([2.0]))
    model.configure()
    subnet = Subnetwork(
        name=name,
        model=model,
        scheme=HeunDeterministic(dt=DT),
        nnodes=1,
    ).configure()
    subnet.node_indices = np.array([node_index])
    return subnet


def make_projection(
    source: Subnetwork,
    target: Subnetwork,
    weight: float,
    delay_steps: int,
) -> InterProjection:
    weights = sp.csr_matrix(np.array([[weight]], dtype=np.float64))
    lengths = sp.csr_matrix(np.array([[delay_steps * CV * DT]], dtype=np.float64))
    return InterProjection(
        source=source,
        target=target,
        source_cvar=np.array([0]),  # source r
        target_cvar=np.array([0]),  # target Coupling_Term_r slot
        weights=weights,
        lengths=lengths,
        cv=CV,
        dt=DT,
        scale=1.0,
    )


def make_initial_state(subnet: Subnetwork, r: float, v: float) -> np.ndarray:
    x0 = np.zeros(
        (subnet.model.nvar, subnet.nnodes, subnet.model.number_of_modes),
        dtype=np.float64,
    )
    x0[0, :, :] = r
    x0[1, :, :] = v
    return x0


def run_python_hybrid_simulator(
    network: NetworkSet,
    initial_a: np.ndarray,
    initial_b: np.ndarray,
    nstep: int,
    chunk_size: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run the standard Python hybrid Simulator API and split subnet outputs.

    This follows the same public pattern as
    tvb_documentation/demos/simulate_hybrid_getting_started.py: build a
    NetworkSet, create a Simulator, call sim.run(initial_conditions=...), then
    slice the concatenated monitor output back into per-subnetwork arrays.
    """
    simulator = Simulator(
        nets=network,
        simulation_length=nstep * DT,
        monitors=[TemporalAverage(period=chunk_size * DT)],
    )
    simulator.configure()
    ((times, data),) = simulator.run(
        initial_conditions=[initial_a.copy(), initial_b.copy()]
    )

    # Both subnetworks are one-node MPR models with the same two VOIs.
    # NetworkSet.observe(flat=True) places them at node_indices 0 and 1.
    data_a = np.asarray(data[:, :, 0:1, :], dtype=np.float64)
    data_b = np.asarray(data[:, :, 1:2, :], dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    return [(times, data_a), (times, data_b)]


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a minimal reciprocal inter-subnetwork delayed projection check "
            "against the standard Python hybrid Simulator API."
        )
    )
    parser.add_argument("--nstep", type=int, default=80)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--delay-steps", type=int, default=2)
    parser.add_argument("--weight-ab", type=float, default=0.20)
    parser.add_argument("--weight-ba", type=float, default=0.35)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving the Python-vs-C++ overlay plot.",
    )
    args = parser.parse_args()

    subnet_a = make_subnet("A", node_index=0)
    subnet_b = make_subnet("B", node_index=1)
    proj_ab = make_projection(subnet_a, subnet_b, args.weight_ab, args.delay_steps)
    proj_ba = make_projection(subnet_b, subnet_a, args.weight_ba, args.delay_steps)
    network = NetworkSet(
        subnets=[subnet_a, subnet_b],
        projections=[proj_ab, proj_ba],
    )
    network.configure()

    initial_a = make_initial_state(subnet_a, r=0.8, v=-0.20)
    initial_b = make_initial_state(subnet_b, r=1.3, v=-0.35)
    py_results = run_python_hybrid_simulator(
        network=network,
        initial_a=initial_a,
        initial_b=initial_b,
        nstep=args.nstep,
        chunk_size=args.chunk_size,
    )

    backend = CppHybridBackend(build_root=EXAMPLES_DIR / ".build")
    compiled = backend.compile(
        network,
        monitors=[TemporalAverage(period=args.chunk_size * DT)],
        user_source_hint="minimal_inter_delayed_projection_order",
    )
    native_results = compiled.run(
        initial_states=[initial_a.copy(), initial_b.copy()],
        nstep=args.nstep,
        chunk_size=args.chunk_size,
    )

    payload = {
        "description": (
            "Reciprocal A<->B inter-projections. Python uses the standard "
            "tvb.simulator.hybrid.Simulator path; C++ uses CppHybridBackend."
        ),
        "config": {
            "dt": DT,
            "nstep": args.nstep,
            "chunk_size": args.chunk_size,
            "delay_steps": args.delay_steps,
            "weight_ab": args.weight_ab,
            "weight_ba": args.weight_ba,
            "projection_idelays": [
                proj_ab.idelays.astype(int).tolist(),
                proj_ba.idelays.astype(int).tolist(),
            ],
        },
        "native": compiled.debug_summary(),
        "diffs": {
            "note": "Python Simulator TemporalAverage timestamps are 0.5*dt earlier than C++/Numba timestamps.",
            "A_times_aligned_max_abs": max_abs_diff(
                py_results[0][0] + 0.5 * DT, native_results[0][0]
            ),
            "A_data_max_abs": max_abs_diff(py_results[0][1], native_results[0][1]),
            "B_times_aligned_max_abs": max_abs_diff(
                py_results[1][0] + 0.5 * DT, native_results[1][0]
            ),
            "B_data_max_abs": max_abs_diff(py_results[1][1], native_results[1][1]),
        },
        # "preview": {
        #     "cpp_times": native_results[0][0].tolist(),
        #     "python_times": py_results[0][0].tolist(),
        #     "python_times_aligned": (py_results[0][0] + 0.5 * DT).tolist(),
        #     "A_cpp_r": native_results[0][1][:, 0, 0, 0].tolist(),
        #     "A_python_simulator_r": py_results[0][1][:, 0, 0, 0].tolist(),
        #     "B_cpp_r": native_results[1][1][:, 0, 0, 0].tolist(),
        #     "B_python_simulator_r": py_results[1][1][:, 0, 0, 0].tolist(),
        # },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    if not args.no_plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping plot", file=sys.stderr)
            return

        out_dir = EXAMPLES_DIR / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "minimal_inter_delayed_projection_order.png"

        times_a = native_results[0][0]
        times_b = native_results[1][0]
        a_cpp_r = native_results[0][1][:, 0, 0, 0]
        python_times_aligned = py_results[0][0] + 0.5 * DT
        a_py_r = py_results[0][1][:, 0, 0, 0]
        b_cpp_r = native_results[1][1][:, 0, 0, 0]
        b_py_r = py_results[1][1][:, 0, 0, 0]

        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        axes[0].plot(times_a, a_cpp_r, color="steelblue", linewidth=1.5, label="C++")
        axes[0].plot(
            python_times_aligned,
            a_py_r,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="Python Simulator",
        )
        axes[0].set_title("Subnet A: r(t)")
        axes[0].set_ylabel("r")
        axes[0].grid(alpha=0.3)
        axes[0].legend(loc="best")

        axes[1].plot(times_b, b_cpp_r, color="firebrick", linewidth=1.5, label="C++")
        axes[1].plot(
            python_times_aligned,
            b_py_r,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="Python Simulator",
        )
        axes[1].set_title("Subnet B: r(t)")
        axes[1].set_xlabel("Time (ms)")
        axes[1].set_ylabel("r")
        axes[1].grid(alpha=0.3)
        axes[1].legend(loc="best")

        fig.suptitle(
            "Inter-subnetwork delayed projections: Python Simulator vs C++",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
