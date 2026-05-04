#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a single-subnetwork MPR connectome simulation with C++ backend BOLD output.

The script builds one 84-node Montbrio-Pazo-Roxin subnetwork, couples nodes
through an intra-subnetwork projection from neural activity ``r`` to the MPR
``Coupling_Term_r`` input, sweeps a global coupling scale in [0, 1], and stores
the resulting BOLD signal sampled every TR=500 ms.

This checkout ships ``connectivity_76.zip``, ``connectivity_96.zip`` and
``connectivity_192.zip``.  To get 84 nodes by default, the script loads
``connectivity_96.zip`` and selects its first 84 regions.  Pass
``--connectivity`` and/or ``--nodes`` to use a different connectome selection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

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

from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.backend_cpp import CppHybridBackend
from tvb.simulator.hybrid import IntraProjection, NetworkSet, Subnetwork
from tvb.simulator.hybrid.coupling import Linear
from tvb.simulator.integrators import HeunStochastic
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.monitors import Bold
from tvb.simulator.noise import Additive

DT = 0.1
TR_MS = 500.0
DEFAULT_CONNECTIVITY = "connectivity_96.zip"
DEFAULT_NODES = 84
DEFAULT_SIMULATION_MS = 10_000.0
DEFAULT_COUPLINGS = np.linspace(0.0, 1.0, 11)
OUTPUT_DIR = EXAMPLES_DIR / "outputs"


def load_connectome_selection(
    source_file: str,
    n_nodes: int,
    scaling: str,
) -> dict[str, np.ndarray]:
    conn = Connectivity.from_file(source_file)
    conn.configure()
    if conn.weights.shape[0] < n_nodes:
        raise ValueError(
            f"{source_file!r} has {conn.weights.shape[0]} regions, "
            f"cannot select {n_nodes} nodes."
        )

    weights = np.asarray(conn.scaled_weights(mode=scaling), dtype=np.float64)
    weights = weights[:n_nodes, :n_nodes].copy()
    np.fill_diagonal(weights, 0.0)

    tract_lengths = np.asarray(conn.tract_lengths, dtype=np.float64)[
        :n_nodes, :n_nodes
    ].copy()
    tract_lengths[weights == 0.0] = 0.0

    region_labels = np.asarray(conn.region_labels[:n_nodes], dtype=str)
    return {
        "weights": weights,
        "tract_lengths": tract_lengths,
        "region_labels": region_labels,
    }


def make_subnet(
    n_nodes: int,
    weights: np.ndarray,
    tract_lengths: np.ndarray,
    global_coupling: float,
    conduction_speed: float,
) -> Subnetwork:
    model = MontbrioPazoRoxin(
        variables_of_interest=["r"],
    )
    model.configure()

    weights_sparse = sp.csr_matrix(weights, dtype=np.float64)
    lengths_sparse = sp.csr_matrix(tract_lengths, dtype=np.float64)
    projection = IntraProjection(
        source_cvar=np.array([0]),  # source state variable: r
        target_cvar=np.array([0]),  # target coupling slot: Coupling_Term_r
        weights=weights_sparse,
        lengths=lengths_sparse,
        cv=conduction_speed,
        dt=DT,
        scale=float(global_coupling),
        cfun=Linear(),
    )

    subnet = Subnetwork(
        name="mpr_connectome",
        model=model,
        scheme=HeunStochastic(
            dt=DT, noise=Additive(nsig=np.array([0.037, 2 * 0.037]), noise_seed=42)
        ),
        nnodes=n_nodes,
        projections=[projection],
    ).configure()
    subnet.node_indices = np.arange(n_nodes)
    return subnet


def make_network(subnet: Subnetwork) -> NetworkSet:
    network = NetworkSet(subnets=[subnet], projections=[], stimuli=[])
    network.configure()
    return network


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


def run_coupling(
    weights: np.ndarray,
    tract_lengths: np.ndarray,
    nstep: int,
    n_nodes: int,
    global_coupling: float,
    conduction_speed: float,
    backend: "CppHybridBackend | None" = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compile (or reuse cached binary) and run one coupling-scale configuration.

    Passing `backend` is optional; a fresh one is created when omitted.
    Because the projection scale is a runtime parameter that does not influence
    the generated C++ structure, `backend.compile()` returns a cache hit on
    every call after the first — so only the first iteration pays the compile
    cost.  Subsequent calls skip compilation and go straight to `run()`.
    """
    if backend is None:
        backend = CppHybridBackend()

    subnet = make_subnet(
        n_nodes=n_nodes,
        weights=weights,
        tract_lengths=tract_lengths,
        global_coupling=global_coupling,
        conduction_speed=conduction_speed,
    )
    network = make_network(subnet)
    initial_state = make_initial_state(subnet)
    bold = Bold(period=TR_MS)

    t0 = time.perf_counter()
    compiled = backend.compile(
        network,
        monitors=[bold],
        user_source_hint="mpr_bold_connectome",
    )
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    ((times, bold_data),) = compiled.run(
        nstep=nstep,
        chunk_size=1,
        initial_states=[initial_state.copy()],
    )
    run_s = time.perf_counter() - t0

    # Model variables_of_interest=["r"], so the BOLD data has one VOI.
    bold_r = np.asarray(bold_data, dtype=np.float64)[:, 0, :, 0]
    summary = compiled.debug_summary()
    summary.update(
        {
            "global_coupling": float(global_coupling),
            "compile_s": compile_s,
            "run_s": run_s,
            "n_bold_samples": int(bold_r.shape[0]),
        }
    )
    return np.asarray(times, dtype=np.float64), bold_r, summary


def parse_couplings(args: argparse.Namespace) -> np.ndarray:
    if args.couplings is not None:
        values = np.asarray(args.couplings, dtype=np.float64)
    else:
        values = np.linspace(args.coupling_min, args.coupling_max, args.n_couplings)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("coupling sweep must contain at least one value.")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("global coupling values must stay in [0, 1].")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep MPR global coupling and store C++ backend BOLD output."
    )
    parser.add_argument("--connectivity", default=DEFAULT_CONNECTIVITY)
    parser.add_argument("--nodes", type=int, default=DEFAULT_NODES)
    parser.add_argument("--simulation-ms", type=float, default=DEFAULT_SIMULATION_MS)
    parser.add_argument("--conduction-speed", type=float, default=3.0)
    parser.add_argument(
        "--weight-scaling",
        default="tract",
        choices=["tract", "edge", "region", "node", "none"],
    )
    parser.add_argument("--coupling-min", type=float, default=0.0)
    parser.add_argument("--coupling-max", type=float, default=1.0)
    parser.add_argument("--n-couplings", type=int, default=len(DEFAULT_COUPLINGS))
    parser.add_argument(
        "--couplings",
        type=float,
        nargs="+",
        default=None,
        help="Explicit coupling values. Overrides min/max/n-couplings.",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=OUTPUT_DIR / "mpr_bold_connectome_cpp_sweep.npz",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=OUTPUT_DIR / "mpr_bold_connectome_cpp_sweep_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.nodes < 1:
        raise ValueError("nodes must be >= 1.")
    if args.simulation_ms <= 0.0:
        raise ValueError("simulation-ms must be > 0.")
    if args.conduction_speed <= 0.0:
        raise ValueError("conduction-speed must be > 0.")
    if args.n_couplings < 1:
        raise ValueError("n-couplings must be >= 1.")

    couplings = parse_couplings(args)
    nstep = int(round(args.simulation_ms / DT))
    connectome = load_connectome_selection(
        source_file=args.connectivity,
        n_nodes=args.nodes,
        scaling=None if args.weight_scaling == "none" else args.weight_scaling,
    )

    print("=== MPR Connectome BOLD Sweep with C++ Backend ===")
    print(f"connectivity      = {args.connectivity}")
    print(f"nodes             = {args.nodes}")
    print(f"dt                = {DT} ms")
    print(f"simulation        = {args.simulation_ms} ms ({nstep} steps)")
    print(f"BOLD TR           = {TR_MS} ms")
    print(f"couplings         = {', '.join(f'{v:.3g}' for v in couplings)}")
    print()

    # One shared backend: the C++ module is compiled on the first coupling value
    # and reused for all subsequent ones (scale is a runtime parameter).
    shared_backend = CppHybridBackend()

    all_bold = []
    all_times = []
    summaries = []
    for coupling_value in couplings:
        print(f"running global coupling {coupling_value:.6g} ...", flush=True)
        times, bold_r, summary = run_coupling(
            weights=connectome["weights"],
            tract_lengths=connectome["tract_lengths"],
            nstep=nstep,
            n_nodes=args.nodes,
            global_coupling=float(coupling_value),
            conduction_speed=args.conduction_speed,
            backend=shared_backend,
        )
        all_times.append(times)
        all_bold.append(bold_r)
        summaries.append(summary)
        cached = summary["pipeline_stage"] == "extension_cached"
        print(
            f"  samples={bold_r.shape[0]} compile={summary['compile_s']:.3f}s "
            f"run={summary['run_s']:.3f}s"
            + (" (cached)" if cached else ""),
            flush=True,
        )

    reference_times = all_times[0]
    for times in all_times[1:]:
        np.testing.assert_allclose(times, reference_times)

    bold = np.stack(all_bold, axis=0)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        bold=bold,
        times_ms=reference_times,
        global_coupling=couplings,
        region_labels=connectome["region_labels"],
        dt_ms=np.array(DT),
        tr_ms=np.array(TR_MS),
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "connectivity": args.connectivity,
                "nodes": args.nodes,
                "simulation_ms": args.simulation_ms,
                "dt_ms": DT,
                "tr_ms": TR_MS,
                "conduction_speed": args.conduction_speed,
                "weight_scaling": args.weight_scaling,
                "output_npz": str(args.output_npz),
                "bold_shape": list(bold.shape),
                "bold_shape_axes": ["global_coupling", "time", "node"],
                "runs": summaries,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print()
    print(f"BOLD array shape: {bold.shape} (coupling, time, node)")
    print(f"Saved BOLD data to {args.output_npz}")
    print(f"Saved run summary to {args.output_json}")


if __name__ == "__main__":
    main()
