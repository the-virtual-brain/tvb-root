#!/usr/bin/env python
# coding: utf-8
"""Benchmark single-core vs multi-core C++ hybrid parameter sweeps.

This script uses the cortex + thalamus network from
``simulate_hybrid_parameter_sweeps_backend_cpp.py`` and compares
``CppHybridBackend.sweep(..., n_workers=1)`` against multiple worker counts.

The benchmark intentionally runs a compile-cache warmup before timing so the
reported times mostly measure sweep execution, not C++ extension compilation.
It also compares every multi-worker result with the single-worker reference.
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


def find_repo_root(start: Path | None = None) -> Path:
    try:
        script_dir: Path | None = Path(__file__).resolve().parent
    except NameError:
        script_dir = None
    for search_root in (start, script_dir, Path.cwd()):
        if search_root is None:
            continue
        for candidate in (search_root.resolve(), *search_root.resolve().parents):
            if (candidate / "tvb_library" / "tvb" / "simulator").exists():
                return candidate
    raise RuntimeError("Could not locate the tvb-root repository root.")


REPO_ROOT = find_repo_root()
TVB_LIBRARY_ROOT = REPO_ROOT / "tvb_library"
DEMO_DIR = REPO_ROOT / "tvb_documentation" / "demos" / "hybrid" / "backend_cpp"
BUILD_ROOT = Path(os.environ.get("TVB_CPP_BUILD_DIR", str(DEMO_DIR / ".build"))).resolve()
OUTPUT_DIR = DEMO_DIR / "outputs"

if str(TVB_LIBRARY_ROOT) not in sys.path:
    sys.path.insert(0, str(TVB_LIBRARY_ROOT))

os.environ.setdefault("TVB_USER_HOME", str(Path(tempfile.gettempdir()) / "tvb-user"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
warnings.filterwarnings("ignore", message="Hybrid simulation is experimental.*")

from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.backend_cpp import CppHybridBackend
from tvb.simulator.hybrid import IntraProjection, InterProjection, NetworkSet, Subnetwork
from tvb.simulator.hybrid.coupling import Linear
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.models.jansen_rit import JansenRit
from tvb.simulator.monitors import TemporalAverage


DT = 0.1
N_CORTEX = 68
N_THALAMUS = 8
N_TOTAL = N_CORTEX + N_THALAMUS


def build_cortex_thalamus_network() -> tuple[NetworkSet, dict[str, np.ndarray]]:
    """Build the same network used by the parameter-sweep demo."""
    conn = Connectivity.from_file("connectivity_76.zip")
    conn.configure()

    def slice_weights(row_slice, col_slice):
        return np.asarray(
            conn.weights[row_slice[0]:row_slice[1], col_slice[0]:col_slice[1]],
            dtype=np.float64,
        )

    def slice_lengths(row_slice, col_slice):
        return np.asarray(
            conn.tract_lengths[row_slice[0]:row_slice[1], col_slice[0]:col_slice[1]],
            dtype=np.float64,
        )

    ctx_model = JansenRit()
    ctx_model.configure()
    ctx = Subnetwork(
        name="cortex",
        model=ctx_model,
        scheme=HeunDeterministic(dt=DT),
        nnodes=N_CORTEX,
    )
    ctx.node_indices = np.arange(N_CORTEX)
    ctx.projections = [
        IntraProjection(
            source_cvar=np.array([0], dtype=np.int_),
            target_cvar=np.array([0], dtype=np.int_),
            weights=sp.csr_matrix(slice_weights((0, N_CORTEX), (0, N_CORTEX))),
            lengths=sp.csr_matrix(slice_lengths((0, N_CORTEX), (0, N_CORTEX))),
            cv=1.0,
            dt=DT,
            scale=1.0,
            cfun=Linear(a=np.array([0.03])),
        )
    ]
    ctx.configure()

    thal_model = JansenRit()
    thal_model.configure()
    thal = Subnetwork(
        name="thalamus",
        model=thal_model,
        scheme=HeunDeterministic(dt=DT),
        nnodes=N_THALAMUS,
    )
    thal.node_indices = np.arange(N_CORTEX, N_TOTAL)
    thal.configure()

    cortex_to_thalamus = InterProjection(
        source=ctx,
        target=thal,
        source_cvar=1,
        target_cvar=0,
        weights=sp.csr_matrix(slice_weights((N_CORTEX, N_TOTAL), (0, N_CORTEX))),
        lengths=sp.csr_matrix(slice_lengths((N_CORTEX, N_TOTAL), (0, N_CORTEX))),
        cv=1.0,
        dt=DT,
        scale=1.0,
        cfun=Linear(a=np.array([0.01])),
    )

    network = NetworkSet(subnets=[ctx, thal], projections=[cortex_to_thalamus])
    network.configure()

    node_indices = {
        "cortex": np.arange(N_CORTEX),
        "thalamus": np.arange(N_CORTEX, N_TOTAL),
    }
    return network, node_indices


def parse_worker_counts(value: str, cpu_count: int) -> list[int]:
    if value == "auto":
        counts = [1]
        n = 2
        while n < cpu_count:
            counts.append(n)
            n *= 2
        if cpu_count not in counts:
            counts.append(cpu_count)
        return counts

    counts = []
    for item in value.split(","):
        worker_count = int(item.strip())
        if worker_count < 1:
            raise ValueError("worker counts must be >= 1")
        counts.append(worker_count)
    counts = list(dict.fromkeys(counts))
    if 1 not in counts:
        counts.insert(0, 1)
    return counts


def compare_results(reference: Any, candidate: Any) -> dict[str, Any]:
    comparisons = {}
    ok = True
    max_abs = 0.0
    max_rel = 0.0

    for name, ref_arr in reference.tavg.items():
        cand_arr = candidate.tavg[name]
        abs_diff = np.abs(ref_arr - cand_arr)
        local_max_abs = float(abs_diff.max(initial=0.0))
        denom = np.maximum(np.abs(ref_arr), 1e-300)
        local_max_rel = float((abs_diff / denom).max(initial=0.0))
        local_ok = bool(np.allclose(ref_arr, cand_arr, rtol=1e-10, atol=1e-10))
        comparisons[name] = {
            "ok": local_ok,
            "max_abs_diff": local_max_abs,
            "max_rel_diff": local_max_rel,
            "shape": list(ref_arr.shape),
        }
        ok = ok and local_ok
        max_abs = max(max_abs, local_max_abs)
        max_rel = max(max_rel, local_max_rel)

    if reference.merged_tavg is not None and candidate.merged_tavg is not None:
        abs_diff = np.abs(reference.merged_tavg - candidate.merged_tavg)
        merged_ok = bool(
            np.allclose(reference.merged_tavg, candidate.merged_tavg, rtol=1e-10, atol=1e-10)
        )
        comparisons["merged_tavg"] = {
            "ok": merged_ok,
            "max_abs_diff": float(abs_diff.max(initial=0.0)),
            "shape": list(reference.merged_tavg.shape),
        }
        ok = ok and merged_ok

    return {
        "ok": ok,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "by_output": comparisons,
    }


def time_sweep(
    backend: CppHybridBackend,
    network: NetworkSet,
    sweep_values: np.ndarray,
    nstep: int,
    tavg_period: float,
    node_indices: dict[str, np.ndarray],
    n_workers: int,
) -> tuple[Any, float]:
    monitor = TemporalAverage(period=tavg_period)
    start = time.perf_counter()
    result = backend.sweep(
        network,
        params={"coupling_scale": sweep_values},
        nstep=nstep,
        monitors=[monitor],
        node_indices=node_indices,
        n_workers=n_workers,
    )
    return result, time.perf_counter() - start


def summarize_run(
    result: Any,
    wall_s: float,
    baseline_s: float,
    n_sweeps: int,
    nstep: int,
) -> dict[str, Any]:
    return {
        "backend": result.backend,
        "wall_s": wall_s,
        "backend_elapsed_s": float(result.elapsed),
        "speedup_vs_1_worker": baseline_s / wall_s,
        "sweep_points_per_s": n_sweeps / wall_s,
        "ksteps_per_s": n_sweeps * nstep / wall_s / 1000.0,
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    print()
    print("workers | backend    | wall_s | speedup | sweep/s | ksteps/s | correctness")
    print("--------+------------+--------+---------+---------+----------+------------")
    for row in rows:
        correctness = "reference" if row["workers"] == 1 else (
            "ok" if row["correctness"]["ok"] else "FAIL"
        )
        print(
            f"{row['workers']:>7} | "
            f"{row['backend']:<10} | "
            f"{row['wall_s']:>6.2f} | "
            f"{row['speedup_vs_1_worker']:>7.2f} | "
            f"{row['sweep_points_per_s']:>7.2f} | "
            f"{row['ksteps_per_s']:>8.2f} | "
            f"{correctness}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark C++ hybrid parameter sweeps with one and many CPU workers."
    )
    parser.add_argument("--nstep", type=int, default=5000)
    parser.add_argument("--sweeps", type=int, default=50)
    parser.add_argument("--tavg-period", type=float, default=1.0)
    parser.add_argument("--sweep-start", type=float, default=0.002)
    parser.add_argument("--sweep-stop", type=float, default=0.1)
    parser.add_argument(
        "--workers",
        default="auto",
        help="Comma-separated worker counts, or 'auto' for 1, powers of two, and os.cpu_count().",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "parallel_sweep_cpp_cpu_benchmark.json",
        help="JSON summary path.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip the one-point compile-cache warmup. Timings may include compilation.",
    )
    args = parser.parse_args()

    if args.nstep < 1:
        raise ValueError("--nstep must be >= 1")
    if args.sweeps < 1:
        raise ValueError("--sweeps must be >= 1")

    cpu_count = os.cpu_count() or 1
    worker_counts = parse_worker_counts(args.workers, cpu_count)
    sweep_values = np.linspace(args.sweep_start, args.sweep_stop, args.sweeps, dtype=np.float32)

    print(f"Repository root: {REPO_ROOT}")
    print(f"C++ build root:  {BUILD_ROOT}")
    print(f"CPU count:       {cpu_count}")
    print(f"Sweep:           {args.sweeps} points x {args.nstep} steps")
    print(f"Workers:         {worker_counts}")

    network, node_indices = build_cortex_thalamus_network()
    backend = CppHybridBackend(build_root=BUILD_ROOT)

    if not args.skip_warmup:
        print("Warmup: compiling/reusing C++ extension with one sweep point...")
        time_sweep(
            backend,
            network,
            sweep_values[:1],
            args.nstep,
            args.tavg_period,
            node_indices,
            n_workers=1,
        )

    rows: list[dict[str, Any]] = []
    reference = None
    baseline_s = None

    for n_workers in worker_counts:
        print(f"Running n_workers={n_workers}...")
        result, wall_s = time_sweep(
            backend,
            network,
            sweep_values,
            args.nstep,
            args.tavg_period,
            node_indices,
            n_workers=n_workers,
        )
        if n_workers == 1:
            reference = result
            baseline_s = wall_s
            correctness = {"ok": True, "by_output": {}}
        else:
            correctness = compare_results(reference, result)

        row = {
            "workers": n_workers,
            **summarize_run(result, wall_s, baseline_s, args.sweeps, args.nstep),
            "correctness": correctness,
        }
        rows.append(row)

    print_table(rows)

    failed = [row for row in rows if not row["correctness"]["ok"]]
    summary = {
        "repo_root": str(REPO_ROOT),
        "build_root": str(BUILD_ROOT),
        "cpu_count": cpu_count,
        "nstep": args.nstep,
        "sweeps": args.sweeps,
        "tavg_period": args.tavg_period,
        "sweep_start": args.sweep_start,
        "sweep_stop": args.sweep_stop,
        "rows": rows,
        "all_correct": not failed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote JSON summary: {args.output}")

    if failed:
        raise SystemExit("Correctness check failed for at least one worker count.")


if __name__ == "__main__":
    main()
