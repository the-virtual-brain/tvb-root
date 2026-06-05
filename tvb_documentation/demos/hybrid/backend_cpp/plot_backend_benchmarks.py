#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot hybrid backend benchmark summaries.

By default this script reads the JSON files written by the benchmark scripts in
this directory and writes publication-friendly PNG figures plus a normalized CSV
table.  It can also refresh the benchmark JSONs before plotting.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


EXAMPLES_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXAMPLES_DIR / "outputs"
DEFAULT_HYBRID_JSON = OUTPUT_DIR / "hybrid_backends_scenario_benchmark.json"
DEFAULT_PARALLEL_JSON = OUTPUT_DIR / "parallel_sweep_cpp_cpu_benchmark.json"
DEFAULT_PLOT_DIR = OUTPUT_DIR / "benchmark_plots"

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_BACKENDS = ("python", "numba", "numba_cuda", "cpp")
BACKEND_LABELS = {
    "python": "Python/NumPy",
    "numba": "Numba CPU",
    "numba_cuda": "Numba CUDA",
    "cpp": "C++ single",
}
BACKEND_COLORS = {
    "python": "#5B6770",
    "numba": "#2A9D8F",
    "numba_cuda": "#457B9D",
    "cpp": "#E76F51",
}
EXTRA_COLORS = ("#8E7CC3", "#C77DFF", "#6D597A", "#B56576", "#355070")
SCENARIO_LABELS = {
    "single_subnet": "Single subnet",
    "two_subnets_uncoupled": "Two subnets, uncoupled",
    "two_subnets_delayed_projection": "Two subnets, delayed projection",
}


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run the matching benchmark script first, "
            "or pass --refresh-hybrid/--refresh-parallel."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def ok_metric(result: dict[str, Any], metric: str) -> float | None:
    if result.get("status") != "ok":
        return None
    value = result.get(metric)
    return None if value is None else float(value)


def case_label(row: dict[str, Any]) -> str:
    scenario = SCENARIO_LABELS.get(row["scenario"], row["scenario"])
    return f"{scenario}\nchunk={row['chunk_size']}"


def backend_sort_key(name: str) -> tuple[int, int | str]:
    if name in BASE_BACKENDS:
        return (0, BASE_BACKENDS.index(name))
    if name.startswith("cpp_parallel_"):
        try:
            return (1, int(name.rsplit("_", 1)[1]))
        except ValueError:
            return (1, name)
    return (2, name)


def discover_backends(rows: list[dict[str, Any]]) -> list[str]:
    names = set(BASE_BACKENDS)
    for row in rows:
        for key, value in row.items():
            if key.startswith("cpp_parallel_") and isinstance(value, dict):
                names.add(key)
    return sorted(names, key=backend_sort_key)


def backend_label(name: str) -> str:
    if name in BACKEND_LABELS:
        return BACKEND_LABELS[name]
    if name.startswith("cpp_parallel_"):
        workers = name.rsplit("_", 1)[1]
        return f"C++ {workers} workers"
    return name


def backend_color(name: str, index: int) -> str:
    if name in BACKEND_COLORS:
        return BACKEND_COLORS[name]
    return EXTRA_COLORS[index % len(EXTRA_COLORS)]


def write_hybrid_csv(
    rows: list[dict[str, Any]],
    backends: list[str],
    output_path: Path,
) -> None:
    fieldnames = [
        "scenario",
        "chunk_size",
        "backend",
        "status",
        "run_s",
        "per_sim_s",
        "speedup_vs_python_total",
        "speedup_vs_python_per_sim",
        "sweeps",
        "cpu_times_extrapolated",
        "notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            py_total = ok_metric(row["python"], "run_s")
            py_per_sim = ok_metric(row["python"], "per_sim_s")
            for backend in backends:
                result = row.get(backend, {"status": "not_applicable"})
                total = ok_metric(result, "run_s")
                per_sim = ok_metric(result, "per_sim_s")
                notes = []
                if result.get("extrapolated"):
                    notes.append(
                        f"extrapolated from {result.get('measured_sweeps')} "
                        f"to {result.get('target_sweeps')} sweeps"
                    )
                if result.get("proxy_sweep"):
                    notes.append(result["proxy_sweep"])
                if result.get("reason"):
                    notes.append(result["reason"])
                writer.writerow(
                    {
                        "scenario": row["scenario"],
                        "chunk_size": row["chunk_size"],
                        "backend": backend,
                        "status": result.get("status"),
                        "run_s": total,
                        "per_sim_s": per_sim,
                        "speedup_vs_python_total": (
                            py_total / total if py_total and total else None
                        ),
                        "speedup_vs_python_per_sim": (
                            py_per_sim / per_sim if py_per_sim and per_sim else None
                        ),
                        "sweeps": row.get("sweeps"),
                        "cpu_times_extrapolated": bool(result.get("extrapolated")),
                        "notes": "; ".join(notes),
                    }
                )


def grouped_backend_plot(
    rows: list[dict[str, Any]],
    backends: list[str],
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    *,
    speedup_vs_python: bool = False,
    log_y: bool = False,
) -> None:
    labels = [case_label(row) for row in rows]
    x = np.arange(len(rows), dtype=float)
    width = min(0.16, 0.82 / max(1, len(backends)))
    offsets = (np.arange(len(backends)) - (len(backends) - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(max(11.0, 1.5 * len(rows)), 5.8))
    plotted_any = False
    for idx, (offset, backend) in enumerate(zip(offsets, backends)):
        values: list[float] = []
        for row in rows:
            value = ok_metric(row.get(backend, {}), metric)
            if speedup_vs_python:
                baseline = ok_metric(row["python"], metric)
                value = baseline / value if baseline and value else None
            values.append(np.nan if value is None else value)
        if not np.isfinite(values).any():
            continue
        plotted_any = True
        ax.bar(
            x + offset,
            values,
            width=width,
            label=backend_label(backend),
            color=backend_color(backend, idx),
        )

    if not plotted_any:
        raise ValueError(f"No plottable values for {metric}.")

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    if log_y:
        ax.set_yscale("log")
    if speedup_vs_python:
        ax.axhline(1.0, color="#333333", linewidth=0.9, linestyle="--")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        ncol=min(6, len(backends)),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_parallel_summary(summary: dict[str, Any], output_dir: Path) -> list[Path]:
    rows = summary.get("rows", [])
    if not rows:
        return []

    workers = [int(row["workers"]) for row in rows]
    wall_s = [float(row["wall_s"]) for row in rows]
    speedups = [float(row["speedup_vs_1_worker"]) for row in rows]
    outputs = []

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(workers, wall_s, marker="o", color=BACKEND_COLORS["cpp"])
    ax.set_title("C++ CPU Parameter Sweep Wall Time")
    ax.set_xlabel("Workers")
    ax.set_ylabel("Wall time (s)")
    ax.grid(alpha=0.25)
    ax.set_xticks(workers)
    path = output_dir / "cpp_parallel_wall_time.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(workers, speedups, marker="o", color=BACKEND_COLORS["numba"])
    ax.plot(workers, workers, linestyle="--", color="#777777", label="Ideal")
    ax.set_title("C++ CPU Parameter Sweep Speedup")
    ax.set_xlabel("Workers")
    ax.set_ylabel("Speedup vs 1 worker")
    ax.grid(alpha=0.25)
    ax.set_xticks(workers)
    ax.legend(frameon=False)
    path = output_dir / "cpp_parallel_speedup.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)

    return outputs


def run_command(command: list[str]) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in command))
    subprocess.run(command, cwd=EXAMPLES_DIR, check=True)


def refresh_benchmarks(args: argparse.Namespace) -> None:
    if args.refresh_hybrid:
        command = [
            sys.executable,
            str(EXAMPLES_DIR / "benchmark_hybrid_backends.py"),
            "--output-json",
            str(args.hybrid_json),
        ]
        command.extend(shlex.split(args.hybrid_benchmark_args))
        run_command(command)

    if args.refresh_parallel:
        command = [
            sys.executable,
            str(EXAMPLES_DIR / "benchmark_parallel_sweep.py"),
            "--output",
            str(args.parallel_json),
        ]
        command.extend(shlex.split(args.parallel_benchmark_args))
        run_command(command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot hybrid backend benchmark JSON summaries."
    )
    parser.add_argument("--hybrid-json", type=Path, default=DEFAULT_HYBRID_JSON)
    parser.add_argument("--parallel-json", type=Path, default=DEFAULT_PARALLEL_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument(
        "--skip-parallel",
        action="store_true",
        help="Only plot backend comparisons; ignore the C++ worker-scaling JSON.",
    )
    parser.add_argument(
        "--refresh-hybrid",
        action="store_true",
        help="Run benchmark_hybrid_backends.py before plotting.",
    )
    parser.add_argument(
        "--refresh-parallel",
        action="store_true",
        help="Run benchmark_parallel_sweep.py before plotting.",
    )
    parser.add_argument(
        "--hybrid-benchmark-args",
        default="",
        help=(
            "Extra quoted arguments passed to benchmark_hybrid_backends.py, "
            "for example: \"--nstep 5000 --sweeps 256\"."
        ),
    )
    parser.add_argument(
        "--parallel-benchmark-args",
        default="",
        help=(
            "Extra quoted arguments passed to benchmark_parallel_sweep.py, "
            "for example: \"--workers 1,4,8 --sweeps 20\"."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    refresh_benchmarks(args)

    hybrid_rows = load_json(args.hybrid_json)
    if not isinstance(hybrid_rows, list):
        raise TypeError(f"{args.hybrid_json} must contain a list of benchmark rows.")

    written: list[Path] = []
    csv_path = args.output_dir / "hybrid_backend_benchmark_table.csv"
    backends = discover_backends(hybrid_rows)
    write_hybrid_csv(hybrid_rows, backends, csv_path)
    written.append(csv_path)

    plot_specs = [
        (
            "run_s",
            "Hybrid Backend Total Sweep Wall Time",
            "Wall time for full sweep (s)",
            "hybrid_backend_wall_time_total.png",
            False,
            True,
        ),
        (
            "per_sim_s",
            "Hybrid Backend Per-Simulation Wall Time",
            "Wall time per simulation (s)",
            "hybrid_backend_wall_time_per_sim.png",
            False,
            True,
        ),
        (
            "run_s",
            "Hybrid Backend Total Sweep Speedup vs Python",
            "Speedup vs Python/NumPy",
            "hybrid_backend_speedup_vs_python_total.png",
            True,
            True,
        ),
        (
            "per_sim_s",
            "Hybrid Backend Per-Simulation Speedup vs Python",
            "Speedup vs Python/NumPy",
            "hybrid_backend_speedup_vs_python_per_sim.png",
            True,
            True,
        ),
    ]
    for metric, title, ylabel, filename, speedup, log_y in plot_specs:
        path = args.output_dir / filename
        grouped_backend_plot(
            hybrid_rows,
            backends,
            metric,
            title,
            ylabel,
            path,
            speedup_vs_python=speedup,
            log_y=log_y,
        )
        written.append(path)

    if not args.skip_parallel and args.parallel_json.exists():
        parallel_summary = load_json(args.parallel_json)
        written.extend(plot_parallel_summary(parallel_summary, args.output_dir))
    elif not args.skip_parallel:
        print(f"Skipping parallel plots; {args.parallel_json} does not exist.")

    print("Wrote benchmark plots and tables:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
