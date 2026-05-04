#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal demo: C++ backend rebuild avoidance.

The same tiny network is compiled twice with the same build root. The first
compile builds a native extension; the second compile should reuse the cached
extension selected by the SimulationSpec hash.

Run from the repo root:

    python tvb_library/tvb/simulator/backend_cpp/examples/rebuild_cache_reuse_demo.py

Optional: set TVB_CPP_BUILD_DIR to use a persistent cache directory.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np


EXAMPLES_DIR = Path(__file__).resolve().parent
TVB_LIBRARY_ROOT = EXAMPLES_DIR.parent.parent.parent.parent
if str(TVB_LIBRARY_ROOT) not in sys.path:
    sys.path.insert(0, str(TVB_LIBRARY_ROOT))

os.environ.setdefault("TVB_USER_HOME", str(Path(tempfile.gettempdir()) / "tvb-user"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
warnings.filterwarnings("ignore", message="Hybrid simulation is experimental.*")

from tvb.simulator.backend_cpp import CppHybridBackend
from tvb.simulator.hybrid import NetworkSet, Subnetwork
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.monitors import TemporalAverage


DT = 0.1
N_NODES = 2
N_STEP = 4
CHUNK_SIZE = 2


def make_network() -> tuple[NetworkSet, Subnetwork]:
    model = MontbrioPazoRoxin(I=np.array([2.0]))
    model.configure()

    subnet = Subnetwork(
        name="toy",
        model=model,
        scheme=HeunDeterministic(dt=DT),
        nnodes=N_NODES,
        projections=[],
    ).configure()
    subnet.node_indices = np.arange(N_NODES)

    network = NetworkSet(subnets=[subnet], projections=[], stimuli=[])
    network.configure()
    return network, subnet


def make_initial_state(subnet: Subnetwork) -> np.ndarray:
    model = subnet.model
    state = np.zeros(
        (model.nvar, subnet.nnodes, model.number_of_modes),
        dtype=np.float64,
    )
    for i, state_var in enumerate(model.state_variables):
        if state_var not in model.state_variable_range:
            continue
        low, high = map(float, model.state_variable_range[state_var])
        state[i, :, :] = (low + high) / 2.0
    return state


def timed_compile(backend: CppHybridBackend, network: NetworkSet, monitor: TemporalAverage):
    start = time.perf_counter()
    compiled = backend.compile(
        network,
        monitors=[monitor],
        user_source_hint="rebuild_cache_reuse_demo",
        verbose=True
    )
    return compiled, time.perf_counter() - start


def main() -> None:
    network, subnet = make_network()
    monitor = TemporalAverage(period=CHUNK_SIZE * DT)

    with tempfile.TemporaryDirectory(prefix="tvb-cpp-cache-demo-") as tmpdir:
        build_root = Path(os.environ.get("TVB_CPP_BUILD_DIR", tmpdir)).resolve()
        backend = CppHybridBackend(build_root=build_root)

        first, first_seconds = timed_compile(backend, network, monitor)
        second, second_seconds = timed_compile(backend, network, monitor)

        first_summary = first.debug_summary()
        second_summary = second.debug_summary()

        print(f"Build root       : {build_root}")
        print(f"Cache key        : {first_summary['cache_key']}")
        print(f"First stage      : {first.pipeline_stage} ({first_seconds:.3f}s)")
        print(f"Second stage     : {second.pipeline_stage} ({second_seconds:.3f}s)")
        print(f"Same extension   : {first_summary['extension_path'] == second_summary['extension_path']}")
        print(f"Extension path   : {second_summary['extension_path']}")

        initial_state = make_initial_state(subnet)
        ((times, data),) = second.run(
            initial_states=[initial_state],
            nstep=N_STEP,
            chunk_size=CHUNK_SIZE,
        )
        print(f"Cached run output: times={times.shape}, data={data.shape}")


if __name__ == "__main__":
    main()
