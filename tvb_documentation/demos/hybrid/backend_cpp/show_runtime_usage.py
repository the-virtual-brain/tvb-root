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

from tvb.simulator.hybrid import NetworkSet, Subnetwork
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.monitors import TemporalAverage

from backend_cpp import CppHybridBackend


DT = 0.1


def make_subnet(name: str, n_nodes: int) -> Subnetwork:
    model = MontbrioPazoRoxin(I=np.array([2.0]))
    model.configure()
    return Subnetwork(
        name=name,
        model=model,
        scheme=HeunDeterministic(dt=DT),
        nnodes=n_nodes,
    ).configure()


def make_network(n_nodes: int) -> NetworkSet:
    subnet = make_subnet("sn", n_nodes)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show how the fixed C++ runtime is used by a generated backend module."
    )
    parser.add_argument("--nodes", type=int, default=3)
    parser.add_argument("--nstep", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=2)
    args = parser.parse_args()

    network = make_network(args.nodes)
    initial_state = make_initial_state(network.subnets[0])

    backend = CppHybridBackend()
    compiled = backend.compile(
        network,
        monitors=[TemporalAverage(period=args.chunk_size * DT)],
        user_source_hint="show_runtime_usage",
    )
    module = compiled.load_module()
    times, data = compiled.run(
        initial_states=[initial_state],
        nstep=args.nstep,
        chunk_size=args.chunk_size,
    )

    generated_cpp = compiled.generated_cpp_path.read_text(encoding="utf-8")
    runtime_header = compiled.generated_source.runtime_header_path.read_text(
        encoding="utf-8"
    )

    payload = {
        "summary": {
            "python_entrypoint": "CppHybridBackend.compile() / CompiledCppNetwork.run()",
            "generated_module": compiled.module_name,
            "generated_cpp_path": str(compiled.generated_cpp_path),
            "bindings_cpp_path": str(compiled.generated_source.bindings_cpp_path),
            "runtime_header_path": str(compiled.generated_source.runtime_header_path),
            "extension_path": str(compiled.generated_source.extension_path),
        },
        "usage_chain": [
            "Python lowers the NetworkSet into SimulationSpec.",
            "codegen.py writes a generated C++ translation unit for this spec.",
            "that generated C++ file includes runtime/runtime.hpp.",
            "the generated module delegates describe() and run_simulation() into tvb::hybrid::runtime.",
            "the pybind11 bindings expose run_simulation() back to Python.",
        ],
        "generated_cpp_runtime_calls": {
            "includes_runtime_header": '#include "runtime/runtime.hpp"' in generated_cpp,
            "uses_runtime_describe": "tvb::hybrid::runtime::describe<GeneratedModel>()"
            in generated_cpp,
            "uses_runtime_run_simulation": "tvb::hybrid::runtime::run_simulation<GeneratedModel>("
            in generated_cpp,
        },
        "runtime_header_symbols": {
            "has_simulation_metadata": "struct SimulationMetadata" in runtime_header,
            "has_simulation_result": "struct SimulationResult" in runtime_header,
            "has_heun_step": "inline void heun_step" in runtime_header,
            "has_run_simulation": "inline SimulationResult run_simulation" in runtime_header,
        },
        "native_run": {
            "times": times.tolist(),
            "data_shape": list(data.shape),
            "first_chunk_first_voi": data[0, 0, :, 0].tolist(),
        },
        "extension_metadata": module.describe_metadata(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
