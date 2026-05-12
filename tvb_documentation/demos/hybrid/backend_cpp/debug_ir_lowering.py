#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
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

from tvb.simulator.hybrid import InterProjection, NetworkSet, Subnetwork
from tvb.simulator.hybrid.coupling import Linear
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.monitors import TemporalAverage

from tvb.simulator.backend_cpp import CppHybridBackend

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


def make_single_network(n_nodes: int) -> NetworkSet:
    subnet = make_subnet("sn", n_nodes)
    network = NetworkSet(subnets=[subnet], projections=[])
    network.configure()
    return network


def make_coupled_network(n_nodes: int) -> NetworkSet:
    sn1 = make_subnet("sn1", n_nodes)
    sn2 = make_subnet("sn2", n_nodes)
    weights = sp.eye(n_nodes, format="csr", dtype=np.float64) * 0.1
    lengths = sp.csr_matrix((n_nodes, n_nodes), dtype=np.float64)
    projection = InterProjection(
        source=sn1,
        target=sn2,
        source_cvar=np.array([0]),
        target_cvar=np.array([0]),
        weights=weights,
        lengths=lengths,
        cv=3.0,
        dt=DT,
        scale=1.0,
        cfun=Linear(),
    )
    network = NetworkSet(subnets=[sn1, sn2], projections=[projection])
    network.configure()
    return network


def summarize_projection(projection) -> dict[str, object]:
    return {
        "name": projection.name,
        "source_subnet": projection.source_subnet,
        "target_subnet": projection.target_subnet,
        "cfun_type": projection.cfun_type,
        "cvar_mapping_mode": projection.cvar_mapping_mode,
        "horizon": projection.horizon,
        "scale": projection.scale,
        "weights_nnz": int(projection.weights_data.shape[0]),
        "source_cvar_shape": list(projection.source_cvar.shape),
        "target_cvar_shape": list(projection.target_cvar.shape),
        "mode_map_shape": None if projection.mode_map is None else list(projection.mode_map.shape),
    }


def summarize_subnetwork(subnetwork) -> dict[str, object]:
    return {
        "name": subnetwork.name,
        "model_type": subnetwork.model_type,
        "integrator": subnetwork.integrator.type_name,
        "dt": subnetwork.integrator.dt,
        "n_nodes": subnetwork.n_nodes,
        "n_modes": subnetwork.n_modes,
        "n_state_vars": subnetwork.n_state_vars,
        "n_coupling_vars": subnetwork.n_coupling_vars,
        "variables_of_interest": list(subnetwork.variables_of_interest),
        "state_variables": list(subnetwork.state_variables),
        "parameter_names": sorted(subnetwork.parameter_values.keys()),
        "initial_state_shape": list(subnetwork.initial_state_shape),
        "has_stimulus": subnetwork.has_stimulus,
    }


def build_demo_network(kind: str, n_nodes: int) -> NetworkSet:
    if kind == "single":
        return make_single_network(n_nodes)
    if kind == "coupled":
        return make_coupled_network(n_nodes)
    raise ValueError(f"Unsupported network kind: {kind}")


def make_demo_initial_states(network: NetworkSet) -> list[np.ndarray]:
    return [make_initial_state(subnet) for subnet in network.subnets]


def main() -> None:
    parser = argparse.ArgumentParser(description="Lower a hybrid demo NetworkSet and print its C++ backend spec.")
    parser.add_argument("--kind", choices=("single", "coupled"), default="single")
    parser.add_argument("--nodes", type=int, default=4)
    args = parser.parse_args()

    network = build_demo_network(args.kind, args.nodes)
    initial_states = make_demo_initial_states(network)
    backend = CppHybridBackend()
    compiled = backend.compile(
        network,
        monitors=[TemporalAverage(period=1.0)],
        user_source_hint=f"debug_lowering_demo:{args.kind}",
    )
    spec = compiled.spec
    extension_module = compiled.load_module()
    native_times, native_data = compiled.run(
        initial_states=initial_states,
        nstep=10,
        chunk_size=2,
    )

    payload = {
        "compiled_stub": compiled.debug_summary(),
        "extension_metadata": extension_module.describe_metadata(),
        "native_run": {
            "times_shape": list(native_times.shape),
            "data_shape": list(native_data.shape),
            "times": native_times.tolist(),
            "first_chunk_first_voi": native_data[0, 0, :, 0].tolist(),
        },
        "simulation": {
            "backend_version": spec.backend_version,
            "dt": spec.dt,
            "cache_key": spec.cache_key(),
            "user_source_hint": spec.user_source_hint,
            "source_horizons": spec.source_horizons,
        },
        "monitors": [dataclasses.asdict(monitor) for monitor in spec.monitors],
        "stimuli": [dataclasses.asdict(stimulus) for stimulus in spec.stimuli],
        "subnetworks": [summarize_subnetwork(sn) for sn in spec.subnetworks],
        "inter_projections": [summarize_projection(p) for p in spec.inter_projections],
        "intra_projections": [summarize_projection(p) for p in spec.intra_projections],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
