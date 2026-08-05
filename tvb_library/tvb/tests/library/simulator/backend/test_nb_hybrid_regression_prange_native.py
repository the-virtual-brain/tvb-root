# -*- coding: utf-8 -*-
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package.
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

"""Process-isolated regressions for the native CPU prange sweep failure."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


_CHILD_TIMEOUT_SECONDS = 180
_LIBRARY_ROOT = Path(__file__).resolve().parents[5]
_CHILD_MODULE = (
    "tvb.tests.library.simulator.backend."
    "test_nb_hybrid_regression_prange_native"
)


def _make_sjr_network(cmin=0.0):
    import numpy as np
    import scipy.sparse as sp

    from tvb.simulator.hybrid.coupling import SigmoidalJansenRit
    from tvb.simulator.hybrid.intra_projection import IntraProjection
    from tvb.simulator.hybrid.network import NetworkSet
    from tvb.simulator.hybrid.subnetwork import Subnetwork
    from tvb.simulator.integrators import HeunDeterministic
    from tvb.simulator.models.jansen_rit import JansenRit

    dt = 0.1
    n_nodes = 5
    model = JansenRit()
    model.configure()
    subnet = Subnetwork(
        name="jr", model=model, scheme=HeunDeterministic(dt=dt), nnodes=n_nodes
    )
    rng = np.random.RandomState(1)
    dense_weights = rng.uniform(0.0, 0.5, (n_nodes, n_nodes))
    np.fill_diagonal(dense_weights, 0.0)
    weights = sp.csr_matrix(dense_weights)
    coupling = SigmoidalJansenRit(
        a=np.array([1.0]),
        cmin=np.array([cmin]),
        cmax=np.array([2.0]),
        r=np.array([1.0]),
        midpoint=np.array([0.5]),
    )
    projection = IntraProjection(
        source_cvar=np.array([1], dtype=np.int_),
        target_cvar=np.array([0], dtype=np.int_),
        weights=weights,
        lengths=sp.csr_matrix(weights.shape),
        cv=1.0,
        dt=dt,
        scale=1.0,
        cfun=coupling,
    )
    subnet.projections = [projection]
    subnet.configure()
    network = NetworkSet(subnets=[subnet], projections=[])
    network.configure()
    initial_state = np.zeros((model.nvar, n_nodes, model.number_of_modes))
    return network, initial_state


def _assert_parallel_result(result, requested_threads):
    import numba
    import numpy as np

    from tvb.simulator.backend.nb_hybrid_sweep_cpu import _SWEEP_KERNEL_CACHE

    assert result.backend == "cpu-prange", result.backend
    assert result.merged_tavg is not None
    assert np.isfinite(result.merged_tavg).all()
    assert numba.get_num_threads() == requested_threads
    assert numba.threading_layer() in {"omp", "tbb", "workqueue"}
    assert _SWEEP_KERNEL_CACHE, "the CPU prange sweep kernel was not compiled"
    kernel = next(reversed(_SWEEP_KERNEL_CACHE.values()))
    assert kernel.targetoptions.get("parallel") is True
    assert kernel.signatures, "the CPU prange sweep kernel has no compiled signature"
    print(
        "PARALLEL_BACKEND "
        f"result={result.backend} layer={numba.threading_layer()} "
        f"threads={numba.get_num_threads()} signatures={len(kernel.signatures)}",
        flush=True,
    )


def _run_sjr_native_child(requested_workers, requested_threads):
    import numba
    import numpy as np

    from tvb.simulator.backend.nb_hybrid import NbHybridBackend

    sweep_values = np.array([0.0, 1.0], dtype=np.float32)
    nstep = 20

    # Independent safe oracle: separate non-sweep simulations, each with its
    # coupling object constructed at the requested value.
    oracle = []
    for cmin in sweep_values:
        network, initial_state = _make_sjr_network(float(cmin))
        direct = NbHybridBackend().run_network(
            network,
            nstep=nstep,
            chunk_size=1,
            initial_states=[initial_state],
        )
        oracle.append(direct[0][1])
    oracle = np.stack(oracle)
    assert np.isfinite(oracle).all()

    network, initial_state = _make_sjr_network()
    print(
        "START_SJR_PRANGE "
        f"boundscheck={os.environ['NUMBA_BOUNDSCHECK']} "
        f"workers={requested_workers} configured_threads={numba.get_num_threads()}",
        flush=True,
    )
    result = NbHybridBackend().sweep(
        network,
        params={"jr.intra.cmin": sweep_values},
        nstep=nstep,
        backend="cpu",
        n_workers=requested_workers,
        initial_states=[initial_state],
    )
    _assert_parallel_result(result, requested_threads)
    max_abs_error = float(np.max(np.abs(result.merged_tavg - oracle)))
    print(
        f"SJR_RESULT shape={result.merged_tavg.shape} "
        f"oracle_shape={oracle.shape} max_abs_error={max_abs_error:.9g}",
        flush=True,
    )
    np.testing.assert_allclose(result.merged_tavg, oracle, rtol=1e-5, atol=1e-5)


def _run_selection_child(requested_workers, requested_threads):
    import numpy as np
    import scipy.sparse as sp

    from tvb.simulator.backend.nb_hybrid import NbHybridBackend
    from tvb.simulator.hybrid.coupling import Linear
    from tvb.simulator.hybrid.intra_projection import IntraProjection
    from tvb.simulator.hybrid.network import NetworkSet
    from tvb.simulator.hybrid.subnetwork import Subnetwork
    from tvb.simulator.integrators import EulerDeterministic
    from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin

    model = MontbrioPazoRoxin()
    model.configure()
    subnet = Subnetwork(
        name="mpr", model=model, scheme=EulerDeterministic(dt=0.01), nnodes=2
    )
    weights = sp.csr_matrix(np.array([[0.0, 0.1], [0.1, 0.0]]))
    subnet.projections = [
        IntraProjection(
            source_cvar=np.array([0], dtype=np.int_),
            target_cvar=np.array([0], dtype=np.int_),
            weights=weights,
            lengths=sp.csr_matrix(weights.shape),
            cv=1.0,
            dt=0.01,
            scale=1.0,
            cfun=Linear(a=np.array([1.0])),
        )
    ]
    subnet.configure()
    network = NetworkSet(subnets=[subnet], projections=[])
    network.configure()
    result = NbHybridBackend().sweep(
        network,
        params={"mpr.intra.a": np.array([0.1, 0.2], dtype=np.float32)},
        nstep=2,
        backend="cpu",
        n_workers=requested_workers,
    )
    _assert_parallel_result(result, requested_threads)


def _child_main():
    import faulthandler
    import platform

    faulthandler.enable(all_threads=True)
    action = os.environ["TVB_PRANGE_CHILD_ACTION"]
    requested_workers = int(os.environ["TVB_PRANGE_REQUESTED_WORKERS"])
    requested_threads = int(os.environ["NUMBA_NUM_THREADS"])
    print(
        "CHILD_CONFIG "
        f"action={action} pid={os.getpid()} python={platform.python_version()} "
        f"boundscheck={os.environ['NUMBA_BOUNDSCHECK']} "
        f"workers={requested_workers} threads={requested_threads} "
        f"tmpdir={os.environ['TMPDIR']} cache={os.environ['NUMBA_CACHE_DIR']}",
        flush=True,
    )
    if action == "sjr-native":
        _run_sjr_native_child(requested_workers, requested_threads)
    elif action == "selection":
        _run_selection_child(requested_workers, requested_threads)
    else:
        raise ValueError(f"unknown child action: {action}")


def _child_diagnostics(completed, description):
    return (
        f"{description}\n"
        f"returncode: {completed.returncode}\n"
        f"--- child stdout ---\n{completed.stdout or '<empty>'}\n"
        f"--- child stderr ---\n{completed.stderr or '<empty>'}"
    )


def _run_isolated_child(tmp_path, action, boundscheck, workers, threads):
    case_dir = tmp_path / f"{action}-bounds{boundscheck}-workers{workers}-threads{threads}"
    temp_dir = case_dir / "tmp"
    cache_dir = case_dir / "numba-cache"
    mpl_dir = case_dir / "matplotlib"
    for directory in (temp_dir, cache_dir, mpl_dir):
        directory.mkdir(parents=True, exist_ok=False)

    env = os.environ.copy()
    env.update(
        {
            "MPLCONFIGDIR": str(mpl_dir),
            "NUMBA_BOUNDSCHECK": str(boundscheck),
            "NUMBA_CACHE_DIR": str(cache_dir),
            "NUMBA_NUM_THREADS": str(threads),
            "PYTHONFAULTHANDLER": "1",
            "TMP": str(temp_dir),
            "TMPDIR": str(temp_dir),
            "TEMP": str(temp_dir),
            "TVB_PRANGE_CHILD_ACTION": action,
            "TVB_PRANGE_REQUESTED_WORKERS": str(workers),
            "XDG_CACHE_HOME": str(case_dir / "xdg-cache"),
        }
    )
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(_LIBRARY_ROOT) + (
        os.pathsep + old_pythonpath if old_pythonpath else ""
    )
    command = [
        sys.executable,
        "-c",
        f"from {_CHILD_MODULE} import _child_main; _child_main()",
    ]
    description = (
        f"child action={action}, NUMBA_BOUNDSCHECK={boundscheck}, "
        f"n_workers={workers}, NUMBA_NUM_THREADS={threads}"
    )
    try:
        completed = subprocess.run(
            command,
            cwd=_LIBRARY_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=_CHILD_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
        return (
            f"{description} exceeded hard timeout of {_CHILD_TIMEOUT_SECONDS}s\n"
            f"--- child stdout ---\n{stdout or '<empty>'}\n"
            f"--- child stderr ---\n{stderr or '<empty>'}"
        )
    if completed.returncode != 0:
        return _child_diagnostics(completed, description)
    return None


@pytest.mark.parametrize(
    "boundscheck,threads",
    [
        pytest.param(0, 2, id="unchecked-2-threads"),
        pytest.param(1, 2, marks=pytest.mark.slow, id="checked-2-threads"),
        pytest.param(0, 4, marks=pytest.mark.slow, id="unchecked-4-threads"),
        pytest.param(1, 4, marks=pytest.mark.slow, id="checked-4-threads"),
    ],
)
def test_sjr_native_abort_prange_sweep_bounds_and_thread_matrix(
    tmp_path, boundscheck, threads
):
    """The native-abort reproducer must survive and match direct-run oracles."""
    failure = _run_isolated_child(
        tmp_path,
        action="sjr-native",
        boundscheck=boundscheck,
        workers=threads,
        threads=threads,
    )
    assert failure is None, failure


def test_requested_workers_and_threads_reach_real_parallel_backend(tmp_path):
    """A requested multi-worker sweep must compile and run the prange backend."""
    failure = _run_isolated_child(
        tmp_path,
        action="selection",
        boundscheck=1,
        workers=2,
        threads=2,
    )
    assert failure is None, failure
