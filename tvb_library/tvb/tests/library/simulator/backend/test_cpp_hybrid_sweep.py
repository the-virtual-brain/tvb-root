# -*- coding: utf-8 -*-
"""Sweep tests for the C++ backend: SIMD lanes = sweep members.

Verifies that a batched multithreaded sweep reproduces per-point single
simulations, and that the sweep API matches nb_hybrid's semantics.
"""

import numpy as np
import pytest

from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.hybrid.inter_projection import InterProjection
from tvb.simulator.hybrid.coupling import Linear
from tvb.simulator.backend.cpp_hybrid.backend import CppHybridBackend
from .test_cpp_hybrid_parity import _mpr_subnetwork, _projection, DT


def _ns(coupling_scale=1.5):
    A = _mpr_subnetwork("A", 3)
    B = _mpr_subnetwork("B", 4)
    proj = _projection(A, B, 3, 4, scale=coupling_scale)
    ns = NetworkSet(subnets=[A, B], projections=[proj])
    ns.configure()
    return ns


def test_sweep_matches_single_runs():
    nstep = 30
    values = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0], np.float32)
    backend = CppHybridBackend()

    result = backend.sweep(_ns(), {"coupling_scale": values}, nstep=nstep)
    assert result.tavg["B"].shape[0] == len(values)
    assert result.backend.startswith("cpp-simd")

    # per-point single runs must match the sweep lanes
    for k, v in enumerate(values):
        ns = _ns(coupling_scale=float(v))
        out = backend.run_network(ns, nstep)
        single_B = out[1][1]  # (n_samp, n_voi, n_nodes, 1)
        swept = result.tavg["B"][k]  # (n_samp, n_voi, n_nodes, 1)
        np.testing.assert_allclose(swept, single_B, rtol=1e-4, atol=1e-5)


def test_run_sweep_legacy_descriptor():
    nstep = 30
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0], np.float32)
    backend = CppHybridBackend()
    out = backend.run_sweep(
        _ns(), values, nstep=nstep,
        sweep_descriptor=[{"type": "cfun", "projection": "A_to_B",
                           "param_idx": 0}],
    )
    assert len(out) == len(values)
    # compare against explicit single runs
    for k, v in enumerate(values):
        ns = _ns(coupling_scale=float(v))
        single = backend.run_network(ns, nstep)[1][1]
        np.testing.assert_allclose(out[k][1][1], single, rtol=1e-4, atol=1e-5)


def test_sweep_model_param():
    nstep = 30
    taus = np.array([0.8, 1.0, 1.2, 1.5], np.float32)
    backend = CppHybridBackend()
    result = backend.sweep(_ns(), {"A.tau": taus}, nstep=nstep)
    for k, tv in enumerate(taus):
        ns = _ns()
        ns.subnets[0].model.tau = np.array([float(tv)])
        ns.subnets[0].model.configure()
        out = backend.run_network(ns, nstep)
        np.testing.assert_allclose(
            result.tavg["A"][k], out[0][1], rtol=1e-4, atol=1e-5)
