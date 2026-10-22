# -*- coding: utf-8 -*-
"""Parity tests: CppHybridBackend vs NbHybridBackend on real NetworkSets."""

import numpy as np
import pytest
import scipy.sparse as sp

from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.integrators import HeunDeterministic, EulerDeterministic
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.hybrid.inter_projection import InterProjection
from tvb.simulator.hybrid.coupling import Linear
from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.backend.cpp_hybrid.backend import CppHybridBackend

DT = 0.01


def _mpr_subnetwork(name, n_nodes, integrator_cls=HeunDeterministic):
    model = MontbrioPazoRoxin()
    model.configure()
    scheme = integrator_cls(dt=DT)
    sn = Subnetwork(name=name, model=model, scheme=scheme, nnodes=n_nodes)
    sn.configure()
    return sn


def _projection(src, tgt, n_src, n_tgt, density=0.5, seed=42, scale=1.5):
    rng = np.random.RandomState(seed)
    W = np.zeros((n_tgt, n_src), np.float32)  # rows = targets
    for i in range(n_tgt):
        for j in range(n_src):
            if i != j and rng.rand() < density:
                W[i, j] = rng.rand()
    lengths = W * 10.0
    proj = InterProjection(
        source=src,
        target=tgt,
        source_cvar=np.array([0], dtype=np.int_),
        target_cvar=np.array([0], dtype=np.int_),
        weights=sp.csr_matrix(W),
        lengths=sp.csr_matrix(lengths),
        cfun=Linear(),
        scale=scale,
        cv=1.0,
        dt=DT,
    )
    return proj


def _network_set(nA=3, nB=4, coupled=True, integrator_cls=HeunDeterministic):
    A = _mpr_subnetwork("A", nA, integrator_cls)
    B = _mpr_subnetwork("B", nB, integrator_cls)
    projections = []
    if coupled:
        projections.append(_projection(A, B, nA, nB))
    ns = NetworkSet(subnets=[A, B], projections=projections)
    ns.configure()
    return ns


@pytest.mark.parametrize("integrator_cls", [HeunDeterministic, EulerDeterministic])
def test_cpp_matches_numba_single(integrator_cls):
    nstep = 40
    ns = _network_set(coupled=True, integrator_cls=integrator_cls)
    nb = NbHybridBackend().compile(ns, eager=True)
    nb_out = nb.run(nstep)

    ns2 = _network_set(coupled=True, integrator_cls=integrator_cls)
    cpp = CppHybridBackend()
    cpp_out = cpp.run_network(ns2, nstep)

    assert len(cpp_out) == len(nb_out)
    for (t_nb, d_nb, c_nb), (t_cp, d_cp, c_cp) in zip(nb_out, cpp_out):
        assert d_cp.shape == d_nb.shape, (d_cp.shape, d_nb.shape)
        np.testing.assert_allclose(d_cp, d_nb, rtol=1e-3, atol=1e-4)
