# -*- coding: utf-8 -*-
"""Coupling-function parity tests: CppHybridBackend vs NbHybridBackend.

Covers every coupling function/variant nb_hybrid supports, including the
SigmoidalJansenRit legacy form and the PreSigmoidal dynamic form (static and
globalT dynamic threshold).
"""

import numpy as np
import pytest
import scipy.sparse as sp

from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.hybrid.inter_projection import InterProjection
from tvb.simulator.hybrid.coupling import (
    Linear, Scaling, Sigmoidal, SigmoidalJansenRit, Kuramoto as KuramotoCfun,
    Difference, HyperbolicTangent, PreSigmoidal,
)
from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.backend.cpp_hybrid.backend import CppHybridBackend

DT = 0.01
NSTEP = 40


def _subnetwork(name, n_nodes):
    model = MontbrioPazoRoxin()
    model.configure()
    scheme = HeunDeterministic(dt=DT)
    sn = Subnetwork(name=name, model=model, scheme=scheme, nnodes=n_nodes)
    sn.configure()
    return sn


def _projection(src, tgt, cfun, source_cvar=(0,), seed=42):
    n_src, n_tgt = src.nnodes, tgt.nnodes
    rng = np.random.RandomState(seed)
    W = np.zeros((n_tgt, n_src), np.float32)
    for i in range(n_tgt):
        for j in range(n_src):
            if i != j and rng.rand() < 0.5:
                W[i, j] = rng.rand()
    proj = InterProjection(
        source=src, target=tgt,
        source_cvar=np.array(source_cvar, dtype=np.int_),
        target_cvar=np.array([0], dtype=np.int_),
        weights=sp.csr_matrix(W),
        lengths=sp.csr_matrix(W * 10.0),
        cfun=cfun, scale=1.5, cv=1.0, dt=DT,
    )
    return proj


def _run_pair(cfun, source_cvar=(0,)):
    """Build identical NetworkSets and run both backends; return outputs."""
    A = _subnetwork("A", 3)
    B = _subnetwork("B", 4)
    ns = NetworkSet(subnets=[A, B],
                    projections=[_projection(A, B, cfun, source_cvar)])
    ns.configure()
    nb_out = NbHybridBackend().compile(ns, eager=True).run(NSTEP)

    A2 = _subnetwork("A", 3)
    B2 = _subnetwork("B", 4)
    ns2 = NetworkSet(subnets=[A2, B2],
                     projections=[_projection(A2, B2, cfun, source_cvar)])
    ns2.configure()
    cpp_out = CppHybridBackend().run_network(ns2, NSTEP)
    return nb_out, cpp_out


def _assert_close(nb_out, cpp_out):
    assert len(cpp_out) == len(nb_out)
    for (t_nb, d_nb, c_nb), (t_cp, d_cp, c_cp) in zip(nb_out, cpp_out):
        np.testing.assert_allclose(d_cp, d_nb, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("cfun,src_cvars", [
    (Linear(a=np.array([0.7]), b=np.array([0.2])), (0,)),
    (Scaling(a=np.array([1.3])), (0,)),
    (Sigmoidal(a=np.array([0.9]), sigma=np.array([0.5]),
               midpoint=np.array([2.0]), cmin=np.array([-0.2]),
               cmax=np.array([1.1])), (0,)),
    (Difference(a=np.array([1.1])), (0,)),
    (KuramotoCfun(a=np.array([1.0])), (0,)),
    (HyperbolicTangent(a=np.array([1.0]), b=np.array([0.5]),
                       midpoint=np.array([1.0]), sigma=np.array([2.0])), (0,)),
    (SigmoidalJansenRit(), (0, 1)),               # classic (2 src cvars)
    (SigmoidalJansenRit(use_classic=0), (0,)),    # legacy
    (PreSigmoidal(dynamic=False, theta=np.array([2.0])), (0,)),   # static
    (PreSigmoidal(dynamic=True), (0, 1)),            # dynamic
    (PreSigmoidal(dynamic=True, globalT=True), (0, 1)), # dynamic + global threshold
])
def test_coupling_parity(cfun, src_cvars):
    nb_out, cpp_out = _run_pair(cfun, src_cvars)
    _assert_close(nb_out, cpp_out)
