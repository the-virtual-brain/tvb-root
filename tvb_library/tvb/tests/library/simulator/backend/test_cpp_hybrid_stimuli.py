# -*- coding: utf-8 -*-
"""Stimulus parity tests: identical stimulus configurations must produce
identical trajectories in CppHybridBackend and NbHybridBackend.

Covers constant, pulse-train, sinusoid stimuli (single node and all-node
spatial patterns) and the stochastic noise pathway (HeunStochastic with
Additive noise and matched seeds).
"""

import numpy as np
import pytest
import scipy.sparse as sp

from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.integrators import HeunDeterministic, HeunStochastic
from tvb.simulator.noise import Additive
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.hybrid.inter_projection import InterProjection
from tvb.simulator.hybrid.coupling import Linear
from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.backend.cpp_hybrid.backend import CppHybridBackend

DT = 0.1
NSTEP = 50


def _subnetwork(name, n_nodes, stochastic=False, seed=42):
    model = MontbrioPazoRoxin()
    model.configure()
    if stochastic:
        scheme = HeunStochastic(
            dt=DT, noise=Additive(nsig=np.array([1e-4, 1e-4]),
                                  noise_seed=seed))
    else:
        scheme = HeunDeterministic(dt=DT)
    sn = Subnetwork(name=name, model=model, scheme=scheme, nnodes=n_nodes)
    sn.configure()
    return sn


def _projection(src, tgt):
    rng = np.random.RandomState(7)
    W = np.zeros((tgt.nnodes, src.nnodes), np.float32)
    mask = rng.rand(tgt.nnodes, src.nnodes) < 0.6
    W[mask] = rng.rand(mask.sum())
    return InterProjection(
        source=src, target=tgt,
        source_cvar=np.array([0], dtype=np.int_),
        target_cvar=np.array([0], dtype=np.int_),
        weights=sp.csr_matrix(W), lengths=sp.csr_matrix(W * 1.0),
        cfun=Linear(), scale=1.0, cv=1.0, dt=DT)


def _network(stochastic=False):
    A = _subnetwork("A", 3, stochastic)
    B = _subnetwork("B", 4, stochastic)
    ns = NetworkSet(subnets=[A, B], projections=[_projection(A, B)])
    return ns


def _run_pair_factory(make_stims, stochastic=False):
    """make_stims(subnets) -> list of Stim for the freshly built network."""
    ns_nb = _network(stochastic=stochastic)
    stims_nb = make_stims(ns_nb.subnets)
    for sn in ns_nb.subnets:
        sn.stimuli = [s for s in stims_nb if s.target is sn]
    ns_nb.configure()
    nb_out = NbHybridBackend().compile(ns_nb, eager=True).run(NSTEP)

    ns_cpp = _network(stochastic=stochastic)
    stims_cpp = make_stims(ns_cpp.subnets)
    for sn in ns_cpp.subnets:
        sn.stimuli = [s for s in stims_cpp if s.target is sn]
    ns_cpp.configure()
    cpp_out = CppHybridBackend().run_network(ns_cpp, NSTEP)
    return nb_out, cpp_out


def _assert_close(nb_out, cpp_out, tag):
    assert len(cpp_out) == len(nb_out), tag
    for (t_nb, d_nb, c_nb), (t_cp, d_cp, c_cp) in zip(nb_out, cpp_out):
        np.testing.assert_allclose(
            d_cp, d_nb, rtol=1e-3, atol=1e-4,
            err_msg=f"{tag} trajectory mismatch")


def _mk_constant(subnets):
    from tvb.simulator.hybrid.stimulus_utils import constant_stim
    return [constant_stim(subnets[1], amplitude=1.5, target_node=1,
                          target_cvar=0, projection_scale=2.0,
                          simulation_length=DT * NSTEP)]


def _mk_constant_all(subnets):
    from tvb.simulator.hybrid.stimulus_utils import constant_stim
    return [constant_stim(subnets[0], amplitude=0.8, target_node=None,
                          target_cvar=0, projection_scale=1.0,
                          simulation_length=DT * NSTEP)]


def _mk_pulse(subnets):
    from tvb.simulator.hybrid.stimulus_utils import pulse_stim
    return [pulse_stim(subnets[1], amplitude=2.0, onset=1.0, period=2.0,
                       pulse_width=0.5, target_node=0, target_cvar=0,
                       simulation_length=DT * NSTEP)]


def _mk_sinusoid(subnets):
    from tvb.simulator.hybrid.stimulus_utils import sinusoid_stim
    return [sinusoid_stim(subnets[0], amplitude=1.0, frequency=2.0,
                          target_node=2, target_cvar=0,
                          simulation_length=DT * NSTEP)]


@pytest.mark.parametrize("make_stims,stoch", [
    (_mk_constant, False),
    (_mk_constant_all, False),
    (_mk_pulse, False),
    (_mk_sinusoid, False),
])
def test_stimulus_parity(make_stims, stoch):
    tag = make_stims.__name__
    nb_out, cpp_out = _run_pair_factory(make_stims, stoch)
    _assert_close(nb_out, cpp_out, tag)


def test_multi_subnet_stimuli_and_noise():
    """Stimuli on both subnetworks + both subnetworks stochastic."""
    def make(subnets):
        from tvb.simulator.hybrid.stimulus_utils import constant_stim
        return [constant_stim(subnets[0], amplitude=0.5, target_node=0,
                              target_cvar=0, simulation_length=DT * NSTEP),
                constant_stim(subnets[1], amplitude=1.0, target_node=2,
                              target_cvar=0, simulation_length=DT * NSTEP)]
    nb_out, cpp_out = _run_pair_factory(make, stochastic=True)
    _assert_close(nb_out, cpp_out, "multi-stim-noise")


def test_stochastic_noise_parity():
    """Stochastic pathway: same Additive noise seed -> same trajectories."""
    def no_stims(subnets):
        return []
    nb_out, cpp_out = _run_pair_factory(no_stims, stochastic=True)
    _assert_close(nb_out, cpp_out, "stochastic")
