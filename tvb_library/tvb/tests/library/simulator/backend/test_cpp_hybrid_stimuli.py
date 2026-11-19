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


@pytest.mark.parametrize("eq_name,params,dt_val", [
    ("Linear", {"a": 0.2, "b": 0.5}, DT),
    ("Sinusoid", {"amp": 1.2, "frequency": 4.0}, DT),
    ("Cosine", {"amp": 1.2, "frequency": 4.0}, DT),
    ("PulseTrain", {"T": 2.0, "tau": 0.5, "amp": 1.5, "onset": 1.0}, 0.125),
    ("Alpha", {"onset": 0.5, "alpha": 1.5, "beta": 0.5}, DT),
    ("GeneralizedSigmoid", {"low": 0.0, "high": 2.0, "midpoint": 2.0, "sigma": 0.5}, DT),
    ("Gaussian", {"amp": 1.5, "sigma": 1.0, "midpoint": 2.5, "offset": 0.1}, DT),
    ("DoubleGaussian", {
        "amp_1": 1.0, "sigma_1": 0.5, "midpoint_1": 1.5,
        "amp_2": 0.8, "sigma_2": 0.7, "midpoint_2": 3.5,
    }, DT),
    ("Sigmoid", {"amp": 1.5, "radius": 2.5, "sigma": 0.5, "offset": 0.0}, DT),
])
def test_separable_analytic_equations_parity(eq_name, params, dt_val):
    """All 9 separable analytic C++ stimulus equations match NbHybridBackend."""
    from tvb.datatypes import equations as eq
    from tvb.datatypes.patterns import StimuliRegion
    from tvb.simulator.hybrid.stimulus import Stim

    def build_net():
        def mk_sn(name, nn):
            m = MontbrioPazoRoxin()
            m.configure()
            return Subnetwork(
                name=name, model=m, scheme=HeunDeterministic(dt=dt_val), nnodes=nn,
            ).configure()

        A, B = mk_sn("A", 3), mk_sn("B", 4)
        proj = _projection(A, B)
        proj.dt = dt_val
        eq_cls = getattr(eq, eq_name)
        pat = StimuliRegion.from_weights(
            weight=np.array([0.0, 1.0, 0.5, 0.0]),
            temporal=eq_cls(parameters=params),
        )
        pat.configure_space()
        pat.configure_time(np.arange(0.0, NSTEP * dt_val, dt_val))
        B.stimuli = [
            Stim(target=B, stimulus=pat, target_cvar=np.array([0]), projection_scale=1.5)
        ]
        ns = NetworkSet(subnets=[A, B], projections=[proj])
        ns.configure()
        return ns

    nb_out = NbHybridBackend().compile(build_net(), eager=True).run(NSTEP)
    cpp_out = CppHybridBackend().run_network(build_net(), NSTEP)
    _assert_close(nb_out, cpp_out, f"analytic_{eq_name}")


def test_custom_temporal_equation_discrete_1d_fallback_parity():
    """Custom non-analytic temporal equation lowers to STIM_DISCRETE_1D and matches NbHybrid."""
    from tvb.basic.neotraits.api import Attr, Final
    from tvb.datatypes import equations as eq
    from tvb.datatypes.patterns import StimuliRegion
    from tvb.simulator.hybrid.stimulus import Stim

    class CustomChirp(eq.TemporalApplicableEquation):
        equation = Final(label="Chirp", default="amp * sin(omega * var * var)")
        parameters = Attr(
            field_type=dict,
            label="Parameters",
            default={"amp": 1.4, "omega": 0.5},
        )

    def build_net():
        A = _subnetwork("A", 3)
        B = _subnetwork("B", 4)
        pat = StimuliRegion.from_weights(
            weight=np.array([1.0, 0.25, 0.0, 0.5]),
            temporal=CustomChirp(),
        )
        pat.configure_space()
        pat.configure_time(np.arange(0.0, NSTEP * DT, DT))
        B.stimuli = [
            Stim(target=B, stimulus=pat, target_cvar=np.array([0]), projection_scale=1.0)
        ]
        return NetworkSet(subnets=[A, B], projections=[_projection(A, B)]).configure()

    nb_out = NbHybridBackend().compile(build_net(), eager=True).run(NSTEP)
    cpp_out = CppHybridBackend().run_network(build_net(), NSTEP)
    _assert_close(nb_out, cpp_out, "discrete_1d_fallback")


def test_multi_dt_stimulus_master_grid_parity():
    """Stimulus on slow subnet (dt_B = 2 * dt_A) evaluates on master dt0 grid matching NbHybrid."""
    from tvb.simulator.hybrid.stimulus_utils import sinusoid_stim

    def build_net():
        m_a = MontbrioPazoRoxin(); m_a.configure()
        m_b = MontbrioPazoRoxin(); m_b.configure()
        A = Subnetwork(name="A", model=m_a, scheme=HeunDeterministic(dt=0.1), nnodes=3).configure()
        B = Subnetwork(name="B", model=m_b, scheme=HeunDeterministic(dt=0.2), nnodes=4).configure()
        proj = _projection(A, B)
        proj.dt = 0.1
        B.stimuli = [
            sinusoid_stim(
                B, amplitude=1.3, frequency=3.0, target_node=2,
                target_cvar=0, simulation_length=NSTEP * 0.1,
            )
        ]
        return NetworkSet(subnets=[A, B], projections=[proj]).configure()

    nb_out = NbHybridBackend().compile(build_net(), eager=True).run(NSTEP)
    cpp_out = CppHybridBackend().run_network(build_net(), NSTEP)
    _assert_close(nb_out, cpp_out, "multi_dt_stim")


def test_stimuli_active_in_sweep_match_nb_hybrid():
    """Separable stimuli (Alpha + PulseTrain) work inside sweep() and match NbHybrid."""
    from tvb.datatypes import equations as eq
    from tvb.datatypes.patterns import StimuliRegion
    from tvb.simulator.hybrid.stimulus import Stim

    nstep = 30
    taus = np.array([0.8, 1.0, 1.2, 1.5], dtype=np.float32)

    def build_net():
        A = _subnetwork("A", 3)
        B = _subnetwork("B", 4)
        pat_a = StimuliRegion.from_weights(
            weight=np.array([1.0, 0.0, 0.5]),
            temporal=eq.Alpha(parameters={"onset": 0.5, "alpha": 1.5, "beta": 0.5}),
        )
        pat_a.configure_space()
        pat_a.configure_time(np.arange(0.0, nstep * DT, DT))
        A.stimuli = [Stim(target=A, stimulus=pat_a, target_cvar=np.array([0]))]

        pat_b = StimuliRegion.from_weights(
            weight=np.array([0.0, 1.0, 0.0, 0.5]),
            temporal=eq.PulseTrain(parameters={"T": 2.0, "tau": 0.5, "amp": 1.5, "onset": 0.5}),
        )
        pat_b.configure_space()
        pat_b.configure_time(np.arange(0.0, nstep * DT, DT))
        B.stimuli = [Stim(target=B, stimulus=pat_b, target_cvar=np.array([0]))]
        return NetworkSet(subnets=[A, B], projections=[_projection(A, B)]).configure()

    nb_sw = NbHybridBackend().sweep(build_net(), {"A.tau": taus}, nstep=nstep, backend="cpu")
    cp_sw = CppHybridBackend().sweep(build_net(), {"A.tau": taus}, nstep=nstep, width=4)
    np.testing.assert_allclose(cp_sw.tavg["A"], nb_sw.tavg["A"], rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(cp_sw.tavg["B"], nb_sw.tavg["B"], rtol=1e-3, atol=1e-4)
