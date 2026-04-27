# -*- coding: utf-8 -*-
"""
Stimulus equation parity: Python vs Numba backend across all temporal equations.

For every ``TemporalApplicableEquation`` subclass, this module constructs a
single-subnet simulation with a ``StimuliRegion`` stimulus driven by that
equation, runs it through both the pure-Python ``NetworkSet.step`` loop and
the ``NbHybridBackend``, and asserts that the final states agree within
floating-point tolerance.

If an equation's evaluation diverges between the two backends (e.g. due to
float32 truncation in the pre-computed Numba stimulus array), the
corresponding test will fail, surface the equation class, and report the
maximum absolute / relative error.

Equation classes tested
-----------------------
- :class:`~tvb.datatypes.equations.Linear`
- :class:`~tvb.datatypes.equations.Sinusoid`
- :class:`~tvb.datatypes.equations.Cosine`
- :class:`~tvb.datatypes.equations.PulseTrain`
- :class:`~tvb.datatypes.equations.Alpha`
- :class:`~tvb.datatypes.equations.GeneralizedSigmoid`
- :class:`~tvb.datatypes.equations.Gaussian` (temporal, via finite support)
- :class:`~tvb.datatypes.equations.Sigmoid` (temporal, via finite support)
"""

import unittest

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_allclose

from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.models.oscillator import Generic2dOscillator
from tvb.simulator.integrators import (
    HeunDeterministic, EulerDeterministic,
    HeunStochastic, EulerStochastic,
)
from tvb.simulator.noise import Additive
from tvb.simulator.hybrid import Subnetwork, NetworkSet
from tvb.simulator.hybrid.stimulus import Stim
from tvb.simulator.hybrid.inter_projection import InterProjection
from tvb.simulator.hybrid.coupling import Linear as LinearCoupling
from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.datatypes.patterns import StimuliRegion
from tvb.datatypes import equations as eqs
from tvb.datatypes.connectivity import Connectivity

DT = 0.1


# ── helpers ───────────────────────────────────────────────────────────────

def _make_connectivity(n_nodes):
    """Minimal Connectivity for StimuliRegion."""
    conn = Connectivity(
        centres=np.zeros((n_nodes, 3)),
        weights=np.zeros((n_nodes, n_nodes)),
        tract_lengths=np.zeros((n_nodes, n_nodes)),
        region_labels=np.array([str(i) for i in range(n_nodes)]),
        speed=np.array([1.0]),
    )
    conn.configure()
    return conn


def _make_subnet(name, model, n_nodes, integrator_cls=HeunDeterministic, dt=DT, **kw):
    scheme = integrator_cls(dt=dt, **kw)
    sn = Subnetwork(name=name, model=model, scheme=scheme, nnodes=n_nodes)
    return sn


def _configure_all(*objs):
    for o in objs:
        if hasattr(o, "configure"):
            o.configure()
    return objs


def _seeded_ic(model, n_nodes, n_modes, seed=42):
    rng = np.random.RandomState(seed)
    ic = np.zeros((model.nvar, n_nodes, n_modes), dtype=np.float64)
    for k, sv in enumerate(model.state_variables):
        lo, hi = model.state_variable_range[sv]
        ic[k] = rng.uniform(lo, hi, (n_nodes, n_modes))
    return ic


def _build_stim(subnet, temporal_eq, weight, nstep, target_cvar=None,
                projection_scale=1.0):
    """Wire up a Stim from a temporal equation and spatial weight vector."""
    if target_cvar is None:
        target_cvar = np.array([0], dtype=np.int_)
    conn = _make_connectivity(subnet.nnodes)
    stim_pattern = StimuliRegion(
        temporal=temporal_eq,
        connectivity=conn,
        weight=weight,
    )
    stim = Stim(
        target=subnet,
        stimulus=stim_pattern,
        target_cvar=target_cvar,
        projection_scale=projection_scale,
    )
    stim.configure(simulation_length=nstep * DT)
    return stim


def _run_python(ns, nstep, ics):
    """Run pure-Python loop; return final states (float64)."""
    x = ns.zero_states(initial_states=[ic.copy() for ic in ics])
    ns.init_projection_buffers(x)
    for step in range(1, nstep + 1):
        x = ns.step(step, x)
    return [np.asarray(xi, dtype=np.float64) for xi in x]


def _run_numba(ns, nstep, ics):
    """Run Numba backend; return final states (float32)."""
    be = NbHybridBackend()
    compiled = be.compile(ns)
    _outputs, snapshot = compiled.run(
        nstep, chunk_size=1,
        initial_states=[ic.copy() for ic in ics],
        return_snapshot=True,
    )
    return [s.astype(np.float32) for s in snapshot["states"]]


# ══════════════════════════════════════════════════════════════════════════
# Parameterised equation parity
# ══════════════════════════════════════════════════════════════════════════

# Each entry: (test_id, equation_factory, integrator, nstep, rtol, atol)
#
# The ``equation_factory`` must return a *configured* equation instance when
# called with no arguments.

_EQUATION_CASES = [
    # ── Linear ──────────────────────────────────────────────────────
    (
        "linear_constant",
        lambda: _linear_eq(a=0.0, b=0.05),
        HeunDeterministic, 100, 1e-5, 1e-6,
    ),
    (
        "linear_ramp",
        lambda: _linear_eq(a=0.01, b=0.0),
        HeunDeterministic, 100, 1e-5, 1e-6,
    ),
    # ── Sinusoid ────────────────────────────────────────────────────
    (
        "sinusoid_slow",
        lambda: _sinusoid_eq(amp=0.05, frequency=0.01),
        HeunDeterministic, 150, 1e-3, 1e-4,
    ),
    (
        "sinusoid_fast",
        lambda: _sinusoid_eq(amp=0.03, frequency=0.1),
        EulerDeterministic, 150, 1e-3, 1e-4,
    ),
    # ── Cosine ──────────────────────────────────────────────────────
    (
        "cosine_slow",
        lambda: _cosine_eq(amp=0.05, frequency=0.01),
        HeunDeterministic, 150, 1e-3, 1e-4,
    ),
    (
        "cosine_fast",
        lambda: _cosine_eq(amp=0.03, frequency=0.1),
        EulerDeterministic, 150, 1e-3, 1e-4,
    ),
    # ── PulseTrain ──────────────────────────────────────────────────
    (
        "pulsetrain_narrow",
        lambda: _pulsetrain_eq(onset=2.0, T=10.0, tau=2.0, amp=0.05),
        HeunDeterministic, 150, 1e-4, 1e-5,
    ),
    (
        "pulsetrain_wide",
        lambda: _pulsetrain_eq(onset=1.0, T=5.0, tau=3.0, amp=0.04),
        EulerDeterministic, 150, 1e-4, 1e-5,
    ),
    # ── Alpha ───────────────────────────────────────────────────────
    (
        "alpha_default",
        lambda: _alpha_eq(onset=0.5, alpha=13.0, beta=42.0),
        HeunDeterministic, 100, 1e-4, 1e-5,
    ),
    (
        "alpha_slow",
        lambda: _alpha_eq(onset=1.0, alpha=2.0, beta=5.0),
        EulerDeterministic, 150, 1e-4, 1e-5,
    ),
    # ── GeneralizedSigmoid ──────────────────────────────────────────
    (
        "gen_sigmoid_default",
        lambda: _gen_sigmoid_eq(low=0.0, high=0.05, midpoint=5.0, sigma=2.0),
        HeunDeterministic, 100, 1e-3, 1e-4,
    ),
    (
        "gen_sigmoid_steep",
        lambda: _gen_sigmoid_eq(low=0.0, high=0.1, midpoint=3.0, sigma=0.5),
        EulerDeterministic, 100, 1e-3, 1e-4,
    ),
    # ── Gaussian (used temporally via FiniteSupportEquation) ─────────
    (
        "gaussian_temporal",
        lambda: _gaussian_eq(amp=0.05, sigma=2.0, midpoint=5.0, offset=0.0),
        HeunDeterministic, 100, 1e-3, 1e-4,
    ),
    # ── Sigmoid (used temporally via FiniteSupportEquation) ──────────
    (
        "sigmoid_temporal",
        lambda: _sigmoid_eq(amp=0.05, radius=5.0, sigma=1.0, offset=0.0),
        EulerDeterministic, 100, 1e-3, 1e-4,
    ),
    # ── DoubleGaussian (used temporally via FiniteSupportEquation) ───
    (
        "double_gaussian_temporal",
        lambda: _double_gaussian_eq(
            amp_1=0.05, sigma_1=3.0, midpoint_1=5.0,
            amp_2=0.03, sigma_2=1.5, midpoint_2=5.0,
        ),
        HeunDeterministic, 100, 1e-3, 1e-4,
    ),
]


# ── equation factories ───────────────────────────────────────────────────

def _linear_eq(a, b):
    eq = eqs.Linear()
    eq.parameters["a"] = float(a)
    eq.parameters["b"] = float(b)
    return eq


def _sinusoid_eq(amp, frequency):
    eq = eqs.Sinusoid()
    eq.parameters["amp"] = float(amp)
    eq.parameters["frequency"] = float(frequency)
    return eq


def _cosine_eq(amp, frequency):
    eq = eqs.Cosine()
    eq.parameters["amp"] = float(amp)
    eq.parameters["frequency"] = float(frequency)
    return eq


def _pulsetrain_eq(onset, T, tau, amp):
    eq = eqs.PulseTrain()
    eq.parameters["onset"] = float(onset)
    eq.parameters["T"] = float(T)
    eq.parameters["tau"] = float(tau)
    eq.parameters["amp"] = float(amp)
    return eq


def _alpha_eq(onset, alpha, beta):
    eq = eqs.Alpha()
    eq.parameters["onset"] = float(onset)
    eq.parameters["alpha"] = float(alpha)
    eq.parameters["beta"] = float(beta)
    return eq


def _gen_sigmoid_eq(low, high, midpoint, sigma):
    eq = eqs.GeneralizedSigmoid()
    eq.parameters["low"] = float(low)
    eq.parameters["high"] = float(high)
    eq.parameters["midpoint"] = float(midpoint)
    eq.parameters["sigma"] = float(sigma)
    return eq


def _gaussian_eq(amp, sigma, midpoint, offset=0.0):
    eq = eqs.Gaussian()
    eq.parameters["amp"] = float(amp)
    eq.parameters["sigma"] = float(sigma)
    eq.parameters["midpoint"] = float(midpoint)
    eq.parameters["offset"] = float(offset)
    return eq


def _sigmoid_eq(amp, radius, sigma, offset=0.0):
    eq = eqs.Sigmoid()
    eq.parameters["amp"] = float(amp)
    eq.parameters["radius"] = float(radius)
    eq.parameters["sigma"] = float(sigma)
    eq.parameters["offset"] = float(offset)
    return eq


def _double_gaussian_eq(amp_1, sigma_1, midpoint_1, amp_2, sigma_2, midpoint_2):
    eq = eqs.DoubleGaussian()
    eq.parameters["amp_1"] = float(amp_1)
    eq.parameters["sigma_1"] = float(sigma_1)
    eq.parameters["midpoint_1"] = float(midpoint_1)
    eq.parameters["amp_2"] = float(amp_2)
    eq.parameters["sigma_2"] = float(sigma_2)
    eq.parameters["midpoint_2"] = float(midpoint_2)
    return eq


# ══════════════════════════════════════════════════════════════════════════
# Test class
# ══════════════════════════════════════════════════════════════════════════


class TestStimulusEquationParity(unittest.TestCase):
    """Stimulus equation parity: Python vs Numba for each temporal equation.

    Each sub-test builds a single-subnet MPR network with a stimulus driven
    by a specific equation, runs both backends, and checks the final state.
    """

    N = 6

    def _run_equation_case(self, test_id, eq_factory, integ_cls, nstep,
                           rtol, atol, target_node=0, seed=5000):
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, self.N, integ_cls)
        _configure_all(m, sn)

        weight = np.zeros(self.N)
        weight[target_node] = 1.0
        temporal_eq = eq_factory()
        stim = _build_stim(sn, temporal_eq, weight, nstep)

        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[stim])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=seed)
        py = _run_python(ns, nstep, [ic])
        nb = _run_numba(ns, nstep, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=rtol, atol=atol,
            err_msg=f"{test_id}: diverged between Python and Numba",
        )


# ── generate one test method per equation case ────────────────────────

def _make_equation_test(test_id, eq_factory, integ_cls, nstep, rtol, atol):
    """Create a test method for a single equation case."""

    def test_method(self):
        self._run_equation_case(test_id, eq_factory, integ_cls, nstep,
                                rtol, atol)

    test_method.__doc__ = f"Equation parity: {test_id}"
    return test_method


for _i, (_tid, _eq_fact, _integ, _nstep, _rtol, _atol) in enumerate(
    _EQUATION_CASES
):
    _method = _make_equation_test(
        _tid, _eq_fact, _integ, _nstep, _rtol, _atol,
    )
    _method.__name__ = f"test_{_tid}"
    setattr(TestStimulusEquationParity, f"test_{_tid}", _method)
del _i, _tid, _eq_fact, _integ, _nstep, _rtol, _atol, _method


# ══════════════════════════════════════════════════════════════════════════
# Additional targeted tests
# ══════════════════════════════════════════════════════════════════════════


class TestStimulusEquationEdgeCases(unittest.TestCase):
    """Edge-case stimulus equation parity tests."""

    N = 5
    NSTEP = 100

    def test_linear_zero_is_identity(self):
        """Linear(a=0, b=0) should produce zero coupling → match no-stim baseline."""
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn)

        # No stimulus baseline
        ns_base = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        ns_base.configure()
        ic = _seeded_ic(m, self.N, 1, seed=6001)
        py_base = _run_python(ns_base, self.NSTEP, [ic])
        nb_base = _run_numba(ns_base, self.NSTEP, [ic])

        # Zero stimulus
        sn2 = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn2)
        weight = np.ones(self.N)
        stim = _build_stim(sn2, _linear_eq(0.0, 0.0), weight, self.NSTEP)
        ns_stim = NetworkSet(subnets=[sn2], projections=[], stimuli=[stim])
        ns_stim.configure()

        py_stim = _run_python(ns_stim, self.NSTEP, [ic])
        nb_stim = _run_numba(ns_stim, self.NSTEP, [ic])

        assert_allclose(
            py_stim[0], py_base[0],
            rtol=1e-12, atol=1e-14,
            err_msg="Zero Linear Python path differs from no-stim baseline",
        )
        assert_allclose(
            nb_stim[0].astype(np.float64), nb_base[0].astype(np.float64),
            rtol=1e-5, atol=1e-6,
            err_msg="Zero Linear Numba path differs from no-stim baseline",
        )

    def test_pulsetrain_before_onset_is_zero(self):
        """PulseTrain before onset should contribute zero coupling."""
        m = MontbrioPazoRoxin()
        nstep = 30  # 3.0 ms, all before onset=50 ms
        sn = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn)

        # Baseline no stim
        ns_base = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        ns_base.configure()
        ic = _seeded_ic(m, self.N, 1, seed=6002)
        py_base = _run_python(ns_base, nstep, [ic])
        nb_base = _run_numba(ns_base, nstep, [ic])

        # PulseTrain with onset well beyond nstep*DT = 3.0 ms
        sn2 = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn2)
        weight = np.ones(self.N)
        stim = _build_stim(sn2, _pulsetrain_eq(onset=50.0, T=100.0, tau=10.0,
                                                amp=1.0),
                           weight, nstep)
        ns_stim = NetworkSet(subnets=[sn2], projections=[], stimuli=[stim])
        ns_stim.configure()

        py_stim = _run_python(ns_stim, nstep, [ic])
        nb_stim = _run_numba(ns_stim, nstep, [ic])

        assert_allclose(
            py_stim[0], py_base[0],
            rtol=1e-12, atol=1e-14,
            err_msg="Pre-onset PulseTrain Python differs from baseline",
        )
        assert_allclose(
            nb_stim[0].astype(np.float64), nb_base[0].astype(np.float64),
            rtol=1e-5, atol=1e-6,
            err_msg="Pre-onset PulseTrain Numba differs from baseline",
        )

    def test_all_nodes_stimulated_parity(self):
        """Stimulus on all nodes (uniform weight) matches between backends."""
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn)
        weight = np.ones(self.N)
        stim = _build_stim(sn, _sinusoid_eq(amp=0.04, frequency=0.05),
                           weight, self.NSTEP)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[stim])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=6003)
        py = _run_python(ns, self.NSTEP, [ic])
        nb = _run_numba(ns, self.NSTEP, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-3, atol=1e-4,
            err_msg="All-nodes Sinusoid parity diverged",
        )

    def test_multi_cvar_stimulus_parity(self):
        """Stimulus targeting both coupling variables (r and V)."""
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn)
        weight = np.zeros(self.N)
        weight[0] = 1.0
        stim = _build_stim(
            sn, _linear_eq(0.0, 0.03), weight, self.NSTEP,
            target_cvar=np.array([0, 1], dtype=np.int_),
        )
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[stim])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=6004)
        py = _run_python(ns, self.NSTEP, [ic])
        nb = _run_numba(ns, self.NSTEP, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-5, atol=1e-6,
            err_msg="Multi-cvar Linear stimulus diverged",
        )

    def test_projection_scale_large_parity(self):
        """Large projection_scale (5.0) applied consistently in both backends."""
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn)
        weight = np.zeros(self.N)
        weight[0] = 1.0
        stim = _build_stim(
            sn, _linear_eq(0.0, 0.01), weight, self.NSTEP,
            projection_scale=5.0,
        )
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[stim])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=6005)
        py = _run_python(ns, self.NSTEP, [ic])
        nb = _run_numba(ns, self.NSTEP, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-4, atol=1e-5,
            err_msg="Large projection_scale stimulus diverged",
        )

    def test_cosine_vs_sinusoid_orthogonal(self):
        """Cosine and Sinusoid at same frequency produce different results.

        This sanity-checks that the equation evaluation is actually using the
        correct waveform — if both gave the same result, something would be
        wrong with equation dispatch.
        """
        m = MontbrioPazoRoxin()
        nstep = 80

        # Sinusoid run
        sn_sin = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn_sin)
        weight = np.zeros(self.N)
        weight[0] = 1.0
        stim_sin = _build_stim(
            sn_sin, _sinusoid_eq(amp=0.1, frequency=0.05), weight, nstep,
        )
        ns_sin = NetworkSet(subnets=[sn_sin], projections=[], stimuli=[stim_sin])
        ns_sin.configure()

        ic = _seeded_ic(m, self.N, 1, seed=6006)
        py_sin = _run_python(ns_sin, nstep, [ic])[0]

        # Cosine run
        sn_cos = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn_cos)
        stim_cos = _build_stim(
            sn_cos, _cosine_eq(amp=0.1, frequency=0.05), weight, nstep,
        )
        ns_cos = NetworkSet(subnets=[sn_cos], projections=[], stimuli=[stim_cos])
        ns_cos.configure()

        py_cos = _run_python(ns_cos, nstep, [ic])[0]

        # They must differ (sin and cos are phase-shifted)
        self.assertFalse(
            np.allclose(py_sin, py_cos, rtol=1e-5, atol=1e-6),
            "Sinusoid and Cosine should produce different states at same frequency",
        )


# ══════════════════════════════════════════════════════════════════════════
# Coupling-level parity: directly compare Stim.get_coupling() values
# ══════════════════════════════════════════════════════════════════════════


class TestStimulusCouplingValues(unittest.TestCase):
    """Compare Stim.get_coupling() step-by-step against Numba pre-computed array.

    The Numba backend pre-computes a ``(n_cvar, n_nodes, n_modes, nstep)``
    float32 array by calling ``Stim.get_coupling(step)`` for each step.
    This test class directly compares those coupling values to ensure the
    pre-computation pipeline is numerically faithful.
    """

    N = 4
    NSTEP = 50

    def _collect_python_coupling(self, stim, nstep):
        """Collect Stim.get_coupling() output at every step (float64)."""
        couplings = []
        for step in range(1, nstep + 1):
            c = np.asarray(stim.get_coupling(step), dtype=np.float64)
            couplings.append(c)
        # Shape: (nstep, n_cvar, n_nodes, n_modes)
        return np.stack(couplings, axis=0)

    def _collect_numba_coupling(self, ns, nstep, ics):
        """Run Numba backend and return the raw ctavg (coupling temporal average).

        ctavg has shape (nstep, n_cvar, n_nodes, n_modes) at chunk_size=1.
        With a single stim and no projections, ctavg should equal the stimulus
        coupling per step.
        """
        be = NbHybridBackend()
        results = be.run_network(
            ns, nstep=nstep, chunk_size=1,
            initial_states=[ic.copy() for ic in ics],
        )
        _times, _data, ctavg = results[0]
        return ctavg.astype(np.float64)

    def test_linear_coupling_values_match(self):
        """Linear equation coupling values match between Python and Numba."""
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn)
        weight = np.zeros(self.N)
        weight[0] = 1.0
        stim = _build_stim(sn, _linear_eq(0.0, 0.05), weight, self.NSTEP)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[stim])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=7001)
        py_c = self._collect_python_coupling(stim, self.NSTEP)
        nb_c = self._collect_numba_coupling(ns, self.NSTEP, [ic])

        # ctavg shape: (nstep, n_cvar, n_nodes, 1) where n_cvar=2 for MPR.
        # Stim targets cvar 0, so compare only that slice.
        assert_allclose(
            nb_c[:, 0:1, :, :], py_c,
            rtol=1e-5, atol=1e-6,
            err_msg="Linear coupling values diverged",
        )

    def test_sinusoid_coupling_values_match(self):
        """Sinusoid equation coupling values match."""
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn)
        weight = np.zeros(self.N)
        weight[0] = 1.0
        stim = _build_stim(sn, _sinusoid_eq(amp=0.05, frequency=0.1),
                           weight, self.NSTEP)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[stim])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=7002)
        py_c = self._collect_python_coupling(stim, self.NSTEP)
        nb_c = self._collect_numba_coupling(ns, self.NSTEP, [ic])

        assert_allclose(
            nb_c[:, 0:1, :, :], py_c,
            rtol=1e-5, atol=1e-6,
            err_msg="Sinusoid coupling values diverged",
        )

    def test_pulsetrain_coupling_values_match(self):
        """PulseTrain coupling values match, including zero regions."""
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn)
        weight = np.zeros(self.N)
        weight[0] = 1.0
        stim = _build_stim(sn, _pulsetrain_eq(onset=1.0, T=5.0, tau=1.0, amp=0.1),
                           weight, self.NSTEP)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[stim])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=7003)
        py_c = self._collect_python_coupling(stim, self.NSTEP)
        nb_c = self._collect_numba_coupling(ns, self.NSTEP, [ic])

        assert_allclose(
            nb_c[:, 0:1, :, :], py_c,
            rtol=1e-5, atol=1e-6,
            err_msg="PulseTrain coupling values diverged",
        )

    def test_cosine_coupling_values_match(self):
        """Cosine equation coupling values match."""
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn)
        weight = np.zeros(self.N)
        weight[0] = 1.0
        stim = _build_stim(sn, _cosine_eq(amp=0.05, frequency=0.1),
                           weight, self.NSTEP)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[stim])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=7004)
        py_c = self._collect_python_coupling(stim, self.NSTEP)
        nb_c = self._collect_numba_coupling(ns, self.NSTEP, [ic])

        assert_allclose(
            nb_c[:, 0:1, :, :], py_c,
            rtol=1e-5, atol=1e-6,
            err_msg="Cosine coupling values diverged",
        )

    def test_alpha_coupling_values_match(self):
        """Alpha equation coupling values match."""
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn)
        weight = np.zeros(self.N)
        weight[0] = 1.0
        stim = _build_stim(sn, _alpha_eq(onset=0.5, alpha=13.0, beta=42.0),
                           weight, self.NSTEP)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[stim])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=7005)
        py_c = self._collect_python_coupling(stim, self.NSTEP)
        nb_c = self._collect_numba_coupling(ns, self.NSTEP, [ic])

        assert_allclose(
            nb_c[:, 0:1, :, :], py_c,
            rtol=1e-5, atol=1e-6,
            err_msg="Alpha coupling values diverged",
        )


# ══════════════════════════════════════════════════════════════════════════
# Per-step trajectory parity with stimulus
# ══════════════════════════════════════════════════════════════════════════


class TestStimulusPerStepParity(unittest.TestCase):
    """Per-step observed trajectory parity with stimulus."""

    N = 5

    def _run_python_observed(self, ns, nstep, ics):
        """Run Python path; return (nstep, nvoi, nnodes, 1) per subnet."""
        x = ns.zero_states(initial_states=[ic.copy() for ic in ics])
        ns.init_projection_buffers(x)
        outputs = []
        for step in range(1, nstep + 1):
            x = ns.step(step, x)
            per_sn = []
            for sn, nx in zip(ns.subnets, x):
                obs = sn.model.observe(np.asarray(nx)).sum(axis=-1)[..., None]
                per_sn.append(obs)
            outputs.append(per_sn)
        result = []
        for si in range(len(ns.subnets)):
            arr = np.stack([outputs[t][si] for t in range(nstep)], axis=0)
            result.append(arr)
        return result

    def test_sinusoid_per_step_trajectory(self):
        """Sinusoid stimulus: full per-step observed trajectory parity."""
        m = MontbrioPazoRoxin()
        nstep = 100
        sn = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn)
        weight = np.zeros(self.N)
        weight[0] = 1.0
        stim = _build_stim(sn, _sinusoid_eq(amp=0.04, frequency=0.05),
                           weight, nstep)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[stim])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=8001)
        py_obs = self._run_python_observed(ns, nstep, [ic])

        be = NbHybridBackend()
        results = be.run_network(
            ns, nstep=nstep, chunk_size=1,
            initial_states=[ic.copy()],
        )
        _times, data, _ctavg = results[0]

        assert_allclose(
            data.astype(np.float64), py_obs[0],
            rtol=1e-3, atol=1e-4,
            err_msg="Sinusoid per-step trajectory diverged",
        )

    def test_pulsetrain_per_step_trajectory(self):
        """PulseTrain stimulus: per-step observed trajectory parity."""
        m = MontbrioPazoRoxin()
        nstep = 100
        sn = _make_subnet("ctx", m, self.N, EulerDeterministic)
        _configure_all(m, sn)
        weight = np.zeros(self.N)
        weight[0] = 1.0
        stim = _build_stim(sn, _pulsetrain_eq(onset=2.0, T=8.0, tau=2.0, amp=0.05),
                           weight, nstep)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[stim])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=8002)
        py_obs = self._run_python_observed(ns, nstep, [ic])

        be = NbHybridBackend()
        results = be.run_network(
            ns, nstep=nstep, chunk_size=1,
            initial_states=[ic.copy()],
        )
        _times, data, _ctavg = results[0]

        assert_allclose(
            data.astype(np.float64), py_obs[0],
            rtol=1e-3, atol=1e-4,
            err_msg="PulseTrain per-step trajectory diverged",
        )

    def test_linear_per_step_trajectory(self):
        """Constant Linear stimulus: per-step observed trajectory parity."""
        m = MontbrioPazoRoxin()
        nstep = 80
        sn = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn)
        weight = np.zeros(self.N)
        weight[0] = 1.0
        stim = _build_stim(sn, _linear_eq(0.0, 0.04), weight, nstep)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[stim])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=8003)
        py_obs = self._run_python_observed(ns, nstep, [ic])

        be = NbHybridBackend()
        results = be.run_network(
            ns, nstep=nstep, chunk_size=1,
            initial_states=[ic.copy()],
        )
        _times, data, _ctavg = results[0]

        assert_allclose(
            data.astype(np.float64), py_obs[0],
            rtol=1e-3, atol=1e-4,
            err_msg="Linear per-step trajectory diverged",
        )


# ══════════════════════════════════════════════════════════════════════════
# Stochastic + stimulus parity
# ══════════════════════════════════════════════════════════════════════════


class TestStimulusStochasticParity(unittest.TestCase):
    """Stochastic integrator + stimulus: matched noise seed parity."""

    N = 4
    NSTEP = 50
    NSIG = 1e-4

    def _make_noise(self, seed=42):
        noise = Additive(nsig=np.array([self.NSIG]))
        noise.noise_seed = seed
        noise.random_stream = np.random.RandomState(seed)
        noise.configure_white(DT)
        return noise

    def test_heun_stochastic_with_stimulus(self):
        """HeunStochastic + constant stimulus with matched noise seed."""
        m = MontbrioPazoRoxin()
        _configure_all(m)
        seed = 42

        # Python path
        noise_py = self._make_noise(seed)
        sn_py = _make_subnet("ctx", m, self.N, HeunStochastic, noise=noise_py)
        _configure_all(sn_py)
        weight = np.zeros(self.N)
        weight[0] = 1.0
        stim_py = _build_stim(sn_py, _linear_eq(0.0, 0.03), weight, self.NSTEP)
        ns_py = NetworkSet(subnets=[sn_py], projections=[], stimuli=[stim_py])
        ns_py.configure()
        ic = _seeded_ic(m, self.N, 1, seed=9001)
        py = _run_python(ns_py, self.NSTEP, [ic])

        # Numba path
        noise_nb = self._make_noise(seed)
        sn_nb = _make_subnet("ctx", m, self.N, HeunStochastic, noise=noise_nb)
        _configure_all(sn_nb)
        stim_nb = _build_stim(sn_nb, _linear_eq(0.0, 0.03), weight, self.NSTEP)
        ns_nb = NetworkSet(subnets=[sn_nb], projections=[], stimuli=[stim_nb])
        ns_nb.configure()
        nb = _run_numba(ns_nb, self.NSTEP, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-4, atol=1e-5,
            err_msg="HeunStochastic+stimulus diverged with matched noise seed",
        )

    def test_euler_stochastic_with_sinusoid(self):
        """EulerStochastic + Sinusoid stimulus with matched noise seed."""
        m = MontbrioPazoRoxin()
        _configure_all(m)
        seed = 43

        # Python path
        noise_py = self._make_noise(seed)
        sn_py = _make_subnet("ctx", m, self.N, EulerStochastic, noise=noise_py)
        _configure_all(sn_py)
        weight = np.zeros(self.N)
        weight[0] = 1.0
        stim_py = _build_stim(sn_py, _sinusoid_eq(amp=0.05, frequency=0.05),
                              weight, self.NSTEP)
        ns_py = NetworkSet(subnets=[sn_py], projections=[], stimuli=[stim_py])
        ns_py.configure()
        ic = _seeded_ic(m, self.N, 1, seed=9002)
        py = _run_python(ns_py, self.NSTEP, [ic])

        # Numba path
        noise_nb = self._make_noise(seed)
        sn_nb = _make_subnet("ctx", m, self.N, EulerStochastic, noise=noise_nb)
        _configure_all(sn_nb)
        stim_nb = _build_stim(sn_nb, _sinusoid_eq(amp=0.05, frequency=0.05),
                              weight, self.NSTEP)
        ns_nb = NetworkSet(subnets=[sn_nb], projections=[], stimuli=[stim_nb])
        ns_nb.configure()
        nb = _run_numba(ns_nb, self.NSTEP, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-4, atol=1e-5,
            err_msg="EulerStochastic+Sinusoid diverged with matched noise seed",
        )


# ══════════════════════════════════════════════════════════════════════════
# Long-run divergence bound
# ══════════════════════════════════════════════════════════════════════════


class TestStimulusLongRunDivergence(unittest.TestCase):
    """Verify that stimulus parity holds over long simulation runs."""

    N = 6
    NSTEP = 500

    def test_sinusoid_500_steps_bounded(self):
        """Sinusoid stimulus: max divergence stays bounded over 500 steps."""
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn)
        weight = np.zeros(self.N)
        weight[0] = 1.0
        stim = _build_stim(sn, _sinusoid_eq(amp=0.03, frequency=0.02),
                           weight, self.NSTEP)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[stim])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=10001)
        py = _run_python(ns, self.NSTEP, [ic])
        nb = _run_numba(ns, self.NSTEP, [ic])

        max_err = np.max(np.abs(nb[0].astype(np.float64) - py[0]))
        self.assertLess(
            max_err, 1e-3,
            f"Sinusoid 500-step divergence too large: {max_err:.2e}",
        )

    def test_linear_500_steps_bounded(self):
        """Constant stimulus: max divergence stays bounded over 500 steps."""
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, self.N, EulerDeterministic)
        _configure_all(m, sn)
        weight = np.zeros(self.N)
        weight[0] = 1.0
        stim = _build_stim(sn, _linear_eq(0.0, 0.03), weight, self.NSTEP)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[stim])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=10002)
        py = _run_python(ns, self.NSTEP, [ic])
        nb = _run_numba(ns, self.NSTEP, [ic])

        max_err = np.max(np.abs(nb[0].astype(np.float64) - py[0]))
        self.assertLess(
            max_err, 1e-3,
            f"Linear 500-step divergence too large: {max_err:.2e}",
        )


# ══════════════════════════════════════════════════════════════════════════
# Multi-subnet with stimulus on each subnet
# ══════════════════════════════════════════════════════════════════════════


class TestStimulusMultiSubnet(unittest.TestCase):
    """Two subnets each receiving independent stimuli."""

    def test_dual_stimulus_parity(self):
        """Each subnet gets its own stimulus: both match between backends."""
        m1 = MontbrioPazoRoxin()
        m2 = MontbrioPazoRoxin()
        _configure_all(m1, m2)

        n1, n2 = 4, 5
        sn1 = _make_subnet("ctx", m1, n1, HeunDeterministic)
        sn1.node_indices = np.arange(n1)
        sn2 = _make_subnet("thal", m2, n2, HeunDeterministic)
        sn2.node_indices = np.arange(n1, n1 + n2)
        _configure_all(sn1, sn2)

        # Weak coupling
        weights = sp.csr_matrix(np.ones((n2, n1)) * 0.005)
        lengths = sp.csr_matrix(np.ones((n2, n1)) * DT)
        proj = InterProjection(
            source=sn1, target=sn2,
            weights=weights, lengths=lengths,
            source_cvar=np.array([0]), target_cvar=np.array([0]),
            scale=1.0, dt=DT, cv=np.array([1.0]),
            cfun=LinearCoupling(),
        )

        nstep = 60
        # Stimulus on cortex node 0
        w1 = np.zeros(n1); w1[0] = 1.0
        stim1 = _build_stim(sn1, _linear_eq(0.0, 0.03), w1, nstep)
        # Stimulus on thalamus node 0
        w2 = np.zeros(n2); w2[0] = 1.0
        stim2 = _build_stim(sn2, _sinusoid_eq(amp=0.04, frequency=0.05), w2, nstep)

        ns = NetworkSet(
            subnets=[sn1, sn2], projections=[proj],
            stimuli=[stim1, stim2],
        )
        ns.configure()

        ic1 = _seeded_ic(m1, n1, 1, seed=11001)
        ic2 = _seeded_ic(m2, n2, 1, seed=11002)
        py = _run_python(ns, nstep, [ic1, ic2])
        nb = _run_numba(ns, nstep, [ic1, ic2])

        for i, name in enumerate(["ctx", "thal"]):
            assert_allclose(
                nb[i].astype(np.float64), py[i],
                rtol=1e-4, atol=1e-5,
                err_msg=f"Dual-stimulus subnet '{name}' diverged",
            )

    def test_stimulus_only_on_target_subnet(self):
        """Stimulus on one subnet should not affect the other subnet's parity."""
        m1 = MontbrioPazoRoxin()
        m2 = MontbrioPazoRoxin()
        _configure_all(m1, m2)

        n1, n2 = 3, 4
        sn1 = _make_subnet("ctx", m1, n1, HeunDeterministic)
        sn1.node_indices = np.arange(n1)
        sn2 = _make_subnet("thal", m2, n2, HeunDeterministic)
        sn2.node_indices = np.arange(n1, n1 + n2)
        _configure_all(sn1, sn2)

        # Weak coupling
        weights = sp.csr_matrix(np.ones((n2, n1)) * 0.005)
        lengths = sp.csr_matrix(np.ones((n2, n1)) * DT)
        proj = InterProjection(
            source=sn1, target=sn2,
            weights=weights, lengths=lengths,
            source_cvar=np.array([0]), target_cvar=np.array([0]),
            scale=1.0, dt=DT, cv=np.array([1.0]),
            cfun=LinearCoupling(),
        )

        nstep = 60
        # Stimulus ONLY on cortex
        w1 = np.zeros(n1); w1[0] = 1.0
        stim1 = _build_stim(sn1, _linear_eq(0.0, 0.05), w1, nstep)

        ns = NetworkSet(
            subnets=[sn1, sn2], projections=[proj],
            stimuli=[stim1],
        )
        ns.configure()

        ic1 = _seeded_ic(m1, n1, 1, seed=11003)
        ic2 = _seeded_ic(m2, n2, 1, seed=11004)
        py = _run_python(ns, nstep, [ic1, ic2])
        nb = _run_numba(ns, nstep, [ic1, ic2])

        for i, name in enumerate(["ctx", "thal"]):
            assert_allclose(
                nb[i].astype(np.float64), py[i],
                rtol=1e-4, atol=1e-5,
                err_msg=f"Single-stim subnet '{name}' diverged",
            )


# ══════════════════════════════════════════════════════════════════════════
# Different model + stimulus
# ══════════════════════════════════════════════════════════════════════════


class TestStimulusDifferentModels(unittest.TestCase):
    """Stimulus parity with models other than MontbrioPazoRoxin."""

    def test_generic2d_oscillator_sinusoid_parity(self):
        """Generic2dOscillator + Sinusoid stimulus parity."""
        m = Generic2dOscillator()
        nstep = 80
        n = 5
        sn = _make_subnet("osc", m, n, HeunDeterministic)
        _configure_all(m, sn)
        weight = np.zeros(n)
        weight[0] = 1.0
        stim = _build_stim(sn, _sinusoid_eq(amp=0.05, frequency=0.05),
                           weight, nstep)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[stim])
        ns.configure()

        ic = _seeded_ic(m, n, 1, seed=12001)
        py = _run_python(ns, nstep, [ic])
        nb = _run_numba(ns, nstep, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-5, atol=1e-6,
            err_msg="Generic2dOscillator+Sinusoid diverged",
        )

    def test_generic2d_oscillator_pulsetrain_parity(self):
        """Generic2dOscillator + PulseTrain stimulus parity."""
        m = Generic2dOscillator()
        nstep = 100
        n = 4
        sn = _make_subnet("osc", m, n, EulerDeterministic)
        _configure_all(m, sn)
        weight = np.zeros(n)
        weight[0] = 1.0
        stim = _build_stim(sn, _pulsetrain_eq(onset=2.0, T=8.0, tau=2.0, amp=0.1),
                           weight, nstep)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[stim])
        ns.configure()

        ic = _seeded_ic(m, n, 1, seed=12002)
        py = _run_python(ns, nstep, [ic])
        nb = _run_numba(ns, nstep, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-4, atol=1e-5,
            err_msg="Generic2dOscillator+PulseTrain diverged",
        )


if __name__ == "__main__":
    unittest.main()
