import pytest
import numpy as np
from numpy.testing import assert_allclose

from tvb.simulator.models.stefanescu_jirsa import (
    ReducedSetFitzHughNagumo,
    ReducedSetHindmarshRose,
)
from tvb.simulator.models.wong_wang_exc_inh import ReducedWongWangExcInh
from tvb.simulator import integrators


def _baseline_fhn_reference(state, coupling, model):
    xi = state[0, :]
    eta = state[1, :]
    alpha = state[2, :]
    beta = state[3, :]

    c_0 = coupling[0, :].sum(axis=1)[:, np.newaxis]

    dot_xi_A = xi @ model.Aik
    dot_alpha_B = alpha @ model.Bik
    dot_xi_C = xi @ model.Cik

    deriv = np.empty_like(state)

    deriv[0] = (
        model.tau * (xi - model.e_i * (xi**3) / 3.0 - eta)
        + model.K11 * (dot_xi_A - xi)
        - model.K12 * (dot_alpha_B - xi)
        + model.tau * (model.IE_i + c_0)
    )
    deriv[1] = (xi - model.b * eta + model.m_i) / model.tau
    deriv[2] = (
        model.tau * (alpha - model.f_i * (alpha**3) / 3.0 - beta)
        + model.K21 * (dot_xi_C - alpha)
        + model.tau * (model.II_i + c_0)
    )
    deriv[3] = (alpha - model.b * beta + model.n_i) / model.tau

    return deriv


def _baseline_hr_reference(state, coupling, model):
    xi = state[0, :]
    eta = state[1, :]
    tau = state[2, :]
    alpha = state[3, :]
    beta = state[4, :]
    gamma = state[5, :]

    c_0 = coupling[0, :].sum(axis=1)[:, np.newaxis]

    dot_xi_A = xi @ model.A_ik
    dot_alpha_B = alpha @ model.B_ik
    dot_xi_C = xi @ model.C_ik

    xi_sq = xi * xi
    alpha_sq = alpha * alpha

    deriv = np.empty_like(state)

    deriv[0] = (
        eta
        - model.a_i * (xi_sq * xi)
        + model.b_i * xi_sq
        - tau
        + model.K11 * (dot_xi_A - xi)
        - model.K12 * (dot_alpha_B - xi)
        + model.IE_i
        + c_0
    )
    deriv[1] = model.c_i - model.d_i * xi_sq - eta
    deriv[2] = model.r * model.s * xi - model.r * tau - model.m_i
    deriv[3] = (
        beta
        - model.e_i * (alpha_sq * alpha)
        + model.f_i * alpha_sq
        - gamma
        + model.K21 * (dot_xi_C - alpha)
        + model.II_i
        + c_0
    )
    deriv[4] = model.h_i - model.p_i * alpha_sq - beta
    deriv[5] = model.r * model.s * alpha - model.r * gamma - model.n_i

    return deriv


@pytest.mark.parametrize("n_nodes", [68, 100])
def test_fhn_parity_and_noncontiguous_slices(n_nodes):
    model = ReducedSetFitzHughNagumo()
    model.configure()

    state = np.random.randn(4, n_nodes, 3)
    raw_buffer = np.random.randn(2, n_nodes, 6)
    coupling = raw_buffer[:1, :, ::2]

    expected = _baseline_fhn_reference(state, coupling, model)
    actual = model.dfun(state, coupling)

    assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("n_nodes", [68, 100])
def test_hindmarsh_rose_parity_and_attributes(n_nodes):
    model = ReducedSetHindmarshRose()
    model.configure()

    state = np.random.randn(6, n_nodes, 3)
    raw_buffer = np.random.randn(2, n_nodes, 6)
    coupling = raw_buffer[:1, :, ::2]

    expected = _baseline_hr_reference(state, coupling, model)
    actual = model.dfun(state, coupling)

    assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)


def test_wong_wang_heun_integrator_numerical_parity():
    model = ReducedWongWangExcInh()
    model.configure()
    integ = integrators.HeunDeterministic(dt=0.1)

    n_nodes = 68
    state = np.random.uniform(0.1, 0.8, size=(2, n_nodes, 1))
    coupling = np.zeros((1, n_nodes, 1))

    # Next state computed through the compiled routine
    state_compiled = integ.scheme(state.copy(), model.dfun, coupling, 0.0, 0.0)

    # Next state computed through the pure NumPy reference routine
    state_baseline = integ.scheme(state.copy(), model._numpy_dfun, coupling, 0.0, 0.0)

    # Proves the multi-stage predictor-corrector evaluates correctly without stage-one overwrites
    assert_allclose(state_compiled, state_baseline, rtol=1e-5, atol=1e-7)


def test_wong_wang_shape_and_broadcast_semantics():
    model = ReducedWongWangExcInh()
    n_nodes = 68

    # Heterogeneous parameter array across nodes
    model.a_e = np.linspace(300.0, 320.0, n_nodes)
    model.configure()

    # Multiple modes and non-contiguous coupling slice
    n_modes = 3
    state = np.random.uniform(0.1, 0.8, size=(2, n_nodes, n_modes))
    raw_coupling = np.random.randn(2, n_nodes, n_modes * 2)
    coupling = raw_coupling[:1, :, ::2]

    deriv = model.dfun(state, coupling)
    assert deriv.shape == (2, n_nodes, n_modes)
    assert not np.isnan(deriv).any()

@pytest.mark.parametrize("model_cls,n_vars,n_modes", [
    (ReducedSetFitzHughNagumo, 4, 3),
    (ReducedSetHindmarshRose, 6, 3),
    (ReducedWongWangExcInh, 2, 1),
])
def test_forward_heun_trajectory_parity(model_cls, n_vars, n_modes):
    n_nodes = 68
    dt = 0.05
    n_steps = 30

    model = model_cls()
    model.configure()
    integ = integrators.HeunDeterministic(dt=dt)

    np.random.seed(42)
    state_new = np.random.uniform(0.1, 0.5, size=(n_vars, n_nodes, n_modes)).astype(np.float64)
    state_ref = state_new.copy()

    # Coupling slice with realistic structure
    raw_coupling = np.random.randn(2, n_nodes, n_modes * 2)
    coupling = raw_coupling[:1, :, ::2]

    # Accept *args so the integrator can pass local_coupling without a signature mismatch
    if model_cls is ReducedSetFitzHughNagumo:
        ref_dfun = lambda s, c, *args: _baseline_fhn_reference(s, c, model)
    elif model_cls is ReducedSetHindmarshRose:
        ref_dfun = lambda s, c, *args: _baseline_hr_reference(s, c, model)
    else:
        ref_dfun = model._numpy_dfun

    for _ in range(n_steps):
        state_new = integ.scheme(state_new, model.dfun, coupling, 0.0, 0.0)
        state_ref = integ.scheme(state_ref, ref_dfun, coupling, 0.0, 0.0)

    assert_allclose(state_new, state_ref, rtol=1e-4, atol=1e-6)