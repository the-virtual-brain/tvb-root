import numpy as np
import pytest

from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.integrators import HeunDeterministic, HeunStochastic
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.models.stefanescu_jirsa import ReducedSetFitzHughNagumo
from tvb.simulator.noise import Additive


DT = 0.1
SEED = 1729


CASES = pytest.mark.parametrize(
    "combined, stochastic",
    [
        pytest.param(False, False, id="standard-deterministic"),
        pytest.param(False, True, id="standard-stochastic"),
        pytest.param(True, False, id="combined-deterministic"),
        pytest.param(True, True, id="combined-stochastic"),
    ],
)


def _make_network(combined, stochastic, clamp_value=None):
    if combined:
        model = ReducedSetFitzHughNagumo()
        constrained_index = model.state_variables.index("xi")
        # Reduced-set models have no physiological hard bounds by default.  A
        # model-level bound exercises the generated cross-mode Heun path.
        model.state_variable_boundaries = {"xi": np.array([0.0, np.inf])}
        initial_state = np.zeros((model.nvar, 1, model.number_of_modes))
        initial_state[0, 0] = 0.1
        initial_state[1, 0] = 10.0
    else:
        model = MontbrioPazoRoxin()
        constrained_index = model.state_variables.index("r")
        initial_state = np.zeros((model.nvar, 1, model.number_of_modes))
        initial_state[:, 0, 0] = (0.1, -20.0)

    if stochastic:
        noise = Additive(nsig=np.array([0.01]))
        noise.random_stream = np.random.RandomState(SEED)
        scheme = HeunStochastic(dt=DT, noise=noise)
    else:
        scheme = HeunDeterministic(dt=DT)

    if clamp_value is not None:
        scheme.clamped_state_variable_indices = np.array(
            [constrained_index], dtype=np.int32
        )
        scheme.clamped_state_variable_values = np.array([[clamp_value]])

    subnet = Subnetwork(name="clamp_regression", model=model, scheme=scheme, nnodes=1)
    subnet.configure()
    network = NetworkSet(subnets=[subnet], projections=[])
    network.configure()
    return network, initial_state, constrained_index


def _noise_increment(scheme, state):
    if not isinstance(scheme, HeunStochastic):
        return np.zeros_like(state)
    noise = scheme.noise.generate(state.shape)
    return noise * scheme.noise.gfun(state)


def _python_step(network, initial_state):
    state = network.States(initial_state.copy())
    network.init_projection_buffers(state)
    return np.asarray(network.step(1, state)[0])


def _late_constraint_reference(network, initial_state):
    """Heun with no predictor constraint, but the normal final constraint."""
    subnet = network.subnets[0]
    coupling = subnet.zero_cvars()
    noise = _noise_increment(subnet.scheme, initial_state)
    k1 = subnet.model.dfun(initial_state, coupling, 0.0)
    predictor = initial_state + DT * k1 + noise
    k2 = subnet.model.dfun(predictor, coupling, 0.0)
    result = initial_state + DT * (k1 + k2) / 2.0 + noise
    subnet.scheme.bound_and_clamp(result)
    return predictor, result


def _numba_step(network, initial_state):
    compiled = NbHybridBackend().compile(network)
    _, snapshot = compiled.run(
        nstep=1,
        chunk_size=1,
        initial_states=[initial_state.copy()],
        return_snapshot=True,
    )
    return snapshot["states"][0]


def _assert_second_dfun_is_sensitive(correct, late, constrained_index):
    unconstrained = np.delete(correct - late, constrained_index, axis=0)
    assert np.max(np.abs(unconstrained)) > 1e-3, (
        "the case does not distinguish predictor clamping from final-only clamping"
    )


@CASES
def test_heun_clamps_boundary_crossing_predictor_before_second_dfun(
    combined, stochastic
):
    py_network, initial_state, constrained_index = _make_network(combined, stochastic)
    late_network, _, _ = _make_network(combined, stochastic)
    nb_network, _, _ = _make_network(combined, stochastic)

    python_state = _python_step(py_network, initial_state)
    predictor, late_state = _late_constraint_reference(late_network, initial_state)
    numba_state = _numba_step(nb_network, initial_state)

    assert np.min(predictor[constrained_index]) < -0.1
    _assert_second_dfun_is_sensitive(python_state, late_state, constrained_index)
    np.testing.assert_allclose(
        numba_state,
        python_state,
        rtol=2e-5,
        atol=2e-5,
        err_msg="Numba evaluated Heun's second dfun before clamping its predictor",
    )


@CASES
def test_heun_integrator_clamped_state_variables_match_python(
    combined, stochastic
):
    clamp_value = 0.25
    py_network, initial_state, constrained_index = _make_network(
        combined, stochastic, clamp_value
    )
    late_network, _, _ = _make_network(combined, stochastic, clamp_value)
    nb_network, _, _ = _make_network(combined, stochastic, clamp_value)

    python_state = _python_step(py_network, initial_state)
    predictor, late_state = _late_constraint_reference(late_network, initial_state)
    numba_state = _numba_step(nb_network, initial_state)

    assert np.max(np.abs(predictor[constrained_index] - clamp_value)) > 0.1
    _assert_second_dfun_is_sensitive(python_state, late_state, constrained_index)
    np.testing.assert_array_equal(
        numba_state[constrained_index],
        np.full_like(numba_state[constrained_index], clamp_value),
    )
    np.testing.assert_allclose(
        numba_state,
        python_state,
        rtol=2e-5,
        atol=2e-5,
        err_msg=(
            "Numba did not honor clamped_state_variable_indices/values during "
            "the Heun predictor and corrector"
        ),
    )
