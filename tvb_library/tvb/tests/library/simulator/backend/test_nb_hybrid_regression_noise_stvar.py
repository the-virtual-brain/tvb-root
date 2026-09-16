import numpy as np

from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.hybrid import NetworkSet, Subnetwork
from tvb.simulator.integrators import EulerStochastic
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.noise import Additive


DT = 0.1
SEED = 1729


def _make_network():
    model = MontbrioPazoRoxin(
        Delta=np.array([0.0]),
        eta=np.array([0.0]),
        I=np.array([0.0]),
    )
    noise = Additive(nsig=np.array([0.25]))
    noise.noise_seed = SEED
    noise.random_stream = np.random.RandomState(SEED)
    subnet = Subnetwork(
        name="zero_drift",
        model=model,
        scheme=EulerStochastic(dt=DT, noise=noise),
        nnodes=8,
    )
    subnet.configure()
    network = NetworkSet(subnets=[subnet], projections=[])
    network.configure()
    return network


def test_euler_stochastic_scalar_noise_broadcasts_independently_of_stvar():
    initial = np.zeros((2, 8, 1), dtype=np.float64)

    python_network = _make_network()
    python_subnet = python_network.subnets[0]
    np.testing.assert_array_equal(python_subnet.model.stvar, np.array([1]))

    noise_gfun = python_subnet.scheme.noise.gfun(initial)
    assert noise_gfun.shape == (1,)
    np.testing.assert_allclose(noise_gfun, np.sqrt(2.0 * np.array([0.25])))

    coupling = python_subnet.zero_cvars()
    gaussian = np.sqrt(DT) * np.random.RandomState(SEED).normal(size=initial.shape)
    expected = (
        initial
        + DT * python_subnet.model.dfun(initial, coupling, 0.0)
        + noise_gfun * gaussian
    )
    python_subnet.scheme.integration_bound_and_clamp(expected)

    python_state = python_network.zero_states(initial_states=[initial.copy()])
    python_network.init_projection_buffers(python_state)
    python_final = np.asarray(python_network.step(1, python_state)[0])

    numba_network = _make_network()
    compiled = NbHybridBackend().compile(numba_network)
    _, snapshot = compiled.run(
        1,
        initial_states=[initial.copy()],
        return_snapshot=True,
    )
    numba_final = np.asarray(snapshot["states"][0])

    np.testing.assert_allclose(python_final, expected, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(numba_final, python_final, rtol=1e-6, atol=1e-7)

    for state_row in range(initial.shape[0]):
        assert np.ptp(python_final[state_row, :, 0]) > 0.0
        assert np.ptp(numba_final[state_row, :, 0]) > 0.0
