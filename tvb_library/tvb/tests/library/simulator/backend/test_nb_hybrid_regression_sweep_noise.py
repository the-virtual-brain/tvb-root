"""Regression coverage for stochastic CPU-prange sweep realizations."""

import numpy as np
import scipy.sparse as sp

from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.hybrid.coupling import Linear
from tvb.simulator.hybrid.intra_projection import IntraProjection
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.integrators import EulerStochastic
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.noise import Additive


DT = 0.01
BASE_SEED = 7319
NSTEPS = 40


def _stochastic_network():
    model = MontbrioPazoRoxin()
    model.configure()

    noise = Additive(nsig=np.array([0.75]))
    noise.noise_seed = BASE_SEED
    noise.random_stream = np.random.RandomState(BASE_SEED)
    noise.configure_white(DT)
    scheme = EulerStochastic(dt=DT, noise=noise)
    scheme.configure_boundaries(model)

    subnet = Subnetwork(name="noise_oracle", model=model, scheme=scheme, nnodes=4)
    weights = sp.csr_matrix((4, 4), dtype=np.float64)
    subnet.projections = [
        IntraProjection(
            source_cvar=np.array([0], dtype=np.int_),
            target_cvar=np.array([0], dtype=np.int_),
            weights=weights,
            lengths=weights.copy(),
            cv=1.0,
            dt=DT,
            scale=1.0,
            cfun=Linear(a=np.array([1.0])),
        )
    ]
    subnet.configure()
    network = NetworkSet(subnets=[subnet], projections=[])
    network.configure()
    return network


def _reset_noise(network, seed=BASE_SEED):
    noise = network.subnets[0].scheme.noise
    noise.noise_seed = seed
    noise.random_stream = np.random.RandomState(seed)


def _run(network, values, n_workers):
    return NbHybridBackend().sweep(
        network,
        params={"coupling_scale": np.asarray(values, dtype=np.float32)},
        nstep=NSTEPS,
        backend="cpu",
        n_workers=n_workers,
    )


def _collapsed(result):
    return result.merged_tavg


def test_prange_duplicate_rows_get_independent_replayable_noise():
    network = _stochastic_network()
    identical_rows = np.ones(4, dtype=np.float32)

    # This is the path under test: all identical configurations coexist in one
    # prange launch rather than being tested in separate one-row sweeps.
    _reset_noise(network)
    parallel = _run(network, identical_rows, n_workers=4)
    _reset_noise(network)
    parallel_replay = _run(network, identical_rows, n_workers=4)

    # Sequential sweep defines the random-stream ordering: configurations are
    # independent, while resetting the base seed replays the complete sweep.
    _reset_noise(network)
    sequential = _run(network, identical_rows, n_workers=1)
    _reset_noise(network)
    sequential_replay = _run(network, identical_rows, n_workers=1)

    # Sensitivity oracle: resetting before each singleton deliberately gives
    # two configurations shared noise. With zero coupling and identical initial
    # conditions their stochastic outputs must then be exactly identical.
    _reset_noise(network)
    shared_a = _collapsed(_run(network, [1.0], n_workers=1))[0]
    _reset_noise(network)
    shared_b = _collapsed(_run(network, [1.0], n_workers=1))[0]
    _reset_noise(network, BASE_SEED + 1)
    different_noise = _collapsed(_run(network, [1.0], n_workers=1))[0]

    assert parallel.backend == "cpu-prange"
    assert sequential.backend == "cpu-seq"
    np.testing.assert_array_equal(parallel.sweep_values[:, 0], identical_rows)
    np.testing.assert_array_equal(shared_a, shared_b)
    assert not np.allclose(shared_a, different_noise, rtol=1e-4, atol=1e-4), (
        "noise is too weak for this fixture to detect a shared realization"
    )

    parallel_data = _collapsed(parallel)
    sequential_data = _collapsed(sequential)
    assert not np.allclose(parallel_data[0], parallel_data[1], rtol=1e-6, atol=1e-6), (
        "identical prange rows reused one stochastic realization"
    )
    assert not np.allclose(sequential_data[0], sequential_data[1], rtol=1e-6, atol=1e-6)

    np.testing.assert_array_equal(parallel_data, _collapsed(parallel_replay))
    np.testing.assert_array_equal(sequential_data, _collapsed(sequential_replay))
    np.testing.assert_allclose(parallel_data, sequential_data, rtol=2e-5, atol=2e-5)
