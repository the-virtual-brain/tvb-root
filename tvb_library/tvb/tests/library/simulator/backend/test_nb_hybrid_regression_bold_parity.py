"""Regression coverage for Numba BOLD parity with TVB's HRF monitor."""

import numpy as np

from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.monitors import Bold


DT = 0.1
PERIOD = 500.0
HRF_LENGTH = 8000.0
SIMULATION_LENGTH = 10000.0
NSTEP = int(SIMULATION_LENGTH / DT)
INITIAL_STATE = np.array(
    [
        [[0.06], [0.09]],
        [[-1.45], [-1.10]],
    ],
    dtype=np.float64,
)
LEGACY_BALLOON_FINAL = np.array(
    [[[0.16085401], [0.17156173]]], dtype=np.float32
)


def _network():
    model = MontbrioPazoRoxin(
        eta=np.array([-5.0, -4.6]),
        variables_of_interest=("r",),
    )
    model.configure()
    subnet = Subnetwork(
        name="bold_parity",
        model=model,
        scheme=HeunDeterministic(dt=DT),
        nnodes=2,
    )
    subnet.configure()
    network = NetworkSet(subnets=[subnet], projections=[])
    network.configure()
    return network


def _monitor():
    return Bold(
        period=PERIOD,
        hrf_length=HRF_LENGTH,
        variables_of_interest=np.array([0], dtype=int),
    )


def _python_bold(raw_states):
    monitor = _monitor()
    monitor._config_dt(DT)
    monitor.voi = np.array([0], dtype=int)
    monitor.compute_hrf()
    monitor._config_stock(num_vars=1, num_nodes=2, num_modes=1)

    samples = [
        sample
        for step, state in enumerate(raw_states, start=1)
        if (sample := monitor.sample(step, state)) is not None
    ]
    return np.asarray([sample[0] for sample in samples]), np.stack(
        [sample[1] for sample in samples]
    )


def test_numba_bold_matches_python_hrf_convolution():
    """Numba BOLD must implement the Python monitor, not a Balloon substitute."""
    raw = NbHybridBackend().run_network(
        _network(),
        nstep=NSTEP,
        chunk_size=1,
        initial_states=[INITIAL_STATE.copy()],
    )
    raw_times, raw_states, _ = raw[0]
    python_times, python_data = _python_bold(raw_states)

    result = NbHybridBackend().run_network(
        _network(),
        nstep=NSTEP,
        chunk_size=1,
        monitors=[_monitor()],
        initial_states=[INITIAL_STATE.copy()],
    )
    numba_times, numba_data = result[0][0]

    assert raw_times[-1] >= HRF_LENGTH
    assert np.count_nonzero(python_times > HRF_LENGTH) >= 3
    assert np.all(np.isfinite(raw_states))
    assert np.max(np.abs(raw_states)) < 0.2
    # Numba accumulates its time vector in float32, so align schedules to much
    # less than one integration step rather than requiring bitwise equality.
    np.testing.assert_allclose(numba_times, python_times, rtol=0.0, atol=2e-4)
    assert numba_data.shape == python_data.shape
    assert np.all(np.isfinite(python_data))

    # This frozen final sample is from the superseded Numba Balloon ODE. The
    # fixture must distinguish that behavior from TVB's HRF-convolution oracle.
    assert not np.allclose(
        python_data[-1], LEGACY_BALLOON_FINAL, rtol=5e-4, atol=2e-5
    ), "fixture would accept the Balloon result as the HRF-convolution oracle"

    # The backend evolves neural state in float32; this allows accumulated
    # rounding while remaining tight relative to the non-trivial BOLD signal.
    np.testing.assert_allclose(
        numba_data,
        python_data,
        rtol=5e-4,
        atol=2e-5,
        err_msg="Numba BOLD differs from tvb.simulator.monitors.Bold",
    )
