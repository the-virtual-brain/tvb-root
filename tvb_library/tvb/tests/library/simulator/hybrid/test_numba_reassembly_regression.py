"""Regression coverage for public hybrid Simulator Numba output reassembly."""

import warnings

import numpy as np

from tvb.simulator.hybrid import NetworkSet, Simulator, Subnetwork
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.monitors import Raw


DT = 0.1
SIMULATION_LENGTH = 0.2


def _subnetwork(name, nnodes):
    return Subnetwork(
        name=name,
        model=MontbrioPazoRoxin(),
        scheme=HeunDeterministic(dt=DT),
        nnodes=nnodes,
    ).configure()


def _run_python(name, initial_state):
    network = NetworkSet(
        subnets=[_subnetwork(name, initial_state.shape[1])],
        projections=[],
    )
    simulator = Simulator(
        nets=network,
        monitors=[Raw()],
        simulation_length=SIMULATION_LENGTH,
        backend="python",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        simulator.configure()
    return simulator.run(initial_conditions=[initial_state.copy()])[0]


def test_numba_nonmerged_reassembly_concatenates_nodes_and_preserves_vois():
    left_initial = np.array(
        [[[0.03], [0.05]], [[-1.2], [-0.7]]], dtype=np.float64
    )
    right_initial = np.array(
        [[[0.11], [0.13], [0.17]], [[0.4], [0.9], [1.6]]], dtype=np.float64
    )

    left_times, left_python = _run_python("left", left_initial)
    right_times, right_python = _run_python("right", right_initial)
    reference = np.concatenate([left_python, right_python], axis=2)

    subnets = [
        _subnetwork("left", left_initial.shape[1]),
        _subnetwork("right", right_initial.shape[1]),
    ]
    assert subnets[0].nnodes != subnets[1].nnodes
    assert all(subnet.node_indices is None for subnet in subnets)
    voi_counts = [len(subnet.model.variables_of_interest) for subnet in subnets]
    assert voi_counts == [2, 2]

    simulator = Simulator(
        nets=NetworkSet(subnets=subnets, projections=[]),
        monitors=[Raw()],
        simulation_length=SIMULATION_LENGTH,
        backend="numba",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        simulator.configure()
    (numba_times, numba_output), = simulator.run(
        initial_conditions=[left_initial.copy(), right_initial.copy()]
    )

    np.testing.assert_array_equal(left_times, right_times)
    np.testing.assert_array_equal(numba_times, left_times)
    assert numba_output.shape == (
        len(numba_times),
        voi_counts[0],
        left_initial.shape[1] + right_initial.shape[1],
        1,
    )
    np.testing.assert_allclose(numba_output, reference, rtol=1e-5, atol=1e-6)
