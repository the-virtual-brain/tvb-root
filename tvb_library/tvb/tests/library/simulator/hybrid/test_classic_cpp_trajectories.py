"""Public classic/CPP Simulator trajectory contracts on synthetic networks.

Both APIs receive the same constant prehistory, not random model defaults.
Classic accepts H history samples and starts its clock at (H - 1) * dt;
hybrid accepts one state and fills its history with that state, starting at
zero. We retain both natural clocks and compare elapsed time. The delayed
fixture has H - 1 divisible by the TemporalAverage window so averaging bins
align without resetting simulator state or trimming transient samples.
TemporalAverage(period=dt) exposes each integrated state with a midpoint
timestamp; these tests do not cover Raw (the public CPP Simulator currently
rejects Raw with its default chunk size).

Weights use [target, source] ordering in both APIs, without transposition or
row/whole-network normalization. Classic Kuramoto is a SparseCoupling and
normalizes by the number of coupling variables, NOT the number of nodes;
that number is one here, so identical amplitudes need no compensating factor.
SigmoidalJansenRit uses the classic pre-summation sigmoid of y1 - y2 in both
APIs (not the legacy hybrid e0/v0 convention), feeding coupling slot zero.
CPP integrates in float32, classic in float64; tolerances allow roundoff,
not a scientifically different coupling, delay, integration, or monitor rule.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from tvb.datatypes.connectivity import Connectivity
from tvb.simulator import coupling, integrators, models, monitors
from tvb.simulator.hybrid import NetworkSet, Simulator as HybridSimulator, Subnetwork
from tvb.simulator.hybrid import coupling as hybrid_coupling
from tvb.simulator.hybrid.projection_utils import create_intra_projection
from tvb.simulator.simulator import Simulator as ClassicSimulator


DT = 0.0625
PERIOD = 0.5
# Deliberately asymmetric, including self-edges. All edges are nonzero so the
# two sparse representations have the same graph and length ordering.
WEIGHTS = np.array([[0.125, 0.75, 0.25],
                    [0.5, 0.0625, 0.375],
                    [0.875, 0.25, 0.1875]])
SPEED = 2.0


@pytest.fixture
def cpp_extension():
    # Match the optional-extension convention used by the hybrid suite. CPP CI
    # builds the extension; a run with this fixture skipped is not parity evidence.
    pytest.importorskip("tvb.simulator.backend.cpp_hybrid")


def _make_connectivity(delay_steps):
    conn = Connectivity(
        weights=WEIGHTS.copy(),
        tract_lengths=np.asarray(delay_steps, dtype=float) * SPEED * DT,
        speed=np.array([SPEED]),
        centres=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        region_labels=np.array(["left", "middle", "right"]),
    )
    conn.configure()
    return conn


def _monitors(selected_voi):
    # Different output rates/shapes, in this order, catch monitor reordering.
    return [monitors.TemporalAverage(period=PERIOD, variables_of_interest=selected_voi),
            monitors.TemporalAverage(period=DT)]


def _compare_public_simulators(model_factory, classic_cfun, hybrid_cfun,
                               initial_state, delay_steps, expected_delays,
                               duration, selected_voi):
    conn = _make_connectivity(delay_steps)
    horizon = int(np.max(expected_delays)) + 1
    history = np.repeat(initial_state[np.newaxis], horizon, axis=0)
    classic = ClassicSimulator(
        connectivity=conn,
        model=model_factory(),
        coupling=classic_cfun,
        integrator=integrators.HeunDeterministic(dt=DT),
        initial_conditions=history.copy(),
        monitors=_monitors(selected_voi),
    ).configure()
    np.testing.assert_array_equal(conn.idelays, expected_delays)
    np.testing.assert_array_equal(classic.current_state, initial_state)
    assert classic.current_step == horizon - 1
    offset = (horizon - 1) * DT
    assert (horizon - 1) % int(PERIOD / DT) == 0
    classic_outputs = classic.run(simulation_length=duration)

    subnet = Subnetwork(
        name="all_regions",
        model=model_factory(),
        scheme=integrators.HeunDeterministic(dt=DT),
        nnodes=3,
        node_indices=np.arange(3),
    )
    # Source indices are model-state indices; target [0] is the model's first
    # COUPLING slot, not necessarily its first state variable (JR: y1, not y0).
    projection = create_intra_projection(
        subnet=subnet,
        source_cvar=np.asarray(subnet.model.cvar),
        target_cvar=np.array([0]),
        weights=csr_matrix(WEIGHTS.copy()),
        lengths=csr_matrix(conn.tract_lengths.copy()),
        cv=SPEED,
        dt=DT,
        scale=1.0,
        coupling=hybrid_cfun,
    )
    subnet.projections = [projection]
    subnet.configure()
    rows, cols = projection.weights.nonzero()
    np.testing.assert_array_equal(projection.idelays, expected_delays[rows, cols])
    hybrid = HybridSimulator(
        nets=NetworkSet(subnets=[subnet]),
        simulation_length=duration,
        monitors=_monitors(selected_voi),
        backend="cpp",
    )
    hybrid.configure()
    hybrid_outputs = hybrid.run(initial_conditions=[initial_state.copy()])

    nstep = int(duration / DT)
    naverage = int(duration / PERIOD)
    elapsed_times = [PERIOD * (np.arange(naverage) + 0.5),
                     DT * (np.arange(nstep) + 0.5)]
    nvois = [len(selected_voi), len(subnet.model.variables_of_interest)]
    assert len(classic_outputs) == len(hybrid_outputs) == 2
    for index, ((tc, yc), (th, yh)) in enumerate(zip(classic_outputs, hybrid_outputs)):
        expected_time = elapsed_times[index]
        # Check each absolute clock independently before normalizing its origin.
        np.testing.assert_array_equal(tc, expected_time + offset)
        np.testing.assert_array_equal(th, expected_time)
        np.testing.assert_array_equal(tc - offset, th)
        assert yc.shape == yh.shape == (len(expected_time), nvois[index], 3, 1)
        assert np.all(np.isfinite(yc))
        assert np.all(np.isfinite(yh))
        np.testing.assert_allclose(yh, yc, rtol=3e-5, atol=3e-6,
                                   err_msg=f"monitor {index}: classic vs CPP trajectory")

    # Verify selected-VOI order and the averaging window against every per-step
    # trajectory, independently of agreement between the two implementations.
    window = int(PERIOD / DT)
    for output in (classic_outputs, hybrid_outputs):
        averaged, per_step = output[0][1], output[1][1]
        expected_average = per_step.reshape(naverage, window, per_step.shape[1], 3, 1).mean(axis=1)
        np.testing.assert_allclose(averaged, expected_average[:, selected_voi],
                                   rtol=3e-6, atol=3e-7)


def test_linear_classic_cpp_public_trajectory(cpp_extension):
    """Damped, linearly coupled heterogeneous initial values; no delays."""
    _compare_public_simulators(
        lambda: models.Linear(gamma=np.array([-0.5])),
        coupling.Linear(a=np.array([0.25]), b=np.array([0.015625])),
        hybrid_coupling.Linear(a=np.array([0.25]), b=np.array([0.015625])),
        np.array([[[0.25], [-0.5], [0.75]]]),
        np.zeros((3, 3)), np.zeros((3, 3), dtype=int), 8.0, np.array([0]),
    )


def test_kuramoto_classic_cpp_public_trajectory(cpp_extension):
    """Nonuniform phases/frequencies exercise target-dependent sine coupling."""
    _compare_public_simulators(
        lambda: models.Kuramoto(omega=np.array([0.3, 0.5, 0.7])),
        coupling.Kuramoto(a=np.array([0.375])),
        hybrid_coupling.Kuramoto(a=np.array([0.375])),
        np.array([[[0.2], [1.1], [2.4]]]),
        np.zeros((3, 3)), np.zeros((3, 3), dtype=int), 8.0, np.array([0]),
    )


def test_delayed_jansen_rit_classic_cpp_public_trajectory(cpp_extension):
    """Rounded heterogeneous delays, all six states, and reordered monitor VOIs."""
    # Noninteger propagation times exercise nearest-step quantization, away
    # from half-integer ties. Maximum delay 8 yields a classic clock offset
    # of 0.5 ms, exactly one TemporalAverage period.
    delay_steps = np.array([[0.2, 1.2, 3.8], [2.8, 0.2, 8.2], [5.2, 6.8, 0.2]])
    expected_delays = np.array([[0, 1, 4], [3, 0, 8], [5, 7, 0]])
    initial_state = np.array([
        [0.1, 0.15, 0.2], [2.0, 4.0, 6.0], [0.4, 1.2, 2.4],
        [0.01, -0.02, 0.03], [0.1, 0.2, -0.1], [-0.1, 0.05, 0.15],
    ])[:, :, np.newaxis]
    cfun_params = dict(a=np.array([4.0]), cmin=np.array([0.0]),
                       cmax=np.array([0.005]), midpoint=np.array([6.0]),
                       r=np.array([0.56]))
    _compare_public_simulators(
        lambda: models.JansenRit(variables_of_interest=("y0", "y1", "y2", "y3", "y4", "y5")),
        coupling.SigmoidalJansenRit(**cfun_params),
        hybrid_coupling.SigmoidalJansenRit(use_classic=np.array([1]), **cfun_params),
        initial_state, delay_steps, expected_delays, 16.0, np.array([2, 0]),
    )
