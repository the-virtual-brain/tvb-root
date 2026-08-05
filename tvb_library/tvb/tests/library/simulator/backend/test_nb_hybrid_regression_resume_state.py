"""Regression coverage for complete NbHybridBackend checkpoint state."""

import numpy as np
import scipy.sparse as sp

from tvb.datatypes import equations
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.patterns import StimuliRegion
from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.hybrid.coupling import Linear
from tvb.simulator.hybrid.intra_projection import IntraProjection
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.stimulus import Stim
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.integrators import EulerStochastic
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.monitors import Bold, Raw, TemporalAverage
from tvb.simulator.noise import Additive


DT = 0.1
N_NODES = 3
DELAY_STEPS = 10
HORIZON = DELAY_STEPS + 1
SPLIT = 13
TOTAL = 41
TAVG_STEPS = 7


def _network(seed=73):
    """Return a fresh, equivalent stochastic network with delay and stimulus."""
    model = MontbrioPazoRoxin()
    model.configure()

    noise = Additive(nsig=np.array([2.0e-4]))
    noise.noise_seed = seed
    noise.random_stream = np.random.RandomState(seed)
    noise.configure_white(DT)
    scheme = EulerStochastic(dt=DT, noise=noise)
    scheme.configure_boundaries(model)

    subnet = Subnetwork(name="resume", model=model, scheme=scheme, nnodes=N_NODES)
    weights = sp.csr_matrix(
        np.array(
            [[0.0, 0.14, 0.03], [0.05, 0.0, 0.11], [0.09, 0.04, 0.0]],
            dtype=np.float64,
        )
    )
    lengths = weights.copy()
    lengths.data[:] = DELAY_STEPS * DT
    projection = IntraProjection(
        source_cvar=np.array([0], dtype=np.int32),
        target_cvar=np.array([0], dtype=np.int32),
        weights=weights,
        lengths=lengths,
        cv=1.0,
        dt=DT,
        scale=0.7,
        cfun=Linear(a=np.array([0.8]), b=np.array([0.01])),
    )
    subnet.projections = [projection]

    connectivity = Connectivity(
        centres=np.zeros((N_NODES, 3)),
        weights=np.zeros((N_NODES, N_NODES)),
        tract_lengths=np.zeros((N_NODES, N_NODES)),
        region_labels=np.array(["a", "b", "c"]),
        speed=np.array([1.0]),
    )
    connectivity.configure()
    temporal = equations.Sinusoid()
    temporal.parameters["amp"] = 0.025
    temporal.parameters["frequency"] = 0.37
    pattern = StimuliRegion(
        temporal=temporal,
        connectivity=connectivity,
        weight=np.array([1.0, 0.35, 0.0]),
    )
    stimulus = Stim(
        target=subnet,
        stimulus=pattern,
        target_cvar=np.array([0], dtype=np.int32),
        projection_scale=1.0,
    )
    stimulus.configure(simulation_length=TOTAL * DT)
    subnet.stimuli = [stimulus]
    subnet.configure()

    network = NetworkSet(subnets=[subnet], projections=[])
    network.configure()
    assert projection._horizon == HORIZON
    return network


def _initial_state():
    rng = np.random.RandomState(19)
    state = rng.uniform(0.01, 0.08, (2, N_NODES, 1)).astype(np.float64)
    state[0] = np.abs(state[0])
    return [state]


def _compiled():
    return NbHybridBackend().compile(_network())


def _assert_snapshot_exact(actual, expected):
    assert actual.keys() == expected.keys()
    for actual_state, expected_state in zip(actual["states"], expected["states"]):
        np.testing.assert_array_equal(actual_state, expected_state)
    assert actual["buffers"].keys() == expected["buffers"].keys()
    for name in actual["buffers"]:
        np.testing.assert_array_equal(actual["buffers"][name], expected["buffers"][name])


def test_resume_preserves_exact_absolute_timeline_delay_state_and_rng():
    """The resumed raw trajectory must be the exact tail of one uninterrupted run."""
    assert SPLIT % HORIZON != 0
    initial = _initial_state()

    full, full_snapshot = _compiled().run(
        TOTAL,
        chunk_size=1,
        initial_states=[state.copy() for state in initial],
        monitors=[Raw()],
        return_snapshot=True,
    )

    split_compiled = _compiled()
    _, split_snapshot = split_compiled.run(
        SPLIT,
        chunk_size=1,
        initial_states=[state.copy() for state in initial],
        monitors=[Raw()],
        return_snapshot=True,
    )
    resumed, resumed_snapshot = _compiled().resume(
        split_snapshot,
        TOTAL - SPLIT,
        chunk_size=1,
        monitors=[Raw()],
        return_snapshot=True,
    )

    full_times, full_data = full[0][0]
    resumed_times, resumed_data = resumed[0][0]
    np.testing.assert_array_equal(resumed_times, full_times[SPLIT:])
    np.testing.assert_array_equal(resumed_data, full_data[SPLIT:])
    _assert_snapshot_exact(resumed_snapshot, full_snapshot)


def test_resume_continues_temporal_average_window_and_absolute_times():
    """A split inside a TemporalAverage window must not reset or emit that window."""
    assert SPLIT % TAVG_STEPS != 0
    initial = _initial_state()

    full_monitor = TemporalAverage(period=TAVG_STEPS * DT)
    full, full_snapshot = _compiled().run(
        TOTAL,
        initial_states=[state.copy() for state in initial],
        monitors=[full_monitor],
        return_snapshot=True,
    )

    split_compiled = _compiled()
    first, split_snapshot = split_compiled.run(
        SPLIT,
        initial_states=[state.copy() for state in initial],
        monitors=[TemporalAverage(period=TAVG_STEPS * DT)],
        return_snapshot=True,
    )
    resumed, resumed_snapshot = _compiled().resume(
        split_snapshot,
        TOTAL - SPLIT,
        monitors=[TemporalAverage(period=TAVG_STEPS * DT)],
        return_snapshot=True,
    )

    full_times, full_data = full[0][0]
    first_times, first_data = first[0][0]
    resumed_times, resumed_data = resumed[0][0]
    completed_before_split = SPLIT // TAVG_STEPS

    np.testing.assert_array_equal(first_times, full_times[:completed_before_split])
    np.testing.assert_array_equal(first_data, full_data[:completed_before_split])
    np.testing.assert_array_equal(resumed_times, full_times[completed_before_split:])
    np.testing.assert_array_equal(resumed_data, full_data[completed_before_split:])
    _assert_snapshot_exact(resumed_snapshot, full_snapshot)


def test_resume_continues_bold_runtime_state_and_sampling_phase():
    """Checkpointing carries BOLD state/phase; Balloon-model semantics live elsewhere."""
    bold_steps = 5
    total = 21
    initial = _initial_state()

    full, _ = _compiled().run(
        total,
        chunk_size=1,
        initial_states=[state.copy() for state in initial],
        monitors=[Bold(period=bold_steps * DT)],
        return_snapshot=True,
    )

    split_compiled = _compiled()
    first, snapshot = split_compiled.run(
        SPLIT,
        chunk_size=1,
        initial_states=[state.copy() for state in initial],
        monitors=[Bold(period=bold_steps * DT)],
        return_snapshot=True,
    )
    resumed, _ = _compiled().resume(
        snapshot,
        total - SPLIT,
        chunk_size=1,
        monitors=[Bold(period=bold_steps * DT)],
        return_snapshot=True,
    )

    full_times, full_data = full[0][0]
    first_times, first_data = first[0][0]
    resumed_times, resumed_data = resumed[0][0]
    completed_before_split = SPLIT // bold_steps

    np.testing.assert_array_equal(first_times, full_times[:completed_before_split])
    np.testing.assert_array_equal(first_data, full_data[:completed_before_split])
    np.testing.assert_array_equal(resumed_times, full_times[completed_before_split:])
    np.testing.assert_array_equal(resumed_data, full_data[completed_before_split:])
