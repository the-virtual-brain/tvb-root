"""Regression contracts for backend-neutral hybrid helpers and Numba aliases."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from tvb.simulator.backend import hybrid_common as common
from tvb.simulator.hybrid import coupling
from tvb.simulator.monitors import (
    AfferentCoupling, GlobalAverage, Raw, SubSample, TemporalAverage,
)


@pytest.mark.parametrize("name", [
    "_CFUN_PARAM_ATTRS", "_CFUN_ATTR_TO_IDX", "_NAMED_PARAM_ALIASES",
    "_compute_chunk_size", "_aggregate_chunks_to_period", "_aggregate_raw_outputs",
    "_copy_monitor_states", "_apply_monitors", "_validate_multi_dt",
    "_stim_master_grid", "_can_merge_subnets", "_merge_subnet_outputs",
    "_merge_and_global_average", "_merge_and_spatial_average", "_cfun_type",
])
def test_numba_keeps_identical_helper_aliases(name):
    from tvb.simulator.backend import nb_hybrid

    assert getattr(nb_hybrid, name) is getattr(common, name)


def test_numba_keeps_static_named_parameter_alias():
    from tvb.simulator.backend.nb_hybrid import NbHybridBackend

    assert NbHybridBackend._resolve_named_params is common._resolve_named_params
    assert NbHybridBackend()._resolve_named_params is common._resolve_named_params


def test_monitor_chunk_size_and_partial_raw_aggregation():
    assert common._compute_chunk_size([Raw(), AfferentCoupling()], 0.1) == 1
    assert common._compute_chunk_size([
        TemporalAverage(period=0.6), TemporalAverage(period=0.4), AfferentCoupling(),
    ], 0.1) == 2
    for monitor in (Raw(), SubSample(period=0.4)):
        assert common._compute_chunk_size([TemporalAverage(period=0.6), monitor], 0.1) == 1
    times = np.arange(1, 6) * 0.1
    data = np.arange(5, dtype=np.float32).reshape(5, 1, 1, 1)
    raw = [(times, data, data + 10)]
    assert common._aggregate_raw_outputs(raw, 1) is raw
    [(t, d, c)] = common._aggregate_raw_outputs(raw, 2)
    np.testing.assert_allclose(t, [0.15, 0.35, 0.5])
    np.testing.assert_allclose(d.ravel(), [0.5, 2.5, 4])
    np.testing.assert_allclose(c.ravel(), [10.5, 12.5, 14])
    t, d = common._aggregate_chunks_to_period(times, data, 0.2, 0.1, 1)
    np.testing.assert_allclose(t, [0.1, 0.3])
    np.testing.assert_allclose(d.ravel(), [0.5, 2.5])
    t, d = common._aggregate_chunks_to_period(times[:1], data[:1], 0.2, 0.1, 1)
    np.testing.assert_array_equal(t, times[:1])
    np.testing.assert_array_equal(d, data[:1])


def _info(indices, dt=0.1, nvoi=1):
    return NS(node_indices=np.asarray(indices) if indices is not None else None,
              dt=dt, model=NS(variables_of_interest=list(range(nvoi))))


def test_merge_guards_and_connectome_ordered_averages():
    infos = [_info([2, 0]), _info([1])]
    assert common._can_merge_subnets(infos)
    for invalid in ([], [_info(None)], [_info([0]), _info([1], dt=0.2)],
                    [_info([0], dt=None)], [_info([0]), _info([1], nvoi=2)]):
        assert not common._can_merge_subnets(invalid)
    times = np.array([0.1, 0.2])
    first = np.array([[30, 10], [60, 40]], dtype=np.float32).reshape(2, 1, 2, 1)
    second = np.array([20, 50], dtype=np.float32).reshape(2, 1, 1, 1)
    outputs = [(times, first), (times, second)]
    [(t, data)] = common._merge_subnet_outputs(outputs, infos)
    np.testing.assert_array_equal(t, times)
    np.testing.assert_array_equal(data[:, 0, :, 0], [[10, 20, 30], [40, 50, 60]])
    [(t, data)] = common._merge_and_global_average(outputs, infos)
    np.testing.assert_array_equal(data.ravel(), [20, 50])
    [(t, data)] = common._merge_and_spatial_average(outputs, infos, [[0.5, 0, 0.5]])
    np.testing.assert_array_equal(data.ravel(), [20, 50])
    raw = [(t, d, d + 1) for t, d in outputs]
    monitors = [Raw(), SubSample(period=0.2), GlobalAverage(period=0.1),
                AfferentCoupling()]
    dispatched = common._apply_monitors(raw, monitors, 0.1, subnet_infos=infos)
    np.testing.assert_array_equal(dispatched[0][0][1], first)
    np.testing.assert_array_equal(dispatched[1][0][0], [0.2])
    np.testing.assert_array_equal(dispatched[2][0][1].ravel(), [20, 50])
    # AfferentCoupling inherits Raw: its streams stay per-subnet.
    np.testing.assert_array_equal(dispatched[3][0][1], first + 1)
    np.testing.assert_array_equal(dispatched[3][1][1], second + 1)


def test_monitor_runtime_copy_preserves_fields_and_isolates_arrays():
    monitor = NS(stock=np.arange(4), transient="kept")
    states = {("temporal_average", 0, 0): {"monitor": monitor, "step": 3}}
    copied = common._copy_monitor_states(states)
    runtime = copied[("temporal_average", 0, 0)]
    assert runtime["step"] == 3
    assert runtime["monitor"].transient == "kept"
    runtime["monitor"].stock[0] = 99
    assert monitor.stock[0] == 0
    runtime["step"] = 4
    assert states[("temporal_average", 0, 0)]["step"] == 3
    assert common._copy_monitor_states(None) is None


def _network(dts=(0.1, 0.2)):
    subnets = [NS(name=name, scheme=NS(dt=dt), projections=[])
               for name, dt in zip(("fast", "slow"), dts)]
    return NS(subnets=subnets, projections=[])


def test_master_clock_and_projection_source_dt_validation():
    network = _network()
    assert common._validate_multi_dt(network) == 0.1
    with pytest.raises(ValueError, match="integer multiples"):
        common._validate_multi_dt(_network((0.1, 0.15)))
    projection = NS(source=network.subnets[1], dt=0.1)
    network.projections.append(projection)
    with pytest.raises(ValueError, match="must match source subnetwork 'slow'"):
        common._validate_multi_dt(network)
    projection.dt = 0.2
    assert common._validate_multi_dt(network) == 0.1
    network.projections.clear()
    network.subnets[0].projections.append(NS(dt=0.2))
    with pytest.raises(ValueError, match="must match source subnetwork 'fast'"):
        common._validate_multi_dt(network)


def test_stim_master_grid_copies_pattern_without_mutating_original():
    from tvb.simulator.hybrid.stimulus_utils import constant_stim
    from tvb.simulator.hybrid.subnetwork import Subnetwork
    from tvb.simulator.integrators import HeunDeterministic
    from tvb.simulator.models.linear import Linear

    model = Linear()
    model.configure()
    subnet = Subnetwork(name="slow", model=model, nnodes=2,
                        scheme=HeunDeterministic(dt=0.2))
    stim = constant_stim(subnet, amplitude=2.0, simulation_length=0.6)
    original_time = stim.time.copy()
    copied = common._stim_master_grid(stim, 0.1, 8)
    assert copied is not stim
    assert copied.stimulus is not stim.stimulus
    assert copied.dt == 0.1
    np.testing.assert_allclose(copied.time, np.arange(9) * 0.1)
    np.testing.assert_array_equal(stim.time, original_time)
    assert stim.dt == 0.2
    assert common._stim_master_grid(copied, 0.1, 8) is copied


@pytest.mark.parametrize("cfun, ncvar, expected", [
    (None, 1, "none"), (coupling.Linear(), 1, "linear"),
    (coupling.Scaling(), 1, "scaling"), (coupling.Sigmoidal(), 1, "sigmoidal"),
    (coupling.Difference(), 1, "difference"), (coupling.Kuramoto(), 1, "kuramoto"),
    (coupling.HyperbolicTangent(), 1, "tanh"),
    (coupling.SigmoidalJansenRit(), 2, "sigmoidal_jr"),
    (coupling.SigmoidalJansenRit(), 1, "sigmoidal_jr_legacy"),
    (coupling.PreSigmoidal(), 1, "pre_sigmoidal"),
    (coupling.PreSigmoidal(dynamic=True), 2, "pre_sigmoidal_dynamic"),
])
def test_cfun_type_dispatch(cfun, ncvar, expected):
    assert common._cfun_type(NS(cfun=cfun, source_cvar=np.arange(ncvar))) == expected


def test_cfun_subclasses_still_rejected():
    class Unsupported(coupling.Linear):
        pass

    with pytest.raises(NotImplementedError, match="Unsupported"):
        common._cfun_type(NS(cfun=Unsupported(), source_cvar=np.array([0])))


def test_named_parameter_resolution_preserves_aliases_projection_and_model_order():
    network = _network()
    network.subnets[0].projections = [NS(name="intra", cfun=coupling.Linear())]
    network.projections = [NS(source=network.subnets[0], target=network.subnets[1],
                              cfun=coupling.Scaling())]
    descriptor, values = common._resolve_named_params(network, {
        "coupling_scale": [1, 2], "fast.intra.b": [3, 4], "slow.gamma": [-1, -2],
    })
    assert descriptor == [
        {"type": "cfun", "projection": "fast_to_slow", "param_idx": 0},
        {"type": "cfun", "projection": "intra", "param_idx": 1},
        {"type": "model", "subnet": "slow", "param": "gamma"},
    ]
    assert values.dtype == np.float32
    np.testing.assert_array_equal(values, [[1, 3, -1], [2, 4, -2]])
    with pytest.raises(ValueError, match="same length"):
        common._resolve_named_params(network, {"scale": [1], "slow.gamma": [1, 2]})
    with pytest.raises(ValueError, match="Cannot resolve"):
        common._resolve_named_params(network, {"not_a_parameter": [1]})
