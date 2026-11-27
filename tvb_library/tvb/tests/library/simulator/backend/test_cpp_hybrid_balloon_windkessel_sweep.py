"""Public BW sweep streams retain scalar-run semantics and sweep row order."""

import numpy as np
import pytest

from tvb.simulator.backend.cpp_hybrid.backend import CppHybridBackend
from .test_cpp_hybrid_local_balloon_windkessel import (
    _initial, _monitor, _network, _subnet,
)


def _network_with_slope(slope):
    subnet = _subnet()
    subnet.model.a = np.full(3, slope)
    return _network(subnet)


@pytest.mark.parametrize("width", [1, 4, 8, 16])
def test_global_sweep_matches_independent_public_runs(width):
    slopes = np.array([0.001, 0.003, 0.005, 0.007, 0.009], dtype=np.float32)
    backend = CppHybridBackend()
    result = backend.sweep(
        _network_with_slope(float(slopes[0])), {"regions.a": slopes}, nstep=80,
        width=width, n_workers=2, monitor="none", initial_states=[_initial()],
        bw={"global": _monitor()},
    )
    times, data = result.bw_global
    assert times.dtype == np.float64
    assert data.dtype == np.float32
    assert data.shape == (len(slopes), len(times), 1, 3, 1)
    np.testing.assert_array_equal(result.sweep_values[:, 0], slopes)
    for row, slope in enumerate(slopes):
        outputs = backend.run_network(
            _network_with_slope(float(slope)), 80,
            initial_states=[_initial()], monitors=[_monitor()],
        )
        expected_times, expected_data = outputs[0][0]
        np.testing.assert_array_equal(times, expected_times)
        np.testing.assert_allclose(data[row], expected_data, rtol=1e-6, atol=1e-10)
    assert np.max(np.abs(data)) > 0
    assert not np.array_equal(data[0], data[-1])


def _mixed_network(slope):
    fast = _subnet("fast", node_indices=[2, 0, 1])
    slow = _subnet("slow", cadence=2, node_indices=[5, 3, 4])
    fast.model.a = np.full(3, slope)
    return _network(fast, slow)


def test_mixed_clocks_global_local_and_empty_windows():
    slopes = np.array([0.002, 0.004, 0.006], dtype=np.float32)
    backend = CppHybridBackend()
    bw = {"global": _monitor(period=20),
          "local": {"slow": _monitor(period=40)}}
    result = backend.sweep(_mixed_network(slopes[0]), {"fast.a": slopes},
                           nstep=80, width=4, bw=bw,
                           initial_states=[_initial(), _initial()])
    assert set(result.bw_local) == {"slow"}
    for i, value in enumerate(slopes):
        output, local = backend.run_network(
            _mixed_network(value), 80, monitors=[_monitor(period=20)],
            subnet_monitors={"slow": [_monitor(period=40)]},
            initial_states=[_initial(), _initial()],
        )
        for actual, expected in ((result.bw_global, output[0][0]),
                                 (result.bw_local["slow"], local["slow"][0])):
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_allclose(actual[1][i], expected[1], rtol=1e-6, atol=1e-10)
    empty = backend.sweep(_mixed_network(slopes[0]), {"fast.a": slopes},
                          nstep=3, width=4, monitor="none", bw=bw,
                          initial_states=[_initial(), _initial()])
    assert empty.bw_global[1].shape == (3, 0, 1, 6, 1)
    assert empty.bw_local["slow"][1].shape == (3, 0, 1, 3, 1)


def test_sweep_permutation_workers_and_positive_padding():
    values = np.linspace(0.8, 1.6, 17, dtype=np.float32)
    backend = CppHybridBackend()
    options = dict(nstep=80, width=16, monitor="none", bw={"global": _monitor()},
                   initial_states=[_initial()])
    first = backend.sweep(_network(), {"regions.tau": values}, n_workers=1, **options)
    reversed_result = backend.sweep(_network(), {"regions.tau": values[::-1].copy()},
                                    n_workers=2, **options)
    repeated = backend.sweep(_network(), {"regions.tau": values}, n_workers=2, **options)
    np.testing.assert_array_equal(first.bw_global[1], reversed_result.bw_global[1][::-1])
    np.testing.assert_array_equal(first.bw_global[1], repeated.bw_global[1])


@pytest.mark.parametrize("bw", [{}, {"other": 1}, {"local": {}},
                                  {"local": {"missing": _monitor()}},
                                  {"global": "bold"}, {"local": []}])
def test_invalid_bw_configuration(bw):
    with pytest.raises((TypeError, ValueError)):
        CppHybridBackend().sweep(_network(), {"regions.tau": [1.0]}, nstep=8, bw=bw)


@pytest.mark.parametrize("monitor", ["raw", "subsample", "bold", "features"])
def test_unsupported_legacy_combinations_are_not_silent(monitor):
    with pytest.raises(ValueError, match="bw supports"):
        CppHybridBackend().sweep(_network(), {"regions.tau": [1.0]}, nstep=8,
                                 monitor=monitor, bw={"global": _monitor()})


def test_notebook_cartesian_grid_and_paired_rows():
    from tvb.simulator.backend.cpp_hybrid import notebook

    slopes = np.array([0.001, 0.004], dtype=np.float32)
    taus = np.array([0.8, 1.0, 1.2], dtype=np.float32)
    options = dict(nstep=80, width=4, workers=1, monitor="none",
                   initial_states=[_initial()], bw={"global": _monitor()})
    grid = notebook.sweep(_network(), grid=True, **options,
                          **{"regions.a": slopes, "regions.tau": taus})
    np.testing.assert_array_equal(grid.sweep_values,
                                  [[a, t] for a in slopes for t in taus])
    paired = notebook.sweep(_network(), grid=False, **options,
                            **{"regions.a": grid.sweep_values[:, 0],
                               "regions.tau": grid.sweep_values[:, 1]})
    np.testing.assert_array_equal(grid.bw_global[1], paired.bw_global[1])
    backend = CppHybridBackend()
    for row, (slope, tau) in enumerate(grid.sweep_values):
        network = _network_with_slope(float(slope))
        network.subnets[0].model.tau = np.array([float(tau)])
        network.subnets[0].model.configure()
        expected = backend.run_network(network, 80, initial_states=[_initial()],
                                       monitors=[_monitor()])[0][0]
        np.testing.assert_array_equal(grid.bw_global[0], expected[0])
        np.testing.assert_allclose(grid.bw_global[1][row], expected[1],
                                   rtol=1e-6, atol=1e-10)


def test_invalid_clock_and_calibration():
    backend = CppHybridBackend()
    fast = _subnet("fast")
    incompatible = _subnet("slow", cadence=3)
    network = _network(fast, incompatible)
    with pytest.raises(ValueError):
        backend.sweep(network, {"fast.tau": [1.0]}, nstep=12,
                      bw={"local": {"slow": _monitor()}})
    with pytest.raises(ValueError, match="finite"):
        backend.sweep(_network(), {"regions.tau": [1.0]}, nstep=12,
                      bw={"global": _monitor(drive_scale=float("inf"))})


def test_single_source_permuted_indices_and_tavg_coexistence():
    subnet = _subnet(node_indices=[2, 0, 1])
    network = _network(subnet)
    values = np.array([0.8, 1.2], dtype=np.float32)
    backend = CppHybridBackend()
    kwargs = dict(nstep=80, width=4, initial_states=[_initial()])
    legacy = backend.sweep(network, {"regions.tau": values}, **kwargs)
    both = backend.sweep(network, {"regions.tau": values},
                         bw={"global": _monitor(), "local": {"regions": _monitor()}},
                         **kwargs)
    np.testing.assert_array_equal(legacy.tavg["regions"], both.tavg["regions"])
    np.testing.assert_array_equal(both.bw_global[1], both.bw_local["regions"][1])
    for i, tau in enumerate(values):
        independent = _network(_subnet(node_indices=[2, 0, 1]))
        independent.subnets[0].model.tau = np.array([float(tau)])
        independent.subnets[0].model.configure()
        expected = backend.run_network(independent, 80, monitors=[_monitor()],
                                       initial_states=[_initial()])[0][0]
        np.testing.assert_array_equal(both.bw_global[0], expected[0])
        np.testing.assert_allclose(both.bw_global[1][i], expected[1],
                                   rtol=1e-6, atol=1e-10)


def _coupled_network(gain):
    from scipy.sparse import csr_matrix
    from tvb.simulator.hybrid.coupling import Linear
    from tvb.simulator.hybrid.inter_projection import InterProjection
    from tvb.simulator.hybrid.network import NetworkSet

    target = _subnet("target", node_indices=[2, 0, 1])
    source = _subnet("source", node_indices=[5, 3, 4])
    weights = csr_matrix(np.eye(3, dtype=np.float32) * 0.05)
    projection = InterProjection(source=source, target=target,
                                 source_cvar=np.array([0]), target_cvar=np.array([0]),
                                 weights=weights, lengths=weights * 0,
                                 cfun=Linear(a=np.array([float(gain)])),
                                 scale=1.0, cv=1.0, dt=2.5)
    return NetworkSet(subnets=[target, source], projections=[projection]).configure()


def test_coupling_sweep_first_local_placement_and_global():
    values = np.array([0.0, 0.2, 0.4, 0.6, 0.8], dtype=np.float32)
    backend = CppHybridBackend()
    monitor = lambda: _monitor(variables_of_interest=np.array([1]))
    opts = dict(nstep=80, width=4, initial_states=[_initial(), _initial()])
    result = backend.sweep(_coupled_network(0), {"coupling_scale": values},
                           bw={"global": monitor(), "local": {"target": monitor()}}, **opts)
    control = backend.sweep(_coupled_network(0), {"source_to_target.a": values}, **opts)
    np.testing.assert_array_equal(result.tavg["source"], control.tavg["source"])
    np.testing.assert_array_equal(result.tavg["target"], control.tavg["target"])
    for i, value in enumerate(values):
        global_output, local = backend.run_network(
            _coupled_network(value), 80, initial_states=[_initial(), _initial()],
            monitors=[monitor()], subnet_monitors={"target": [monitor()]})
        for actual, expected in ((result.bw_global, global_output[0][0]),
                                 (result.bw_local["target"], local["target"][0])):
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_allclose(actual[1][i], expected[1], rtol=1e-6, atol=1e-10)
    assert np.max(np.abs(result.bw_local["target"][1][0] -
                         result.bw_local["target"][1][-1])) > 1e-9


@pytest.mark.parametrize("explicit", [False, True])
def test_stochastic_partial_group_seed_padding_and_real_lane_parity(monkeypatch, explicit):
    from tvb.simulator import integrators, noise
    from tvb.simulator.backend.cpp_hybrid import backend as backend_module

    subnet = _subnet()
    subnet.scheme = integrators.EulerStochastic(
        dt=2.5, noise=noise.Additive(nsig=np.array([1e-5]), noise_seed=123))
    network = _network(subnet)
    values = np.array([0.8, 0.9, 1.0, 1.1, 1.2], dtype=np.float32)
    seeds = np.array([101, 102, 103, 104, 105], dtype=np.uint64) if explicit else None
    captured = []
    native = backend_module.Sim

    class TracedSim:
        def __init__(self, width):
            self.sim = native(width)
        def __getattr__(self, name):
            return getattr(self.sim, name)
        def set_subnet_rng(self, index, keys, *args, **kwargs):
            captured.append(list(keys) if isinstance(keys, list) else keys)
            return self.sim.set_subnet_rng(index, keys, *args, **kwargs)

    monkeypatch.setattr(backend_module, "Sim", TracedSim)
    backend = CppHybridBackend()
    options = dict(nstep=80, monitor="none", initial_states=[_initial()],
                   noise_seeds=seeds, bw={"local": {"regions": _monitor()}})
    wide = backend.sweep(network, {"regions.tau": values}, width=4, n_workers=1, **options)
    assert captured[-1] == [captured[-1][0]] * 4
    captured.clear()
    scalar = backend.sweep(network, {"regions.tau": values}, width=1, n_workers=2, **options)
    np.testing.assert_array_equal(wide.bw_local["regions"][0], scalar.bw_local["regions"][0])
    np.testing.assert_allclose(wide.bw_local["regions"][1], scalar.bw_local["regions"][1],
                               rtol=1e-6, atol=1e-10)


def test_notebook_sweep_narrows_parameter_hints():
    from tvb.simulator.backend.cpp_hybrid import notebook

    with pytest.raises(ValueError) as exc_info:
        notebook.sweep(_network(), nstep=8, width=1, workers=1,
                       bw={"other": 1}, **{"regions.tau": [1.0]})
    assert "Try one of: coupling" not in str(exc_info.value)

    with pytest.raises(ValueError, match="Try one of:"):
        notebook.sweep(_network(), nstep=8, width=1, workers=1,
                       **{"unknown_param": [1.0]})
