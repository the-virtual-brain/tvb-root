"""Regression tests for restoring caller-owned parameters after CPU sweeps."""

from types import SimpleNamespace

import numpy as np
import pytest

from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.hybrid.coupling import Linear
from tvb.simulator.integrators import EulerDeterministic
from tvb.simulator.models.linear import Linear as LinearModel


class _SimulationFailure(RuntimeError):
    pass


def _case(target):
    model = LinearModel()
    if target == "model-array":
        original = np.array(1.25, dtype=np.float32)
        model.tau = original
        cfun = Linear()
        parameter = lambda: model.tau
        named_param = "unit.tau"
        descriptor = {"type": "model", "subnet": "unit", "param": "tau"}
    elif target == "model-scalar":
        original = np.float32(1.25)
        model.tau = original
        cfun = Linear()
        parameter = lambda: model.tau
        named_param = "unit.tau"
        descriptor = {"type": "model", "subnet": "unit", "param": "tau"}
    else:
        original = np.array([1.25], dtype=np.float32)
        cfun = Linear()
        cfun.a = original
        parameter = lambda: cfun.a
        named_param = "unit.intra.a"
        descriptor = {"type": "cfun", "projection": "intra", "param_idx": 0}

    projection = SimpleNamespace(name=None, cfun=cfun)
    subnet = SimpleNamespace(
        name="unit",
        model=model,
        scheme=EulerDeterministic(dt=0.1),
        projections=[projection],
    )
    network = SimpleNamespace(subnets=[subnet], projections=[])
    return network, parameter, named_param, descriptor


def _snapshot(value):
    if isinstance(value, np.ndarray):
        return (type(value), value.dtype, value.shape, value.strides, value.tobytes())
    array = np.asarray(value)
    return (type(value), array.dtype, array.shape, array.tobytes())


def _assert_restored(parameter, before):
    restored = parameter()
    assert _snapshot(restored) == before


def _invoke(api, backend, network, named_param, descriptor, values):
    if api == "run_sweep":
        return backend.run_sweep(
            network,
            sweep_values=values,
            sweep_descriptor=[descriptor],
            nstep=1,
        )
    return backend.sweep(
        network,
        params={named_param: values},
        nstep=1,
        backend="cpu",
        n_workers=1,
    )


@pytest.mark.parametrize("api", ["run_sweep", "sweep"])
@pytest.mark.parametrize("target", ["model-array", "model-scalar", "cfun-array"])
def test_sweep_restores_parameter_after_simulation_failure(monkeypatch, api, target):
    network, parameter, named_param, descriptor = _case(target)
    before = _snapshot(parameter())
    backend = NbHybridBackend()
    mutations = []

    def fail_after_mutation(*args, **kwargs):
        mutations.append(float(np.asarray(parameter()).reshape(-1)[0]))
        assert mutations[-1] == pytest.approx(3.5)
        raise _SimulationFailure("deterministic failure after sweep mutation")

    monkeypatch.setattr(backend, "run_network", fail_after_mutation)

    with pytest.raises(_SimulationFailure, match="deterministic failure"):
        _invoke(
            api,
            backend,
            network,
            named_param,
            descriptor,
            np.array([3.5], dtype=np.float32),
        )

    assert mutations == [3.5]
    _assert_restored(parameter, before)


@pytest.mark.parametrize("api", ["run_sweep", "sweep"])
@pytest.mark.parametrize("target", ["model-array", "model-scalar", "cfun-array"])
def test_sweep_restores_parameter_after_normal_completion(monkeypatch, api, target):
    network, parameter, named_param, descriptor = _case(target)
    before = _snapshot(parameter())
    backend = NbHybridBackend()
    mutations = []

    def complete_after_mutation(*args, **kwargs):
        mutations.append(float(np.asarray(parameter()).reshape(-1)[0]))
        data = np.zeros((1, 1, 1, 1), dtype=np.float32)
        return [(np.array([0.5]), data, data.copy())]

    monkeypatch.setattr(backend, "run_network", complete_after_mutation)
    _invoke(
        api,
        backend,
        network,
        named_param,
        descriptor,
        np.array([3.5], dtype=np.float32),
    )

    assert mutations == [3.5]
    _assert_restored(parameter, before)
