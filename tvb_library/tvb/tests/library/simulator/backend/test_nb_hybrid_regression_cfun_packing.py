"""Regressions for named coupling parameters and Numba packed layouts."""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from tvb.simulator.backend.nb_hybrid import NbHybridBackend, _cfun_params
from tvb.simulator.hybrid.coupling import (
    HyperbolicTangent,
    Linear,
    PreSigmoidal,
    Sigmoidal,
    SigmoidalJansenRit,
)
from tvb.simulator.hybrid.intra_projection import IntraProjection
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin


DT = 0.01
NSTEP = 8
NODES = 3


def _array(value):
    return np.array([value], dtype=np.float64)


def _linear(**overrides):
    values = dict(a=0.11, b=0.22)
    values.update(overrides)
    return Linear(**{name: _array(value) for name, value in values.items()})


def _sigmoidal(**overrides):
    values = dict(a=0.31, sigma=0.42, midpoint=0.53, cmin=0.64, cmax=0.75)
    values.update(overrides)
    return Sigmoidal(**{name: _array(value) for name, value in values.items()})


def _sjr(use_classic, **overrides):
    # Set every public numeric attribute, including those unused by the selected mode.
    values = dict(a=0.21, e0=0.32, r=0.43, v0=0.54,
                  cmin=0.65, cmax=0.76, midpoint=0.87)
    values.update(overrides)
    return SigmoidalJansenRit(
        use_classic=use_classic,
        **{name: _array(value) for name, value in values.items()},
    )


def _tanh(**overrides):
    values = dict(a=0.27, b=0.38, midpoint=0.49, sigma=0.61)
    values.update(overrides)
    return HyperbolicTangent(
        **{name: _array(value) for name, value in values.items()}
    )


def _pre_sigmoidal(dynamic=False, **overrides):
    values = dict(H=0.17, Q=0.29, G=0.41, P=0.53, theta=0.67)
    values.update(overrides)
    return PreSigmoidal(
        dynamic=dynamic,
        **{name: _array(value) for name, value in values.items()},
    )


PACKING_CASES = [
    pytest.param(_linear, [0.11, 0.22], id="linear"),
    pytest.param(_sigmoidal, [0.31, 0.42, 0.53, 0.64, 0.75], id="sigmoidal"),
    pytest.param(lambda: _sjr(True), [0.21, 0.65, 0.76, 0.43, 0.87],
                 id="sigmoidal-jansen-rit-classic"),
    pytest.param(lambda: _sjr(False), [0.21, 0.32, 0.43, 0.54],
                 id="sigmoidal-jansen-rit-legacy"),
    pytest.param(_tanh, [0.27, 0.38, 0.49, 0.61], id="hyperbolic-tangent"),
    pytest.param(lambda: _pre_sigmoidal(False), [0.17, 0.29, 0.41, 0.53, 0.67],
                 id="pre-sigmoidal-static"),
    pytest.param(lambda: _pre_sigmoidal(True), [0.17, 0.29, 0.41, 0.53, 0.0],
                 id="pre-sigmoidal-dynamic"),
]


SWEEP_CASES = [
    pytest.param(_linear, "a", 0, [0.2, 0.8], id="linear-a"),
    pytest.param(_linear, "b", 1, [-0.2, 0.3], id="linear-b"),
    pytest.param(_sigmoidal, "a", 0, [0.4, 1.2], id="sigmoidal-a"),
    pytest.param(_sigmoidal, "sigma", 1, [0.3, 1.1], id="sigmoidal-sigma"),
    pytest.param(_sigmoidal, "midpoint", 2, [-0.2, 0.4], id="sigmoidal-midpoint"),
    pytest.param(_sigmoidal, "cmin", 3, [-0.3, 0.1], id="sigmoidal-cmin"),
    pytest.param(_sigmoidal, "cmax", 4, [0.7, 1.3], id="sigmoidal-cmax"),
    pytest.param(lambda **kw: _sjr(True, **kw), "a", 0, [0.2, 0.8], id="sjr-classic-a"),
    pytest.param(lambda **kw: _sjr(True, **kw), "cmin", 1, [-0.2, 0.2],
                 id="sjr-classic-cmin"),
    pytest.param(lambda **kw: _sjr(True, **kw), "cmax", 2, [0.7, 1.3],
                 id="sjr-classic-cmax"),
    pytest.param(lambda **kw: _sjr(True, **kw), "r", 3, [0.4, 1.1],
                 id="sjr-classic-r"),
    pytest.param(lambda **kw: _sjr(True, **kw), "midpoint", 4, [-0.2, 0.4],
                 id="sjr-classic-midpoint"),
    pytest.param(lambda **kw: _sjr(False, **kw), "a", 0, [0.2, 0.8], id="sjr-legacy-a"),
    pytest.param(lambda **kw: _sjr(False, **kw), "e0", 1, [0.3, 0.9], id="sjr-legacy-e0"),
    pytest.param(lambda **kw: _sjr(False, **kw), "r", 2, [0.4, 1.1], id="sjr-legacy-r"),
    pytest.param(lambda **kw: _sjr(False, **kw), "v0", 3, [-0.2, 0.4], id="sjr-legacy-v0"),
    pytest.param(_tanh, "a", 0, [0.2, 0.8], id="tanh-a"),
    pytest.param(_tanh, "b", 1, [0.4, 1.2], id="tanh-b"),
    pytest.param(_tanh, "midpoint", 2, [-0.2, 0.4], id="tanh-midpoint"),
    pytest.param(_tanh, "sigma", 3, [0.3, 1.1], id="tanh-sigma"),
    pytest.param(lambda **kw: _pre_sigmoidal(False, **kw), "H", 0, [0.2, 0.8],
                 id="pre-static-H"),
    pytest.param(lambda **kw: _pre_sigmoidal(False, **kw), "Q", 1, [-0.2, 0.5],
                 id="pre-static-Q"),
    pytest.param(lambda **kw: _pre_sigmoidal(False, **kw), "G", 2, [0.4, 1.2],
                 id="pre-static-G"),
    pytest.param(lambda **kw: _pre_sigmoidal(False, **kw), "P", 3, [0.4, 1.2],
                 id="pre-static-P"),
    pytest.param(lambda **kw: _pre_sigmoidal(False, **kw), "theta", 4, [-0.2, 0.4],
                 id="pre-static-theta"),
    pytest.param(lambda **kw: _pre_sigmoidal(True, **kw), "H", 0, [0.2, 0.8],
                 id="pre-dynamic-H"),
    pytest.param(lambda **kw: _pre_sigmoidal(True, **kw), "Q", 1, [-0.2, 0.5],
                 id="pre-dynamic-Q"),
    pytest.param(lambda **kw: _pre_sigmoidal(True, **kw), "G", 2, [0.4, 1.2],
                 id="pre-dynamic-G"),
    pytest.param(lambda **kw: _pre_sigmoidal(True, **kw), "P", 3, [0.4, 1.2],
                 id="pre-dynamic-P"),
]


def _network(cfun):
    model = MontbrioPazoRoxin()
    model.configure()
    subnetwork = Subnetwork(
        name="ctx",
        model=model,
        scheme=HeunDeterministic(dt=DT),
        nnodes=NODES,
    )
    weights = sp.csr_matrix(np.array([
        [0.0, 0.35, 0.15],
        [0.20, 0.0, 0.30],
        [0.25, 0.10, 0.0],
    ]))
    source_cvar = np.array([0, 1] if getattr(cfun, "n_cvar_in", 1) == 2 else [0],
                           dtype=np.int_)
    projection = IntraProjection(
        source_cvar=source_cvar,
        target_cvar=np.array([0], dtype=np.int_),
        weights=weights,
        lengths=sp.csr_matrix((NODES, NODES), dtype=np.float64),
        cv=1.0,
        dt=DT,
        scale=0.7,
        cfun=cfun,
    )
    subnetwork.projections = [projection]
    subnetwork.configure()
    network = NetworkSet(subnets=[subnetwork], projections=[])
    network.configure()
    return network


def _initial_state():
    return np.array([
        [[0.12], [0.18], [0.24]],
        [[-0.35], [0.05], [0.45]],
    ], dtype=np.float64)


def _single_average(factory, attribute, value):
    # A fresh object graph is intentional: no descriptor mutation or buffers are shared.
    network = _network(factory(**{attribute: value}))
    result = NbHybridBackend().run_network(
        network,
        nstep=NSTEP,
        chunk_size=1,
        initial_states=[_initial_state()],
    )[0]
    return result[1].mean(axis=0), result[2].mean(axis=0)


@pytest.mark.parametrize("factory,packed_prefix", PACKING_CASES)
def test_cfun_params_has_explicit_public_attribute_layout(factory, packed_prefix):
    cfun = factory()
    projection = SimpleNamespace(cfun=cfun, source_cvar=np.array([0, 1]))
    expected = np.zeros(16, dtype=np.float32)
    expected[:len(packed_prefix)] = packed_prefix

    packed = _cfun_params(projection)

    assert packed.dtype == np.float32
    np.testing.assert_array_equal(packed, expected)


@pytest.mark.parametrize("factory,attribute,packed_slot,_values", SWEEP_CASES)
def test_named_descriptor_serializes_to_the_packed_kernel_slot(
    factory, attribute, packed_slot, _values
):
    network = _network(factory())
    projection = network.subnets[0].projections[0]
    cfun = projection.cfun
    packed_before = _cfun_params(projection).copy()

    descriptors, values = NbHybridBackend._resolve_named_params(
        network, {f"ctx.intra.{attribute}": np.array([0.125, 0.875])}
    )

    assert len(descriptors) == 1
    assert descriptors[0]["type"] == "cfun"
    assert descriptors[0]["projection"] == "intra"
    np.testing.assert_array_equal(values[:, 0], np.array([0.125, 0.875], dtype=np.float32))

    # Descriptor indices are a public, backward-compatible namespace and need
    # not equal packed slots. Serialization must still update the named slot.
    NbHybridBackend._cfun_set_param(cfun, descriptors[0]["param_idx"], 0.875)
    packed_after = _cfun_params(projection)
    expected = packed_before.copy()
    expected[packed_slot] = np.float32(0.875)
    np.testing.assert_array_equal(packed_after, expected)


@pytest.mark.parametrize("factory,attribute,_packed_slot,sweep_values", SWEEP_CASES)
def test_cpu_prange_named_sweep_matches_fresh_single_runs(
    factory, attribute, _packed_slot, sweep_values
):
    key = f"ctx.intra.{attribute}"
    parallel = NbHybridBackend().sweep(
        _network(factory()),
        params={key: np.asarray(sweep_values, dtype=np.float32)},
        nstep=NSTEP,
        backend="cpu",
        n_workers=2,
        initial_states=[_initial_state()],
    )
    references = [
        _single_average(factory, attribute, value) for value in sweep_values
    ]

    state_delta = np.max(np.abs(references[0][0] - references[1][0]))
    coupling_delta = np.max(np.abs(references[0][1] - references[1][1]))
    assert max(state_delta, coupling_delta) > 1e-6, (
        f"{type(factory()).__name__}.{attribute} reference runs are not sensitive "
        f"to {sweep_values}"
    )

    for row, (expected_state, expected_coupling) in enumerate(references):
        np.testing.assert_allclose(
            parallel.tavg["ctx"][row].mean(axis=0), expected_state,
            rtol=2e-5, atol=2e-6,
            err_msg=f"prange row {row} did not apply named parameter {key}",
        )
        np.testing.assert_allclose(
            parallel.ctavg["ctx"][row].mean(axis=0), expected_coupling,
            rtol=2e-5, atol=2e-6,
            err_msg=f"prange coupling row {row} did not apply named parameter {key}",
        )
