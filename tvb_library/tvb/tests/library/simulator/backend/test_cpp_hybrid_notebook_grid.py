"""Cartesian notebook sweeps and their alignment with sweep results."""

import numpy as np
import pytest

from tvb.simulator.backend.cpp_hybrid import notebook as nb
from scipy import sparse

from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.hybrid.coupling import Linear
from tvb.simulator.hybrid.inter_projection import InterProjection
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.integrators import HeunDeterministic


def build_coupled_net(n=2):
    subnets = []
    for name in ("A", "B"):
        model = MontbrioPazoRoxin()
        model.configure()
        subnets.append(Subnetwork(name=name, model=model,
                                  scheme=HeunDeterministic(dt=0.1), nnodes=n))
    proj = InterProjection(
        source=subnets[0], target=subnets[1], source_cvar=np.array([0], np.int32),
        target_cvar=np.array([0], np.int32),
        weights=sparse.csr_matrix(np.full((n, n), 0.01) - np.eye(n) * 0.01),
        lengths=sparse.csr_matrix((n, n)), cv=1.0, dt=0.1, scale=2.0, cfun=Linear())
    ns = NetworkSet(subnets=subnets, projections=[proj])
    ns.configure()
    return ns


def test_grid_matches_explicit_zipped_sweep():
    ns = build_coupled_net(n=2)
    initial = [np.zeros((2, 2, 1), dtype=np.float32) for _ in range(2)]
    kwargs = dict(workers=1, width=4, initial_states=initial, features=["mean"])
    grid = nb.sweep(ns, 12, grid=True, coupling_scale=[0.5, 1.0],
                    **{"A.eta": [0.1, 0.2, 0.3]}, **kwargs)
    zipped = nb.sweep(ns, 12, coupling_scale=[0.5, 0.5, 0.5, 1, 1, 1],
                      **{"A.eta": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]}, **kwargs)
    assert grid.param_keys == ["coupling_scale", "A.eta"]
    np.testing.assert_array_equal(grid.sweep_values, zipped.sweep_values)
    np.testing.assert_allclose(grid.merged_tavg, zipped.merged_tavg, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(grid.online_features["A"]["mean"], zipped.online_features["A"]["mean"],
                               rtol=1e-6, atol=1e-7)
    df = grid.to_dataframe()
    np.testing.assert_array_equal(df[grid.param_keys].to_numpy(), grid.sweep_values)


def test_default_zip_and_grid_single_parameter():
    ns = build_coupled_net(n=2)
    params = {"A.eta": [0.1, 0.2]}
    zipped = nb.sweep(ns, 4, workers=1, width=1, **params)
    grid = nb.sweep(ns, 4, workers=1, width=1, grid=True, **params)
    np.testing.assert_array_equal(zipped.sweep_values, grid.sweep_values)
    with pytest.raises(ValueError, match="same length"):
        nb.sweep(ns, 4, workers=1, coupling_scale=[0.5], **params)


@pytest.mark.parametrize("values", [[], [[1, 2]], 1])
def test_grid_rejects_non_vector_or_empty(values):
    with pytest.raises(ValueError, match="nonempty 1-D"):
        nb.sweep(None, grid=True, **{"A.eta": values})


def test_grid_expansion_order_and_auto_width(monkeypatch):
    captured = {}
    def sweep(self, network_set, params, **kwargs):
        captured.update(params=params, kwargs=kwargs)
        return params
    monkeypatch.setattr(nb.CppHybridBackend, "sweep", sweep)
    monkeypatch.setattr(nb, "default_width", lambda n, ns: captured.setdefault("n", n) and 4)
    nb.sweep(None, grid=True, first=[1, 2], second=[3, 4, 5], third=[6, 7])
    assert captured["n"] == 12
    assert list(captured["params"]) == ["first", "second", "third"]
    np.testing.assert_array_equal(np.column_stack(list(captured["params"].values())),
        [[a, b, c] for a in [1, 2] for b in [3, 4, 5] for c in [6, 7]])
