# -*- coding: utf-8 -*-
"""Model parity tests: every nb_hybrid-supported model class must produce
identical trajectories in CppHybridBackend and NbHybridBackend.

Runs each model as a single 3-node subnetwork (Heun, deterministic) with no
projections, comparing full state trajectories at float32 tolerance.
CerebellarMF is covered separately (custom hand-written dfun + template).
"""

import numpy as np
import pytest

from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.backend.cpp_hybrid.backend import CppHybridBackend

DT = 0.01
NSTEP = 30


def _model_instances():
    from tvb.simulator.backend.nb_hybrid import _get_supported_models_classes
    out = []
    for cls in _get_supported_models_classes():
        try:
            m = cls()
            m.configure()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"cannot configure {cls.__name__}") from exc
        out.append(m)
    return out


def _run(model, backend):
    name = "S"
    m2 = type(model)()
    # copy parameter values so both runs use identical settings
    try:
        pnames = list(model.parameter_names)
    except AttributeError:
        # e.g. Zerlaut models (P_e:/P_i: expansion) expose no parameter_names
        from tvb.simulator.backend.cpp_hybrid.backend import _MODEL_PARM_NAMES
        pnames = list(_MODEL_PARM_NAMES.get(type(model).__name__, []))
    for k in pnames:
        if ":" in k:
            continue  # P_e:i-style expansions read from the base array attr
        setattr(m2, k, getattr(model, k))
    m2.configure()
    scheme = HeunDeterministic(dt=DT)
    sn = Subnetwork(name=name, model=m2, scheme=scheme, nnodes=3)
    sn.configure()
    ns = NetworkSet(subnets=[sn], projections=[])
    ns.configure()
    if backend == "nb":
        return NbHybridBackend().compile(ns, eager=True).run(NSTEP)
    return CppHybridBackend().run_network(ns, NSTEP)


@pytest.mark.parametrize("model", _model_instances(),
                         ids=lambda m: type(m).__name__)
def test_model_parity(model):
    name = type(model).__name__
    nb_out = _run(model, "nb")
    cpp_out = _run(model, "cpp")
    assert len(cpp_out) == len(nb_out)
    for (t_nb, d_nb, c_nb), (t_cp, d_cp, c_cp) in zip(nb_out, cpp_out):
        assert d_cp.shape == d_nb.shape, (name, d_cp.shape, d_nb.shape)
        np.testing.assert_allclose(
            d_cp, d_nb, rtol=1e-3, atol=1e-4,
            err_msg=f"model {name} trajectory mismatch")


def _cerebellarMF_with(**overrides):
    from tvb.simulator.models.cerebellar_mf import CerebellarMF
    m = CerebellarMF()
    for k, v in overrides.items():
        setattr(m, k, v)
    m.configure()
    return m


@pytest.mark.parametrize("overrides", [
    {},
    {"use_legacy_goc_e_e": np.array([True])},
    {"add_noise_mli_pc": np.array([True])},
    {"use_legacy_goc_e_e": np.array([True]),
     "add_noise_mli_pc": np.array([True])},
])
def test_cerebellar_mf_flags(overrides):
    """CerebellarMF legacy/noise-flag variants (custom hand-written dfun)."""
    nb_out = _run(_cerebellarMF_with(**overrides), "nb")
    cpp_out = _run(_cerebellarMF_with(**overrides), "cpp")
    for (t_nb, d_nb, c_nb), (t_cp, d_cp, c_cp) in zip(nb_out, cpp_out):
        np.testing.assert_allclose(d_cp, d_nb, rtol=1e-3, atol=1e-4)
