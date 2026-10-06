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


# ---------------------------------------------------------------------------
# ZerlautAdaptationSecondOrder: the hand-written kernels (mako + C++) must
# carry the corrected dC_ei equation (upstream tvb-root#800 / Carlu et al.
# 2020 Eq. 17).  nb-vs-cpp parity alone cannot catch this — both kernels
# shared the stale term order — so both are compared against the Python
# model's own dfun with nonzero covariances, where the two orders differ.
# ---------------------------------------------------------------------------

def _zerlaut_heun_reference(model, x0, dt, nstep):
    """Pure-Python Heun mirroring the generated kernels (no coupling): k1 at
    the state, x1 = x + dt*k1 clamped to the model boundaries, k2 at x1,
    x += dt/2*(k1+k2) clamped again."""
    lo = {}
    for k, name in enumerate(model.state_variables):
        b = (model.state_variable_boundaries or {}).get(name)
        if b is not None and b[0] is not None:
            lo[k] = float(b[0])

    def clamp(v):
        for k, lo_v in lo.items():
            v[k] = np.maximum(v[k], lo_v)
        return v

    zero_c = np.zeros((len(model.cvar),) + np.asarray(x0).shape[1:])
    x = np.array(x0, dtype=np.float64)

    def dfun(state):
        return np.stack([np.asarray(a, dtype=np.float64)
                         for a in model.dfun(state, zero_c, 0.0)])

    for _ in range(nstep):
        k1 = dfun(x)
        x1 = clamp(x + dt * k1)
        k2 = dfun(x1)
        x = clamp(x + dt * 0.5 * (k1 + k2))
    return x


@pytest.mark.parametrize("e0", [0.05, 0.1])
def test_zerlaut_second_order_matches_python_dfun(e0):
    from tvb.simulator.models.zerlaut import ZerlautAdaptationSecondOrder

    def build():
        m = ZerlautAdaptationSecondOrder()
        m.configure()
        # nonzero, UNEQUAL covariances: the corrected and stale dC_ei term
        # orders differ by (C_ee - C_ei)*(dfe_TF_i - dfe_TF_e)
        # + (C_ii - C_ei)*(dfi_TF_e - dfi_TF_i), which vanishes when the
        # covariances are equal — equal values would make the test blind
        x0 = np.zeros((len(m.state_variables), 3, 1))
        x0[0] = 0.01        # E (battery working regime)
        x0[1] = 0.01        # I
        x0[2] = 0.3         # C_ee
        x0[3] = 0.05        # C_ei
        x0[4] = 0.5         # C_ii
        x0[5] = 50.0        # W_e
        sn = Subnetwork(name="S", model=m, scheme=HeunDeterministic(dt=DT),
                        nnodes=3)
        sn.configure()
        ns = NetworkSet(subnets=[sn], projections=[])
        ns.configure()
        return ns, x0

    nstep = 300
    ns, x0 = build()
    _, snap_nb = NbHybridBackend().compile(ns, eager=True).run(
        nstep, initial_states=[x0.copy()], return_snapshot=True)
    ns2, x0_2 = build()
    cpp_cn = CppHybridBackend().compile(ns2)
    _, snap_cpp = cpp_cn.run(nstep, initial_states=[x0_2],
                             return_snapshot=True)

    ref = _zerlaut_heun_reference(
        ZerlautAdaptationSecondOrder(), x0, DT, nstep)

    for tag, state in (("nb", snap_nb["states"][0]),
                       ("cpp", snap_cpp["states"][0])):
        state = np.asarray(state, dtype=np.float64).reshape(
            len(ref), ref.shape[1], ref.shape[2])
        # the corrected term order is a dC_ei property: C_ei is the sensitive
        # state; E follows through the 0.5*C_ei*d2f feedback
        # tight enough to separate the two term orders (the corrected
        # kernel matches the float64 reference to ~1e-8; the stale order
        # drifts by ~1e-4 over 300 steps)
        np.testing.assert_allclose(
            state[3], ref[3], rtol=1e-4, atol=1e-6,
            err_msg=f"ZerlautAdaptationSecondOrder C_ei ({tag} backend) "
                    "diverged from the corrected Python dfun")
        np.testing.assert_allclose(
            state[0], ref[0], rtol=5e-3, atol=1e-6,
            err_msg=f"ZerlautAdaptationSecondOrder E ({tag} backend) "
                    "diverged from the corrected Python dfun")
