# -*- coding: utf-8 -*-
"""Correctness tests for the C++ hybrid simulator core.

Compares the C++ kernel against a pure NumPy reference implementing the same
semantics as the nb_hybrid backend: coupling computed once per step from the
source history (slot t-1-delay), post-cfun applied to the weighted sum, the
same coupling used for both Heun stages, r clamped >= 0 for MPR, and tavg
accumulated over monitor periods.
"""

import numpy as np
import pytest

from tvb.simulator.backend.cpp_hybrid import Sim

DT = 0.01
PI = np.float32(np.pi)
MPR_PARM = 6  # tau I Delta J eta cr


def _make_csr(n_node, density=0.5, delay_max=5, seed=42):
    rng = np.random.RandomState(seed)
    W = np.zeros((n_node, n_node), np.float32)
    for i in range(n_node):
        for j in range(n_node):
            if i != j and rng.rand() < density:
                W[i, j] = rng.rand()
    idx, w_, del_ = [], [], []
    ptr = [0]
    for j in range(n_node):
        for i in range(n_node):  # source i -> target j
            if W[i, j] > 0:
                idx.append(i)
                w_.append(W[i, j])
                del_.append(1 + rng.randint(delay_max))
        ptr.append(len(idx))
    horizon = max(del_) + 2
    return (
        np.array(w_, np.float32),
        np.array(idx, np.uint32),
        np.array(ptr, np.uint32),
        np.array(del_, np.uint32),
        horizon,
    )


def _numpy_ref_mpr(w, idx, ptr, del_, horizon, n_node, nstep, tavg_period, dt,
                   params, cfun_a, integ="heun"):
    """MPR + Linear coupling; params: (6,) tau I Delta J eta cr"""
    tau, I, Delta, J, eta, cr = [np.float32(v) for v in params]
    x = np.zeros((2, n_node), np.float32)
    x[0] = 0.1
    buf = np.zeros((n_node, horizon), np.float32)
    buf[:] = x[0][:, None]
    tavg = []
    for t in range(nstep):
        c = np.zeros(n_node, np.float32)
        for j in range(n_node):
            s = np.float32(0.0)
            for k in range(ptr[j], ptr[j + 1]):
                slot = (t - 1 - int(del_[k])) % horizon
                s += w[k] * buf[idx[k], slot]
            c[j] = cfun_a * s

        def df(r, V):
            r = r * (r > 0)
            dr = (1 / tau) * (Delta / (PI * tau) + 2 * r * V)
            dV = (1 / tau) * (V * V + eta + J * tau * r + I + cr * c
                              - PI * PI * r * r * tau * tau)
            return dr, dV

        r, V = x[0], x[1]
        dr, dV = df(r, V)
        if integ == "heun":
            r1 = np.maximum(r + dt * dr, 0)
            V1 = V + dt * dV
            dr1, dV1 = df(r1, V1)
            x[0] = np.maximum(r + dt * 0.5 * (dr + dr1), 0)
            x[1] = V + dt * 0.5 * (dV + dV1)
        else:
            x[0] = np.maximum(r + dt * dr, 0)
            x[1] = V + dt * dV
        buf[:, t % horizon] = x[0]
        # nb_hybrid semantics: accumulate every step, mean per chunk
        chunk = t // tavg_period
        if len(tavg) <= chunk:
            tavg.append(np.zeros_like(x[0]))
        tavg[chunk] += x[0]
    n_chunks = (nstep + tavg_period - 1) // tavg_period
    counts = np.full(n_chunks, tavg_period, dtype=int)
    if nstep % tavg_period:
        counts[-1] = nstep % tavg_period
    arr = np.array(tavg[:n_chunks], np.float32) / counts[:, None].astype(
        np.float32)
    return arr[:, None, :, None]  # (nsamp,1,n_node,1)


def _run_cpp(w, idx, ptr, del_, horizon, n_node, nstep, tavg_period, dt,
             params, cfun_a, width, integ="heun"):
    sim = Sim(width)
    sim.add_subnet(n_node, 2, MPR_PARM, 1, 0, horizon, 1, dt=DT)  # MPR
    sim.add_projection(0, 0, w, idx, ptr, del_, 0, np.array([0],np.int32), 0, 2, 1.0)  # linear cfun
    sim.set_cfun_params(0, np.broadcast_to(
        np.array([[cfun_a], [0.0]], np.float32), (2, width)).copy())
    sim.set_voi(0, [0])
    sim.set_opts(1 if integ == "heun" else 0, dt, tavg_period)
    sim.set_subnet_params(0, np.broadcast_to(
        np.array(params, np.float32)[None, :, None], (n_node, MPR_PARM, width)
    ).copy())
    x0 = np.zeros((2, n_node, 1, width), np.float32)
    x0[0, :, 0, :] = 0.1
    sim.set_subnet_state(0, x0)
    out = sim.run(nstep, 0, None, None)
    return np.asarray(out[0])  # tavg of subnet 0 (outs = [t0, c0, t1, c1, ...])


@pytest.mark.parametrize("width", [1, 8])
@pytest.mark.parametrize("tavg_period", [1, 7])
@pytest.mark.parametrize("integ", ["heun", "euler"])
def test_mpr_matches_reference(width, tavg_period, integ):
    n_node = 5
    nstep = 50
    w, idx, ptr, del_, horizon = _make_csr(n_node)
    params = [1.0, 0.0, 1.0, 15.0, -5.0, 1.0]
    ref = _numpy_ref_mpr(w, idx, ptr, del_, horizon, n_node, nstep,
                         tavg_period, DT, params, 1.5, integ)
    got = _run_cpp(w, idx, ptr, del_, horizon, n_node, nstep, tavg_period,
                   DT, params, 1.5, width, integ)
    assert got.shape[0:3] == ref.shape[0:3]
    np.testing.assert_allclose(got, np.broadcast_to(ref, got.shape),
                               rtol=1e-3, atol=1e-4)


def test_two_subnets_inter_projection():
    """Two subnets, A -> B inter-projection with Linear coupling (heun)."""
    nA, nB = 3, 4
    nstep, period = 40, 3
    w, idx, ptr, del_, horizon = _make_csr(nB, seed=7)
    # retarget sources into [0, nA)
    idx = np.minimum(idx, nA - 1).astype(np.uint32)
    params = [1.0, 0.0, 1.0, 15.0, -5.0, 1.0]
    tau, I, Delta, J, eta, cr = [np.float32(v) for v in params]
    cfun_a = np.float32(0.8)

    # --- numpy reference: two coupled subnets
    xA = np.zeros((2, nA), np.float32); xA[0] = 0.1
    xB = np.zeros((2, nB), np.float32); xB[0] = 0.2
    bufA = np.zeros((nA, horizon), np.float32); bufA[:] = xA[0][:, None]
    bufB = np.zeros((nB, horizon), np.float32); bufB[:] = xB[0][:, None]

    def df(x, c):
        r, V = x[0], x[1]
        r = r * (r > 0)
        dr = (1 / tau) * (Delta / (PI * tau) + 2 * r * V)
        dV = (1 / tau) * (V * V + eta + J * tau * r + I + cr * c
                          - PI * PI * r * r * tau * tau)
        return dr, dV

    tavgA, tavgB = [], []
    for t in range(nstep):
        cA = np.zeros(nA, np.float32)  # no afferent projection to A
        cB = np.zeros(nB, np.float32)
        for j in range(nB):
            s = np.float32(0.0)
            for k in range(ptr[j], ptr[j + 1]):
                slot = (t - 1 - int(del_[k])) % horizon
                s += w[k] * bufA[idx[k], slot]
            cB[j] = cfun_a * s

        for (x, c, buf, n) in ((xA, cA, bufA, nA), (xB, cB, bufB, nB)):
            r, V = x[0], x[1]
            dr, dV = df(x, c)
            r1 = np.maximum(r + DT * dr, 0)
            V1 = V + DT * dV
            x1 = np.stack([r1, V1])
            dr1, dV1 = df(x1, c)
            x[0] = np.maximum(r + DT * 0.5 * (dr + dr1), 0)
            x[1] = V + DT * 0.5 * (dV + dV1)
            buf[:, t % horizon] = x[0]
        # nb_hybrid semantics: accumulate every step, mean per chunk
        chunk = t // period
        if len(tavgA) <= chunk:
            tavgA.append(np.zeros_like(xA[0]))
            tavgB.append(np.zeros_like(xB[0]))
        tavgA[chunk] += xA[0]
        tavgB[chunk] += xB[0]
    n_chunks = (nstep + period - 1) // period
    counts = np.full(n_chunks, period, dtype=int)
    if nstep % period:
        counts[-1] = nstep % period
    refA = (np.array(tavgA[:n_chunks], np.float32)
            / counts[:, None].astype(np.float32))[:, None, :, None]
    refB = (np.array(tavgB[:n_chunks], np.float32)
            / counts[:, None].astype(np.float32))[:, None, :, None]

    # --- C++
    sim = Sim(8)
    sim.add_subnet(nA, 2, MPR_PARM, 1, 0, horizon, 1, dt=DT)
    sim.add_subnet(nB, 2, MPR_PARM, 1, 0, horizon, 1, dt=DT)
    sim.add_projection(0, 1, w, idx, ptr, del_, 0, np.array([0],np.int32), 0, 2, 1.0)
    sim.set_cfun_params(0, np.broadcast_to(
        np.array([[0.8], [0.0]], np.float32), (2, 8)).copy())
    sim.set_voi(0, [0])
    sim.set_voi(1, [0])
    sim.set_opts(1, DT, period)
    for si, (n, init) in enumerate(((nA, 0.1), (nB, 0.2))):
        sim.set_subnet_params(si, np.broadcast_to(
            np.array(params, np.float32)[None, :, None],
            (n, MPR_PARM, 8)).copy())
        x0 = np.zeros((2, n, 1, 8), np.float32)
        x0[0, :, 0, :] = init
        sim.set_subnet_state(si, x0)
    outs = sim.run(nstep, 0, None, None)
    gotA, gotB = np.asarray(outs[0]), np.asarray(outs[2])
    np.testing.assert_allclose(gotA, np.broadcast_to(refA, gotA.shape),
                               rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(gotB, np.broadcast_to(refB, gotB.shape),
                               rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("width", [1, 8])
def test_uncovered_generic_id_raises(width):
    """Raw Sim API hard-fails on generic ids neither table covers.

    model_id=999 is above the hand-written range, not in the built-in table
    compiled into the extension, and not in the injected runtime library: the
    old dispatch returned without writing dx and the subnet silently never
    evolved.  Now the run raises, naming the id.
    """
    sim = Sim(width)
    sim.add_subnet(2, 2, MPR_PARM, 1, 999, 3, 1)
    sim.set_opts(1, DT, 1)
    with pytest.raises(RuntimeError, match="999"):
        sim.run(3, 0, None, None)


def test_state_persists_across_runs():
    """Calling run twice sequentially equals one run of the total steps."""
    n_node = 4
    w, idx, ptr, del_, horizon = _make_csr(n_node, seed=3)
    params = [1.0, 0.0, 1.0, 15.0, -5.0, 1.0]

    def make():
        sim = Sim(1)
        sim.add_subnet(n_node, 2, MPR_PARM, 1, 0, horizon, 1, dt=DT)
        sim.add_projection(0, 0, w, idx, ptr, del_, 0, np.array([0],np.int32), 0, 2, 1.0)
        sim.set_cfun_params(0, np.broadcast_to(
            np.array([[1.0], [0.0]], np.float32), (2, 1)).copy())
        sim.set_voi(0, [0])
        sim.set_opts(1, DT, 1)
        sim.set_subnet_params(0, np.broadcast_to(
            np.array(params, np.float32)[None, :, None],
            (n_node, MPR_PARM, 1)).copy())
        x0 = np.zeros((2, n_node, 1, 1), np.float32)
        x0[0, :, 0, :] = 0.1
        sim.set_subnet_state(0, x0)
        return sim

    whole = np.asarray(make().run(60, 0, None, None)[0])
    s = make()
    part = np.concatenate([
        np.asarray(s.run(20, 0, None, None)[0]),
        np.asarray(s.run(40, 0, None, None)[0])], axis=0)
    np.testing.assert_allclose(part, whole, rtol=1e-3, atol=1e-4)
