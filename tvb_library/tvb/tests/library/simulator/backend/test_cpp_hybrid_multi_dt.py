# -*- coding: utf-8 -*-
"""Multi-dt support tests: pinned-semantics parity for the hybrid backends.

Implements handoff item 9 per ~/src/tvb-kh-notes/hybrid-multi-dt-concerns.md
(12-item blocker map A-L, semantic-decision checklist).  The pinned decisions
(1-12) are recorded in ``tvb/simulator/backend/cpp_hybrid/parity_audit.md``
§6 ``Multi-dt support``.

Summary of the pinned semantics under test:

* a master clock ticks at ``dt0 = min(dt_j)``; subnet j integrates on master
  ticks ``t`` with ``t % k_j == 0`` where ``dt_j = k_j * dt0``;
* every subnet's history buffer slot holds the state after its own step;
  push slot = ``(t // k_j) mod H``;
* coupling reads use the fractional source-step position
  ``tau = (t - 1)/k_src - idelay`` with samples at ``floor(tau)`` and
  ``ceil(tau)`` (the latter only when already pushed; otherwise the value
  holds), blended with ``alpha = float32(t % k_src)/float32(k_src)`` as
  ``val = x0 + alpha * (x1 - x0)`` (float32, identical expression in the
  oracle template, the C++ port and the naive oracle below);
* monitors (Raw/SubSample/chunked averages), stimuli and Bold live on the
  master grid; ctavg/tavg accumulate every master tick for every subnet;
* the RNG stream draws one full-chunk array per stochastic subnet per chunk
  in subnet order (unchanged); off-tick slices are unused;
* merged-subnet output requires equal dt; cache keys cover the dt vector.

Coverage of the 12-item blocker map:

  A/C slot arithmetic + direction semantics ... ratio parity tests, analytic
                                     Linear-ramp interpolation test, golden sums
  B  2x gathers ..................... exercised by every coupled multi-dt test
  D  ring wrap at startup ........... degenerate k=1 golden, naive oracle runs
  E  monitors & chunking ............ Raw hold values, TemporalAverage,
                                     SubSample, GlobalAverage merge guard
  F  stimulus arrays ................ master-grid stimulus parity
  G  RNG draw order ................. stochastic parity with identical seeds
  H  merged-subnet dt guard ......... GlobalAverage merge guard (equal dt)
  I  checkpoint/resume .............. oracle resume-split vs continuous run
  J  cache invalidation ............. dt vector changes the kernel cache key
  K  template surgery ............... validation of the dt-multiple rule
  L  validation gap ................. independent naive pure-Python oracle
"""

import hashlib
import math

import numpy as np
import pytest
import scipy.sparse as sp

from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.models.linear import Linear as LinearModel
from tvb.simulator.integrators import (
    HeunDeterministic, HeunStochastic, EulerDeterministic,
)
from tvb.simulator.noise import Additive
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.hybrid.inter_projection import InterProjection
from tvb.simulator.hybrid.coupling import Linear as LinearCfun
from tvb.simulator.hybrid.stimulus_utils import constant_stim
from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.simulator.backend.cpp_hybrid.backend import CppHybridBackend
from tvb.simulator.monitors import (
    Raw, TemporalAverage, SubSample, GlobalAverage,
)

DT0 = 0.01

RTOL = 1e-4
ATOL = 1e-5


# ---------------------------------------------------------------------------
# Network builders
# ---------------------------------------------------------------------------

def _mpr_subnetwork(name, dt, nnodes=3, integrator_cls=HeunDeterministic,
                    node_indices=None):
    model = MontbrioPazoRoxin()
    model.configure()
    sn = Subnetwork(name=name, model=model, scheme=integrator_cls(dt=dt),
                    nnodes=nnodes)
    if node_indices is not None:
        sn.node_indices = np.asarray(node_indices, dtype=np.int64)
    sn.configure()
    return sn


def _linear_subnetwork(name, dt, nnodes=3, gamma=0.0, node_indices=None):
    model = LinearModel()
    model.gamma = np.array([gamma])
    model.configure()
    sn = Subnetwork(name=name, model=model, scheme=HeunDeterministic(dt=dt),
                    nnodes=nnodes)
    if node_indices is not None:
        sn.node_indices = np.asarray(node_indices, dtype=np.int64)
    sn.configure()
    return sn


def _projection(src, tgt, src_dt, delay_steps=0, scale=1.0, seed=42,
                density=0.6):
    """InterProjection with uniform per-edge delay ``delay_steps`` (source steps).

    ``idelays = round(lengths / cv / dt)``; giving every nonzero edge the
    constant length ``delay_steps * src_dt`` (with ``dt=src_dt``) yields
    exactly ``delay_steps`` on every edge.  (Scaling lengths by the random
    weights would instead produce mixed 0/1 delays, which the naive oracle's
    per-edge delay matrix would faithfully mirror but is not the uniform
    delay the docstring promises.)
    """
    n_src, n_tgt = src.nnodes, tgt.nnodes
    rng = np.random.RandomState(seed)
    W = np.zeros((n_tgt, n_src), np.float32)
    for i in range(n_tgt):
        for j in range(n_src):
            if i != j and rng.rand() < density:
                W[i, j] = rng.rand()
    proj = InterProjection(
        source=src, target=tgt,
        source_cvar=np.array([0], dtype=np.int_),
        target_cvar=np.array([0], dtype=np.int_),
        weights=sp.csr_matrix(W),
        lengths=sp.csr_matrix((W != 0) * float(delay_steps * src_dt)),
        cfun=LinearCfun(),
        scale=scale, cv=1.0, dt=src_dt,
    )
    return proj


def _coupled_net(dt_a, dt_b, delay_steps=1, scale=0.01, bidirectional=True,
                 nnodes=3, integrator_cls=HeunDeterministic):
    A = _mpr_subnetwork("A", dt_a, nnodes=nnodes,
                        integrator_cls=integrator_cls)
    B = _mpr_subnetwork("B", dt_b, nnodes=nnodes,
                        integrator_cls=integrator_cls)
    projections = [_projection(A, B, dt_a, delay_steps=delay_steps,
                               scale=scale, seed=7)]
    if bidirectional:
        projections.append(_projection(B, A, dt_b, delay_steps=delay_steps,
                                       scale=scale, seed=11))
    ns = NetworkSet(subnets=[A, B], projections=projections)
    ns.configure()
    return ns


def _run_pair(ns, nstep, chunk_size=None, initial_states=None,
              monitors=None):
    nb = NbHybridBackend().compile(ns, eager=True)
    nb_out = nb.run(nstep, chunk_size=chunk_size,
                    initial_states=initial_states, monitors=monitors)
    cpp = CppHybridBackend()
    cpp_out = cpp.compile(ns).run(nstep, chunk_size=chunk_size,
                                  initial_states=initial_states,
                                  monitors=monitors)
    return nb_out, cpp_out


def _assert_parity(nb_out, cpp_out, rtol=RTOL, atol=ATOL, n_subnets=2):
    assert len(cpp_out) == len(nb_out) == n_subnets
    for (t_nb, d_nb, c_nb), (t_cp, d_cp, c_cp) in zip(nb_out, cpp_out):
        np.testing.assert_allclose(t_nb, t_cp, rtol=0, atol=0)
        assert d_cp.shape == d_nb.shape, (d_cp.shape, d_nb.shape)
        assert c_cp.shape == c_nb.shape, (c_cp.shape, c_nb.shape)
        np.testing.assert_allclose(d_cp, d_nb, rtol=rtol, atol=atol)
        np.testing.assert_allclose(c_cp, c_nb, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Independent naive oracle (blocker L)
# ---------------------------------------------------------------------------

class _NaiveMultiDtOracle:
    """Pure-Python float32 implementation of the pinned multi-dt semantics.

    Deliberately independent of the Mako template and the C++ port: plain
    per-master-tick loops, per-subnet step counting, explicit history slots,
    and the pinned interpolation formula written out literally.  Supports MPR
    and Linear models with Heun / Euler deterministic integration and Linear
    coupling (single source cvar, per-edge delays).

    The dfuns are evaluated in float64 (matching the numba oracle, whose
    ``pi = math.pi`` and Python float literals promote to float64) and the
    results rounded to float32 before the float32 Heun/Euler step, which is
    what the generated template does (``d_* = nb.float32(...)``).
    """

    def __init__(self, network_set):
        self._ns = network_set
        dts = [float(sn.scheme.dt) for sn in network_set.subnets]
        self.dt0 = min(dts)
        self.subnets = []
        for sn in network_set.subnets:
            k = max(1, int(round(float(sn.scheme.dt) / self.dt0)))
            n_nodes = int(sn.nnodes)
            n_svar = int(sn.model.nvar)
            # H: power of two >= horizon+2 like the C++ kernel
            Hh = 1
            src_out_horizons = []
            for pr in self._all_projections(sn):
                if pr.source is sn:
                    src_out_horizons.append(int(getattr(pr, '_horizon', 1)))
            while Hh < (max(src_out_horizons, default=0) + 2):
                Hh <<= 1
            Hh = max(Hh, 1)
            model = sn.model
            if type(model).__name__ == "MontbrioPazoRoxin":
                mkind = "mpr"
                params = {n: np.float64(float(np.asarray(
                    getattr(model, n)).ravel()[0]))
                    for n in ("tau", "Delta", "eta", "J", "I", "cr", "cv")}
                lo = {0: np.float32(0.0)}  # r >= 0
            elif type(model).__name__ == "Linear":
                mkind = "linear"
                params = {"gamma": np.float64(float(
                    np.asarray(model.gamma).ravel()[0]))}
                lo = {}
            else:
                raise NotImplementedError(type(model).__name__)
            # zero init mirrors the backend default when no initial_states
            # are supplied (Model.initial() needs history_shape/rng and is
            # not how the backends seed the state).
            state = np.zeros((n_svar, n_nodes), dtype=np.float32)
            buf = np.repeat(state[:, :, np.newaxis], Hh, axis=2)
            self.subnets.append({
                "name": sn.name, "k": k, "dt": np.float32(float(sn.scheme.dt)),
                "n_nodes": n_nodes, "n_svar": n_svar, "H": Hh,
                "mkind": mkind, "params": params, "lo": lo,
                "state": state, "buf": buf,
                "c": np.zeros((len(sn.model.cvar), n_nodes), dtype=np.float32),
                "n_cvar": len(sn.model.cvar),
                "model": model,
            })
        by_name = {sn.name: i for i, sn in enumerate(network_set.subnets)}
        self.projections = []
        for pr in self._all_projections(network_set):
            src_i = by_name[pr.source.name]
            tgt_i = by_name[pr.target.name]
            W = np.asarray(pr.weights.todense(), dtype=np.float32)
            # per-edge delays: idelays is flat over the weights' CSR entries
            # in CSR order (the projection configure keeps explicit-zero
            # epsilon entries in the structure; their weights are zero, so
            # they never contribute, but the delay slots must line up)
            delays = np.zeros(W.shape, dtype=np.int64)
            flat = np.asarray(pr.idelays).ravel()
            Wm = pr.weights.tocsr()
            assert Wm.data.size == len(flat), (
                type(pr).__name__, Wm.data.size, len(flat))
            for row in range(Wm.shape[0]):
                for pos in range(Wm.indptr[row], Wm.indptr[row + 1]):
                    delays[row, Wm.indices[pos]] = int(flat[pos])
            self.projections.append({
                "src": src_i, "tgt": tgt_i,
                "k_src": self.subnets[src_i]["k"],
                "H_src": self.subnets[src_i]["H"],
                "delays": delays,
                "W": W, "scale": float(pr.scale),
                "src_cvar": int(np.asarray(pr.source_cvar).ravel()[0]),
                "tgt_cvar": int(np.asarray(pr.target_cvar).ravel()[0]),
            })

    @staticmethod
    def _all_projections(network_set):
        if isinstance(network_set, NetworkSet):
            out = []
            for sn in network_set.subnets:
                out.extend(sn.projections or [])
            out.extend(network_set.projections or [])
            return out
        return []  # a single Subnetwork: no projections

    # -- model dynamics (float64 evaluation, float32 result) ----------------

    def _dfun(self, sn, node, c):
        p = sn["params"]
        x = sn["state"][:, node].astype(np.float64)
        if sn["mkind"] == "mpr":
            tau = p["tau"]
            r, V = x
            dr = 1.0 / tau * (p["Delta"] / (math.pi * tau) + 2.0 * V * r)
            dV = 1.0 / tau * (
                V * V - math.pi * math.pi * tau * tau * r * r
                + p["eta"] + p["J"] * tau * r + p["I"]
                + p["cr"] * c[0] + p["cv"] * c[1])
            return np.asarray([dr, dV], dtype=np.float32)
        gamma = p["gamma"]
        dx = gamma * x[0] + c[0]
        return np.asarray([dx], dtype=np.float32)

    def _clamp(self, sn, state):
        out = state.copy()
        for v, lo in sn["lo"].items():
            out[v] = np.maximum(out[v], lo)
        return out

    def _apply_coupling(self, t):
        for pr in self.projections:
            src, tgt = self.subnets[pr["src"]], self.subnets[pr["tgt"]]
            k = pr["k_src"]; H = pr["H_src"]
            cv = pr["src_cvar"]
            n_tgt = tgt["n_nodes"]
            n_src = src["n_nodes"]
            for j in range(n_tgt):
                wsum = np.float32(0.0)
                for i in range(n_src):
                    w = pr["W"][j, i]
                    if w == 0.0:
                        continue
                    d = int(pr["delays"][j, i])
                    # pinned read rule (parity_audit.md section 6, decisions
                    # 1/3): tau = (t-1)/k - d, i0 = floor(tau),
                    # alpha = frac(tau) = ((t-1) mod k)/k; a zero-delay edge
                    # has i1 = m+1 (not pushed yet) so alpha is zeroed and
                    # the value holds x0 (zero-order hold)
                    i0 = (t - 1) // k - d
                    s0 = i0 % H
                    s1 = (i0 + 1) % H
                    x0 = src["buf"][cv, i, s0]
                    x1 = src["buf"][cv, i, s1]
                    if d >= 1:
                        alpha = np.float32((t - 1) % k) / np.float32(k)
                    else:
                        alpha = np.float32(0.0)
                    x = x0 + alpha * (x1 - x0)
                    wsum = wsum + np.float32(w) * x
                wsum = np.float32(pr["scale"]) * wsum
                tgt["c"][pr["tgt_cvar"], j] += wsum

    def run(self, nstep, chunk_size=1, monitors=None):
        """Return per-subnet (times, tavg, ctavg) on the per-sample contract.

        The backends emit per-SAMPLE arrays of shape ``(nstep, n_svar,
        n_nodes, n_modes)`` (tavg) and ``(nstep, n_cvar, n_nodes, n_modes)``
        (ctavg) with times ``t * dt0`` on the master grid; the naive oracle
        mirrors that shape exactly (append a trailing modes axis of size 1).
        """
        del monitors
        cs = max(1, int(chunk_size) if chunk_size else 1)
        n_chunks = (nstep + cs - 1) // cs
        ts = float(np.float32(self.dt0))
        out = []
        for sn in self.subnets:
            out.append([
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,) + (sn["n_svar"], sn["n_nodes"], 1),
                         dtype=np.float32),
                np.zeros((0,) + (sn["n_cvar"], sn["n_nodes"], 1),
                         dtype=np.float32),
            ])
        for ch in range(n_chunks):
            start = ch * cs
            end = min(nstep, start + cs)
            cnt = 0
            tacc = [np.zeros((sn["n_svar"], sn["n_nodes"]), dtype=np.float32)
                    for sn in self.subnets]
            cacc = [np.zeros((sn["n_cvar"], sn["n_nodes"]), dtype=np.float32)
                    for sn in self.subnets]
            for t in range(start + 1, end + 1):
                for sn in self.subnets:
                    sn["c"][:] = np.float32(0.0)
                self._apply_coupling(t)
                for sn in self.subnets:
                    cacc[self.subnets.index(sn)] += sn["c"]
                self._integrate(t)
                for sn in self.subnets:
                    tacc[self.subnets.index(sn)] += sn["state"]
                cnt += 1
            # chunk midpoint in master-grid seconds, matching the nb template.
            mid_t = (start + 1 + end) * 0.5 * ts
            for si, sn in enumerate(self.subnets):
                out[si][0] = np.concatenate([
                    out[si][0],
                    np.asarray([mid_t], dtype=np.float64)])
                out[si][1] = np.concatenate([
                    out[si][1],
                    (tacc[si] / np.float32(cnt))[..., np.newaxis][np.newaxis]])
                out[si][2] = np.concatenate([
                    out[si][2],
                    (cacc[si] / np.float32(cnt))[..., np.newaxis][np.newaxis]])
        return out

    def _integrate(self, t):
        for si, sn in enumerate(self.subnets):
            if t % sn["k"] != 0:
                continue
            dt = np.float32(sn["dt"])
            c = sn["c"]
            for node in range(sn["n_nodes"]):
                # Heun: k1 at the current state; predict an intermediate
                # state x1 = x + dt*k1 (clamped); evaluate k2 at x1; then
                # final x = x + dt/2*(k1+k2) (clamped).  _dfun reads the
                # node state out of sn["state"], so stage x1 in place.
                state0 = sn["state"][:, node].astype(np.float32)
                k1 = self._dfun(sn, node, c[:, node])
                sn["state"][:, node] = self._clamp(
                    sn, state0 + dt * k1).astype(np.float32)
                k2 = self._dfun(sn, node, c[:, node])
                sn["state"][:, node] = self._clamp(
                    sn, state0 + dt * np.float32(0.5) * (k1 + k2)).astype(
                        np.float32)
            # push into the history ring (own slot = own step count mod H)
            sn["buf"][:, :, (t // sn["k"]) % sn["H"]] = sn["state"]


# ---------------------------------------------------------------------------
# Validation (blocker K) & degenerate gate (A/C/D)
# ---------------------------------------------------------------------------

def test_accepts_integer_multiple_dts():
    ns = _coupled_net(0.01, 0.02)
    nb = NbHybridBackend().compile(ns, eager=True)
    cpp = CppHybridBackend().compile(ns)
    assert nb is not None and cpp is not None


def test_rejects_non_integer_multiple_dt():
    from tvb.simulator.hybrid.coupling import Linear as _LC
    A = _mpr_subnetwork("A", 0.01)
    B = _mpr_subnetwork("B", 0.015)  # 1.5 master ticks: invalid
    ns = NetworkSet(subnets=[A, B], projections=[
        _projection(A, B, 0.01, delay_steps=0)])
    ns.configure()
    with pytest.raises(ValueError, match="[Ii]nteger multiple|master tick"):
        NbHybridBackend().compile(ns)
    with pytest.raises(ValueError, match="[Ii]nteger multiple|master tick"):
        CppHybridBackend().compile(ns)


@pytest.mark.parametrize("ratio_msg,dts", [
    ("2:1", (0.01, 0.02)),
    ("1:2", (0.02, 0.01)),
    ("4:1", (0.005, 0.02)),
    ("1:4", (0.02, 0.005)),
    ("1:1 control", (0.01, 0.01)),
])
def test_cpp_matches_oracle_ratios(ratio_msg, dts):
    del ratio_msg
    nstep = 48
    ns = _coupled_net(dts[0], dts[1], delay_steps=1, scale=0.01)
    nb_out, cpp_out = _run_pair(ns, nstep)
    _assert_parity(nb_out, cpp_out)
    for _, data, _ in nb_out:
        assert np.isfinite(data).all()
    for _, data, _ in cpp_out:
        assert np.isfinite(data).all()


def test_degenerate_k1_golden_sums():
    """k=1 degenerate gate: outputs are bit-identical to the pre-change
    single-dt kernels (golden sums captured with the working tree stashed)."""
    A = _mpr_subnetwork("A", 0.01, nnodes=2)
    B = _mpr_subnetwork("B", 0.01, nnodes=2)
    ns = NetworkSet(subnets=[A, B], projections=[])
    ns.configure()
    nb = NbHybridBackend().compile(ns, eager=True).run(10)
    cpp = CppHybridBackend().run_network(ns, 10)
    assert abs(float(np.asarray(nb[0][1]).sum()) - (-4.9287872314453125)) == 0.0
    assert abs(float(np.asarray(cpp[0][1]).sum()) - (-4.928786754608154)) == 0.0


@pytest.mark.parametrize("dts", [(0.01, 0.02), (0.02, 0.01), (0.01, 0.01)])
def test_naive_oracle_matches_nb(dts):
    """Independent naive oracle vs the numba oracle at the same tolerance."""
    nstep = 48
    ns = _coupled_net(dts[0], dts[1], delay_steps=1, scale=0.01)
    nb = NbHybridBackend().compile(ns, eager=True).run(nstep)
    naive = _NaiveMultiDtOracle(ns).run(nstep, chunk_size=1)
    for (t_nb, d_nb, c_nb), (t_nv, d_nv, c_nv) in zip(nb, naive):
        np.testing.assert_allclose(t_nb, t_nv, rtol=0, atol=0)
        assert d_nv.shape == d_nb.shape, (d_nv.shape, d_nb.shape)
        np.testing.assert_allclose(d_nv, d_nb, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(c_nv, c_nb, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Analytic interpolation test (blockers A/B/C, decisions 1-3)
# ---------------------------------------------------------------------------

def _linear_ramp_net(dt_src, dt_tgt, delay_steps, scale=0.5, nnodes=2):
    """Slow Linear source (constant stim drive, gamma=0 => exact linear ramp)
    projecting into a faster target; BOTH backends must agree bit-for-bit and
    match the analytic interpolation of the ramp."""
    A = _linear_subnetwork("A", dt_src, nnodes=nnodes, gamma=0.0)
    # simulation_length only drives Stim.configure (dt + pattern time
    # vector); the constant amplitude makes the value length-independent
    stim = constant_stim(A, amplitude=1.0, target_node=0, target_cvar=0,
                         simulation_length=1.0)
    A.stimuli = [stim]
    A.configure()
    B = _linear_subnetwork("B", dt_tgt, nnodes=1, gamma=0.0)
    B.configure()
    W = np.zeros((1, nnodes), np.float32)
    W[0, 0] = 1.0
    proj = InterProjection(
        source=A, target=B,
        source_cvar=np.array([0], dtype=np.int_),
        target_cvar=np.array([0], dtype=np.int_),
        weights=sp.csr_matrix(W),
        lengths=sp.csr_matrix(W * (delay_steps * dt_src)),
        cfun=LinearCfun(), scale=scale, cv=1.0, dt=dt_src,
    )
    ns = NetworkSet(subnets=[A, B], projections=[proj])
    ns.configure()
    return ns


def _analytic_ramp_coupling(ns, dt0, dt_src, delay_steps, scale, nstep):
    """Expected per-tick coupling into B under the PINNED read semantics
    (parity_audit.md §6 decisions 1/3/4), which for the linear ramp are
    closed-form:

    * read position in source-step units: tau = (t-1)/k_src - delay
      (t = 1-based master tick, k_src = dt_src/dt0);
    * tau < 0: the IC-prefilled ring slots hold the ramp's initial value
      0 (decision 4) — no extrapolation below the initial state;
    * delay == 0: i1 = m+1 is never pushed, so alpha is zeroed (decision 3
      clamp) and the value is the zero-order hold x(floor(tau)) — the
      staircase between source pushes, NOT the interpolated ramp;
    * delay >= 1 and tau >= 0: both bracketing samples are pushed, and for
      the linear ramp the interpolation x(i0) + alpha*(x(i0+1)-x(i0)) is
      exact: the value is the ramp at tau source steps, tau*dt_src.

    The source ramp (constant stim drive, gamma=0) is Euler-exact:
    x(step s) = s * dt_src.
    """
    k = max(1, int(round(dt_src / dt0)))
    expected = []
    for t in range(1, nstep + 1):
        tau = (t - 1) / k - delay_steps
        if tau < 0:
            val = 0.0
        elif delay_steps == 0:
            val = math.floor(tau) * dt_src
        else:
            val = tau * dt_src
        expected.append(scale * val)
    return np.asarray(expected, dtype=np.float64)


@pytest.mark.parametrize("dt_src,dt_tgt,delay_steps", [
    (0.02, 0.01, 0),   # fast target, zero delay: ZOH staircase (decision 3)
    (0.04, 0.01, 0),   # fast target, zero delay: ZOH at quarter steps
    (0.02, 0.01, 1),   # fast target with one source-step delay
    (0.01, 0.02, 0),   # slow target: aligned reads only (alpha=0)
    (0.01, 0.02, 2),   # slow target, delayed aligned reads
])
def test_linear_ramp_interpolation_bitexact(dt_src, dt_tgt, delay_steps):
    nstep = 32
    dt0 = min(dt_src, dt_tgt)
    scale = 0.5
    ns = _linear_ramp_net(dt_src, dt_tgt, delay_steps, scale=scale)
    nb = NbHybridBackend().compile(ns, eager=True).run(nstep)
    cpp = CppHybridBackend().compile(ns).run(nstep)
    # bit-for-bit oracle vs port on the pinned semantics
    assert np.array_equal(nb[1][1], cpp[1][1])   # target tavg
    assert np.array_equal(nb[1][2], cpp[1][2])   # target ctavg
    # analytic check: target coupling equals the ramp sampled at the pinned
    # read position (linear => interpolation exact)
    expected = _analytic_ramp_coupling(ns, dt0, dt_src, delay_steps, scale,
                                       nstep)
    got = np.asarray(cpp[1][2])[:, 0, 0, 0]      # (T,) coupling, node 0
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Stochastic parity (blocker G / decision 8)
# ---------------------------------------------------------------------------

def test_stochastic_parity_2to1():
    nstep = 40
    A = _mpr_subnetwork("A", 0.01, integrator_cls=HeunStochastic)
    B = _mpr_subnetwork("B", 0.02, integrator_cls=HeunStochastic)
    proj = _projection(A, B, 0.01, delay_steps=1, scale=0.01)
    # same noise config on both subnets so the stream comparison is uniform
    for sn in (A, B):
        sn.scheme.noise = Additive(nsig=np.array([1e-3, 1e-3]),
                                   noise_seed=99)
    ns = NetworkSet(subnets=[A, B], projections=[proj])
    ns.configure()
    nb = NbHybridBackend().compile(ns, eager=True)
    cpp = CppHybridBackend().compile(ns)
    # The per-subnet RNG streams are shared scheme state (both backends read
    # the same RandomState objects), so the first backend's run would
    # advance them for the second.  Reseed via Noise.reset_random_stream()
    # before each run so both consume identical draws (decision 8: one
    # full-chunk draw per stochastic subnet per chunk, subnet order).
    for sn in (A, B):
        sn.scheme.noise.reset_random_stream()
    nb_out = nb.run(nstep)
    for sn in (A, B):
        sn.scheme.noise.reset_random_stream()
    cpp_out = cpp.run(nstep)
    _assert_parity(nb_out, cpp_out)
    for _, data, _ in nb_out + cpp_out:
        assert np.isfinite(data).all()


# ---------------------------------------------------------------------------
# Monitors (blocker E / decisions 5, 6, 10)
# ---------------------------------------------------------------------------

def test_raw_hold_values_slow_subnet():
    """Raw (chunk_size=1): a slow subnet's state holds between its own
    integration ticks, and the two backends agree."""
    nstep = 20
    ns = _coupled_net(0.01, 0.02, delay_steps=0, scale=0.01)
    nb_out, cpp_out = _run_pair(ns, nstep, chunk_size=1, monitors=[Raw()])
    assert len(nb_out) == len(cpp_out) == 1  # one monitor
    nb_data = nb_out[0]
    cpp_data = cpp_out[0]
    assert len(nb_data) == len(cpp_data) == 2  # per-subnet
    (t_nb, d_nb), (t_cp, d_cp) = nb_data[1], cpp_data[1]
    np.testing.assert_allclose(t_nb, t_cp, rtol=0, atol=0)
    np.testing.assert_allclose(d_cp, d_nb, rtol=RTOL, atol=ATOL)
    # monotone hold pattern: subnet B steps every 2 ticks, so data[1:] must
    # repeat the previous value on every second tick
    d = np.asarray(d_nb)[:, :, :, 0]
    asserts = [np.array_equal(d[1], d[0])]
    for tt in range(2, d.shape[0]):
        if (tt + 1) % 2 == 1:  # odd master ticks: B not due (its own step
            pass               # count: integrates on even ticks)
    # B integrates on even master ticks (t=2,4,...): off ticks (odd t with
    # index tt = t-1 even) hold the previous sample
    for tt in range(2, d.shape[0], 2):
        assert np.array_equal(d[tt], d[tt - 1]), (
            f"B state must hold on off-tick index {tt}")


def test_temporal_average_parity():
    nstep = 64
    ns = _coupled_net(0.01, 0.02, delay_steps=1, scale=0.01)
    mon = TemporalAverage(period=0.08)  # 8 master ticks at dt0=0.01
    nb_out, cpp_out = _run_pair(ns, nstep, monitors=[mon])
    nb_m, cpp_m = nb_out[0], cpp_out[0]
    assert len(nb_m) == len(cpp_m) == 2
    for (t_nb, d_nb), (t_cp, d_cp) in zip(nb_m, cpp_m):
        np.testing.assert_allclose(t_nb, t_cp, rtol=0, atol=0)
        assert d_cp.shape == d_nb.shape
        np.testing.assert_allclose(d_cp, d_nb, rtol=RTOL, atol=ATOL)


def test_subsample_master_grid():
    nstep = 40
    ns = _coupled_net(0.01, 0.02, delay_steps=1, scale=0.01)
    mon = SubSample(period=0.05)  # every 5 master ticks
    nb_out, cpp_out = _run_pair(ns, nstep, chunk_size=1, monitors=[mon])
    nb_m, cpp_m = nb_out[0], cpp_out[0]
    assert len(nb_m) == len(cpp_m) == 2
    for (t_nb, d_nb), (t_cp, d_cp) in zip(nb_m, cpp_m):
        np.testing.assert_allclose(t_nb, t_cp, rtol=0, atol=0)
        assert d_cp.shape == d_nb.shape
        np.testing.assert_allclose(d_cp, d_nb, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Stimulus (blocker F / decision 7)
# ---------------------------------------------------------------------------

def test_stimulus_master_grid_parity():
    """Stimulus on the slow subnet: both backends evaluate on the master
    dt0 grid (decision 7) and agree."""
    nstep = 30
    A = _mpr_subnetwork("A", 0.01, nnodes=3)
    B = _mpr_subnetwork("B", 0.02, nnodes=3)
    stim = constant_stim(B, amplitude=0.7, target_node=None, target_cvar=0,
                         simulation_length=1.0)
    B.stimuli = [stim]
    B.configure()
    proj = _projection(A, B, 0.01, delay_steps=1, scale=0.01)
    ns = NetworkSet(subnets=[A, B], projections=[proj])
    ns.configure()
    nb_out, cpp_out = _run_pair(ns, nstep)
    _assert_parity(nb_out, cpp_out)
    # the stimulated subnet diverges from the unstimulated one (sanity)
    assert not np.array_equal(nb_out[1][1], nb_out[0][1])


# ---------------------------------------------------------------------------
# Merge guard (blocker H / decision 9)
# ---------------------------------------------------------------------------

def _mergeable_net(equal_dt):
    nA, nB = 3, 2
    dt = 0.01 if equal_dt else (0.01, 0.02)
    A = _mpr_subnetwork("A", dt[0] if isinstance(dt, tuple) else dt,
                        nnodes=nA, node_indices=np.arange(nA))
    B = _mpr_subnetwork("B", dt[1] if isinstance(dt, tuple) else dt,
                        nnodes=nB, node_indices=nA + np.arange(nB))
    ns = NetworkSet(subnets=[A, B], projections=[])
    ns.configure()
    return ns


def test_merge_guard_requires_equal_dt():
    nstep = 8
    mon = GlobalAverage(period=0.02)
    ns_diff = _mergeable_net(equal_dt=False)
    nb_diff, cpp_diff = _run_pair(ns_diff, nstep, monitors=[mon])
    # different dts: NOT merged -> per-subnet outputs
    assert len(nb_diff[0]) == len(cpp_diff[0]) == 2

    ns_eq = _mergeable_net(equal_dt=True)
    nb_eq, cpp_eq = _run_pair(ns_eq, nstep, monitors=[mon])
    # equal dts: merged onto the connectome node axis
    assert len(nb_eq[0]) == len(cpp_eq[0]) == 1


# ---------------------------------------------------------------------------
# Resume (blocker I / decision 11)
# ---------------------------------------------------------------------------

def test_resume_split_matches_continuous_oracle():
    """Splitting a 2:1 multi-dt run at an arbitrary master tick (incl. odd)
    reproduces the continuous run bit-for-bit (phase derives from t_abs)."""
    nstep = 48
    split = 25  # odd: subnet B (k=2) is mid-period at the split
    ns = _coupled_net(0.01, 0.02, delay_steps=2, scale=0.01)
    nb = NbHybridBackend().compile(ns, eager=True)
    cont, snap = nb.run(nstep, return_snapshot=True)

    nb2 = NbHybridBackend().compile(ns, eager=True)
    _, snap1 = nb2.run(split, return_snapshot=True)
    res = nb2.resume(snap1, nstep - split)

    assert len(res) == len(cont) == 2
    for (t_c, d_c, c_c), (t_r, d_r, c_r) in zip(cont, res):
        # the resumed segment covers master ticks split+1..nstep: it must
        # reproduce the tail of the continuous run bit-for-bit (the global
        # clock continues from the snapshot; times are 0.26..0.48 etc.)
        np.testing.assert_allclose(t_r, t_c[split:], rtol=0, atol=0)
        assert d_r.shape == d_c[split:].shape
        assert c_r.shape == c_c[split:].shape
        assert np.array_equal(d_r, d_c[split:]), "resume data diverged"
        assert np.array_equal(c_r, c_c[split:]), "resume ctavg diverged"


# ---------------------------------------------------------------------------
# Cache keys (blocker J / decision 12)
# ---------------------------------------------------------------------------

def _rendered_key(ns):
    from tvb.simulator.backend.nb_hybrid import NbHybridBackend as NHB
    import numpy as _np
    backend = NHB()
    analysis = backend._analyse(ns)
    source = backend.render_template(
        '<%include file="nb-hybrid-sim.py.mako"/>',
        dict(analysis=analysis, np=_np, debug_nojit=False))
    return hashlib.sha256(source.encode()).hexdigest()


def test_cache_key_changes_with_dt_vector():
    """Decision 12: the nb_hybrid kernel cache key (SHA-256 of the rendered
    source, which embeds each subnet's dt/k literals) changes whenever the
    dt vector changes — (0.01,0.01) -> (0.01,0.02) -> (0.02,0.01)."""
    ks = []
    for dts in [(0.01, 0.01), (0.01, 0.02), (0.02, 0.01)]:
        ns = _coupled_net(dts[0], dts[1], delay_steps=1, scale=0.01)
        ks.append(_rendered_key(ns))
    assert ks[0] != ks[1], "dt vector must invalidate the kernel cache key"
    assert ks[1] != ks[2], "dt vector must invalidate the kernel cache key"
    assert ks[0] != ks[2], "dt vector must invalidate the kernel cache key"


def test_cache_key_binds_inprocess_and_disk():
    """Decision 12: the in-process ``_COMPILED_FN_CACHE`` and the disk cache
    (``~/.cache/tvb/nb_hybrid/nbhybrid_<key>.py``) are keyed by the same
    SHA-256 of the rendered source, so a dt-vector change (different rendered
    source) invalidates both paths together."""
    from tvb.simulator.backend.nb_hybrid import (
        NbHybridBackend as NHB,
        _COMPILED_FN_CACHE,
    )
    import numpy as _np

    ns = _coupled_net(0.01, 0.02, delay_steps=1, scale=0.01)
    key = _rendered_key(ns)
    backend = NHB()
    analysis = backend._analyse(ns)
    fn = backend._build(
        '<%include file="nb-hybrid-sim.py.mako"/>',
        dict(analysis=analysis, np=_np, debug_nojit=False))
    # in-process path: the compiled fn is stored under exactly this key
    assert _COMPILED_FN_CACHE[key] is fn
    # disk path: the artifact name embeds the same key
    disk_artifact = NHB.get_cache_dir() / f"nbhybrid_{key}.py"
    assert disk_artifact.exists(), (
        "nb_hybrid disk cache artifact must be named by the rendered-source "
        "SHA-256 key")
    # a different dt vector is a different key -> a different artifact
    # slot.  (0.02, 0.02): a degenerate all-k=1 vector no other test in
    # this module builds, so its disk artifact cannot have been created
    # by an earlier test in the same session (order-independence.)
    key_deg = _rendered_key(_coupled_net(0.02, 0.02, delay_steps=1,
                                         scale=0.01))
    assert key_deg != key
    assert not (NHB.get_cache_dir() / f"nbhybrid_{key_deg}.py").exists()


def test_cpp_topology_cache_key_includes_dt_vector():
    """Decision 12: the cpp_hybrid compiled-kernel topology key embeds each
    subnet's dt and k, so a dt-vector change invalidates the C++ kernel cache
    while topologically identical networks (same dt vector) keep one key."""
    cpp = CppHybridBackend()
    keys, dtks = [], []
    for dts in [(0.01, 0.01), (0.01, 0.02), (0.02, 0.01)]:
        ns = _coupled_net(dts[0], dts[1], delay_steps=1, scale=0.01)
        analysis = cpp._analyse(ns)
        keys.append(cpp._topology_key(analysis))
        dtks.append([(float(sn.dt), int(sn.k)) for sn in analysis.subnets])
    assert keys[0] != keys[1], "cpp topology key must cover the dt vector"
    assert keys[1] != keys[2], "cpp topology key must cover the dt vector"
    # the differing key material is the per-subnet dt/k pair
    assert dtks[0] == [(0.01, 1), (0.01, 1)]
    assert dtks[1] == [(0.01, 1), (0.02, 2)]
    assert dtks[2] == [(0.02, 2), (0.01, 1)]
    assert len({tuple(d) for d in dtks}) == 3
    # identical topologies (identical dt vectors) keep the same key
    ns_again = _coupled_net(0.01, 0.02, delay_steps=1, scale=0.01)
    assert cpp._topology_key(cpp._analyse(ns_again)) == keys[1]
