# -*- coding: utf-8 -*-
"""
Numerical validation: Python (classic) backend vs Numba hybrid backend.

Runs identical simulations through both backends and asserts that results
match within a small tolerance.  Any divergence indicates a numerical bug
in the JIT kernel or the code-generation template.

Test categories
---------------
1. Single-subnet deterministic (Euler, Heun) — state-level comparison
2. Different models (Generic2dOscillator)
3. Multi-subnet with inter-projection coupling
4. Stochastic integrators (matched noise seed)
5. TemporalAverage monitor output comparison
6. Boundary clamping
7. Float32 vs float64 precision bounds
8. Large network (68 nodes)
9. Zero initial conditions
"""
import math
import unittest
import warnings

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_allclose

from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.models.oscillator import Generic2dOscillator
from tvb.simulator.integrators import (
    HeunDeterministic,
    EulerDeterministic,
    HeunStochastic,
    EulerStochastic,
)
from tvb.simulator.noise import Additive
from tvb.simulator.monitors import TemporalAverage
from tvb.simulator.hybrid import Subnetwork, NetworkSet
from tvb.simulator.hybrid.inter_projection import InterProjection
from tvb.simulator.hybrid.coupling import Linear as LinearCoupling
from tvb.simulator.backend.nb_hybrid import NbHybridBackend


DT = 0.1  # integration timestep


# ── helpers ───────────────────────────────────────────────────────────────

def _make_subnet(name, model, n_nodes, integrator_cls=HeunDeterministic, dt=DT, **kw):
    """Create a configured Subnetwork."""
    scheme = integrator_cls(dt=dt, **kw)
    sn = Subnetwork(name=name, model=model, scheme=scheme, nnodes=n_nodes)
    return sn


def _configure_all(*objs):
    """Call .configure() on everything that has it."""
    for o in objs:
        if hasattr(o, "configure"):
            o.configure()
    return objs


def _seeded_ic(model, n_nodes, n_modes, seed=42):
    """Return deterministic initial conditions from a seeded RNG."""
    rng = np.random.RandomState(seed)
    ic = np.zeros((model.nvar, n_nodes, n_modes), dtype=np.float64)
    for k, sv in enumerate(model.state_variables):
        lo, hi = model.state_variable_range[sv]
        ic[k] = rng.uniform(lo, hi, (n_nodes, n_modes))
    return ic


def _run_python(ns, nstep, ics):
    """Run the Python hybrid path (NetworkSet.step loop) and return final states.

    Returns list[ndarray] — one (nvar, nnodes, nmodes) float64 array per subnet.
    """
    x = ns.zero_states(initial_states=[ic.copy() for ic in ics])
    ns.init_projection_buffers(x)
    for step in range(1, nstep + 1):
        x = ns.step(step, x)
    return [np.asarray(xi, dtype=np.float64) for xi in x]


def _run_python_observed(ns, nstep, ics):
    """Run Python path and return observed (summed-over-modes) states per step.

    Returns list[ndarray] with shape (nstep, nvoi, nnodes, 1) per subnet.
    """
    x = ns.zero_states(initial_states=[ic.copy() for ic in ics])
    ns.init_projection_buffers(x)
    outputs = []
    for step in range(1, nstep + 1):
        x = ns.step(step, x)
        per_sn = []
        for sn, nx in zip(ns.subnets, x):
            obs = sn.model.observe(np.asarray(nx)).sum(axis=-1)[..., None]
            per_sn.append(obs)
        outputs.append(per_sn)
    # outputs[t][sn] = (nvoi, nnodes, 1)
    result = []
    for si in range(len(ns.subnets)):
        arr = np.stack([outputs[t][si] for t in range(nstep)], axis=0)
        result.append(arr)
    return result


def _run_numba(ns, nstep, ics):
    """Run Numba backend and return final states via return_snapshot.

    Returns list[ndarray] — one (nvar, nnodes, nmodes) float32 array per subnet.
    """
    be = NbHybridBackend()
    compiled = be.compile(ns)
    outputs, snapshot = compiled.run(
        nstep, chunk_size=1,
        initial_states=[ic.copy() for ic in ics],
        return_snapshot=True,
    )
    return [s.astype(np.float32) for s in snapshot["states"]]


def _run_numba_tavg(ns, nstep, ics, period):
    """Run Numba backend with TemporalAverage and return monitor output.

    Returns list[(times, data)] per subnet.
    """
    be = NbHybridBackend()
    ta = TemporalAverage(period=period)
    ta._config_dt(DT)
    results = be.run_network(
        ns, nstep=nstep, monitors=[ta],
        initial_states=[ic.copy() for ic in ics],
    )
    return results[0]  # list of (times, data) per subnet


# ══════════════════════════════════════════════════════════════════════════
# Test classes
# ══════════════════════════════════════════════════════════════════════════


class TestSingleSubnetEuler(unittest.TestCase):
    """Euler deterministic: single MPR subnet, Python vs Numba."""

    N = 6
    NSTEP = 100

    def test_final_state_close(self):
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, self.N, EulerDeterministic)
        _configure_all(m, sn)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=123)
        py = _run_python(ns, self.NSTEP, [ic])
        nb = _run_numba(ns, self.NSTEP, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-5, atol=1e-6,
            err_msg="Euler final state diverged between Python and Numba",
        )

    def test_per_step_trajectory(self):
        """Full per-step trajectory: compare observed output at every step."""
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, self.N, EulerDeterministic)
        _configure_all(m, sn)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=456)
        py_obs = _run_python_observed(ns, self.NSTEP, [ic])

        be = NbHybridBackend()
        results = be.run_network(
            ns, nstep=self.NSTEP, chunk_size=1,
            initial_states=[ic.copy()],
        )
        times, data, ctavg = results[0]
        # data shape: (nstep, nvoi, nnodes, 1)
        assert_allclose(
            data.astype(np.float64), py_obs[0],
            rtol=1e-3, atol=1e-4,
            err_msg="Euler per-step observed trajectory diverged",
        )

    def test_multiple_node_counts(self):
        """Validate across different network sizes."""
        for n_nodes in [2, 4, 10, 32]:
            with self.subTest(n_nodes=n_nodes):
                m = MontbrioPazoRoxin()
                sn = _make_subnet("ctx", m, n_nodes, EulerDeterministic)
                _configure_all(m, sn)
                ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
                ns.configure()

                ic = _seeded_ic(m, n_nodes, 1, seed=789)
                py = _run_python(ns, 50, [ic])
                nb = _run_numba(ns, 50, [ic])

                assert_allclose(
                    nb[0].astype(np.float64), py[0],
                    rtol=1e-5, atol=1e-6,
                    err_msg=f"Euler diverged at n_nodes={n_nodes}",
                )


class TestSingleSubnetHeun(unittest.TestCase):
    """Heun deterministic: single MPR subnet, Python vs Numba."""

    N = 8
    NSTEP = 200

    def test_final_state_close(self):
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=101)
        py = _run_python(ns, self.NSTEP, [ic])
        nb = _run_numba(ns, self.NSTEP, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-5, atol=1e-6,
            err_msg="Heun final state diverged between Python and Numba",
        )

    def test_per_step_trajectory(self):
        """Full per-step observed trajectory comparison (Heun)."""
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=202)
        py_obs = _run_python_observed(ns, self.NSTEP, [ic])

        be = NbHybridBackend()
        results = be.run_network(
            ns, nstep=self.NSTEP, chunk_size=1,
            initial_states=[ic.copy()],
        )
        times, data, ctavg = results[0]

        assert_allclose(
            data.astype(np.float64), py_obs[0],
            rtol=1e-3, atol=1e-4,
            err_msg="Heun per-step observed trajectory diverged",
        )


class TestDifferentModels(unittest.TestCase):
    """Validate with models other than MPR."""

    def test_generic2d_oscillator_euler(self):
        m = Generic2dOscillator()
        sn = _make_subnet("osc", m, 5, EulerDeterministic)
        _configure_all(m, sn)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        ns.configure()

        ic = _seeded_ic(m, 5, 1, seed=303)
        py = _run_python(ns, 150, [ic])
        nb = _run_numba(ns, 150, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-5, atol=1e-6,
            err_msg="Generic2dOscillator Euler diverged",
        )

    def test_generic2d_oscillator_heun(self):
        m = Generic2dOscillator()
        sn = _make_subnet("osc", m, 5, HeunDeterministic)
        _configure_all(m, sn)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        ns.configure()

        ic = _seeded_ic(m, 5, 1, seed=304)
        py = _run_python(ns, 150, [ic])
        nb = _run_numba(ns, 150, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-5, atol=1e-6,
            err_msg="Generic2dOscillator Heun diverged",
        )


class TestMultiSubnetCoupled(unittest.TestCase):
    """Two coupled subnets with InterProjection — Python vs Numba."""

    def _make_coupled_pair(self, int_cls, n1=4, n2=4, coupling_weight=0.01, seed_offset=0):
        m1 = MontbrioPazoRoxin()
        m2 = MontbrioPazoRoxin()
        _configure_all(m1, m2)

        sn1 = _make_subnet("ctx", m1, n1, int_cls)
        sn1.node_indices = np.arange(n1)
        sn2 = _make_subnet("thal", m2, n2, int_cls)
        sn2.node_indices = np.arange(n1, n1 + n2)
        _configure_all(sn1, sn2)

        # Weak coupling with 1-step delay
        # TVB convention: weights shape (n_tgt, n_src)
        weights = sp.csr_matrix(np.ones((n2, n1)) * coupling_weight)
        lengths = sp.csr_matrix(np.ones((n2, n1)) * DT)  # delay = 1 step at speed=1

        proj = InterProjection(
            source=sn1, target=sn2,
            weights=weights, lengths=lengths,
            source_cvar=np.array([0]), target_cvar=np.array([0]),
            scale=1.0, dt=DT, cv=np.array([1.0]),
            cfun=LinearCoupling(),
        )

        ns = NetworkSet(subnets=[sn1, sn2], projections=[proj], stimuli=[])
        ns.configure()

        ic1 = _seeded_ic(m1, n1, 1, seed=400 + seed_offset)
        ic2 = _seeded_ic(m2, n2, 1, seed=401 + seed_offset)
        return ns, [ic1, ic2], ["ctx", "thal"]

    def test_two_coupled_euler(self):
        ns, ics, names = self._make_coupled_pair(EulerDeterministic)
        py = _run_python(ns, 100, ics)
        nb = _run_numba(ns, 100, ics)

        for i, name in enumerate(names):
            assert_allclose(
                nb[i].astype(np.float64), py[i],
                rtol=1e-4, atol=1e-5,
                err_msg=f"Coupled Euler subnet '{name}' diverged",
            )

    def test_two_coupled_heun(self):
        ns, ics, names = self._make_coupled_pair(HeunDeterministic, coupling_weight=0.02)
        py = _run_python(ns, 80, ics)
        nb = _run_numba(ns, 80, ics)

        for i, name in enumerate(names):
            assert_allclose(
                nb[i].astype(np.float64), py[i],
                rtol=1e-4, atol=1e-5,
                err_msg=f"Coupled Heun subnet '{name}' diverged",
            )

    def test_different_sizes(self):
        """Asymmetric subnet sizes: 3→6 nodes."""
        ns, ics, names = self._make_coupled_pair(
            EulerDeterministic, n1=3, n2=6, seed_offset=10
        )
        py = _run_python(ns, 60, ics)
        nb = _run_numba(ns, 60, ics)

        for i, name in enumerate(names):
            assert_allclose(
                nb[i].astype(np.float64), py[i],
                rtol=1e-4, atol=1e-5,
                err_msg=f"Asymmetric coupled subnet '{name}' diverged",
            )


class TestStochasticEuler(unittest.TestCase):
    """EulerStochastic: Python vs Numba with matched noise seed."""

    N = 4
    NSTEP = 50
    NSIG = 1e-4

    def _make_noise(self, seed=42):
        noise = Additive(nsig=np.array([self.NSIG]))
        noise.noise_seed = seed
        noise.random_stream = np.random.RandomState(seed)
        noise.configure_white(DT)
        return noise

    def test_matched_noise_seed(self):
        """Same noise seed → identical results."""
        m = MontbrioPazoRoxin()
        _configure_all(m)

        # Python
        noise_py = self._make_noise(seed=42)
        sn_py = _make_subnet("ctx", m, self.N, EulerStochastic, noise=noise_py)
        _configure_all(sn_py)
        ns_py = NetworkSet(subnets=[sn_py], projections=[], stimuli=[])
        ns_py.configure()
        ic = _seeded_ic(m, self.N, 1, seed=600)
        py = _run_python(ns_py, self.NSTEP, [ic])

        # Numba (fresh RNG with same seed)
        noise_nb = self._make_noise(seed=42)
        sn_nb = _make_subnet("ctx", m, self.N, EulerStochastic, noise=noise_nb)
        _configure_all(sn_nb)
        ns_nb = NetworkSet(subnets=[sn_nb], projections=[], stimuli=[])
        ns_nb.configure()
        nb = _run_numba(ns_nb, self.NSTEP, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-4, atol=1e-5,
            err_msg="EulerStochastic diverged with matched noise seed",
        )


class TestStochasticHeun(unittest.TestCase):
    """HeunStochastic: Python vs Numba with matched noise seed."""

    N = 4
    NSTEP = 50
    NSIG = 1e-4

    def _make_noise(self, seed=42):
        noise = Additive(nsig=np.array([self.NSIG]))
        noise.noise_seed = seed
        noise.random_stream = np.random.RandomState(seed)
        noise.configure_white(DT)
        return noise

    def test_matched_noise_seed(self):
        m = MontbrioPazoRoxin()
        _configure_all(m)

        noise_py = self._make_noise(seed=42)
        sn_py = _make_subnet("ctx", m, self.N, HeunStochastic, noise=noise_py)
        _configure_all(sn_py)
        ns_py = NetworkSet(subnets=[sn_py], projections=[], stimuli=[])
        ns_py.configure()
        ic = _seeded_ic(m, self.N, 1, seed=700)
        py = _run_python(ns_py, self.NSTEP, [ic])

        noise_nb = self._make_noise(seed=42)
        sn_nb = _make_subnet("ctx", m, self.N, HeunStochastic, noise=noise_nb)
        _configure_all(sn_nb)
        ns_nb = NetworkSet(subnets=[sn_nb], projections=[], stimuli=[])
        ns_nb.configure()
        nb = _run_numba(ns_nb, self.NSTEP, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-4, atol=1e-5,
            err_msg="HeunStochastic diverged with matched noise seed",
        )


class TestTemporalAverageMonitor(unittest.TestCase):
    """TemporalAverage monitor: Python manual average vs Numba backend."""

    N = 5
    NSTEP = 100
    PERIOD = 1.0  # 10 steps at dt=0.1

    def test_tavg_output_matches(self):
        """Manually computed Python TemporalAverage matches Numba."""
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=800)

        # Python: step-by-step, collect observed states, compute manual TAVG
        ns_py = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        ns_py.configure()
        x = ns_py.zero_states(initial_states=[ic.copy()])
        ns_py.init_projection_buffers(x)

        istep = int(round(self.PERIOD / DT))
        stock = []
        py_periods = []
        for step in range(1, self.NSTEP + 1):
            x = ns_py.step(step, x)
            for sn_obj, nx in zip(ns_py.subnets, x):
                obs = sn_obj.model.observe(np.asarray(nx)).sum(axis=-1)[..., None]
                stock.append(obs)
            if step % istep == 0:
                avg = np.mean(stock[-istep:], axis=0)
                py_periods.append(avg)

        py_data = np.stack(py_periods, axis=0)  # (n_periods, nvoi, nnodes, 1)

        # Numba: with TemporalAverage monitor
        nb_results = _run_numba_tavg(ns, self.NSTEP, [ic], self.PERIOD)
        nb_times, nb_data = nb_results[0]

        n_periods = min(py_data.shape[0], nb_data.shape[0])
        self.assertGreater(n_periods, 0, "No TemporalAverage output")

        assert_allclose(
            nb_data[:n_periods].astype(np.float64),
            py_data[:n_periods],
            rtol=1e-4, atol=1e-5,
            err_msg="TemporalAverage output diverged between backends",
        )

    def test_tavg_period_alignment(self):
        """Verify time stamps are centered within each period."""
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, self.N, EulerDeterministic)
        _configure_all(m, sn)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=801)

        nb_results = _run_numba_tavg(ns, self.NSTEP, [ic], self.PERIOD)
        nb_times, _ = nb_results[0]

        # Numba backend computes mid_t = (2*t_start + chunk_size - 1) * 0.5 * dt
        # where t_start is 1-indexed step of chunk start.
        # For period=1.0, dt=0.1, chunk_size=10:
        # First period: t_start=1, mid_t = (2 + 9) * 0.05 = 0.55
        # Second period: t_start=11, mid_t = (22 + 9) * 0.05 = 1.55
        istep = int(round(self.PERIOD / DT))
        expected_times = [
            (2 * (k * istep + 1) + istep - 1) * 0.5 * DT
            for k in range(len(nb_times))
        ]
        assert_allclose(
            nb_times.astype(np.float64),
            expected_times,
            rtol=1e-6, atol=1e-3,
            err_msg="TemporalAverage time stamps not at expected midpoints",
        )


class TestBoundaryClamping(unittest.TestCase):
    """Verify that bound_and_clamp produces identical results."""

    def test_lower_boundary_euler(self):
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, 3, EulerDeterministic)
        _configure_all(m, sn)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        ns.configure()

        # ICs at lower boundary
        ic = np.zeros((m.nvar, 3, 1), dtype=np.float64)
        for k, sv in enumerate(m.state_variables):
            ic[k, :, :] = m.state_variable_range[sv][0]

        py = _run_python(ns, 20, [ic])
        nb = _run_numba(ns, 20, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-5, atol=1e-6,
            err_msg="Boundary clamping diverged",
        )

    def test_upper_boundary_heun(self):
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, 3, HeunDeterministic)
        _configure_all(m, sn)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        ns.configure()

        # ICs at upper boundary
        ic = np.zeros((m.nvar, 3, 1), dtype=np.float64)
        for k, sv in enumerate(m.state_variables):
            ic[k, :, :] = m.state_variable_range[sv][1]

        py = _run_python(ns, 20, [ic])
        nb = _run_numba(ns, 20, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-5, atol=1e-6,
            err_msg="Upper boundary clamping diverged",
        )


class TestFloatingPointPrecision(unittest.TestCase):
    """Verify that float32 vs float64 differences stay bounded."""

    N = 8
    NSTEP = 500

    def test_max_divergence_bounded_euler(self):
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, self.N, EulerDeterministic)
        _configure_all(m, sn)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=999)
        py = _run_python(ns, self.NSTEP, [ic])
        nb = _run_numba(ns, self.NSTEP, [ic])

        max_err = np.max(np.abs(nb[0].astype(np.float64) - py[0]))
        self.assertLess(
            max_err, 1e-3,
            f"Float32/64 divergence too large after {self.NSTEP} Euler steps: {max_err:.2e}"
        )

    def test_relative_error_bounded_heun(self):
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=888)
        py = _run_python(ns, self.NSTEP, [ic])
        nb = _run_numba(ns, self.NSTEP, [ic])

        py_arr = py[0]
        nb_arr = nb[0].astype(np.float64)
        mask = np.abs(py_arr) > 1e-6
        if mask.any():
            rel_err = np.max(np.abs((nb_arr[mask] - py_arr[mask]) / py_arr[mask]))
            self.assertLess(
                rel_err, 1e-3,
                f"Relative error too large after {self.NSTEP} Heun steps: {rel_err:.2e}"
            )


class TestLargeNetwork(unittest.TestCase):
    """Validate with a 68-node network (realistic cortical parcellation)."""

    N = 68
    NSTEP = 200

    def test_euler_68_nodes(self):
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, self.N, EulerDeterministic)
        _configure_all(m, sn)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=1234)
        py = _run_python(ns, self.NSTEP, [ic])
        nb = _run_numba(ns, self.NSTEP, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-4, atol=1e-5,
            err_msg="68-node Euler diverged",
        )

    def test_heun_68_nodes(self):
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=5678)
        py = _run_python(ns, self.NSTEP, [ic])
        nb = _run_numba(ns, self.NSTEP, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-4, atol=1e-5,
            err_msg="68-node Heun diverged",
        )


class TestZeroInitialConditions(unittest.TestCase):
    """Edge case: all-zero initial conditions."""

    def test_zero_ic_euler(self):
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, 4, EulerDeterministic)
        _configure_all(m, sn)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        ns.configure()

        ic = np.zeros((m.nvar, 4, 1), dtype=np.float64)
        py = _run_python(ns, 50, [ic])
        nb = _run_numba(ns, 50, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-5, atol=1e-6,
            err_msg="Zero IC Euler diverged",
        )
        self.assertTrue(np.all(np.isfinite(nb[0])), "NaN in Numba output")
        self.assertTrue(np.all(np.isfinite(py[0])), "NaN in Python output")

    def test_zero_ic_heun(self):
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, 4, HeunDeterministic)
        _configure_all(m, sn)
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        ns.configure()

        ic = np.zeros((m.nvar, 4, 1), dtype=np.float64)
        py = _run_python(ns, 50, [ic])
        nb = _run_numba(ns, 50, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-5, atol=1e-6,
            err_msg="Zero IC Heun diverged",
        )


if __name__ == "__main__":
    unittest.main()
