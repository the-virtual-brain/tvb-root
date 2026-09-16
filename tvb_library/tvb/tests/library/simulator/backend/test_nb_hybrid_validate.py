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
10. Stimulus numerical parity (constant, time-varying, scale, multi-cvar,
    spatial selectivity, combined with projections)
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
from tvb.simulator.hybrid.intra_projection import IntraProjection
from tvb.simulator.hybrid.coupling import Linear as LinearCoupling, Scaling as ScalingCoupling
from tvb.simulator.hybrid.stimulus import Stim
from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.datatypes.patterns import StimuliRegion
from tvb.datatypes import equations as eqs
from tvb.datatypes.connectivity import Connectivity


DT = 0.1  # integration timestep


# ── connectivity / stimulus helpers ─────────────────────────────────────

def _make_minimal_connectivity(n_nodes: int) -> Connectivity:
    """Create a minimal Connectivity object required by StimuliRegion."""
    conn = Connectivity(
        centres=np.zeros((n_nodes, 3)),
        weights=np.zeros((n_nodes, n_nodes)),
        tract_lengths=np.zeros((n_nodes, n_nodes)),
        region_labels=np.array([str(i) for i in range(n_nodes)]),
        speed=np.array([1.0]),
    )
    conn.configure()
    return conn


def _make_constant_stim(subnet, amplitude, target_node=0,
                        target_cvar=None, projection_scale=1.0,
                        simulation_length=None):
    """Create a constant-amplitude (Linear a=0) StimuliRegion → Stim.

    Parameters
    ----------
    subnet : Subnetwork
        Target subnetwork.
    amplitude : float
        Constant stimulus amplitude.
    target_node : int or None
        Which node to stimulate.  None → all nodes.
    target_cvar : ndarray or None
        Target coupling-variable indices.  Default ``[0]``.
    projection_scale : float
        Global scaling factor.
    simulation_length : float or None
        Total simulation duration (ms).  Default ``100 * DT``.
    """
    if target_cvar is None:
        target_cvar = np.array([0], dtype=np.int_)
    if simulation_length is None:
        simulation_length = 100 * DT
    n = subnet.nnodes
    conn = _make_minimal_connectivity(n)
    temporal = eqs.Linear()
    temporal.parameters["a"] = 0.0
    temporal.parameters["b"] = float(amplitude)
    weight = np.zeros(n)
    if target_node is None:
        weight[:] = 1.0
    else:
        weight[target_node] = 1.0
    stim_pattern = StimuliRegion(
        temporal=temporal, connectivity=conn, weight=weight,
    )
    stim = Stim(
        target=subnet,
        stimulus=stim_pattern,
        target_cvar=target_cvar,
        projection_scale=projection_scale,
    )
    stim.configure(simulation_length=simulation_length)
    return stim


def _make_sinusoidal_stim(subnet, amp, frequency, target_node=0,
                          target_cvar=None, projection_scale=1.0,
                          simulation_length=None):
    """Sinusoid-driven StimuliRegion → Stim."""
    if target_cvar is None:
        target_cvar = np.array([0], dtype=np.int_)
    if simulation_length is None:
        simulation_length = 200 * DT
    n = subnet.nnodes
    conn = _make_minimal_connectivity(n)
    temporal = eqs.Sinusoid()
    temporal.parameters["amp"] = float(amp)
    temporal.parameters["frequency"] = float(frequency)
    weight = np.zeros(n)
    weight[target_node] = 1.0
    stim_pattern = StimuliRegion(
        temporal=temporal, connectivity=conn, weight=weight,
    )
    stim = Stim(
        target=subnet,
        stimulus=stim_pattern,
        target_cvar=target_cvar,
        projection_scale=projection_scale,
    )
    stim.configure(simulation_length=simulation_length)
    return stim


def _make_pulse_stim(subnet, onset, period, tau, amp, target_node=0,
                     target_cvar=None, projection_scale=1.0,
                     simulation_length=None):
    """PulseTrain-driven StimuliRegion → Stim."""
    if target_cvar is None:
        target_cvar = np.array([0], dtype=np.int_)
    if simulation_length is None:
        simulation_length = 200 * DT
    n = subnet.nnodes
    conn = _make_minimal_connectivity(n)
    temporal = eqs.PulseTrain()
    temporal.parameters["onset"] = float(onset)
    temporal.parameters["T"] = float(period)
    temporal.parameters["tau"] = float(tau)
    temporal.parameters["amp"] = float(amp)
    weight = np.zeros(n)
    weight[target_node] = 1.0
    stim_pattern = StimuliRegion(
        temporal=temporal, connectivity=conn, weight=weight,
    )
    stim = Stim(
        target=subnet,
        stimulus=stim_pattern,
        target_cvar=target_cvar,
        projection_scale=projection_scale,
    )
    stim.configure(simulation_length=simulation_length)
    return stim


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
        ns = NetworkSet(subnets=[sn], projections=[])
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
        ns = NetworkSet(subnets=[sn], projections=[])
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
                ns = NetworkSet(subnets=[sn], projections=[])
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
        ns = NetworkSet(subnets=[sn], projections=[])
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
        ns = NetworkSet(subnets=[sn], projections=[])
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
        ns = NetworkSet(subnets=[sn], projections=[])
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
        ns = NetworkSet(subnets=[sn], projections=[])
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

        ns = NetworkSet(subnets=[sn1, sn2], projections=[proj])
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
        ns_py = NetworkSet(subnets=[sn_py], projections=[])
        ns_py.configure()
        ic = _seeded_ic(m, self.N, 1, seed=600)
        py = _run_python(ns_py, self.NSTEP, [ic])

        # Numba (fresh RNG with same seed)
        noise_nb = self._make_noise(seed=42)
        sn_nb = _make_subnet("ctx", m, self.N, EulerStochastic, noise=noise_nb)
        _configure_all(sn_nb)
        ns_nb = NetworkSet(subnets=[sn_nb], projections=[])
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
        ns_py = NetworkSet(subnets=[sn_py], projections=[])
        ns_py.configure()
        ic = _seeded_ic(m, self.N, 1, seed=700)
        py = _run_python(ns_py, self.NSTEP, [ic])

        noise_nb = self._make_noise(seed=42)
        sn_nb = _make_subnet("ctx", m, self.N, HeunStochastic, noise=noise_nb)
        _configure_all(sn_nb)
        ns_nb = NetworkSet(subnets=[sn_nb], projections=[])
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
        ns = NetworkSet(subnets=[sn], projections=[])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=800)

        # Python: step-by-step, collect observed states, compute manual TAVG
        ns_py = NetworkSet(subnets=[sn], projections=[])
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
        ns = NetworkSet(subnets=[sn], projections=[])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=801)

        nb_results = _run_numba_tavg(ns, self.NSTEP, [ic], self.PERIOD)
        nb_times, _ = nb_results[0]

        # Match TemporalAverage.sample(): time = (step - istep / 2) * dt.
        # For period=1.0, dt=0.1, the first two samples are 0.5 and 1.5.
        istep = int(round(self.PERIOD / DT))
        expected_times = [
            ((k + 1) * istep - istep / 2.0) * DT
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
        ns = NetworkSet(subnets=[sn], projections=[])
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
        ns = NetworkSet(subnets=[sn], projections=[])
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
        ns = NetworkSet(subnets=[sn], projections=[])
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
        ns = NetworkSet(subnets=[sn], projections=[])
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
        ns = NetworkSet(subnets=[sn], projections=[])
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
        ns = NetworkSet(subnets=[sn], projections=[])
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
        ns = NetworkSet(subnets=[sn], projections=[])
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
        ns = NetworkSet(subnets=[sn], projections=[])
        ns.configure()

        ic = np.zeros((m.nvar, 4, 1), dtype=np.float64)
        py = _run_python(ns, 50, [ic])
        nb = _run_numba(ns, 50, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-5, atol=1e-6,
            err_msg="Zero IC Heun diverged",
        )


# ══════════════════════════════════════════════════════════════════════════
# 10. Stimulus numerical parity tests
# ══════════════════════════════════════════════════════════════════════════


class TestStimulusParity(unittest.TestCase):
    """Stimulus numerical parity: Python vs Numba backend with external stimuli.

    Each test constructs a network that includes one or more :class:`Stim`
    objects, runs it through both backends with identical initial conditions,
    and asserts that the results match within floating-point tolerance.
    """

    N = 6
    NSTEP = 100

    # ── constant stimulus ────────────────────────────────────────────

    def test_constant_stimulus_euler(self):
        """Constant-amplitude stimulus: Euler parity."""
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, self.N, EulerDeterministic)
        _configure_all(m, sn)
        stim = _make_constant_stim(
            sn, amplitude=0.05, target_node=0,
            simulation_length=self.NSTEP * DT,
        )
        sn.stimuli = [stim]
        ns = NetworkSet(subnets=[sn], projections=[])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=1001)
        py = _run_python(ns, self.NSTEP, [ic])
        nb = _run_numba(ns, self.NSTEP, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-5, atol=1e-6,
            err_msg="Constant-stimulus Euler diverged between Python and Numba",
        )

    def test_constant_stimulus_heun(self):
        """Constant-amplitude stimulus: Heun parity."""
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn)
        stim = _make_constant_stim(
            sn, amplitude=0.05, target_node=0,
            simulation_length=self.NSTEP * DT,
        )
        sn.stimuli = [stim]
        ns = NetworkSet(subnets=[sn], projections=[])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=1002)
        py = _run_python(ns, self.NSTEP, [ic])
        nb = _run_numba(ns, self.NSTEP, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-5, atol=1e-6,
            err_msg="Constant-stimulus Heun diverged between Python and Numba",
        )

    # ── time-varying stimulus ────────────────────────────────────────

    def test_sinusoidal_stimulus_parity(self):
        """Sinusoid stimulus: per-step observed trajectory parity."""
        m = MontbrioPazoRoxin()
        nstep = 150
        sn = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn)
        stim = _make_sinusoidal_stim(
            sn, amp=0.03, frequency=0.05, target_node=0,
            simulation_length=nstep * DT,
        )
        sn.stimuli = [stim]
        ns = NetworkSet(subnets=[sn], projections=[])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=1003)
        py_obs = _run_python_observed(ns, nstep, [ic])

        be = NbHybridBackend()
        results = be.run_network(
            ns, nstep=nstep, chunk_size=1,
            initial_states=[ic.copy()],
        )
        _times, data, _ctavg = results[0]

        assert_allclose(
            data.astype(np.float64), py_obs[0],
            rtol=1e-3, atol=1e-4,
            err_msg="Sinusoid stimulus per-step trajectory diverged",
        )

    def test_pulse_train_stimulus_parity(self):
        """PulseTrain stimulus: final state parity."""
        m = MontbrioPazoRoxin()
        nstep = 150
        sn = _make_subnet("ctx", m, self.N, EulerDeterministic)
        _configure_all(m, sn)
        stim = _make_pulse_stim(
            sn, onset=2.0, period=10.0, tau=2.0, amp=0.05,
            target_node=0,
            simulation_length=nstep * DT,
        )
        sn.stimuli = [stim]
        ns = NetworkSet(subnets=[sn], projections=[])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=1004)
        py = _run_python(ns, nstep, [ic])
        nb = _run_numba(ns, nstep, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-4, atol=1e-5,
            err_msg="PulseTrain stimulus diverged between Python and Numba",
        )

    # ── projection scale ────────────────────────────────────────────

    def test_stimulus_projection_scale(self):
        """projection_scale ≠ 1.0: both backends apply same scaling."""
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn)
        stim = _make_constant_stim(
            sn, amplitude=0.05, target_node=0,
            projection_scale=2.5,
            simulation_length=self.NSTEP * DT,
        )
        sn.stimuli = [stim]
        ns = NetworkSet(subnets=[sn], projections=[])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=1005)
        py = _run_python(ns, self.NSTEP, [ic])
        nb = _run_numba(ns, self.NSTEP, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-5, atol=1e-6,
            err_msg="Stimulus projection_scale parity diverged",
        )

    # ── multi-cvar stimulus ──────────────────────────────────────────

    def test_stimulus_multi_cvar(self):
        """Stimulus targeting multiple coupling variables (r and V)."""
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn)
        stim = _make_constant_stim(
            sn, amplitude=0.02, target_node=0,
            target_cvar=np.array([0, 1], dtype=np.int_),
            simulation_length=self.NSTEP * DT,
        )
        sn.stimuli = [stim]
        ns = NetworkSet(subnets=[sn], projections=[])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=1006)
        py = _run_python(ns, self.NSTEP, [ic])
        nb = _run_numba(ns, self.NSTEP, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-5, atol=1e-6,
            err_msg="Multi-cvar stimulus diverged between Python and Numba",
        )

    # ── spatial selectivity ─────────────────────────────────────────

    def test_stimulus_spatial_selectivity_parity(self):
        """Only node 0 stimulated: both stimulated and unstimulated nodes match."""
        m = MontbrioPazoRoxin()
        nstep = 80
        sn = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn)
        stim = _make_constant_stim(
            sn, amplitude=0.1, target_node=0,
            simulation_length=nstep * DT,
        )
        sn.stimuli = [stim]
        ns = NetworkSet(subnets=[sn], projections=[])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=1007)
        py = _run_python(ns, nstep, [ic])
        nb = _run_numba(ns, nstep, [ic])

        # Stimulated node (0)
        assert_allclose(
            nb[0][:, 0, :].astype(np.float64), py[0][:, 0, :],
            rtol=1e-5, atol=1e-6,
            err_msg="Stimulated node 0 diverged",
        )
        # Non-stimulated node (1)
        assert_allclose(
            nb[0][:, 1, :].astype(np.float64), py[0][:, 1, :],
            rtol=1e-5, atol=1e-6,
            err_msg="Non-stimulated node 1 diverged",
        )

    # ── stimulus + inter-projection ─────────────────────────────────

    def test_stimulus_with_interprojection_parity(self):
        """Stimulus and inter-projection coupling applied simultaneously."""
        m1 = MontbrioPazoRoxin()
        m2 = MontbrioPazoRoxin()
        _configure_all(m1, m2)

        n1, n2 = 4, 4
        sn1 = _make_subnet("ctx", m1, n1, HeunDeterministic)
        sn1.node_indices = np.arange(n1)
        sn2 = _make_subnet("thal", m2, n2, HeunDeterministic)
        sn2.node_indices = np.arange(n1, n1 + n2)
        _configure_all(sn1, sn2)

        # Weak coupling: ctx → thal
        weights = sp.csr_matrix(np.ones((n2, n1)) * 0.01)
        lengths = sp.csr_matrix(np.ones((n2, n1)) * DT)
        proj = InterProjection(
            source=sn1, target=sn2,
            weights=weights, lengths=lengths,
            source_cvar=np.array([0]), target_cvar=np.array([0]),
            scale=1.0, dt=DT, cv=np.array([1.0]),
            cfun=LinearCoupling(),
        )

        # Stimulus on cortex node 0
        nstep = 80
        stim = _make_constant_stim(
            sn1, amplitude=0.05, target_node=0,
            simulation_length=nstep * DT,
        )

        sn1.stimuli = [stim]
        ns = NetworkSet(subnets=[sn1, sn2], projections=[proj])
        ns.configure()

        ic1 = _seeded_ic(m1, n1, 1, seed=1008)
        ic2 = _seeded_ic(m2, n2, 1, seed=1009)
        py = _run_python(ns, nstep, [ic1, ic2])
        nb = _run_numba(ns, nstep, [ic1, ic2])

        for i, name in enumerate(["ctx", "thal"]):
            assert_allclose(
                nb[i].astype(np.float64), py[i],
                rtol=1e-4, atol=1e-5,
                err_msg=f"Stimulus+projection subnet '{name}' diverged",
            )

    # ── multiple stimuli ────────────────────────────────────────────

    def test_multiple_stimuli_parity(self):
        """Two stimuli on the same subnet targeting different cvars."""
        m = MontbrioPazoRoxin()
        nstep = 80
        sn = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn)

        sim_len = nstep * DT
        stim_r = _make_constant_stim(
            sn, amplitude=0.03, target_node=0,
            target_cvar=np.array([0], dtype=np.int_),
            simulation_length=sim_len,
        )
        stim_v = _make_constant_stim(
            sn, amplitude=0.02, target_node=1,
            target_cvar=np.array([1], dtype=np.int_),
            simulation_length=sim_len,
        )

        sn.stimuli = [stim_r, stim_v]
        ns = NetworkSet(
            subnets=[sn], projections=[],
        )
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=1010)
        py = _run_python(ns, nstep, [ic])
        nb = _run_numba(ns, nstep, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=5e-3, atol=1e-3,
            err_msg="Multiple-stimuli diverged between Python and Numba",
        )

    # ── zero-amplitude ≡ baseline ───────────────────────────────────

    def test_zero_amplitude_stimulus_matches_baseline(self):
        """Zero-amplitude stimulus should produce identical output to no stimulus."""
        m = MontbrioPazoRoxin()
        nstep = 50

        # Baseline: no stimulus
        sn_base = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn_base)
        ns_base = NetworkSet(subnets=[sn_base], projections=[])
        ns_base.configure()

        ic = _seeded_ic(m, self.N, 1, seed=1011)
        py_base = _run_python(ns_base, nstep, [ic])
        nb_base = _run_numba(ns_base, nstep, [ic])

        # With zero-amplitude stimulus
        sn_stim = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn_stim)
        stim = _make_constant_stim(
            sn_stim, amplitude=0.0, target_node=0,
            simulation_length=nstep * DT,
        )
        sn_stim.stimuli = [stim]
        ns_stim = NetworkSet(subnets=[sn_stim], projections=[])
        ns_stim.configure()

        py_stim = _run_python(ns_stim, nstep, [ic])
        nb_stim = _run_numba(ns_stim, nstep, [ic])

        # Zero-stimulus Python ≡ baseline Python
        assert_allclose(
            py_stim[0], py_base[0],
            rtol=1e-12, atol=1e-14,
            err_msg="Zero-amplitude stimulus python differs from no-stimulus baseline",
        )

        # Zero-stimulus Numba ≡ baseline Numba
        assert_allclose(
            nb_stim[0].astype(np.float64), nb_base[0].astype(np.float64),
            rtol=1e-5, atol=1e-6,
            err_msg="Zero-amplitude stimulus Numba differs from no-stimulus baseline",
        )

    # ── stimulus + intra-projection ──────────────────────────────────

    def test_stimulus_with_intraprojection_parity(self):
        """Single subnet with both intra-projection coupling and stimulus."""
        m = MontbrioPazoRoxin()
        nstep = 80
        n = self.N
        sn = _make_subnet("ctx", m, n, HeunDeterministic)
        # Intra-projection: weak local coupling with zero delay
        w = sp.csr_matrix(np.ones((n, n)) * 0.005)
        w.setdiag(0.0)
        intra = IntraProjection(
            source_cvar=np.array([0], dtype=np.int_),
            target_cvar=np.array([0], dtype=np.int_),
            weights=w,
            lengths=sp.csr_matrix(np.zeros((n, n))),
            cv=1.0, dt=DT, scale=1.0,
        )
        sn.projections = [intra]
        _configure_all(m, sn)
        stim = _make_constant_stim(
            sn, amplitude=0.03, target_node=0,
            simulation_length=nstep * DT,
        )
        sn.stimuli = [stim]
        ns = NetworkSet(subnets=[sn], projections=[])
        ns.configure()

        ic = _seeded_ic(m, n, 1, seed=1012)
        py = _run_python(ns, nstep, [ic])
        nb = _run_numba(ns, nstep, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-4, atol=1e-5,
            err_msg="Stimulus+intra-projection diverged between Python and Numba",
        )

    # ── stimulus on all nodes ────────────────────────────────────────

    def test_stimulus_all_nodes_parity(self):
        """Stimulus applied uniformly to all nodes: parity check."""
        m = MontbrioPazoRoxin()
        sn = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn)
        stim = _make_constant_stim(
            sn, amplitude=0.02, target_node=None,  # all nodes
            simulation_length=self.NSTEP * DT,
        )
        sn.stimuli = [stim]
        ns = NetworkSet(subnets=[sn], projections=[])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=1013)
        py = _run_python(ns, self.NSTEP, [ic])
        nb = _run_numba(ns, self.NSTEP, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-5, atol=1e-6,
            err_msg="All-nodes stimulus diverged between Python and Numba",
        )

    # ── stimulus with Generic2dOscillator ─────────────────────────────

    def test_stimulus_generic2d_oscillator_parity(self):
        """Stimulus on a non-MPR model (Generic2dOscillator)."""
        m = Generic2dOscillator()
        nstep = 80
        sn = _make_subnet("osc", m, 5, HeunDeterministic)
        _configure_all(m, sn)
        stim = _make_constant_stim(
            sn, amplitude=0.05, target_node=0,
            simulation_length=nstep * DT,
        )
        sn.stimuli = [stim]
        ns = NetworkSet(subnets=[sn], projections=[])
        ns.configure()

        ic = _seeded_ic(m, 5, 1, seed=1014)
        py = _run_python(ns, nstep, [ic])
        nb = _run_numba(ns, nstep, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-5, atol=1e-6,
            err_msg="Generic2dOscillator stimulus diverged between Python and Numba",
        )

    # ── large network with stimulus ──────────────────────────────────

    def test_stimulus_large_network_68_nodes(self):
        """68-node network with stimulus: parity at realistic scale."""
        m = MontbrioPazoRoxin()
        nstep = 100
        n = 68
        sn = _make_subnet("ctx", m, n, HeunDeterministic)
        _configure_all(m, sn)
        stim = _make_constant_stim(
            sn, amplitude=0.03, target_node=0,
            simulation_length=nstep * DT,
        )
        sn.stimuli = [stim]
        ns = NetworkSet(subnets=[sn], projections=[])
        ns.configure()

        ic = _seeded_ic(m, n, 1, seed=1015)
        py = _run_python(ns, nstep, [ic])
        nb = _run_numba(ns, nstep, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-4, atol=1e-5,
            err_msg="68-node stimulus diverged between Python and Numba",
        )

    # ── stimulus with stochastic integrator ──────────────────────────

    def test_stimulus_euler_stochastic_parity(self):
        """Stimulus + EulerStochastic with matched noise seed."""
        m = MontbrioPazoRoxin()
        nstep = 50
        n = 4
        nsig = 1e-4
        seed = 42

        # Python path
        noise_py = Additive(nsig=np.array([nsig]))
        noise_py.noise_seed = seed
        noise_py.random_stream = np.random.RandomState(seed)
        noise_py.configure_white(DT)
        sn_py = _make_subnet("ctx", m, n, EulerStochastic, noise=noise_py)
        _configure_all(m, sn_py)
        stim_py = _make_constant_stim(
            sn_py, amplitude=0.03, target_node=0,
            simulation_length=nstep * DT,
        )
        sn_py.stimuli = [stim_py]
        ns_py = NetworkSet(subnets=[sn_py], projections=[])
        ns_py.configure()
        ic = _seeded_ic(m, n, 1, seed=1016)
        py = _run_python(ns_py, nstep, [ic])

        # Numba path (fresh noise with same seed)
        noise_nb = Additive(nsig=np.array([nsig]))
        noise_nb.noise_seed = seed
        noise_nb.random_stream = np.random.RandomState(seed)
        noise_nb.configure_white(DT)
        sn_nb = _make_subnet("ctx", m, n, EulerStochastic, noise=noise_nb)
        _configure_all(m, sn_nb)
        stim_nb = _make_constant_stim(
            sn_nb, amplitude=0.03, target_node=0,
            simulation_length=nstep * DT,
        )
        sn_nb.stimuli = [stim_nb]
        ns_nb = NetworkSet(subnets=[sn_nb], projections=[])
        ns_nb.configure()
        nb = _run_numba(ns_nb, nstep, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-4, atol=1e-5,
            err_msg="EulerStochastic+stimulus diverged with matched noise seed",
        )

    # ── stimulus across node counts ──────────────────────────────────

    def test_stimulus_multiple_node_counts(self):
        """Validate stimulus parity across different network sizes."""
        for n_nodes in [2, 4, 10, 32]:
            with self.subTest(n_nodes=n_nodes):
                m = MontbrioPazoRoxin()
                sn = _make_subnet("ctx", m, n_nodes, HeunDeterministic)
                _configure_all(m, sn)
                stim = _make_constant_stim(
                    sn, amplitude=0.05, target_node=0,
                    simulation_length=50 * DT,
                )
                sn.stimuli = [stim]
                ns = NetworkSet(subnets=[sn], projections=[])
                ns.configure()

                ic = _seeded_ic(m, n_nodes, 1, seed=1017)
                py = _run_python(ns, 50, [ic])
                nb = _run_numba(ns, 50, [ic])

                assert_allclose(
                    nb[0].astype(np.float64), py[0],
                    rtol=1e-4, atol=1e-5,
                    err_msg=f"Stimulus diverged at n_nodes={n_nodes}",
                )

    # ── stimulus + intra + inter combined ────────────────────────────

    def test_stimulus_with_intra_and_inter_parity(self):
        """Full-featured: intra-projection + inter-projection + stimulus."""
        m1 = MontbrioPazoRoxin()
        m2 = MontbrioPazoRoxin()
        _configure_all(m1, m2)

        n1, n2 = 4, 4
        sn1 = _make_subnet("ctx", m1, n1, HeunDeterministic)
        sn1.node_indices = np.arange(n1)
        # Intra-projection on source
        w_intra = sp.csr_matrix(np.ones((n1, n1)) * 0.005)
        w_intra.setdiag(0.0)
        intra = IntraProjection(
            source_cvar=np.array([0], dtype=np.int_),
            target_cvar=np.array([0], dtype=np.int_),
            weights=w_intra,
            lengths=sp.csr_matrix(np.zeros((n1, n1))),
            cv=1.0, dt=DT, scale=1.0,
            cfun=ScalingCoupling(a=np.array([1.5])),
        )
        sn1.projections = [intra]

        sn2 = _make_subnet("thal", m2, n2, HeunDeterministic)
        sn2.node_indices = np.arange(n1, n1 + n2)
        _configure_all(sn1, sn2)

        # Inter-projection ctx → thal
        w_inter = sp.csr_matrix(np.ones((n2, n1)) * 0.01)
        lengths = sp.csr_matrix(np.ones((n2, n1)) * DT)
        proj = InterProjection(
            source=sn1, target=sn2,
            weights=w_inter, lengths=lengths,
            source_cvar=np.array([0]), target_cvar=np.array([0]),
            scale=1.0, dt=DT, cv=np.array([1.0]),
            cfun=LinearCoupling(),
        )

        # Stimulus on cortex
        nstep = 60
        stim = _make_constant_stim(
            sn1, amplitude=0.03, target_node=0,
            simulation_length=nstep * DT,
        )

        sn1.stimuli = [stim]
        ns = NetworkSet(subnets=[sn1, sn2], projections=[proj])
        ns.configure()

        ic1 = _seeded_ic(m1, n1, 1, seed=1018)
        ic2 = _seeded_ic(m2, n2, 1, seed=1019)
        py = _run_python(ns, nstep, [ic1, ic2])
        nb = _run_numba(ns, nstep, [ic1, ic2])

        for i, name in enumerate(["ctx", "thal"]):
            assert_allclose(
                nb[i].astype(np.float64), py[i],
                rtol=1e-4, atol=1e-5,
                err_msg=f"Stimulus+intra+inter subnet '{name}' diverged",
            )


    # ── resume with stimulus ────────────────────────────────────────

    def test_stimulus_snapshot_resume_parity(self):
        """Snapshot/resume: stimulus is correctly preserved across calls."""
        m = MontbrioPazoRoxin()
        n1, n2 = 30, 30
        sn = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn)
        stim = _make_constant_stim(
            sn, amplitude=0.03, target_node=0,
            simulation_length=(n1 + n2) * DT,
        )
        sn.stimuli = [stim]
        ns = NetworkSet(subnets=[sn], projections=[])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=1020)

        # --- Python: two-phase run ---
        x = ns.zero_states(initial_states=[ic.copy()])
        ns.init_projection_buffers(x)
        for step in range(1, n1 + 1):
            x = ns.step(step, x)
        py_mid = np.asarray(list(x)[0], dtype=np.float64).copy()
        for step in range(n1 + 1, n1 + n2 + 1):
            x = ns.step(step, x)
        py_final = np.asarray(list(x)[0], dtype=np.float64)

        # --- Numba: two-phase run via snapshot/resume ---
        be = NbHybridBackend()
        compiled = be.compile(ns)
        outputs, snapshot = compiled.run(
            n1, chunk_size=1,
            initial_states=[ic.copy()],
            return_snapshot=True,
        )
        nb_mid = snapshot["states"][0].astype(np.float64)

        # Resume from snapshot
        outputs2, snapshot2 = compiled.run(
            n2, chunk_size=1,
            initial_states=snapshot["states"],
            _initial_buffers=snapshot["buffers"],
            return_snapshot=True,
        )
        nb_final = snapshot2["states"][0].astype(np.float64)

        # Mid-point parity
        assert_allclose(
            nb_mid, py_mid,
            rtol=1e-5, atol=1e-6,
            err_msg="Stimulus snapshot mid-point diverged",
        )
        # Final parity (full run ≡ two-phase run)
        py_full = _run_python(ns, n1 + n2, [ic])
        assert_allclose(
            nb_final, py_full[0],
            rtol=1e-5, atol=1e-6,
            err_msg="Stimulus resume final state diverged from continuous run",
        )

    # ── spatially-varying weights ─────────────────────────────────────

    def test_stimulus_gradient_weights_parity(self):
        """Gradient spatial weights: all nodes stimulated at different strengths."""
        m = MontbrioPazoRoxin()
        nstep = 80
        n = self.N
        sn = _make_subnet("ctx", m, n, HeunDeterministic)
        _configure_all(m, sn)

        # Gradient weights: node 0 → 1.0, node 5 → 0.0
        conn = _make_minimal_connectivity(n)
        temporal = eqs.Linear()
        temporal.parameters["a"] = 0.0
        temporal.parameters["b"] = 0.05
        weight = np.linspace(1.0, 0.0, n)
        stim_pattern = StimuliRegion(
            temporal=temporal, connectivity=conn, weight=weight,
        )
        stim = Stim(
            target=sn, stimulus=stim_pattern,
            target_cvar=np.array([0], dtype=np.int_),
            projection_scale=1.0,
        )
        stim.configure(simulation_length=nstep * DT)

        sn.stimuli = [stim]
        ns = NetworkSet(subnets=[sn], projections=[])
        ns.configure()

        ic = _seeded_ic(m, n, 1, seed=1021)
        py = _run_python(ns, nstep, [ic])
        nb = _run_numba(ns, nstep, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-5, atol=1e-6,
            err_msg="Gradient-weights stimulus diverged",
        )

    def test_stimulus_checkerboard_weights_parity(self):
        """Alternating on/off spatial weights."""
        m = MontbrioPazoRoxin()
        nstep = 80
        n = self.N
        sn = _make_subnet("ctx", m, n, HeunDeterministic)
        _configure_all(m, sn)

        conn = _make_minimal_connectivity(n)
        temporal = eqs.Sinusoid()
        temporal.parameters["amp"] = 0.05
        temporal.parameters["frequency"] = 0.05
        weight = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])[:n]
        stim_pattern = StimuliRegion(
            temporal=temporal, connectivity=conn, weight=weight,
        )
        stim = Stim(
            target=sn, stimulus=stim_pattern,
            target_cvar=np.array([0], dtype=np.int_),
            projection_scale=1.0,
        )
        stim.configure(simulation_length=nstep * DT)

        sn.stimuli = [stim]
        ns = NetworkSet(subnets=[sn], projections=[])
        ns.configure()

        ic = _seeded_ic(m, n, 1, seed=1022)
        py = _run_python(ns, nstep, [ic])
        nb = _run_numba(ns, nstep, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-3, atol=1e-4,
            err_msg="Checkerboard-weights Sinusoid stimulus diverged",
        )

    # ── Cosine stimulus on validate side ─────────────────────────────

    def test_cosine_stimulus_parity(self):
        """Cosine-driven stimulus: per-step observed trajectory parity."""
        m = MontbrioPazoRoxin()
        nstep = 100
        sn = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn)

        conn = _make_minimal_connectivity(self.N)
        temporal = eqs.Cosine()
        temporal.parameters["amp"] = 0.04
        temporal.parameters["frequency"] = 0.05
        weight = np.zeros(self.N)
        weight[0] = 1.0
        stim_pattern = StimuliRegion(
            temporal=temporal, connectivity=conn, weight=weight,
        )
        stim = Stim(
            target=sn, stimulus=stim_pattern,
            target_cvar=np.array([0], dtype=np.int_),
        )
        stim.configure(simulation_length=nstep * DT)

        sn.stimuli = [stim]
        ns = NetworkSet(subnets=[sn], projections=[])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=1023)
        py_obs = _run_python_observed(ns, nstep, [ic])

        be = NbHybridBackend()
        results = be.run_network(
            ns, nstep=nstep, chunk_size=1,
            initial_states=[ic.copy()],
        )
        _times, data, _ctavg = results[0]

        assert_allclose(
            data.astype(np.float64), py_obs[0],
            rtol=1e-3, atol=1e-4,
            err_msg="Cosine stimulus per-step trajectory diverged",
        )

    # ── Alpha stimulus on validate side ──────────────────────────────

    def test_alpha_stimulus_parity(self):
        """Alpha-function stimulus: final state parity."""
        m = MontbrioPazoRoxin()
        nstep = 100
        sn = _make_subnet("ctx", m, self.N, EulerDeterministic)
        _configure_all(m, sn)

        conn = _make_minimal_connectivity(self.N)
        temporal = eqs.Alpha()
        temporal.parameters["onset"] = 1.0
        temporal.parameters["alpha"] = 5.0
        temporal.parameters["beta"] = 15.0
        weight = np.zeros(self.N)
        weight[0] = 1.0
        stim_pattern = StimuliRegion(
            temporal=temporal, connectivity=conn, weight=weight,
        )
        stim = Stim(
            target=sn, stimulus=stim_pattern,
            target_cvar=np.array([0], dtype=np.int_),
        )
        stim.configure(simulation_length=nstep * DT)

        sn.stimuli = [stim]
        ns = NetworkSet(subnets=[sn], projections=[])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=1024)
        py = _run_python(ns, nstep, [ic])
        nb = _run_numba(ns, nstep, [ic])

        assert_allclose(
            nb[0].astype(np.float64), py[0],
            rtol=1e-4, atol=1e-5,
            err_msg="Alpha stimulus diverged between Python and Numba",
        )

    # ── pulsetrain transition capture ────────────────────────────────

    def test_pulsetrain_transition_parity(self):
        """PulseTrain on→off and off→on transitions captured identically.

        Compares per-step observed output to verify that every transition
        frame matches between backends.
        """
        m = MontbrioPazoRoxin()
        nstep = 100  # 10 ms; pulse at onset=2ms, T=5ms, tau=1ms → pulses at 2–3, 7–8 ms
        sn = _make_subnet("ctx", m, self.N, HeunDeterministic)
        _configure_all(m, sn)

        conn = _make_minimal_connectivity(self.N)
        temporal = eqs.PulseTrain()
        temporal.parameters["onset"] = 2.0
        temporal.parameters["T"] = 5.0
        temporal.parameters["tau"] = 1.0
        temporal.parameters["amp"] = 0.1
        weight = np.zeros(self.N)
        weight[0] = 1.0
        stim_pattern = StimuliRegion(
            temporal=temporal, connectivity=conn, weight=weight,
        )
        stim = Stim(
            target=sn, stimulus=stim_pattern,
            target_cvar=np.array([0], dtype=np.int_),
        )
        stim.configure(simulation_length=nstep * DT)

        sn.stimuli = [stim]
        ns = NetworkSet(subnets=[sn], projections=[])
        ns.configure()

        ic = _seeded_ic(m, self.N, 1, seed=1025)
        py_obs = _run_python_observed(ns, nstep, [ic])

        be = NbHybridBackend()
        results = be.run_network(
            ns, nstep=nstep, chunk_size=1,
            initial_states=[ic.copy()],
        )
        _times, data, _ctavg = results[0]

        assert_allclose(
            data.astype(np.float64), py_obs[0],
            rtol=1e-3, atol=1e-4,
            err_msg="PulseTrain transition per-step trajectory diverged",
        )

    # ── Heun stochastic + stimulus per-step ──────────────────────────

    def test_heun_stochastic_stimulus_per_step_parity(self):
        """HeunStochastic + constant stimulus: per-step trajectory parity."""
        m = MontbrioPazoRoxin()
        nstep = 40
        n = 4
        nsig = 1e-4
        seed = 55

        # Python
        noise_py = Additive(nsig=np.array([nsig]))
        noise_py.noise_seed = seed
        noise_py.random_stream = np.random.RandomState(seed)
        noise_py.configure_white(DT)
        sn_py = _make_subnet("ctx", m, n, HeunStochastic, noise=noise_py)
        _configure_all(m, sn_py)
        stim_py = _make_constant_stim(
            sn_py, amplitude=0.03, target_node=0,
            simulation_length=nstep * DT,
        )
        sn_py.stimuli = [stim_py]
        ns_py = NetworkSet(subnets=[sn_py], projections=[])
        ns_py.configure()
        ic = _seeded_ic(m, n, 1, seed=1026)
        py_obs = _run_python_observed(ns_py, nstep, [ic])

        # Numba (fresh noise with same seed)
        noise_nb = Additive(nsig=np.array([nsig]))
        noise_nb.noise_seed = seed
        noise_nb.random_stream = np.random.RandomState(seed)
        noise_nb.configure_white(DT)
        sn_nb = _make_subnet("ctx", m, n, HeunStochastic, noise=noise_nb)
        _configure_all(m, sn_nb)
        stim_nb = _make_constant_stim(
            sn_nb, amplitude=0.03, target_node=0,
            simulation_length=nstep * DT,
        )
        sn_nb.stimuli = [stim_nb]
        ns_nb = NetworkSet(subnets=[sn_nb], projections=[])
        ns_nb.configure()

        be = NbHybridBackend()
        results = be.run_network(
            ns_nb, nstep=nstep, chunk_size=1,
            initial_states=[ic.copy()],
        )
        _times, data, _ctavg = results[0]

        assert_allclose(
            data.astype(np.float64), py_obs[0],
            rtol=1e-2, atol=1e-3,
            err_msg="HeunStochastic+stimulus per-step trajectory diverged",
        )


if __name__ == "__main__":
    unittest.main()
