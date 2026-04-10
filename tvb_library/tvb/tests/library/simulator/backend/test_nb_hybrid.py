# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
Tests for the Numba hybrid backend (NbHybridBackend).

Each test runs an equivalent configuration through both the pure-Python
NetworkSet step loop and NbHybridBackend.run_network() and asserts that
the output trajectories agree within floating-point tolerance.
"""

import unittest
import math
import time
import numpy as np
import scipy.sparse as sp

from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.integrators import HeunDeterministic, EulerDeterministic
from tvb.simulator.integrators import HeunStochastic, EulerStochastic
from tvb.simulator.noise import Additive
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.hybrid.inter_projection import InterProjection
from tvb.simulator.hybrid.intra_projection import IntraProjection
from tvb.simulator.hybrid.coupling import Linear, Scaling
from tvb.simulator.backend.nb_hybrid import NbHybridBackend
from tvb.datatypes.patterns import StimuliRegion
from tvb.datatypes import equations as eqs
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.hybrid.stimulus import Stim


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

DT = 0.01


def _mpr_subnetwork(name: str, n_nodes: int, integrator_cls=HeunDeterministic) -> Subnetwork:
    model = MontbrioPazoRoxin()
    model.configure()
    scheme = integrator_cls(dt=DT)
    sn = Subnetwork(
        name=name,
        model=model,
        scheme=scheme,
        nnodes=n_nodes,
    )
    return sn


def _mpr_stochastic_subnetwork(
    name: str,
    n_nodes: int,
    integrator_cls=EulerStochastic,
    nsig: float = 1e-4,
    seed: int = 42,
) -> Subnetwork:
    """Create an MPR subnetwork with a stochastic integrator and fixed noise seed."""
    model = MontbrioPazoRoxin()
    model.configure()
    noise = Additive(nsig=np.array([nsig]))  # scalar nsig → correct broadcast for 3D state
    noise.noise_seed = seed
    noise.random_stream = np.random.RandomState(seed)
    noise.configure_white(DT)  # required for generate() to work in Python path
    scheme = integrator_cls(dt=DT, noise=noise)
    scheme.configure_boundaries(model)
    sn = Subnetwork(
        name=name,
        model=model,
        scheme=scheme,
        nnodes=n_nodes,
    )
    sn.configure()
    return sn


def _make_minimal_connectivity(n_nodes: int) -> Connectivity:
    """Create a minimal connectivity object for StimuliRegion."""
    conn = Connectivity(
        centres=np.zeros((n_nodes, 3)),
        weights=np.zeros((n_nodes, n_nodes)),
        tract_lengths=np.zeros((n_nodes, n_nodes)),
        region_labels=np.array([str(i) for i in range(n_nodes)]),
        speed=np.array([1.0]),
    )
    conn.configure()
    return conn


def _make_stim(subnetwork: Subnetwork, amplitude: float = 0.05) -> Stim:
    """Create a Sinusoid-driven StimuliRegion stimulus targeting cvar 0."""
    n = subnetwork.nnodes
    conn = _make_minimal_connectivity(n)
    temporal = eqs.Sinusoid()
    temporal.parameters['amp'] = np.float64(amplitude)
    temporal.parameters['frequency'] = np.float64(0.1)
    weight = np.zeros(n)
    weight[0] = 1.0  # Only stimulate node 0
    stim_pattern = StimuliRegion(
        temporal=temporal,
        connectivity=conn,
        weight=weight,
    )
    stim = Stim(
        target=subnetwork,
        stimulus=stim_pattern,
        target_cvar=np.array([0], dtype=np.int_),
        projection_scale=1.0,
    )
    nstep_len = 100  # configure for enough steps
    stim.configure(simulation_length=nstep_len * DT)
    return stim


def _sparse_weights(n_tgt: int, n_src: int, seed: int = 0, density: float = 1.0) -> sp.csr_matrix:
    """Random weight matrix with configurable density (default fully dense)."""
    rng = np.random.RandomState(seed)
    w = rng.uniform(0.0, 0.5, (n_tgt, n_src)).astype(np.float64)
    if density < 1.0:
        mask = rng.uniform(0.0, 1.0, (n_tgt, n_src)) > density
        w[mask] = 0.0
    np.fill_diagonal(w, 0.0)
    return sp.csr_matrix(w)


def _zero_lengths(n_tgt: int, n_src: int) -> sp.csr_matrix:
    return sp.csr_matrix(np.zeros((n_tgt, n_src)))


def _run_python_loop(network_set: NetworkSet, nstep: int, x0_list: list) -> list:
    """Run pure-Python NetworkSet loop and return per-step states for each subnetwork."""
    x = network_set.States(*[arr.copy() for arr in x0_list])
    network_set.init_projection_buffers(x)

    # Collect raw outputs: list of lists of state snapshots
    outputs = [[] for _ in network_set.subnets]
    for step in range(1, nstep + 1):
        x = network_set.step(step, x)
        for i, xi in enumerate(x):
            outputs[i].append(xi.copy())

    # Stack: (nstep, n_vars, n_nodes, n_modes)
    return [np.stack(o, axis=0) for o in outputs]


def _run_nb(network_set: NetworkSet, nstep: int, x0_list: list,
            print_source: bool = False) -> list:
    """Run NbHybridBackend and return per-step states (one per subnetwork).

    chunk_size=1 gives temporal average of 1 step = raw output.
    """
    backend = NbHybridBackend()
    results = backend.run_network(
        network_set,
        nstep=nstep,
        chunk_size=1,
        print_source=print_source,
        initial_states=x0_list,
    )
    # results: list of (times, data) where data is (nstep, n_voi, n_nodes, n_modes)
    # Return only the data arrays
    return [data for _, data in results]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNbHybridSingleSubnet(unittest.TestCase):
    """Single MPR subnetwork, no projections — validates integrator code-gen."""

    def _make_net(self, integrator_cls=HeunDeterministic):
        n = 4
        sn = _mpr_subnetwork("ctx", n, integrator_cls)
        sn.configure()
        network_set = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        network_set.configure()
        return network_set, n

    def _run_both(self, integrator_cls, nstep=10):
        network_set, n = self._make_net(integrator_cls)
        rng = np.random.RandomState(7)
        model = network_set.subnets[0].model
        x0 = rng.uniform(0.0, 0.2, (model.nvar, n, 1)).astype(np.float64)
        x0[0] = np.abs(x0[0])  # r must be >= 0

        py_out = _run_python_loop(network_set, nstep, [x0])  # list of 1 array

        # Reset buffers for nb run
        network_set.init_projection_buffers(network_set.States(x0.copy()))
        nb_out = _run_nb(network_set, nstep, [x0])

        return py_out[0], nb_out[0]

    def test_heun_no_projection(self):
        py, nb = self._run_both(HeunDeterministic, nstep=10)
        # py shape: (nstep, n_vars, n_nodes, n_modes)
        # nb shape: (nstep, n_voi, n_nodes, n_modes)
        # For MPR, n_voi == n_vars == 2, so shapes match
        self.assertEqual(py.shape, nb.shape,
                         f"Shape mismatch: python {py.shape} vs nb {nb.shape}")
        np.testing.assert_allclose(
            nb, py, rtol=1e-3, atol=1e-4,
            err_msg="Heun single-subnet: Numba output differs from Python"
        )

    def test_euler_no_projection(self):
        py, nb = self._run_both(EulerDeterministic, nstep=10)
        self.assertEqual(py.shape, nb.shape)
        np.testing.assert_allclose(
            nb, py, rtol=1e-3, atol=1e-4,
            err_msg="Euler single-subnet: Numba output differs from Python"
        )


class TestNbHybridIntraProjection(unittest.TestCase):
    """Single subnetwork with intra-projection (local coupling)."""

    def _make_net(self, n=5, delay=False):
        sn = _mpr_subnetwork("ctx", n)
        w = _sparse_weights(n, n, seed=1)
        lengths = sp.csr_matrix(
            w.toarray() * (10.0 if delay else 0.0)
        )
        intra = IntraProjection(
            source_cvar=np.array([0], dtype=np.int_),
            target_cvar=np.array([0], dtype=np.int_),
            weights=w,
            lengths=lengths,
            cv=1.0,
            dt=DT,
            scale=1.0,
        )
        sn.projections = [intra]
        sn.configure()
        network_set = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        network_set.configure()
        return network_set, n

    def _run_both(self, n=5, delay=False, nstep=8):
        network_set, n = self._make_net(n=n, delay=delay)
        rng = np.random.RandomState(11)
        model = network_set.subnets[0].model
        x0 = rng.uniform(0.0, 0.2, (model.nvar, n, 1)).astype(np.float64)
        x0[0] = np.abs(x0[0])

        py_out = _run_python_loop(network_set, nstep, [x0])
        nb_out = _run_nb(network_set, nstep, [x0])
        return py_out[0], nb_out[0]

    def test_intra_no_delay(self):
        py, nb = self._run_both(delay=False)
        self.assertEqual(py.shape, nb.shape)
        np.testing.assert_allclose(
            nb, py, rtol=1e-3, atol=1e-4,
            err_msg="Intra no-delay: Numba differs from Python"
        )

    def test_intra_with_delay(self):
        py, nb = self._run_both(delay=True)
        self.assertEqual(py.shape, nb.shape)
        np.testing.assert_allclose(
            nb, py, rtol=1e-3, atol=1e-4,
            err_msg="Intra with-delay: Numba differs from Python"
        )


class TestNbHybridInterProjection(unittest.TestCase):
    """Two MPR subnetworks with an inter-projection."""

    def _make_net(self, n_src=4, n_tgt=5, delay=False, cvar_mapping="1_to_1"):
        sn_src = _mpr_subnetwork("src_net", n_src)
        sn_tgt = _mpr_subnetwork("tgt_net", n_tgt)
        sn_src.configure()
        sn_tgt.configure()

        w = _sparse_weights(n_tgt, n_src, seed=3)
        lengths = sp.csr_matrix(
            w.toarray() * (15.0 if delay else 0.0)
        )

        if cvar_mapping == "1_to_1":
            sc = np.array([0], dtype=np.int_)
            tc = np.array([0], dtype=np.int_)
        elif cvar_mapping == "1_to_many":
            sc = np.array([0], dtype=np.int_)
            tc = np.array([0, 1], dtype=np.int_)
        elif cvar_mapping == "many_to_1":
            sc = np.array([0, 1], dtype=np.int_)
            tc = np.array([0], dtype=np.int_)

        inter = InterProjection(
            source=sn_src,
            target=sn_tgt,
            source_cvar=sc,
            target_cvar=tc,
            weights=w,
            lengths=lengths,
            cv=1.0,
            dt=DT,
            scale=1.0,
        )

        network_set = NetworkSet(
            subnets=[sn_src, sn_tgt],
            projections=[inter],
            stimuli=[],
        )
        network_set.configure()
        return network_set, n_src, n_tgt

    def _run_both(self, n_src=4, n_tgt=5, delay=False, cvar_mapping="1_to_1", nstep=8):
        network_set, n_src, n_tgt = self._make_net(
            n_src=n_src, n_tgt=n_tgt, delay=delay, cvar_mapping=cvar_mapping
        )
        rng = np.random.RandomState(13)
        m_src = network_set.subnets[0].model
        m_tgt = network_set.subnets[1].model
        x0_src = rng.uniform(0.0, 0.2, (m_src.nvar, n_src, 1)).astype(np.float64)
        x0_tgt = rng.uniform(0.0, 0.2, (m_tgt.nvar, n_tgt, 1)).astype(np.float64)
        x0_src[0] = np.abs(x0_src[0])
        x0_tgt[0] = np.abs(x0_tgt[0])

        py_out = _run_python_loop(network_set, nstep, [x0_src, x0_tgt])
        nb_out = _run_nb(network_set, nstep, [x0_src, x0_tgt])
        return py_out, nb_out

    def test_inter_no_delay_1_to_1(self):
        py, nb = self._run_both(delay=False, cvar_mapping="1_to_1")
        for i, (p_i, n_i) in enumerate(zip(py, nb)):
            self.assertEqual(p_i.shape, n_i.shape)
            np.testing.assert_allclose(
                n_i, p_i, rtol=1e-3, atol=1e-4,
                err_msg=f"Inter 1-to-1 no-delay subnetwork {i}: Numba differs"
            )

    def test_inter_with_delay_1_to_1(self):
        py, nb = self._run_both(delay=True, cvar_mapping="1_to_1")
        for i, (p_i, n_i) in enumerate(zip(py, nb)):
            self.assertEqual(p_i.shape, n_i.shape)
            np.testing.assert_allclose(
                n_i, p_i, rtol=1e-3, atol=1e-4,
                err_msg=f"Inter 1-to-1 with-delay subnetwork {i}: Numba differs"
            )

    def test_inter_cvar_1_to_many(self):
        py, nb = self._run_both(delay=False, cvar_mapping="1_to_many")
        for i, (p_i, n_i) in enumerate(zip(py, nb)):
            self.assertEqual(p_i.shape, n_i.shape)
            np.testing.assert_allclose(
                n_i, p_i, rtol=1e-3, atol=1e-4,
                err_msg=f"Inter 1-to-many subnetwork {i}: Numba differs"
            )

    def test_inter_cvar_many_to_1(self):
        py, nb = self._run_both(delay=False, cvar_mapping="many_to_1")
        for i, (p_i, n_i) in enumerate(zip(py, nb)):
            self.assertEqual(p_i.shape, n_i.shape)
            np.testing.assert_allclose(
                n_i, p_i, rtol=1e-3, atol=1e-4,
                err_msg=f"Inter many-to-1 subnetwork {i}: Numba differs"
            )


class TestNbHybridCompatibilityCheck(unittest.TestCase):
    """Verify that incompatible models/integrators are rejected cleanly."""

    def test_rejects_unsupported_model(self):
        from tvb.simulator.models import Generic2dOscillator
        from tvb.simulator.integrators import HeunDeterministic as HD

        model = Generic2dOscillator()
        model.configure()
        scheme = HD(dt=DT)
        sn = Subnetwork(name="sn", model=model, scheme=scheme, nnodes=3)
        sn.configure()
        nets = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        nets.configure()

        with self.assertRaises(NotImplementedError):
            NbHybridBackend().run_network(nets, nstep=5)

    def test_rejects_mismatched_dt(self):
        sn1 = _mpr_subnetwork("a", 3, HeunDeterministic)
        sn2 = _mpr_subnetwork("b", 3, EulerDeterministic)
        # Give sn2 a different dt
        sn2.scheme = EulerDeterministic(dt=DT * 2)
        sn1.configure()
        sn2.configure()
        nets = NetworkSet(subnets=[sn1, sn2], projections=[], stimuli=[])
        nets.configure()

        with self.assertRaises(ValueError):
            NbHybridBackend().run_network(nets, nstep=5)


# ---------------------------------------------------------------------------
# Coupling function tests
# ---------------------------------------------------------------------------

class TestNbHybridCfun(unittest.TestCase):
    """Test Linear and Scaling coupling functions match between Python and Numba."""

    def _make_net_with_cfun(self, cfun, n=5, delay=False):
        sn = _mpr_subnetwork("ctx", n)
        w = _sparse_weights(n, n, seed=7)
        lengths = sp.csr_matrix(w.toarray() * (10.0 if delay else 0.0))
        intra = IntraProjection(
            source_cvar=np.array([0], dtype=np.int_),
            target_cvar=np.array([0], dtype=np.int_),
            weights=w,
            lengths=lengths,
            cv=1.0,
            dt=DT,
            scale=1.0,
            cfun=cfun,
        )
        sn.projections = [intra]
        sn.configure()
        network_set = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        network_set.configure()
        return network_set, n

    def _run_both(self, cfun, delay=False, nstep=8):
        network_set, n = self._make_net_with_cfun(cfun, delay=delay)
        rng = np.random.RandomState(99)
        model = network_set.subnets[0].model
        x0 = rng.uniform(0.0, 0.2, (model.nvar, n, 1)).astype(np.float64)
        x0[0] = np.abs(x0[0])

        py = _run_python_loop(network_set, nstep, [x0])[0]
        nb = _run_nb(network_set, nstep, [x0])[0]
        return py, nb

    def test_linear_cfun(self):
        cfun = Linear(a=np.array([0.5]), b=np.array([0.1]))
        py, nb = self._run_both(cfun)
        self.assertEqual(py.shape, nb.shape)
        np.testing.assert_allclose(nb, py, rtol=1e-3, atol=1e-4,
                                   err_msg="Linear cfun: Numba differs from Python")

    def test_scaling_cfun(self):
        cfun = Scaling(a=np.array([2.0]))
        py, nb = self._run_both(cfun)
        self.assertEqual(py.shape, nb.shape)
        np.testing.assert_allclose(nb, py, rtol=1e-3, atol=1e-4,
                                   err_msg="Scaling cfun: Numba differs from Python")


# ---------------------------------------------------------------------------
# Target-scales tests
# ---------------------------------------------------------------------------

class TestNbHybridTargetScales(unittest.TestCase):
    """Test that target_scales are applied correctly by the Numba backend."""

    def _make_net(self, target_scales, n_src=4, n_tgt=5):
        sn_src = _mpr_subnetwork("source", n_src)
        sn_tgt = _mpr_subnetwork("target", n_tgt)
        sn_src.configure()
        sn_tgt.configure()
        w = _sparse_weights(n_tgt, n_src, seed=5)
        inter = InterProjection(
            source=sn_src,
            target=sn_tgt,
            source_cvar=np.array([0, 1], dtype=np.int_),
            target_cvar=np.array([0, 1], dtype=np.int_),
            weights=w,
            lengths=_zero_lengths(n_tgt, n_src),
            cv=1.0,
            dt=DT,
            scale=1.0,
            target_scales=target_scales,
        )
        network_set = NetworkSet(
            subnets=[sn_src, sn_tgt],
            projections=[inter],
            stimuli=[],
        )
        network_set.configure()
        return network_set, n_src, n_tgt

    def test_target_scales_n2n(self):
        ts = np.array([0.3, 0.7])
        network_set, n_src, n_tgt = self._make_net(ts)
        rng = np.random.RandomState(55)
        m = MontbrioPazoRoxin()
        m.configure()
        x0_src = rng.uniform(0.0, 0.2, (m.nvar, n_src, 1)).astype(np.float64)
        x0_tgt = rng.uniform(0.0, 0.2, (m.nvar, n_tgt, 1)).astype(np.float64)
        x0_src[0] = np.abs(x0_src[0])
        x0_tgt[0] = np.abs(x0_tgt[0])

        py = _run_python_loop(network_set, 8, [x0_src, x0_tgt])
        nb = _run_nb(network_set, 8, [x0_src, x0_tgt])

        for i, (p, n_) in enumerate(zip(py, nb)):
            np.testing.assert_allclose(n_, p, rtol=1e-3, atol=1e-4,
                                       err_msg=f"target_scales subnetwork {i}: mismatch")


# ---------------------------------------------------------------------------
# Stochastic integrator tests
# ---------------------------------------------------------------------------

class TestNbHybridStochastic(unittest.TestCase):
    """Test EulerStochastic and HeunStochastic integrators."""

    NSTEP = 10
    N = 4
    NSIG = 1e-4  # small noise for near-deterministic comparison

    def _make_stochastic_net(self, integrator_cls, seed=42):
        sn = _mpr_stochastic_subnetwork("ctx", self.N,
                                        integrator_cls=integrator_cls,
                                        nsig=self.NSIG, seed=seed)
        network_set = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        network_set.configure()
        return network_set

    def _run_python_stochastic(self, network_set, nstep, x0_list):
        """Python loop that also calls scheme correctly for stochastic."""
        return _run_python_loop(network_set, nstep, x0_list)

    def _run_both_same_seed(self, integrator_cls, seed=42):
        network_set = self._make_stochastic_net(integrator_cls, seed=seed)
        rng = np.random.RandomState(77)
        model = network_set.subnets[0].model
        x0 = rng.uniform(0.1, 0.3, (model.nvar, self.N, 1)).astype(np.float64)

        # Save RNG state
        saved_state = network_set.subnets[0].scheme.noise.random_stream.get_state()

        # Python path
        network_set.subnets[0].scheme.noise.random_stream.set_state(saved_state)
        py = _run_python_loop(network_set, self.NSTEP, [x0])[0]

        # Numba path (reset to same seed first)
        network_set.subnets[0].scheme.noise.random_stream.set_state(saved_state)
        nb = _run_nb(network_set, self.NSTEP, [x0])[0]

        return py, nb

    def test_euler_stochastic_shape(self):
        """Output shape is correct for EulerStochastic."""
        network_set = self._make_stochastic_net(EulerStochastic)
        m = network_set.subnets[0].model
        x0 = np.random.RandomState(77).uniform(0.1, 0.3, (m.nvar, self.N, 1)).astype(np.float64)
        nb = _run_nb(network_set, self.NSTEP, [x0])[0]
        self.assertEqual(nb.shape, (self.NSTEP, m.nvar, self.N, 1))

    def test_heun_stochastic_shape(self):
        """Output shape is correct for HeunStochastic."""
        network_set = self._make_stochastic_net(HeunStochastic)
        m = network_set.subnets[0].model
        x0 = np.random.RandomState(77).uniform(0.1, 0.3, (m.nvar, self.N, 1)).astype(np.float64)
        nb = _run_nb(network_set, self.NSTEP, [x0])[0]
        self.assertEqual(nb.shape, (self.NSTEP, m.nvar, self.N, 1))

    def test_euler_stochastic_matches_python(self):
        """EulerStochastic Numba output matches Python with same RNG seed."""
        py, nb = self._run_both_same_seed(EulerStochastic)
        self.assertEqual(py.shape, nb.shape)
        np.testing.assert_allclose(nb, py, rtol=1e-2, atol=1e-3,
                                   err_msg="EulerStochastic: Numba differs from Python")

    def test_heun_stochastic_matches_python(self):
        """HeunStochastic Numba output matches Python with same RNG seed."""
        py, nb = self._run_both_same_seed(HeunStochastic)
        self.assertEqual(py.shape, nb.shape)
        np.testing.assert_allclose(nb, py, rtol=1e-2, atol=1e-3,
                                   err_msg="HeunStochastic: Numba differs from Python")

    def test_noise_has_effect(self):
        """Stochastic output must differ from deterministic (noise is applied)."""
        n = self.N
        model_d = MontbrioPazoRoxin()
        model_d.configure()
        sn_det = Subnetwork(name="ctx", model=model_d,
                            scheme=EulerDeterministic(dt=DT), nnodes=n)
        sn_det.configure()
        nets_det = NetworkSet(subnets=[sn_det], projections=[], stimuli=[])
        nets_det.configure()

        # Large noise to ensure visible effect
        sn_stoch = _mpr_stochastic_subnetwork("ctx", n, EulerStochastic, nsig=0.1)
        nets_stoch = NetworkSet(subnets=[sn_stoch], projections=[], stimuli=[])
        nets_stoch.configure()

        x0 = np.random.RandomState(3).uniform(0.1, 0.3, (2, n, 1)).astype(np.float64)
        nb_det = _run_nb(nets_det, 50, [x0])[0]
        nb_stoch = _run_nb(nets_stoch, 50, [x0])[0]

        # With large nsig, trajectories should diverge
        max_diff = np.max(np.abs(nb_stoch - nb_det))
        self.assertGreater(max_diff, 1e-4,
                           "Stochastic output should differ from deterministic")


# ---------------------------------------------------------------------------
# Stimulus tests
# ---------------------------------------------------------------------------

class TestNbHybridStimulus(unittest.TestCase):
    """Test that stimulus is correctly applied by the Numba backend."""

    NSTEP = 20
    N = 4

    def _make_net_with_stim(self, n=None):
        n = n or self.N
        sn = _mpr_subnetwork("ctx", n)
        sn.configure()
        stim = _make_stim(sn, amplitude=0.05)
        network_set = NetworkSet(subnets=[sn], projections=[], stimuli=[stim])
        network_set.configure()
        return network_set, n

    def test_stimulus_does_not_crash(self):
        """Should complete without exceptions."""
        network_set, n = self._make_net_with_stim()
        model = network_set.subnets[0].model
        x0 = np.random.RandomState(5).uniform(0.1, 0.3, (model.nvar, n, 1)).astype(np.float64)
        nb = _run_nb(network_set, self.NSTEP, [x0])
        self.assertEqual(len(nb), 1)
        self.assertEqual(nb[0].shape[0], self.NSTEP)

    def test_stimulus_matches_python(self):
        """Numba stimulus output must match Python loop within tolerance."""
        network_set, n = self._make_net_with_stim()
        model = network_set.subnets[0].model
        x0 = np.random.RandomState(9).uniform(0.1, 0.3, (model.nvar, n, 1)).astype(np.float64)

        py = _run_python_loop(network_set, self.NSTEP, [x0])[0]
        nb = _run_nb(network_set, self.NSTEP, [x0])[0]

        self.assertEqual(py.shape, nb.shape)
        np.testing.assert_allclose(nb, py, rtol=1e-3, atol=1e-4,
                                   err_msg="Stimulus: Numba differs from Python")

    def test_stimulus_has_effect(self):
        """Stimulated output must differ from unstimulated baseline."""
        n = self.N
        sn_base = _mpr_subnetwork("ctx", n)
        sn_base.configure()
        nets_base = NetworkSet(subnets=[sn_base], projections=[], stimuli=[])
        nets_base.configure()

        sn_stim = _mpr_subnetwork("ctx", n)
        sn_stim.configure()
        stim = _make_stim(sn_stim, amplitude=1.0)
        nets_stim = NetworkSet(subnets=[sn_stim], projections=[], stimuli=[stim])
        nets_stim.configure()

        x0 = np.random.RandomState(17).uniform(0.1, 0.3, (2, n, 1)).astype(np.float64)
        nb_base = _run_nb(nets_base, self.NSTEP, [x0])[0]
        nb_stim = _run_nb(nets_stim, self.NSTEP, [x0])[0]

        max_diff = np.max(np.abs(nb_stim - nb_base))
        self.assertGreater(max_diff, 1e-5,
                           "Stimulated output should differ from baseline")


# ---------------------------------------------------------------------------
# End-to-end tests
# ---------------------------------------------------------------------------

class TestNbHybridEndToEnd(unittest.TestCase):
    """Full pipeline end-to-end tests: multi-subnet with delays, cfun, stimulus.

    Each test runs *both* Python and Numba for the same network and asserts
    that the outputs agree within numerical tolerance.
    """

    NSTEP = 20

    def test_two_subnets_delays_cfun(self):
        """2-subnet network with inter-projection delays and Linear cfun."""
        n_src, n_tgt = 5, 6
        sn_src = _mpr_subnetwork("src", n_src)
        sn_tgt = _mpr_subnetwork("tgt", n_tgt)
        sn_src.configure()
        sn_tgt.configure()
        w = _sparse_weights(n_tgt, n_src, seed=21)
        lengths = sp.csr_matrix(w.toarray() * 20.0)
        inter = InterProjection(
            source=sn_src, target=sn_tgt,
            source_cvar=np.array([0], dtype=np.int_),
            target_cvar=np.array([0], dtype=np.int_),
            weights=w, lengths=lengths,
            cv=1.0, dt=DT, scale=0.5,
            cfun=Linear(a=np.array([2.0]), b=np.array([0.0])),
        )
        network_set = NetworkSet(
            subnets=[sn_src, sn_tgt],
            projections=[inter],
            stimuli=[],
        )
        network_set.configure()

        rng = np.random.RandomState(31)
        m = MontbrioPazoRoxin()
        m.configure()
        x0_src = rng.uniform(0.1, 0.3, (m.nvar, n_src, 1)).astype(np.float64)
        x0_tgt = rng.uniform(0.1, 0.3, (m.nvar, n_tgt, 1)).astype(np.float64)

        py = _run_python_loop(network_set, self.NSTEP, [x0_src, x0_tgt])
        nb = _run_nb(network_set, self.NSTEP, [x0_src, x0_tgt])

        for i, (p_i, n_i) in enumerate(zip(py, nb)):
            np.testing.assert_allclose(n_i, p_i, rtol=1e-3, atol=1e-4,
                                       err_msg=f"E2E 2-subnet delayed cfun subnet {i}")

    def test_two_subnets_intra_and_inter(self):
        """2-subnet network with both intra and inter projections."""
        n_src, n_tgt = 4, 5
        sn_src = _mpr_subnetwork("src", n_src)
        sn_tgt = _mpr_subnetwork("tgt", n_tgt)

        # Intra projection on source subnet
        w_intra = _sparse_weights(n_src, n_src, seed=33)
        intra = IntraProjection(
            source_cvar=np.array([0], dtype=np.int_),
            target_cvar=np.array([0], dtype=np.int_),
            weights=w_intra,
            lengths=_zero_lengths(n_src, n_src),
            cv=1.0, dt=DT, scale=0.3,
        )
        sn_src.projections = [intra]
        sn_src.configure()
        sn_tgt.configure()

        # Inter projection
        w_inter = _sparse_weights(n_tgt, n_src, seed=37)
        inter = InterProjection(
            source=sn_src, target=sn_tgt,
            source_cvar=np.array([0], dtype=np.int_),
            target_cvar=np.array([0], dtype=np.int_),
            weights=w_inter,
            lengths=_zero_lengths(n_tgt, n_src),
            cv=1.0, dt=DT, scale=1.0,
        )

        network_set = NetworkSet(
            subnets=[sn_src, sn_tgt],
            projections=[inter],
            stimuli=[],
        )
        network_set.configure()

        rng = np.random.RandomState(41)
        m = MontbrioPazoRoxin()
        m.configure()
        x0_src = rng.uniform(0.1, 0.3, (m.nvar, n_src, 1)).astype(np.float64)
        x0_tgt = rng.uniform(0.1, 0.3, (m.nvar, n_tgt, 1)).astype(np.float64)

        py = _run_python_loop(network_set, self.NSTEP, [x0_src, x0_tgt])
        nb = _run_nb(network_set, self.NSTEP, [x0_src, x0_tgt])

        for i, (p_i, n_i) in enumerate(zip(py, nb)):
            np.testing.assert_allclose(n_i, p_i, rtol=1e-3, atol=1e-4,
                                       err_msg=f"E2E intra+inter subnet {i}")

    def test_full_featured_network(self):
        """Single-subnet network with intra-projection, cfun, and stimulus."""
        n = 5
        sn = _mpr_subnetwork("ctx", n)
        w = _sparse_weights(n, n, seed=51)
        lengths = sp.csr_matrix(w.toarray() * 15.0)
        intra = IntraProjection(
            source_cvar=np.array([0], dtype=np.int_),
            target_cvar=np.array([0], dtype=np.int_),
            weights=w, lengths=lengths,
            cv=1.0, dt=DT, scale=0.5,
            cfun=Scaling(a=np.array([1.5])),
        )
        sn.projections = [intra]
        sn.configure()
        stim = _make_stim(sn, amplitude=0.03)
        network_set = NetworkSet(subnets=[sn], projections=[], stimuli=[stim])
        network_set.configure()

        rng = np.random.RandomState(61)
        m = MontbrioPazoRoxin()
        m.configure()
        x0 = rng.uniform(0.1, 0.3, (m.nvar, n, 1)).astype(np.float64)

        py = _run_python_loop(network_set, self.NSTEP, [x0])[0]
        nb = _run_nb(network_set, self.NSTEP, [x0])[0]

        np.testing.assert_allclose(nb, py, rtol=1e-3, atol=1e-4,
                                   err_msg="E2E full-featured: Numba differs from Python")


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------

class TestNbHybridBenchmark(unittest.TestCase):
    """Wall-clock timing benchmarks to assess Numba speedup over pure Python.

    These tests verify that the Numba backend is at least as fast as Python
    (after JIT compilation).  The primary metric reported is steps/second.
    """

    NSTEP = 1000
    N_NODES = 100

    def _make_benchmark_net(self):
        """Two-subnet MPR network with realistic tract-length delays.

        Tract lengths drawn from U(0, 100) mm; cv=10 m/s; dt=0.01 ms →
        horizon ≈ 0–1000 steps per connection.  This exercises the circular
        history buffer under realistic memory-bandwidth conditions.
        Both projections use 20% density random weights (realistic connectome).
        """
        n_src, n_tgt = self.N_NODES, self.N_NODES
        sn_src = _mpr_subnetwork("src", n_src)
        sn_tgt = _mpr_subnetwork("tgt", n_tgt)
        rng = np.random.RandomState(71)
        w_intra = _sparse_weights(n_src, n_src, seed=71, density=0.2)
        # Tract lengths 0–100 mm, cv=10 m/s → idelays up to 1000 steps
        lengths_intra = sp.csr_matrix(rng.uniform(0.0, 100.0, (n_src, n_src)))
        intra = IntraProjection(
            source_cvar=np.array([0], dtype=np.int_),
            target_cvar=np.array([0], dtype=np.int_),
            weights=w_intra,
            lengths=lengths_intra,
            cv=10.0, dt=DT, scale=0.5,
        )
        sn_src.projections = [intra]
        sn_src.configure()
        sn_tgt.configure()
        w_inter = _sparse_weights(n_tgt, n_src, seed=73, density=0.2)
        lengths_inter = sp.csr_matrix(rng.uniform(0.0, 50.0, (n_tgt, n_src)))
        inter = InterProjection(
            source=sn_src, target=sn_tgt,
            source_cvar=np.array([0], dtype=np.int_),
            target_cvar=np.array([0], dtype=np.int_),
            weights=w_inter,
            lengths=lengths_inter,
            cv=10.0, dt=DT, scale=1.0,
        )
        return NetworkSet(
            subnets=[sn_src, sn_tgt],
            projections=[inter],
            stimuli=[],
        )

    def _initial_states(self, network_set):
        rng = np.random.RandomState(79)
        states = []
        for sn in network_set.subnets:
            x0 = rng.uniform(0.1, 0.3, (sn.model.nvar, sn.nnodes, 1)).astype(np.float64)
            states.append(x0)
        return states

    def test_numba_runs_and_reports_speedup(self):
        """Compile once, then time Python vs Numba kernel (no re-compilation).

        The compiled kernel is obtained via ``backend.compile()`` so that the
        JIT warm-up cost is paid once and not included in the measured kernel
        time.  The test asserts a modest speedup (≥2×) to validate that
        caching works and the Numba path is actually faster.
        """
        network_set = self._make_benchmark_net()
        network_set.configure()
        x0_list = self._initial_states(network_set)
        backend = NbHybridBackend()

        # --- Compile once (one-time cost: Mako render + exec + Numba JIT) ---
        t_compile_start = time.perf_counter()
        compiled = backend.compile(network_set)
        # Force Numba JIT warm-up with a tiny run before timing
        compiled.run(nstep=5, chunk_size=1, initial_states=x0_list)
        t_compile = time.perf_counter() - t_compile_start

        # --- Python timing ---
        t0 = time.perf_counter()
        _run_python_loop(network_set, self.NSTEP, x0_list)
        t_py = time.perf_counter() - t0

        # --- Numba timing (cached kernel — no re-compilation) ---
        t0 = time.perf_counter()
        result = compiled.run(nstep=self.NSTEP, chunk_size=1, initial_states=x0_list)
        t_nb = time.perf_counter() - t0

        py_sps = self.NSTEP / t_py
        nb_sps = self.NSTEP / t_nb
        speedup = t_py / t_nb if t_nb > 0 else float('inf')

        print(
            f"\n[Benchmark] N={self.N_NODES} nodes x 2 subnets, {self.NSTEP} steps, cv=10 m/s delays, 20% density\n"
            f"  Compile (one-time): {t_compile*1e3:.1f} ms\n"
            f"  Python: {t_py*1e3:.1f} ms  ({py_sps:.0f} steps/s)\n"
            f"  Numba (cached kernel): {t_nb*1e3:.1f} ms  ({nb_sps:.0f} steps/s)\n"
            f"  Speedup (cached): {speedup:.2f}x\n"
        )

        # Verify result validity
        self.assertEqual(len(result), 2, "Expected 2 subnetwork results")
        for times, data in result:
            self.assertEqual(data.shape[0], self.NSTEP)
            self.assertFalse(np.any(np.isnan(data)), "NaN in Numba output")

        # Assert meaningful speedup now that caching works
        self.assertGreater(
            speedup, 2.0,
            f"Expected Numba cached kernel ≥ 2× faster than Python, got {speedup:.2f}×"
        )


class TestNbHybridMprKIonEx(unittest.TestCase):
    """NbHybridBackend with a mixed MPR + KIonEx two-subnet configuration.

    Validates that:
    1. The backend accepts KIonEx without raising NotImplementedError.
    2. The compiled kernel runs without NaN/Inf.
    3. Pure-Python and Numba outputs agree within float32 tolerance.
    """

    N = 6    # nodes per subnet
    NSTEP = 20

    def _build_network(self):
        from tvb.simulator.models.k_ion_exchange import KIonEx

        mpr_model = MontbrioPazoRoxin()
        mpr_model.configure()
        kionex_model = KIonEx()
        kionex_model.configure()

        sn_a = Subnetwork(name="mpr", model=mpr_model,
                          scheme=HeunDeterministic(dt=DT), nnodes=self.N)
        sn_b = Subnetwork(name="kionex", model=kionex_model,
                          scheme=HeunDeterministic(dt=DT), nnodes=self.N)

        rng = np.random.RandomState(7)

        def _w(n_tgt, n_src, seed):
            r = np.random.RandomState(seed)
            w = r.uniform(0.0, 0.05, (n_tgt, n_src)).astype(np.float64)
            np.fill_diagonal(w, 0.0)
            return sp.csr_matrix(w)

        def _l(n_tgt, n_src):
            return sp.csr_matrix(np.zeros((n_tgt, n_src), dtype=np.float64))

        # Intra A
        intra_a = IntraProjection(
            source_cvar=np.array([0], dtype=np.int32),
            target_cvar=np.array([0], dtype=np.int32),
            weights=_w(self.N, self.N, 10),
            lengths=_l(self.N, self.N),
            cv=1.0, dt=DT, scale=1e-4,
        )
        sn_a.projections = [intra_a]
        sn_a.configure()

        # Intra B
        intra_b = IntraProjection(
            source_cvar=np.array([0], dtype=np.int32),
            target_cvar=np.array([0], dtype=np.int32),
            weights=_w(self.N, self.N, 11),
            lengths=_l(self.N, self.N),
            cv=1.0, dt=DT, scale=1e-4,
        )
        sn_b.projections = [intra_b]
        sn_b.configure()

        # Inter A→B (MPR r → KIonEx Coupling_Term)
        inter_ab = InterProjection(
            source=sn_a, target=sn_b,
            source_cvar=np.array([0], dtype=np.int32),
            target_cvar=np.array([0], dtype=np.int32),
            weights=_w(self.N, self.N, 12),
            lengths=_l(self.N, self.N),
            cv=1.0, dt=DT, scale=1e-4,
        )

        # Inter B→A (KIonEx x → MPR Coupling_Term_r)
        inter_ba = InterProjection(
            source=sn_b, target=sn_a,
            source_cvar=np.array([0], dtype=np.int32),
            target_cvar=np.array([0], dtype=np.int32),
            weights=_w(self.N, self.N, 13),
            lengths=_l(self.N, self.N),
            cv=1.0, dt=DT, scale=1e-4,
        )

        nets = NetworkSet(subnets=[sn_a, sn_b], projections=[inter_ab, inter_ba], stimuli=[])
        nets.configure()

        # Initial conditions
        rng = np.random.RandomState(42)
        ic_a = rng.uniform(0.0, 0.1, (mpr_model.nvar, self.N, 1)).astype(np.float64)
        ic_a[0] = np.abs(ic_a[0])   # r ≥ 0

        # KIonEx: keep DKi in [-8, -2] and Kg in [-15, -8] so that
        # K_o = K_o0 − 3·DKi + Kg > 0, avoiding log(negative) NaN.
        ic_b = np.zeros((kionex_model.nvar, self.N, 1), dtype=np.float64)
        safe_ic = {
            'x':   (0.0, 0.3),
            'V':   (-70.0, -50.0),
            'n':   (0.2, 0.5),
            'DKi': (-8.0, -2.0),
            'Kg':  (-15.0, -8.0),
        }
        for idx, svar in enumerate(kionex_model.state_variables):
            lo, hi = safe_ic[svar]
            ic_b[idx, :, 0] = rng.uniform(lo, hi, self.N)

        return nets, ic_a, ic_b

    def test_kionex_accepted(self):
        """NbHybridBackend does not reject KIonEx."""
        nets, ic_a, ic_b = self._build_network()
        backend = NbHybridBackend()
        # Should not raise
        backend.run_network(nets, nstep=self.NSTEP, chunk_size=1,
                            initial_states=[ic_a, ic_b])

    def test_output_shapes(self):
        """Numba backend produces correctly-shaped output for MPR+KIonEx."""
        from tvb.simulator.models.k_ion_exchange import KIonEx
        nets, ic_a, ic_b = self._build_network()
        results = NbHybridBackend().run_network(
            nets, nstep=self.NSTEP, chunk_size=1,
            initial_states=[ic_a, ic_b],
        )
        self.assertEqual(len(results), 2, "Expected 2 subnetwork results")
        mpr_nvoi = len(MontbrioPazoRoxin.variables_of_interest.default)
        kionex_nvoi = len(KIonEx.variables_of_interest.default)
        times_a, data_a = results[0]
        self.assertEqual(data_a.shape, (self.NSTEP, mpr_nvoi, self.N, 1),
                         f"MPR subnet shape mismatch: {data_a.shape}")
        times_b, data_b = results[1]
        self.assertEqual(data_b.shape, (self.NSTEP, kionex_nvoi, self.N, 1),
                         f"KIonEx subnet shape mismatch: {data_b.shape}")

    def test_numba_matches_python(self):
        """Numba and pure-Python outputs agree within float32 tolerance."""
        nets, ic_a, ic_b = self._build_network()

        py_out = _run_python_loop(nets, self.NSTEP, [ic_a, ic_b])

        nets2, ic_a2, ic_b2 = self._build_network()
        nb_out = _run_nb(nets2, self.NSTEP, [ic_a2, ic_b2])

        for i, (py_i, nb_i) in enumerate(zip(py_out, nb_out)):
            self.assertEqual(py_i.shape, nb_i.shape,
                             f"Shape mismatch at subnetwork {i}")
            np.testing.assert_allclose(
                nb_i, py_i, rtol=1e-3, atol=1e-3,
                err_msg=f"Numba vs Python mismatch at subnetwork {i}",
            )


if __name__ == "__main__":
    unittest.main()
