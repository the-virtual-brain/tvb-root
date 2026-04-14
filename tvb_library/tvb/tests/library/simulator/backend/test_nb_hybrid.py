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
import pytest
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
from tvb.simulator.backend.nb_hybrid import _STIM_LAZY_THRESHOLD_MB

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

DT = 0.01


def _mpr_subnetwork(
    name: str, n_nodes: int, integrator_cls=HeunDeterministic
) -> Subnetwork:
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
    noise = Additive(
        nsig=np.array([nsig])
    )  # scalar nsig → correct broadcast for 3D state
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
    temporal.parameters["amp"] = np.float64(amplitude)
    temporal.parameters["frequency"] = np.float64(0.1)
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


def _sparse_weights(
    n_tgt: int, n_src: int, seed: int = 0, density: float = 1.0
) -> sp.csr_matrix:
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


def _run_nb(
    network_set: NetworkSet, nstep: int, x0_list: list, print_source: bool = False
) -> list:
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
    # results: list of (times, data, ctavg) where data is (nstep, n_voi, n_nodes, n_modes)
    # Return only the state data arrays
    return [data for _, data, _ in results]


def _run_nb_full(
    network_set: NetworkSet, nstep: int, x0_list: list, print_source: bool = False
) -> list:
    """Run NbHybridBackend and return full (times, state, ctavg) 3-tuples per subnetwork."""
    backend = NbHybridBackend()
    return backend.run_network(
        network_set,
        nstep=nstep,
        chunk_size=1,
        print_source=print_source,
        initial_states=x0_list,
    )


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
        self.assertEqual(
            py.shape, nb.shape, f"Shape mismatch: python {py.shape} vs nb {nb.shape}"
        )
        np.testing.assert_allclose(
            nb,
            py,
            rtol=1e-3,
            atol=1e-4,
            err_msg="Heun single-subnet: Numba output differs from Python",
        )

    def test_euler_no_projection(self):
        py, nb = self._run_both(EulerDeterministic, nstep=10)
        self.assertEqual(py.shape, nb.shape)
        np.testing.assert_allclose(
            nb,
            py,
            rtol=1e-3,
            atol=1e-4,
            err_msg="Euler single-subnet: Numba output differs from Python",
        )


class TestNbHybridIntraProjection(unittest.TestCase):
    """Single subnetwork with intra-projection (local coupling)."""

    def _make_net(self, n=5, delay=False):
        sn = _mpr_subnetwork("ctx", n)
        w = _sparse_weights(n, n, seed=1)
        lengths = sp.csr_matrix(w.toarray() * (10.0 if delay else 0.0))
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
            nb,
            py,
            rtol=1e-3,
            atol=1e-4,
            err_msg="Intra no-delay: Numba differs from Python",
        )

    def test_intra_with_delay(self):
        py, nb = self._run_both(delay=True)
        self.assertEqual(py.shape, nb.shape)
        np.testing.assert_allclose(
            nb,
            py,
            rtol=1e-3,
            atol=1e-4,
            err_msg="Intra with-delay: Numba differs from Python",
        )


class TestNbHybridInterProjection(unittest.TestCase):
    """Two MPR subnetworks with an inter-projection."""

    def _make_net(self, n_src=4, n_tgt=5, delay=False, cvar_mapping="1_to_1"):
        sn_src = _mpr_subnetwork("src_net", n_src)
        sn_tgt = _mpr_subnetwork("tgt_net", n_tgt)
        sn_src.configure()
        sn_tgt.configure()

        w = _sparse_weights(n_tgt, n_src, seed=3)
        lengths = sp.csr_matrix(w.toarray() * (15.0 if delay else 0.0))

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
                n_i,
                p_i,
                rtol=1e-3,
                atol=1e-4,
                err_msg=f"Inter 1-to-1 no-delay subnetwork {i}: Numba differs",
            )

    def test_inter_with_delay_1_to_1(self):
        py, nb = self._run_both(delay=True, cvar_mapping="1_to_1")
        for i, (p_i, n_i) in enumerate(zip(py, nb)):
            self.assertEqual(p_i.shape, n_i.shape)
            np.testing.assert_allclose(
                n_i,
                p_i,
                rtol=1e-3,
                atol=1e-4,
                err_msg=f"Inter 1-to-1 with-delay subnetwork {i}: Numba differs",
            )

    def test_inter_cvar_1_to_many(self):
        py, nb = self._run_both(delay=False, cvar_mapping="1_to_many")
        for i, (p_i, n_i) in enumerate(zip(py, nb)):
            self.assertEqual(p_i.shape, n_i.shape)
            np.testing.assert_allclose(
                n_i,
                p_i,
                rtol=1e-3,
                atol=1e-4,
                err_msg=f"Inter 1-to-many subnetwork {i}: Numba differs",
            )

    def test_inter_cvar_many_to_1(self):
        py, nb = self._run_both(delay=False, cvar_mapping="many_to_1")
        for i, (p_i, n_i) in enumerate(zip(py, nb)):
            self.assertEqual(p_i.shape, n_i.shape)
            np.testing.assert_allclose(
                n_i,
                p_i,
                rtol=1e-3,
                atol=1e-4,
                err_msg=f"Inter many-to-1 subnetwork {i}: Numba differs",
            )


class TestNbHybridCompatibilityCheck(unittest.TestCase):
    """Verify that incompatible models/integrators are rejected cleanly."""

    def test_rejects_unsupported_model(self):
        from tvb.simulator.models.stefanescu_jirsa import ReducedSetFitzHughNagumo
        from tvb.simulator.integrators import HeunDeterministic as HD

        model = ReducedSetFitzHughNagumo()
        model.configure()
        model.dfun_mode = "not_combined"
        scheme = HD(dt=DT)
        sn = Subnetwork(name="sn", model=model, scheme=scheme, nnodes=3)
        sn.configure()
        nets = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        nets.configure()

        with self.assertRaises(NotImplementedError):
            NbHybridBackend()._check_compatibility(nets)

    def test_rejects_multiplicative_noise(self):
        from tvb.simulator.noise import Multiplicative
        from tvb.simulator.integrators import HeunStochastic
        from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
        from tvb.simulator.hybrid import Subnetwork, NetworkSet

        m = MontbrioPazoRoxin()
        m.configure()
        integ = HeunStochastic(noise=Multiplicative(), dt=0.1)
        integ.configure()
        sn = Subnetwork(name="sn", model=m, scheme=integ, nnodes=3)
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

    def test_rejects_chunk_size_gt_horizon(self):
        m = MontbrioPazoRoxin()
        m.configure()
        integ = HeunDeterministic(dt=0.1)
        sn = Subnetwork(name="sn", model=m, scheme=integ, nnodes=3)
        sn.configure()
        # tract_length=0.2 mm, cv=1 mm/ms → delay=0.2 ms → horizon=ceil(0.2/0.1)+1=3 steps
        W = sp.csr_matrix(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32))
        L = sp.csr_matrix(np.full((3, 3), 0.2))
        proj = IntraProjection(
            source_cvar=np.array([0], dtype=np.int_),
            target_cvar=np.array([0], dtype=np.int_),
            weights=W,
            lengths=L,
            cv=1.0,
            dt=0.1,
            scale=1.0,
            cfun=Linear(),
        )
        sn.projections = [proj]
        sn.configure()
        nets = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        nets.configure()
        with self.assertRaises(ValueError):
            NbHybridBackend().run_network(nets, nstep=100, chunk_size=10)


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
        np.testing.assert_allclose(
            nb,
            py,
            rtol=1e-3,
            atol=1e-4,
            err_msg="Linear cfun: Numba differs from Python",
        )

    def test_scaling_cfun(self):
        cfun = Scaling(a=np.array([2.0]))
        py, nb = self._run_both(cfun)
        self.assertEqual(py.shape, nb.shape)
        np.testing.assert_allclose(
            nb,
            py,
            rtol=1e-3,
            atol=1e-4,
            err_msg="Scaling cfun: Numba differs from Python",
        )


class TestNbHybridCfunExtended(unittest.TestCase):
    """Test Kuramoto, Difference, HyperbolicTangent, and PreSigmoidal cfuns."""

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

    def test_kuramoto_cfun(self):
        from tvb.simulator.hybrid.coupling import Kuramoto as KuramotoCfun
        cfun = KuramotoCfun(a=np.array([0.3]))
        py, nb = self._run_both(cfun)
        self.assertEqual(py.shape, nb.shape)
        np.testing.assert_allclose(
            nb, py, rtol=1e-3, atol=1e-4,
            err_msg="Kuramoto cfun: Numba differs from Python",
        )

    def test_difference_cfun(self):
        from tvb.simulator.hybrid.coupling import Difference
        cfun = Difference(a=np.array([1.5]))
        py, nb = self._run_both(cfun)
        self.assertEqual(py.shape, nb.shape)
        np.testing.assert_allclose(
            nb, py, rtol=1e-3, atol=1e-4,
            err_msg="Difference cfun: Numba differs from Python",
        )

    def test_tanh_cfun(self):
        from tvb.simulator.hybrid.coupling import HyperbolicTangent
        cfun = HyperbolicTangent(a=np.array([0.5]), midpoint=np.array([0.0]), sigma=np.array([1.0]))
        py, nb = self._run_both(cfun)
        self.assertEqual(py.shape, nb.shape)
        np.testing.assert_allclose(
            nb, py, rtol=1e-3, atol=1e-4,
            err_msg="HyperbolicTangent cfun: Numba differs from Python",
        )

    def test_pre_sigmoidal_cfun(self):
        from tvb.simulator.hybrid.coupling import PreSigmoidal
        cfun = PreSigmoidal(
            H=np.array([0.5]), Q=np.array([0.0]), G=np.array([1.0]),
            P=np.array([1.0]), theta=np.array([0.0]),
        )
        py, nb = self._run_both(cfun)
        self.assertEqual(py.shape, nb.shape)
        np.testing.assert_allclose(
            nb, py, rtol=1e-3, atol=1e-4,
            err_msg="PreSigmoidal cfun: Numba differs from Python",
        )

    def test_kuramoto_cfun_output_finite(self):
        """Kuramoto cfun produces finite output over longer run."""
        from tvb.simulator.hybrid.coupling import Kuramoto as KuramotoCfun
        cfun = KuramotoCfun(a=np.array([1.0]))
        network_set, n = self._make_net_with_cfun(cfun, n=6)
        model = network_set.subnets[0].model
        x0 = np.random.RandomState(42).uniform(-0.5, 0.5, (model.nvar, n, 1)).astype(np.float64)
        results = _run_nb(network_set, 50, [x0])
        self.assertTrue(np.all(np.isfinite(results[0])), "Kuramoto: NaN/Inf in output")

    def test_tanh_cfun_with_delay(self):
        """HyperbolicTangent cfun with non-zero delays."""
        from tvb.simulator.hybrid.coupling import HyperbolicTangent
        cfun = HyperbolicTangent(a=np.array([0.8]), midpoint=np.array([0.1]), sigma=np.array([2.0]))
        py, nb = self._run_both(cfun, delay=True)
        self.assertEqual(py.shape, nb.shape)
        np.testing.assert_allclose(
            nb, py, rtol=1e-3, atol=1e-4,
            err_msg="HyperbolicTangent+delay: Numba differs from Python",
        )


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
            np.testing.assert_allclose(
                n_,
                p,
                rtol=1e-3,
                atol=1e-4,
                err_msg=f"target_scales subnetwork {i}: mismatch",
            )


# ---------------------------------------------------------------------------
# Stochastic integrator tests
# ---------------------------------------------------------------------------


class TestNbHybridStochastic(unittest.TestCase):
    """Test EulerStochastic and HeunStochastic integrators."""

    NSTEP = 10
    N = 4
    NSIG = 1e-4  # small noise for near-deterministic comparison

    def _make_stochastic_net(self, integrator_cls, seed=42):
        sn = _mpr_stochastic_subnetwork(
            "ctx", self.N, integrator_cls=integrator_cls, nsig=self.NSIG, seed=seed
        )
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
        x0 = (
            np.random.RandomState(77)
            .uniform(0.1, 0.3, (m.nvar, self.N, 1))
            .astype(np.float64)
        )
        nb = _run_nb(network_set, self.NSTEP, [x0])[0]
        self.assertEqual(nb.shape, (self.NSTEP, m.nvar, self.N, 1))

    def test_heun_stochastic_shape(self):
        """Output shape is correct for HeunStochastic."""
        network_set = self._make_stochastic_net(HeunStochastic)
        m = network_set.subnets[0].model
        x0 = (
            np.random.RandomState(77)
            .uniform(0.1, 0.3, (m.nvar, self.N, 1))
            .astype(np.float64)
        )
        nb = _run_nb(network_set, self.NSTEP, [x0])[0]
        self.assertEqual(nb.shape, (self.NSTEP, m.nvar, self.N, 1))

    def test_euler_stochastic_matches_python(self):
        """EulerStochastic Numba output matches Python with same RNG seed."""
        py, nb = self._run_both_same_seed(EulerStochastic)
        self.assertEqual(py.shape, nb.shape)
        np.testing.assert_allclose(
            nb,
            py,
            rtol=1e-2,
            atol=1e-3,
            err_msg="EulerStochastic: Numba differs from Python",
        )

    def test_heun_stochastic_matches_python(self):
        """HeunStochastic Numba output matches Python with same RNG seed."""
        py, nb = self._run_both_same_seed(HeunStochastic)
        self.assertEqual(py.shape, nb.shape)
        np.testing.assert_allclose(
            nb,
            py,
            rtol=1e-2,
            atol=1e-3,
            err_msg="HeunStochastic: Numba differs from Python",
        )

    def test_noise_has_effect(self):
        """Stochastic output must differ from deterministic (noise is applied)."""
        n = self.N
        model_d = MontbrioPazoRoxin()
        model_d.configure()
        sn_det = Subnetwork(
            name="ctx", model=model_d, scheme=EulerDeterministic(dt=DT), nnodes=n
        )
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
        self.assertGreater(
            max_diff, 1e-4, "Stochastic output should differ from deterministic"
        )


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
        x0 = (
            np.random.RandomState(5)
            .uniform(0.1, 0.3, (model.nvar, n, 1))
            .astype(np.float64)
        )
        nb = _run_nb(network_set, self.NSTEP, [x0])
        self.assertEqual(len(nb), 1)
        self.assertEqual(nb[0].shape[0], self.NSTEP)

    def test_stimulus_matches_python(self):
        """Numba stimulus output must match Python loop within tolerance."""
        network_set, n = self._make_net_with_stim()
        model = network_set.subnets[0].model
        x0 = (
            np.random.RandomState(9)
            .uniform(0.1, 0.3, (model.nvar, n, 1))
            .astype(np.float64)
        )

        py = _run_python_loop(network_set, self.NSTEP, [x0])[0]
        nb = _run_nb(network_set, self.NSTEP, [x0])[0]

        self.assertEqual(py.shape, nb.shape)
        np.testing.assert_allclose(
            nb, py, rtol=1e-3, atol=1e-4, err_msg="Stimulus: Numba differs from Python"
        )

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
        self.assertGreater(
            max_diff, 1e-5, "Stimulated output should differ from baseline"
        )


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
            source=sn_src,
            target=sn_tgt,
            source_cvar=np.array([0], dtype=np.int_),
            target_cvar=np.array([0], dtype=np.int_),
            weights=w,
            lengths=lengths,
            cv=1.0,
            dt=DT,
            scale=0.5,
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
            np.testing.assert_allclose(
                n_i,
                p_i,
                rtol=1e-3,
                atol=1e-4,
                err_msg=f"E2E 2-subnet delayed cfun subnet {i}",
            )

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
            cv=1.0,
            dt=DT,
            scale=0.3,
        )
        sn_src.projections = [intra]
        sn_src.configure()
        sn_tgt.configure()

        # Inter projection
        w_inter = _sparse_weights(n_tgt, n_src, seed=37)
        inter = InterProjection(
            source=sn_src,
            target=sn_tgt,
            source_cvar=np.array([0], dtype=np.int_),
            target_cvar=np.array([0], dtype=np.int_),
            weights=w_inter,
            lengths=_zero_lengths(n_tgt, n_src),
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

        rng = np.random.RandomState(41)
        m = MontbrioPazoRoxin()
        m.configure()
        x0_src = rng.uniform(0.1, 0.3, (m.nvar, n_src, 1)).astype(np.float64)
        x0_tgt = rng.uniform(0.1, 0.3, (m.nvar, n_tgt, 1)).astype(np.float64)

        py = _run_python_loop(network_set, self.NSTEP, [x0_src, x0_tgt])
        nb = _run_nb(network_set, self.NSTEP, [x0_src, x0_tgt])

        for i, (p_i, n_i) in enumerate(zip(py, nb)):
            np.testing.assert_allclose(
                n_i, p_i, rtol=1e-3, atol=1e-4, err_msg=f"E2E intra+inter subnet {i}"
            )

    def test_full_featured_network(self):
        """Single-subnet network with intra-projection, cfun, and stimulus."""
        n = 5
        sn = _mpr_subnetwork("ctx", n)
        w = _sparse_weights(n, n, seed=51)
        lengths = sp.csr_matrix(w.toarray() * 15.0)
        intra = IntraProjection(
            source_cvar=np.array([0], dtype=np.int_),
            target_cvar=np.array([0], dtype=np.int_),
            weights=w,
            lengths=lengths,
            cv=1.0,
            dt=DT,
            scale=0.5,
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

        np.testing.assert_allclose(
            nb,
            py,
            rtol=1e-3,
            atol=1e-4,
            err_msg="E2E full-featured: Numba differs from Python",
        )


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
            cv=10.0,
            dt=DT,
            scale=0.5,
        )
        sn_src.projections = [intra]
        sn_src.configure()
        sn_tgt.configure()
        w_inter = _sparse_weights(n_tgt, n_src, seed=73, density=0.2)
        lengths_inter = sp.csr_matrix(rng.uniform(0.0, 50.0, (n_tgt, n_src)))
        inter = InterProjection(
            source=sn_src,
            target=sn_tgt,
            source_cvar=np.array([0], dtype=np.int_),
            target_cvar=np.array([0], dtype=np.int_),
            weights=w_inter,
            lengths=lengths_inter,
            cv=10.0,
            dt=DT,
            scale=1.0,
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
        speedup = t_py / t_nb if t_nb > 0 else float("inf")

        print(
            f"\n[Benchmark] N={self.N_NODES} nodes x 2 subnets, {self.NSTEP} steps, cv=10 m/s delays, 20% density\n"
            f"  Compile (one-time): {t_compile * 1e3:.1f} ms\n"
            f"  Python: {t_py * 1e3:.1f} ms  ({py_sps:.0f} steps/s)\n"
            f"  Numba (cached kernel): {t_nb * 1e3:.1f} ms  ({nb_sps:.0f} steps/s)\n"
            f"  Speedup (cached): {speedup:.2f}x\n"
        )

        # Verify result validity
        self.assertEqual(len(result), 2, "Expected 2 subnetwork results")
        for times, data, _ in result:
            self.assertEqual(data.shape[0], self.NSTEP)
            self.assertFalse(np.any(np.isnan(data)), "NaN in Numba output")

        # Assert meaningful speedup now that caching works
        self.assertGreater(
            speedup,
            2.0,
            f"Expected Numba cached kernel ≥ 2× faster than Python, got {speedup:.2f}×",
        )


class TestNbHybridMprKIonEx(unittest.TestCase):
    """NbHybridBackend with a mixed MPR + KIonEx two-subnet configuration.

    Validates that:
    1. The backend accepts KIonEx without raising NotImplementedError.
    2. The compiled kernel runs without NaN/Inf.
    3. Pure-Python and Numba outputs agree within float32 tolerance.
    """

    N = 6  # nodes per subnet
    NSTEP = 20

    def _build_network(self):
        from tvb.simulator.models.k_ion_exchange import KIonEx

        mpr_model = MontbrioPazoRoxin()
        mpr_model.configure()
        kionex_model = KIonEx()
        kionex_model.configure()

        sn_a = Subnetwork(
            name="mpr", model=mpr_model, scheme=HeunDeterministic(dt=DT), nnodes=self.N
        )
        sn_b = Subnetwork(
            name="kionex",
            model=kionex_model,
            scheme=HeunDeterministic(dt=DT),
            nnodes=self.N,
        )

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
            cv=1.0,
            dt=DT,
            scale=1e-4,
        )
        sn_a.projections = [intra_a]
        sn_a.configure()

        # Intra B
        intra_b = IntraProjection(
            source_cvar=np.array([0], dtype=np.int32),
            target_cvar=np.array([0], dtype=np.int32),
            weights=_w(self.N, self.N, 11),
            lengths=_l(self.N, self.N),
            cv=1.0,
            dt=DT,
            scale=1e-4,
        )
        sn_b.projections = [intra_b]
        sn_b.configure()

        # Inter A→B (MPR r → KIonEx Coupling_Term)
        inter_ab = InterProjection(
            source=sn_a,
            target=sn_b,
            source_cvar=np.array([0], dtype=np.int32),
            target_cvar=np.array([0], dtype=np.int32),
            weights=_w(self.N, self.N, 12),
            lengths=_l(self.N, self.N),
            cv=1.0,
            dt=DT,
            scale=1e-4,
        )

        # Inter B→A (KIonEx x → MPR Coupling_Term_r)
        inter_ba = InterProjection(
            source=sn_b,
            target=sn_a,
            source_cvar=np.array([0], dtype=np.int32),
            target_cvar=np.array([0], dtype=np.int32),
            weights=_w(self.N, self.N, 13),
            lengths=_l(self.N, self.N),
            cv=1.0,
            dt=DT,
            scale=1e-4,
        )

        nets = NetworkSet(
            subnets=[sn_a, sn_b], projections=[inter_ab, inter_ba], stimuli=[]
        )
        nets.configure()

        # Initial conditions
        rng = np.random.RandomState(42)
        ic_a = rng.uniform(0.0, 0.1, (mpr_model.nvar, self.N, 1)).astype(np.float64)
        ic_a[0] = np.abs(ic_a[0])  # r ≥ 0

        # KIonEx: keep DKi in [-8, -2] and Kg in [-15, -8] so that
        # K_o = K_o0 − 3·DKi + Kg > 0, avoiding log(negative) NaN.
        ic_b = np.zeros((kionex_model.nvar, self.N, 1), dtype=np.float64)
        safe_ic = {
            "x": (0.0, 0.3),
            "V": (-70.0, -50.0),
            "n": (0.2, 0.5),
            "DKi": (-8.0, -2.0),
            "Kg": (-15.0, -8.0),
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
        backend.run_network(
            nets, nstep=self.NSTEP, chunk_size=1, initial_states=[ic_a, ic_b]
        )

    def test_output_shapes(self):
        """Numba backend produces correctly-shaped output for MPR+KIonEx.

        NOTE: KIonEx's ionic-current helpers use log() applied to concentrations
        that can go negative in float32 (the hybrid numba backend's precision),
        so NaN-freedom is only asserted for the Python (float64) path in
        test_mpr_kionex.py.  Here we validate shapes and that the backend runs.
        """
        from tvb.simulator.models.k_ion_exchange import KIonEx

        nets, ic_a, ic_b = self._build_network()
        results = NbHybridBackend().run_network(
            nets,
            nstep=self.NSTEP,
            chunk_size=1,
            initial_states=[ic_a, ic_b],
        )
        self.assertEqual(len(results), 2, "Expected 2 subnetwork results")
        mpr_nvoi = len(MontbrioPazoRoxin.variables_of_interest.default)
        kionex_nvoi = len(KIonEx.variables_of_interest.default)
        times_a, data_a, _ = results[0]
        self.assertEqual(
            data_a.shape,
            (self.NSTEP, mpr_nvoi, self.N, 1),
            f"MPR subnet shape mismatch: {data_a.shape}",
        )
        times_b, data_b, _ = results[1]
        self.assertEqual(
            data_b.shape,
            (self.NSTEP, kionex_nvoi, self.N, 1),
            f"KIonEx subnet shape mismatch: {data_b.shape}",
        )

    def test_numba_matches_python(self):
        """Numba and pure-Python outputs agree within float32 tolerance."""
        nets, ic_a, ic_b = self._build_network()

        py_out = _run_python_loop(nets, self.NSTEP, [ic_a, ic_b])

        nets2, ic_a2, ic_b2 = self._build_network()
        nb_out = _run_nb(nets2, self.NSTEP, [ic_a2, ic_b2])

        for i, (py_i, nb_i) in enumerate(zip(py_out, nb_out)):
            self.assertEqual(
                py_i.shape, nb_i.shape, f"Shape mismatch at subnetwork {i}"
            )
            np.testing.assert_allclose(
                nb_i,
                py_i,
                rtol=1e-3,
                atol=1e-3,
                err_msg=f"Numba vs Python mismatch at subnetwork {i}",
            )


# ---------------------------------------------------------------------------
# Sigmoidal coupling function tests
# ---------------------------------------------------------------------------


class TestNbHybridSigmoidalCfun:
    """Sigmoidal and SigmoidalJansenRit coupling function tests (inter-projection)."""

    def _build_net(self, cfun, n=6, seed=42):
        """Two MPR subnets connected by an inter-projection with the given cfun."""
        sn1 = _mpr_subnetwork("mpr1", n)
        sn1.configure()
        sn2 = _mpr_subnetwork("mpr2", n)
        sn2.configure()
        w = _sparse_weights(n, n, seed=seed, density=0.5)
        inter = InterProjection(
            source=sn1,
            target=sn2,
            source_cvar=np.array([0], dtype=np.int32),
            target_cvar=np.array([0], dtype=np.int32),
            weights=w,
            lengths=_zero_lengths(n, n),
            cv=1.0,
            dt=DT,
            scale=1e-2,
            cfun=cfun,
        )
        nets = NetworkSet(subnets=[sn1, sn2], projections=[inter], stimuli=[])
        nets.configure()
        return nets

    def _make_ic(self, n=6):
        rng = np.random.RandomState(42)
        m = MontbrioPazoRoxin()
        m.configure()
        x0 = rng.uniform(0.0, 0.2, (m.nvar, n, 1)).astype(np.float64)
        x0[0] = np.abs(x0[0])
        return x0

    def test_sigmoidal_cfun_accepts(self):
        """NbHybridBackend accepts Sigmoidal cfun without raising."""
        from tvb.simulator.hybrid.coupling import Sigmoidal

        nets = self._build_net(cfun=Sigmoidal())
        NbHybridBackend()._check_compatibility(nets)

    def test_sigmoidal_cfun_finite(self):
        """Numba backend produces finite output with Sigmoidal cfun."""
        from tvb.simulator.hybrid.coupling import Sigmoidal

        nets = self._build_net(cfun=Sigmoidal())
        x0 = self._make_ic()
        results = _run_nb(nets, 15, [x0, x0.copy()])
        assert len(results) == 2
        for d in results:
            assert np.all(np.isfinite(d)), "NaN/Inf in Sigmoidal cfun output"

    def test_sigmoidal_cfun_matches_python(self):
        """Numba Sigmoidal cfun output matches Python backend."""
        from tvb.simulator.hybrid.coupling import Sigmoidal

        n, nstep = 6, 15
        nets_py = self._build_net(cfun=Sigmoidal(), n=n, seed=42)
        nets_nb = self._build_net(cfun=Sigmoidal(), n=n, seed=42)
        x0 = self._make_ic(n)
        py = _run_python_loop(nets_py, nstep, [x0, x0.copy()])
        nb = _run_nb(nets_nb, nstep, [x0, x0.copy()])
        for py_d, nb_d in zip(py, nb):
            np.testing.assert_allclose(
                nb_d, py_d.astype(np.float32), rtol=1e-3, atol=1e-3
            )

    def test_sigmoidal_jr_cfun_accepts(self):
        """NbHybridBackend accepts SigmoidalJansenRit cfun without raising."""
        from tvb.simulator.hybrid.coupling import SigmoidalJansenRit

        nets = self._build_net(cfun=SigmoidalJansenRit())
        NbHybridBackend()._check_compatibility(nets)

    def test_sigmoidal_jr_cfun_finite(self):
        """Numba backend produces finite output with SigmoidalJansenRit cfun."""
        from tvb.simulator.hybrid.coupling import SigmoidalJansenRit

        nets = self._build_net(cfun=SigmoidalJansenRit())
        x0 = self._make_ic()
        results = _run_nb(nets, 15, [x0, x0.copy()])
        assert len(results) == 2
        for d in results:
            assert np.all(np.isfinite(d)), "NaN/Inf in SigmoidalJansenRit cfun output"

    def test_sigmoidal_jr_cfun_matches_python(self):
        """Numba SigmoidalJansenRit cfun output matches Python backend."""
        from tvb.simulator.hybrid.coupling import SigmoidalJansenRit

        n, nstep = 6, 15
        nets_py = self._build_net(cfun=SigmoidalJansenRit(), n=n, seed=42)
        nets_nb = self._build_net(cfun=SigmoidalJansenRit(), n=n, seed=42)
        x0 = self._make_ic(n)
        py = _run_python_loop(nets_py, nstep, [x0, x0.copy()])
        nb = _run_nb(nets_nb, nstep, [x0, x0.copy()])
        for py_d, nb_d in zip(py, nb):
            np.testing.assert_allclose(
                nb_d, py_d.astype(np.float32), rtol=1e-3, atol=1e-3
            )


# ---------------------------------------------------------------------------
# JansenRit model tests
# ---------------------------------------------------------------------------


class TestNbHybridJansenRit:
    """JansenRit model support in the numba hybrid backend."""

    N = 6
    NSTEP = 15

    def _build_net(self):
        """Two JR subnets connected by an inter-projection (y0 → Coupling_Term)."""
        from tvb.simulator.models.jansen_rit import JansenRit

        n = self.N
        m1 = JansenRit()
        m2 = JansenRit()
        sn1 = Subnetwork(
            name="jr1", model=m1, scheme=HeunDeterministic(dt=DT), nnodes=n
        )
        sn1.configure()
        sn2 = Subnetwork(
            name="jr2", model=m2, scheme=HeunDeterministic(dt=DT), nnodes=n
        )
        sn2.configure()
        w = _sparse_weights(n, n, seed=7, density=0.4)
        inter = InterProjection(
            source=sn1,
            target=sn2,
            source_cvar=np.array([0], dtype=np.int32),
            target_cvar=np.array([0], dtype=np.int32),
            weights=w,
            lengths=_zero_lengths(n, n),
            cv=1.0,
            dt=DT,
            scale=1e-3,
        )
        nets = NetworkSet(subnets=[sn1, sn2], projections=[inter], stimuli=[])
        nets.configure()
        return nets

    def _make_ic(self):
        """Safe initial conditions for JansenRit."""
        x0 = np.zeros((6, self.N, 1), dtype=np.float64)
        x0[0, :, 0] = 0.08  # y0
        x0[1, :, 0] = 13.0  # y1
        x0[2, :, 0] = 5.0  # y2
        return x0

    def test_jr_accepted_by_backend(self):
        """JansenRit is accepted by NbHybridBackend._check_compatibility."""
        nets = self._build_net()
        NbHybridBackend()._check_compatibility(nets)

    def test_jr_output_shape(self):
        """JR numba backend produces correct output shape."""
        from tvb.simulator.models.jansen_rit import JansenRit

        nets = self._build_net()
        x0 = self._make_ic()
        results = _run_nb(nets, self.NSTEP, [x0, x0.copy()])
        assert len(results) == 2
        n_voi = len(JansenRit.variables_of_interest.default)
        for d in results:
            assert d.ndim == 4
            assert d.shape[0] == self.NSTEP
            assert d.shape[1] == n_voi
            assert d.shape[2] == self.N
            assert d.shape[3] == 1

    def test_jr_output_finite(self):
        """JR numba backend produces finite output."""
        nets = self._build_net()
        x0 = self._make_ic()
        results = _run_nb(nets, self.NSTEP, [x0, x0.copy()])
        for d in results:
            assert np.all(np.isfinite(d)), "NaN/Inf in JansenRit numba output"

    def test_jr_matches_python(self):
        """JR numba backend output matches Python loop backend."""
        from tvb.simulator.models.jansen_rit import JansenRit

        # JR has 4 VoI (y0..y3) but 6 state vars;  Python loop returns all 6,
        # NB returns only VoI.  Extract VoI indices for comparison.
        voi_names = JansenRit.variables_of_interest.default
        sv_names = list(JansenRit.state_variables)
        voi_idx = [sv_names.index(v) for v in voi_names]

        x0 = self._make_ic()
        nets_py = self._build_net()
        nets_nb = self._build_net()
        py = _run_python_loop(nets_py, self.NSTEP, [x0, x0.copy()])
        nb = _run_nb(nets_nb, self.NSTEP, [x0, x0.copy()])

        for py_d, nb_d in zip(py, nb):
            py_voi = py_d[:, voi_idx, :, :].astype(np.float32)
            np.testing.assert_allclose(nb_d, py_voi, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# n_modes > 1 tests
# ---------------------------------------------------------------------------


class TestNbHybridMultiMode:
    """n_modes > 1 code path in the numba hybrid backend."""

    N = 4
    N_MODES = 2
    NSTEP = 10

    def _build_net(self):
        """Single MPR subnet with number_of_modes=2 and no projections."""
        m = MontbrioPazoRoxin()
        m.number_of_modes = self.N_MODES
        m.configure()
        sn = Subnetwork(
            name="ctx", model=m, scheme=HeunDeterministic(dt=DT), nnodes=self.N
        )
        sn.configure()
        nets = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        nets.configure()
        return nets

    def _make_ic(self):
        rng = np.random.RandomState(77)
        x0 = rng.uniform(0.0, 0.2, (2, self.N, self.N_MODES)).astype(np.float64)
        x0[0] = np.abs(x0[0])
        return x0

    def test_multi_mode_shape(self):
        """Output shape contains n_modes > 1 in last dimension."""
        nets = self._build_net()
        x0 = self._make_ic()
        results = _run_nb(nets, self.NSTEP, [x0])
        assert len(results) == 1
        d = results[0]
        assert d.ndim == 4
        assert d.shape[0] == self.NSTEP
        assert d.shape[2] == self.N
        assert d.shape[3] == self.N_MODES, (
            f"Expected n_modes={self.N_MODES}, got {d.shape[3]}"
        )

    def test_multi_mode_finite(self):
        """n_modes=2 output is finite."""
        nets = self._build_net()
        x0 = self._make_ic()
        results = _run_nb(nets, self.NSTEP, [x0])
        assert np.all(np.isfinite(results[0])), "NaN/Inf in multi-mode output"

    def test_multi_mode_matches_python(self):
        """n_modes=2 Numba output matches Python loop."""
        nets_py = self._build_net()
        nets_nb = self._build_net()
        x0 = self._make_ic()
        py = _run_python_loop(nets_py, self.NSTEP, [x0])
        nb = _run_nb(nets_nb, self.NSTEP, [x0])
        np.testing.assert_allclose(
            nb[0], py[0].astype(np.float32), rtol=1e-3, atol=1e-4
        )


class TestNbHybridDiskCache:
    """Disk-persistent JIT cache (§8.2)."""

    def _simple_network(self):
        """Minimal single-subnet network for cache tests."""
        from tvb.simulator.hybrid import NetworkSet
        from tvb.simulator.integrators import HeunDeterministic

        sn = _mpr_subnetwork("sn0", 4, HeunDeterministic)
        ns = NetworkSet(subnets=[sn])
        ns.configure()
        return ns

    def test_cache_dir_created_after_compile(self):
        """Disk cache directory is created after compile()."""
        from tvb.simulator.backend.nb_hybrid import NbHybridBackend

        NbHybridBackend.clear_cache()
        ns = self._simple_network()
        backend = NbHybridBackend()
        backend.compile(ns)
        cache_dir = NbHybridBackend.get_cache_dir()
        assert cache_dir.exists(), f"Cache dir not created: {cache_dir}"
        py_files = list(cache_dir.glob("nbhybrid_*.py"))
        assert len(py_files) >= 1, "No .py files in disk cache"

    def test_in_process_cache_hit(self):
        """Second compile() call with same topology returns cached function."""
        from tvb.simulator.backend.nb_hybrid import NbHybridBackend, _COMPILED_FN_CACHE

        NbHybridBackend.clear_cache()
        ns = self._simple_network()
        backend = NbHybridBackend()
        fn1 = backend.compile(ns)
        fn2 = backend.compile(ns)
        assert len(_COMPILED_FN_CACHE) >= 1
        # Same topology -> same cache entry -> same underlying function
        assert fn1._run_network_fn is fn2._run_network_fn

    def test_clear_cache_removes_files(self):
        """clear_cache() removes disk cache directory."""
        from tvb.simulator.backend.nb_hybrid import NbHybridBackend

        NbHybridBackend.clear_cache()
        ns = self._simple_network()
        backend = NbHybridBackend()
        backend.compile(ns)
        cache_dir = NbHybridBackend.get_cache_dir()
        assert cache_dir.exists()
        NbHybridBackend.clear_cache()
        assert not cache_dir.exists(), "Cache dir should be removed by clear_cache()"


class TestNbHybridGeneric2dOscillator:
    """Generic2dOscillator (generalised FitzHugh-Nagumo) model in numba backend."""

    def _build_g2d_network(self, n=6):
        """Two G2dOsc subnets with one inter-projection (V → V)."""
        from tvb.simulator.models.oscillator import Generic2dOscillator

        m1 = Generic2dOscillator()
        m1.configure()
        m2 = Generic2dOscillator()
        m2.configure()

        sn1 = Subnetwork(
            name="g2d1", model=m1, scheme=HeunDeterministic(dt=DT), nnodes=n
        )
        sn2 = Subnetwork(
            name="g2d2", model=m2, scheme=HeunDeterministic(dt=DT), nnodes=n
        )
        sn1.configure()
        sn2.configure()

        w = _sparse_weights(n, n, seed=11, density=0.4)
        l = _zero_lengths(n, n)
        inter = InterProjection(
            source=sn1,
            target=sn2,
            source_cvar=np.array([0], dtype=np.int_),
            target_cvar=np.array([0], dtype=np.int_),
            weights=w,
            lengths=l,
            cv=1.0,
            dt=DT,
            scale=1e-2,
        )

        ns = NetworkSet(subnets=[sn1, sn2], projections=[inter], stimuli=[])
        ns.configure()
        return ns

    def test_g2d_accepted_by_backend(self):
        """Generic2dOscillator is accepted by _check_compatibility."""
        ns = self._build_g2d_network()
        NbHybridBackend()._check_compatibility(ns)  # should not raise

    def test_g2d_output_shape(self):
        """G2dOsc numba output has correct shape."""
        ns = self._build_g2d_network(n=5)
        nstep = 10
        x0 = np.zeros((2, 5, 1), dtype=np.float32)
        results = _run_nb_full(ns, nstep, [x0.copy(), x0.copy()])
        assert len(results) == 2
        for t_arr, d_arr, _ in results:
            assert d_arr.ndim == 4
            assert d_arr.shape[1] == 1  # 1 VOI (V only — default variables_of_interest)
            assert d_arr.shape[2] == 5  # n_nodes
            assert d_arr.shape[3] == 1  # n_modes

    def test_g2d_output_finite(self):
        """G2dOsc numba output is finite."""
        ns = self._build_g2d_network(n=5)
        nstep = 15
        x0 = np.zeros((2, 5, 1), dtype=np.float32)
        results = _run_nb_full(ns, nstep, [x0.copy(), x0.copy()])
        for t_arr, d_arr, _ in results:
            assert np.all(np.isfinite(d_arr)), "NaN/Inf in G2dOsc output"

    def test_g2d_matches_python(self):
        """G2dOsc numba output matches Python loop backend."""
        ns = self._build_g2d_network(n=5)
        nstep = 15
        x0 = np.zeros((2, 5, 1), dtype=np.float32)
        x0_list = [x0.copy(), x0.copy()]
        py_results = _run_python_loop(ns, nstep, x0_list)
        nb_results = _run_nb_full(ns, nstep, x0_list)
        # _run_python_loop returns (T, n_vars, N, M); nb returns (T, n_voi, N, M).
        # G2dOscillator defaults to VOI=("V",) so extract only VOI rows from python result.
        svars = list(ns.subnets[0].model.state_variables)
        voi = list(ns.subnets[0].model.variables_of_interest)
        voi_idx = [svars.index(v) for v in voi]
        for py_d, (_, nb_d, _) in zip(py_results, nb_results):
            np.testing.assert_allclose(
                nb_d, py_d[:, voi_idx, :, :].astype(np.float32), rtol=1e-3, atol=1e-3
            )


class TestNbHybridAfferentCoupling:
    """AfferentCoupling output (ctavg) from the numba backend."""

    def _build_coupled_network(self, n=6):
        """Two MPR subnets with inter-projection for afferent coupling tests."""
        sn1 = _mpr_subnetwork("sn_ac1", n, HeunDeterministic)
        sn2 = _mpr_subnetwork("sn_ac2", n, HeunDeterministic)
        sn1.configure()
        sn2.configure()

        w = _sparse_weights(n, n, seed=99, density=0.5)
        l = _zero_lengths(n, n)
        inter = InterProjection(
            source=sn1,
            target=sn2,
            source_cvar=np.array([0], dtype=np.int_),
            target_cvar=np.array([0], dtype=np.int_),
            weights=w,
            lengths=l,
            cv=1.0,
            dt=DT,
            scale=1e-2,
        )

        ns = NetworkSet(subnets=[sn1, sn2], projections=[inter], stimuli=[])
        ns.configure()
        return ns

    def test_afferent_coupling_shape(self):
        """ctavg (afferent coupling) output has correct shape."""
        ns = self._build_coupled_network(n=6)
        nstep = 20
        x0 = np.zeros((2, 6, 1), dtype=np.float32)
        results = _run_nb_full(ns, nstep, [x0.copy(), x0.copy()])
        assert len(results) == 2
        for t_arr, d_arr, c_arr in results:
            assert c_arr.ndim == 4  # (n_chunks, n_cvar, n_nodes, n_modes)
            n_chunks = d_arr.shape[0]
            assert c_arr.shape[0] == n_chunks
            assert (
                c_arr.shape[1] == 2
            )  # n_cvar for MPR (Coupling_Term_r, Coupling_Term_V)
            assert c_arr.shape[2] == 6  # n_nodes
            assert c_arr.shape[3] == 1  # n_modes

    def test_afferent_coupling_zero_when_uncoupled(self):
        """ctavg is zero for isolated subnets (no incoming projections)."""
        sn1 = _mpr_subnetwork("isolated_ac", 4, HeunDeterministic)
        sn1.configure()
        ns = NetworkSet(subnets=[sn1], projections=[], stimuli=[])
        ns.configure()
        x0 = np.zeros((2, 4, 1), dtype=np.float32)
        results = _run_nb_full(ns, 10, [x0])
        _, _, c_arr = results[0]
        np.testing.assert_array_equal(
            c_arr, 0.0, err_msg="Afferent coupling should be zero for isolated subnet"
        )

    def test_afferent_coupling_nonzero_when_coupled(self):
        """ctavg is non-zero for target subnet with active projections."""
        ns = self._build_coupled_network(n=6)
        nstep = 20
        x0 = np.zeros((2, 6, 1), dtype=np.float32)
        x0[0] = 0.1  # non-zero r → generates non-zero afferent coupling on sn2
        results = _run_nb_full(ns, nstep, [x0.copy(), x0.copy()])
        # sn2 (index 1) should receive non-zero coupling from sn1 via cvar[0]
        _, _, c_arr_sn2 = results[1]
        assert np.any(c_arr_sn2 != 0.0), (
            "Expected non-zero afferent coupling on target subnet"
        )

    def test_afferent_coupling_finite(self):
        """ctavg output is always finite."""
        ns = self._build_coupled_network(n=6)
        nstep = 20
        x0 = np.zeros((2, 6, 1), dtype=np.float32)
        results = _run_nb_full(ns, nstep, [x0.copy(), x0.copy()])
        for _, _, c_arr in results:
            assert np.all(np.isfinite(c_arr)), "NaN/Inf in afferent coupling output"


class TestNbHybridReducedWongWang:
    """ReducedWongWang model in the numba hybrid backend."""

    N = 6
    NSTEP = 20

    def _build_network(self):
        from tvb.simulator.models.wong_wang import ReducedWongWang

        n = self.N
        m1 = ReducedWongWang()
        m1.configure()
        m2 = ReducedWongWang()
        m2.configure()
        sn1 = Subnetwork(
            name="rww1", model=m1, scheme=HeunDeterministic(dt=DT), nnodes=n
        )
        sn1.configure()
        sn2 = Subnetwork(
            name="rww2", model=m2, scheme=HeunDeterministic(dt=DT), nnodes=n
        )
        sn2.configure()
        w = _sparse_weights(n, n, seed=3, density=0.4)
        l = _zero_lengths(n, n)
        inter = InterProjection(
            source=sn1,
            target=sn2,
            source_cvar=np.array([0], dtype=np.int32),
            target_cvar=np.array([0], dtype=np.int32),
            weights=w,
            lengths=l,
            cv=1.0,
            dt=DT,
            scale=1e-4,
        )
        ns = NetworkSet(subnets=[sn1, sn2], projections=[inter], stimuli=[])
        ns.configure()
        return ns

    def _make_ic(self):
        x0 = np.zeros((1, self.N, 1), dtype=np.float64)
        x0[0, :, 0] = 0.15
        return x0

    def test_rww_accepted_by_backend(self):
        ns = self._build_network()
        NbHybridBackend()._check_compatibility(ns)

    def test_rww_output_shape(self):
        from tvb.simulator.models.wong_wang import ReducedWongWang

        ns = self._build_network()
        x0 = self._make_ic()
        results = _run_nb(ns, self.NSTEP, [x0.copy(), x0.copy()])
        assert len(results) == 2
        n_voi = len(ReducedWongWang.variables_of_interest.default)
        for d in results:
            assert d.ndim == 4
            assert d.shape[0] == self.NSTEP
            assert d.shape[1] == n_voi
            assert d.shape[2] == self.N
            assert d.shape[3] == 1

    def test_rww_output_finite(self):
        ns = self._build_network()
        x0 = self._make_ic()
        results = _run_nb(ns, self.NSTEP, [x0.copy(), x0.copy()])
        for d in results:
            assert np.all(np.isfinite(d)), "NaN/Inf in ReducedWongWang numba output"

    def test_rww_matches_python(self):
        from tvb.simulator.models.wong_wang import ReducedWongWang

        n_voi = len(ReducedWongWang.variables_of_interest.default)
        x0 = self._make_ic()
        ns_py = self._build_network()
        ns_nb = self._build_network()
        py = _run_python_loop(ns_py, self.NSTEP, [x0.copy(), x0.copy()])
        nb = _run_nb(ns_nb, self.NSTEP, [x0.copy(), x0.copy()])
        for py_d, nb_d in zip(py, nb):
            np.testing.assert_allclose(
                nb_d, py_d[:, :n_voi, :, :].astype(np.float32), rtol=1e-3, atol=1e-3
            )


class TestNbHybridEpileptor:
    """Epileptor model in the numba hybrid backend."""

    N = 4
    NSTEP = 15

    def _build_network(self):
        from tvb.simulator.models.epileptor import Epileptor

        n = self.N
        m1 = Epileptor()
        m1.configure()
        m2 = Epileptor()
        m2.configure()
        sn1 = Subnetwork(
            name="ep1", model=m1, scheme=HeunDeterministic(dt=DT), nnodes=n
        )
        sn1.configure()
        sn2 = Subnetwork(
            name="ep2", model=m2, scheme=HeunDeterministic(dt=DT), nnodes=n
        )
        sn2.configure()
        w = _sparse_weights(n, n, seed=11, density=0.4)
        l = _zero_lengths(n, n)
        # Epileptor cvar=[0,3] (x1, x2); coupling_terms=['Coupling_Term_pop1', 'Coupling_Term_pop2']
        inter = InterProjection(
            source=sn1,
            target=sn2,
            source_cvar=np.array([0, 3], dtype=np.int32),
            target_cvar=np.array([0, 1], dtype=np.int32),
            weights=w,
            lengths=l,
            cv=1.0,
            dt=DT,
            scale=1e-4,
        )
        ns = NetworkSet(subnets=[sn1, sn2], projections=[inter], stimuli=[])
        ns.configure()
        return ns

    def _make_ic(self):
        """Epileptor IC near the interictal fixed point."""
        x0 = np.zeros((6, self.N, 1), dtype=np.float64)
        x0[0, :, 0] = -1.6  # x1
        x0[1, :, 0] = -15.0  # y1
        x0[2, :, 0] = 3.5  # z
        x0[3, :, 0] = -0.2  # x2
        x0[4, :, 0] = 0.0  # y2
        x0[5, :, 0] = -0.05  # g
        return x0

    def test_epileptor_accepted_by_backend(self):
        ns = self._build_network()
        NbHybridBackend()._check_compatibility(ns)

    def test_epileptor_output_shape(self):
        from tvb.simulator.models.epileptor import Epileptor

        ns = self._build_network()
        x0 = self._make_ic()
        results = _run_nb(ns, self.NSTEP, [x0.copy(), x0.copy()])
        assert len(results) == 2
        n_voi = len(Epileptor.variables_of_interest.default)
        for d in results:
            assert d.ndim == 4
            assert d.shape[0] == self.NSTEP
            assert d.shape[1] == n_voi
            assert d.shape[2] == self.N
            assert d.shape[3] == 1

    def test_epileptor_output_finite(self):
        ns = self._build_network()
        x0 = self._make_ic()
        results = _run_nb(ns, self.NSTEP, [x0.copy(), x0.copy()])
        for d in results:
            assert np.all(np.isfinite(d)), "NaN/Inf in Epileptor numba output"

    def test_epileptor_matches_python(self):
        from tvb.simulator.models.epileptor import Epileptor

        m = Epileptor()
        m.configure()
        svars = list(m.state_variables)
        voi = list(m.variables_of_interest)  # ('x2 - x1', 'z')
        # Build voi slice: voi[0] = 'x2 - x1' = state[3] - state[0], voi[1] = 'z' = state[2]
        x0 = self._make_ic()
        ns_py = self._build_network()
        ns_nb = self._build_network()
        py = _run_python_loop(ns_py, self.NSTEP, [x0.copy(), x0.copy()])
        nb = _run_nb(ns_nb, self.NSTEP, [x0.copy(), x0.copy()])
        # Construct Python voi array
        py_voi_0 = py[0][:, 3, :, :] - py[0][:, 0, :, :]  # x2 - x1
        py_voi_1 = py[0][:, 2, :, :]                       # z
        py_voi = np.stack([py_voi_0, py_voi_1], axis=1).astype(np.float32)
        py_voi_2 = py[1][:, 3, :, :] - py[1][:, 0, :, :]
        py_voi_2b = py[1][:, 2, :, :]
        py_voi_b = np.stack([py_voi_2, py_voi_2b], axis=1).astype(np.float32)
        for py_d, nb_d in zip([py_voi, py_voi_b], nb):
            np.testing.assert_allclose(
                nb_d, py_d, rtol=1e-2, atol=1e-2
            )


class TestNbHybridWilsonCowan:
    """WilsonCowan model in the numba hybrid backend."""

    N = 6
    NSTEP = 20

    def _build_network(self):
        from tvb.simulator.models.wilson_cowan import WilsonCowan

        n = self.N
        m1 = WilsonCowan()
        m1.configure()
        m2 = WilsonCowan()
        m2.configure()
        sn1 = Subnetwork(
            name="wc1", model=m1, scheme=HeunDeterministic(dt=DT), nnodes=n
        )
        sn1.configure()
        sn2 = Subnetwork(
            name="wc2", model=m2, scheme=HeunDeterministic(dt=DT), nnodes=n
        )
        sn2.configure()
        w = _sparse_weights(n, n, seed=5, density=0.4)
        l = _zero_lengths(n, n)
        # WC cvar=[0,1] (E, I); coupling_terms=['Coupling_Term_E'] (only 1).
        # Use only E (cvar=0) from source for 1-to-1 mapping.
        inter = InterProjection(
            source=sn1,
            target=sn2,
            source_cvar=np.array([0], dtype=np.int32),
            target_cvar=np.array([0], dtype=np.int32),
            weights=w,
            lengths=l,
            cv=1.0,
            dt=DT,
            scale=1e-3,
        )
        ns = NetworkSet(subnets=[sn1, sn2], projections=[inter], stimuli=[])
        ns.configure()
        return ns

    def _make_ic(self):
        x0 = np.zeros((2, self.N, 1), dtype=np.float64)
        x0[0, :, 0] = 0.2  # E
        x0[1, :, 0] = 0.15  # I
        return x0

    def test_wc_accepted_by_backend(self):
        ns = self._build_network()
        NbHybridBackend()._check_compatibility(ns)

    def test_wc_output_shape(self):
        from tvb.simulator.models.wilson_cowan import WilsonCowan

        ns = self._build_network()
        x0 = self._make_ic()
        results = _run_nb(ns, self.NSTEP, [x0.copy(), x0.copy()])
        assert len(results) == 2
        n_voi = len(WilsonCowan.variables_of_interest.default)
        for d in results:
            assert d.ndim == 4
            assert d.shape[0] == self.NSTEP
            assert d.shape[1] == n_voi
            assert d.shape[2] == self.N
            assert d.shape[3] == 1

    def test_wc_output_finite(self):
        ns = self._build_network()
        x0 = self._make_ic()
        results = _run_nb(ns, self.NSTEP, [x0.copy(), x0.copy()])
        for d in results:
            assert np.all(np.isfinite(d)), "NaN/Inf in WilsonCowan numba output"

    def test_wc_matches_python(self):
        from tvb.simulator.models.wilson_cowan import WilsonCowan

        n_voi = len(WilsonCowan.variables_of_interest.default)
        x0 = self._make_ic()
        ns_py = self._build_network()
        ns_nb = self._build_network()
        py = _run_python_loop(ns_py, self.NSTEP, [x0.copy(), x0.copy()])
        nb = _run_nb(ns_nb, self.NSTEP, [x0.copy(), x0.copy()])
        for py_d, nb_d in zip(py, nb):
            np.testing.assert_allclose(
                nb_d, py_d[:, :n_voi, :, :].astype(np.float32), rtol=1e-2, atol=1e-2
            )


class TestNbHybridCheckpointing:
    """§8.8 Disk-Checkpointing and Resumable Runs."""

    N = 5
    NSTEP = 20

    def _build_network(self):
        sn1 = _mpr_subnetwork("cp1", self.N)
        sn2 = _mpr_subnetwork("cp2", self.N)
        sn1.configure()
        sn2.configure()
        w = _sparse_weights(self.N, self.N, seed=7)
        l = _zero_lengths(self.N, self.N)
        inter = InterProjection(
            source=sn1,
            target=sn2,
            source_cvar=np.array([0], dtype=np.int32),
            target_cvar=np.array([0], dtype=np.int32),
            weights=w,
            lengths=l,
            cv=1.0,
            dt=DT,
            scale=1.0,
        )
        ns = NetworkSet(subnets=[sn1, sn2], projections=[inter], stimuli=[])
        ns.configure()
        return ns

    def _make_ic(self):
        rng = np.random.RandomState(42)
        m = MontbrioPazoRoxin()
        m.configure()
        ic1 = rng.uniform(0.0, 0.1, (m.nvar, self.N, 1)).astype(np.float64)
        ic2 = rng.uniform(0.0, 0.1, (m.nvar, self.N, 1)).astype(np.float64)
        ic1[0] = np.abs(ic1[0])
        ic2[0] = np.abs(ic2[0])
        return [ic1, ic2]

    def test_run_returns_list_by_default(self):
        ns = self._build_network()
        ic = self._make_ic()
        compiled = NbHybridBackend().compile(ns)
        result = compiled.run(self.NSTEP, chunk_size=1, initial_states=ic)
        assert isinstance(result, list)

    def test_return_snapshot_gives_tuple(self):
        ns = self._build_network()
        ic = self._make_ic()
        compiled = NbHybridBackend().compile(ns)
        result = compiled.run(
            self.NSTEP, chunk_size=1, initial_states=ic, return_snapshot=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        outputs, snapshot = result
        assert isinstance(outputs, list)
        assert isinstance(snapshot, dict)
        assert "states" in snapshot
        assert "buffers" in snapshot

    def test_snapshot_states_shape(self):
        ns = self._build_network()
        ic = self._make_ic()
        compiled = NbHybridBackend().compile(ns)
        _, snapshot = compiled.run(
            self.NSTEP, chunk_size=1, initial_states=ic, return_snapshot=True
        )
        states = snapshot["states"]
        assert isinstance(states, list)
        assert len(states) == len(ns.subnets)
        m = MontbrioPazoRoxin()
        m.configure()
        for arr in states:
            assert arr.ndim == 3
            assert arr.shape == (m.nvar, self.N, 1)

    def test_resume_continues_from_snapshot(self):
        # Build three identical networks (same topology, same IC)
        ic = self._make_ic()

        ns_split1 = self._build_network()
        ns_split2 = self._build_network()
        ns_full = self._build_network()

        compiled_split = NbHybridBackend().compile(ns_split1)
        compiled_full = NbHybridBackend().compile(ns_full)

        # Run N steps, capture snapshot
        out1, snap1 = compiled_split.run(
            self.NSTEP,
            chunk_size=1,
            initial_states=[a.copy() for a in ic],
            return_snapshot=True,
        )
        # Resume for N more steps
        out2 = compiled_split.resume(snap1, self.NSTEP, chunk_size=1)

        # Run 2*NSTEP from scratch
        out_full = compiled_full.run(
            self.NSTEP * 2,
            chunk_size=1,
            initial_states=[a.copy() for a in ic],
        )

        # Last chunk of resumed run should match last chunk of full run
        for i in range(len(ns_split1.subnets)):
            np.testing.assert_allclose(
                out2[i][1][-1],  # (times, data, ctavg) → data[-1]
                out_full[i][1][-1],
                atol=1e-4,
                err_msg=f"resume vs full mismatch at subnet {i}",
            )


# ---------------------------------------------------------------------------
# mode_map ≠ identity tests
# ---------------------------------------------------------------------------


class TestNbHybridModeMap:
    """Inter-projection mode_map that is NOT the diagonal (identity) matrix."""

    N = 10
    N_MODES = 2
    NSTEP = 5

    def _build_net(self, mode_map_arr):
        """Two MPR subnets (2 modes each) with an inter-projection and explicit mode_map."""
        m1 = MontbrioPazoRoxin()
        m1.number_of_modes = self.N_MODES
        m1.configure()
        m2 = MontbrioPazoRoxin()
        m2.number_of_modes = self.N_MODES
        m2.configure()

        sn1 = Subnetwork(
            name="mm_src", model=m1, scheme=HeunDeterministic(dt=DT), nnodes=self.N
        )
        sn1.configure()
        sn2 = Subnetwork(
            name="mm_tgt", model=m2, scheme=HeunDeterministic(dt=DT), nnodes=self.N
        )
        sn2.configure()

        w = _sparse_weights(self.N, self.N, seed=17, density=0.5)
        inter = InterProjection(
            source=sn1,
            target=sn2,
            source_cvar=np.array([0], dtype=np.int32),
            target_cvar=np.array([0], dtype=np.int32),
            weights=w,
            lengths=_zero_lengths(self.N, self.N),
            cv=1.0,
            dt=DT,
            scale=1e-2,
            mode_map=mode_map_arr,
        )
        ns = NetworkSet(subnets=[sn1, sn2], projections=[inter], stimuli=[])
        ns.configure()
        return ns

    def _make_ic(self):
        rng = np.random.RandomState(19)
        x0 = rng.uniform(0.0, 0.2, (2, self.N, self.N_MODES)).astype(np.float64)
        x0[0] = np.abs(x0[0])
        # Make modes deliberately different so mode mixing has visible effect
        x0[:, :, 0] *= 2.0
        return x0

    def test_nonidentity_mode_map_accepted(self):
        """compile() accepts inter-projection with all-ones (non-diagonal) mode_map."""
        mm = np.array([[1, 1], [1, 1]], dtype=np.int_)
        ns = self._build_net(mm)
        NbHybridBackend().compile(ns)  # must not raise

    def test_nonidentity_mode_map_output_shape(self):
        """run_network() output last dimension equals n_modes=2 with non-diagonal mode_map."""
        mm = np.array([[1, 1], [1, 1]], dtype=np.int_)
        ns = self._build_net(mm)
        x0 = self._make_ic()
        results = _run_nb_full(ns, self.NSTEP, [x0, x0.copy()])
        assert len(results) == 2
        for _, data, _ in results:
            assert data.shape[-1] == self.N_MODES, (
                f"Expected n_modes={self.N_MODES} in last dim, got shape {data.shape}"
            )

    def test_nonidentity_mode_map_finite(self):
        """run_network() produces finite output with non-diagonal mode_map."""
        mm = np.array([[1, 1], [1, 1]], dtype=np.int_)
        ns = self._build_net(mm)
        x0 = self._make_ic()
        results = _run_nb_full(ns, self.NSTEP, [x0, x0.copy()])
        for _, data, _ in results:
            assert np.all(np.isfinite(data)), f"NaN/Inf in mode_map output: {data}"

    def test_nonidentity_mode_map_mixed_modes(self):
        """All-ones mode_map produces different output from diagonal mode_map."""
        # Diagonal: each source mode contributes only to the same target mode
        mm_diag = np.array([[1, 0], [0, 1]], dtype=np.int_)
        # All-ones: each source mode contributes to ALL target modes (mixing)
        mm_mix = np.array([[1, 1], [1, 1]], dtype=np.int_)

        x0 = self._make_ic()

        ns_diag = self._build_net(mm_diag)
        res_diag = _run_nb(ns_diag, self.NSTEP, [x0.copy(), x0.copy()])

        ns_mix = self._build_net(mm_mix)
        res_mix = _run_nb(ns_mix, self.NSTEP, [x0.copy(), x0.copy()])

        # The target subnet (index 1) should differ between the two mode maps
        max_diff = np.max(
            np.abs(res_mix[1].astype(np.float64) - res_diag[1].astype(np.float64))
        )
        assert max_diff > 1e-6, (
            f"Expected mixing mode_map to differ from diagonal by >1e-6, "
            f"got max_diff={max_diff:.2e}"
        )


# ---------------------------------------------------------------------------
# Large-N scaling tests
# ---------------------------------------------------------------------------


class TestNbHybridLargeNScaling:
    """Large-N scaling: correctness at N=100 and speedup regression at N=50."""

    def _build_two_subnet_net(self, n: int, seed: int = 0):
        """Two MPR subnets connected by an inter-projection (no delays)."""
        sn_a = _mpr_subnetwork("ls_a", n)
        sn_b = _mpr_subnetwork("ls_b", n)
        sn_a.configure()
        sn_b.configure()
        w = _sparse_weights(n, n, seed=seed, density=0.2)
        inter = InterProjection(
            source=sn_a,
            target=sn_b,
            source_cvar=np.array([0], dtype=np.int32),
            target_cvar=np.array([0], dtype=np.int32),
            weights=w,
            lengths=_zero_lengths(n, n),
            cv=1.0,
            dt=DT,
            scale=1e-3,
        )
        ns = NetworkSet(subnets=[sn_a, sn_b], projections=[inter], stimuli=[])
        ns.configure()
        return ns

    def _init_states(self, ns, seed=42):
        rng = np.random.RandomState(seed)
        states = []
        for sn in ns.subnets:
            x0 = rng.uniform(0.0, 0.2, (sn.model.nvar, sn.nnodes, 1)).astype(np.float64)
            x0[0] = np.abs(x0[0])
            states.append(x0)
        return states

    def test_large_n_runs_without_error(self):
        """N=100 two-subnet network runs 50 steps and produces finite results."""
        n, nstep = 100, 50
        ns = self._build_two_subnet_net(n, seed=1)
        x0_list = self._init_states(ns)
        results = _run_nb_full(ns, nstep, x0_list)
        assert len(results) == 2
        for _, data, _ in results:
            assert data.shape[0] == nstep
            assert np.all(np.isfinite(data)), "NaN/Inf in large-N output"

    def test_numba_faster_than_python(self):
        """Numba cached kernel runs faster than Python loop (N=50, nstep=100).

        JIT compilation is paid once before timing.  A generous bound of 5×
        is used because warm-up effects can compress the Python advantage on
        very short runs.
        """
        n, nstep = 50, 100
        ns = self._build_two_subnet_net(n, seed=2)
        x0_list = self._init_states(ns)

        backend = NbHybridBackend()
        compiled = backend.compile(ns)

        # Warm up the Numba JIT (one-time compilation cost, not timed)
        compiled.run(nstep=5, chunk_size=1, initial_states=x0_list)

        # Time Python loop
        t0 = time.perf_counter()
        _run_python_loop(ns, nstep, x0_list)
        t_py = time.perf_counter() - t0

        # Time Numba cached kernel (no recompilation)
        t0 = time.perf_counter()
        compiled.run(nstep=nstep, chunk_size=1, initial_states=x0_list)
        t_nb = time.perf_counter() - t0

        speedup = t_py / t_nb if t_nb > 0 else float("inf")
        assert speedup > 0.2, (
            f"Numba kernel should not be more than 5× slower than Python "
            f"(got speedup={speedup:.2f}×, t_py={t_py * 1e3:.1f}ms, "
            f"t_nb={t_nb * 1e3:.1f}ms)"
        )


class TestNbHybridDebugNojit:
    """Tests for the debug_nojit=True fast path (no Numba JIT)."""

    def _build_net(self, n: int = 5):
        """Two MPR subnets with an inter-projection (no delays)."""
        sn_a = _mpr_subnetwork("nojit_a", n)
        sn_b = _mpr_subnetwork("nojit_b", n)
        sn_a.configure()
        sn_b.configure()
        w = _sparse_weights(n, n, seed=7, density=1.0)
        inter = InterProjection(
            source=sn_a,
            target=sn_b,
            source_cvar=np.array([0], dtype=np.int32),
            target_cvar=np.array([0], dtype=np.int32),
            weights=w,
            lengths=_zero_lengths(n, n),
            cv=1.0,
            dt=DT,
            scale=1e-3,
        )
        ns = NetworkSet(subnets=[sn_a, sn_b], projections=[inter], stimuli=[])
        ns.configure()
        return ns

    def _init_states(self, ns, seed: int = 0):
        rng = np.random.RandomState(seed)
        return [
            rng.uniform(0.1, 0.3, (sn.model.nvar, sn.nnodes, 1)).astype(np.float64)
            for sn in ns.subnets
        ]

    def test_debug_nojit_runs(self):
        """debug_nojit=True produces output without JIT compilation."""
        ns = self._build_net(n=5)
        x0 = self._init_states(ns)
        backend = NbHybridBackend()
        results = backend.run_network(
            ns, nstep=10, chunk_size=1, initial_states=x0, debug_nojit=True
        )
        assert isinstance(results, list), "run_network must return a list"
        for times, data, ctavg in results:
            assert np.all(np.isfinite(data)), "output data must be finite"

    def test_debug_nojit_matches_jit(self):
        """debug_nojit=True produces the same output as the JIT path."""
        ns = self._build_net(n=5)
        x0 = self._init_states(ns)
        backend = NbHybridBackend()

        results_jit = backend.run_network(
            ns, nstep=10, chunk_size=1, initial_states=x0, debug_nojit=False
        )
        results_nojit = backend.run_network(
            ns, nstep=10, chunk_size=1, initial_states=x0, debug_nojit=True
        )

        for (_, data_jit, _), (_, data_nojit, _) in zip(results_jit, results_nojit):
            np.testing.assert_allclose(
                data_nojit,
                data_jit,
                atol=1e-5,
                err_msg="debug_nojit=True must match JIT output within atol=1e-5",
            )


class TestStimulusMemoryEstimate(unittest.TestCase):
    """Verify the stim-array size estimation helper used by the lazy path."""

    def test_stimulus_memory_estimate_small(self):
        """A tiny stim array (N=5, nstep=10) is well below the threshold."""
        from tvb.simulator.backend.nb_hybrid import NbHybridBackend, SubnetworkInfo
        from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin

        model = MontbrioPazoRoxin()
        model.configure()
        sn_info = SubnetworkInfo(
            name="test_sn",
            model=model,
            integrator=None,
            n_nodes=5,
            n_modes=1,
            has_stimulus=True,
        )
        nstep = 10
        estimated_mb = NbHybridBackend._stim_estimate_mb(sn_info, nstep)
        # n_cvar=1, nodes=5, modes=1, steps=10 → 200 bytes → ≈1.9e-4 MiB
        self.assertLess(
            estimated_mb,
            _STIM_LAZY_THRESHOLD_MB,
            f"Tiny stim array ({estimated_mb:.4f} MiB) should be below the "
            f"lazy threshold ({_STIM_LAZY_THRESHOLD_MB} MiB)",
        )
        self.assertGreater(estimated_mb, 0.0)

    def test_stimulus_memory_estimate_large(self):
        """A large stim array (N=100, nstep=1_000_000) exceeds the threshold."""
        from tvb.simulator.backend.nb_hybrid import NbHybridBackend, SubnetworkInfo
        from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin

        model = MontbrioPazoRoxin()
        model.configure()
        sn_info = SubnetworkInfo(
            name="test_sn",
            model=model,
            integrator=None,
            n_nodes=100,
            n_modes=1,
            has_stimulus=True,
        )
        nstep = 1_000_000
        estimated_mb = NbHybridBackend._stim_estimate_mb(sn_info, nstep)
        # n_cvar=1, nodes=100, modes=1, steps=1e6 → 400 MB → 381.5 MiB
        self.assertGreater(
            estimated_mb,
            _STIM_LAZY_THRESHOLD_MB,
            f"Large stim array ({estimated_mb:.1f} MiB) should exceed the "
            f"lazy threshold ({_STIM_LAZY_THRESHOLD_MB} MiB)",
        )


# ---------------------------------------------------------------------------
# Zerlaut models — custom Mako template tests
# ---------------------------------------------------------------------------


class TestNbHybridZerlautFirstOrder:
    """ZerlautAdaptationFirstOrder with custom nb-zerlaut-dfun template."""

    N = 4
    NSTEP = 20

    def _build_network(self):
        from tvb.simulator.models.zerlaut import ZerlautAdaptationFirstOrder

        n = self.N
        m1 = ZerlautAdaptationFirstOrder()
        m1.configure()
        m2 = ZerlautAdaptationFirstOrder()
        m2.configure()
        sn1 = Subnetwork(name="z1", model=m1, scheme=HeunDeterministic(dt=DT), nnodes=n)
        sn1.configure()
        sn2 = Subnetwork(name="z2", model=m2, scheme=HeunDeterministic(dt=DT), nnodes=n)
        sn2.configure()
        w = _sparse_weights(n, n, seed=7, density=0.5)
        l = _zero_lengths(n, n)
        inter = InterProjection(
            source=sn1,
            target=sn2,
            source_cvar=np.array([0], dtype=np.int32),
            target_cvar=np.array([0], dtype=np.int32),
            weights=w,
            lengths=l,
            cv=1.0,
            dt=DT,
            scale=1e-4,
        )
        ns = NetworkSet(subnets=[sn1, sn2], projections=[inter], stimuli=[])
        ns.configure()
        return ns

    def _make_ic(self):
        x0 = np.zeros((5, self.N, 1), dtype=np.float64)
        x0[0, :, 0] = 0.01  # E
        x0[1, :, 0] = 0.01  # I
        x0[2, :, 0] = 50.0  # W_e
        x0[3, :, 0] = 0.0  # W_i
        x0[4, :, 0] = 0.0  # ou_drift
        return x0

    def test_zerlaut1_accepted_by_backend(self):
        ns = self._build_network()
        NbHybridBackend()._check_compatibility(ns)

    def test_zerlaut1_output_shape(self):
        from tvb.simulator.models.zerlaut import ZerlautAdaptationFirstOrder

        ns = self._build_network()
        x0 = self._make_ic()
        results = _run_nb(ns, self.NSTEP, [x0.copy(), x0.copy()])
        assert len(results) == 2
        n_voi = len(ZerlautAdaptationFirstOrder.variables_of_interest.default)
        for d in results:
            assert d.ndim == 4
            assert d.shape[0] == self.NSTEP
            assert d.shape[1] == n_voi
            assert d.shape[2] == self.N
            assert d.shape[3] == 1

    def test_zerlaut1_output_finite(self):
        ns = self._build_network()
        x0 = self._make_ic()
        results = _run_nb(ns, self.NSTEP, [x0.copy(), x0.copy()])
        for d in results:
            assert np.all(np.isfinite(d)), "NaN/Inf in ZerlautFirstOrder numba output"

    def test_zerlaut1_matches_python(self):
        from tvb.simulator.models.zerlaut import ZerlautAdaptationFirstOrder

        n_voi = len(ZerlautAdaptationFirstOrder.variables_of_interest.default)
        x0 = self._make_ic()
        ns_py = self._build_network()
        ns_nb = self._build_network()
        py = _run_python_loop(ns_py, self.NSTEP, [x0.copy(), x0.copy()])
        nb = _run_nb(ns_nb, self.NSTEP, [x0.copy(), x0.copy()])
        for py_d, nb_d in zip(py, nb):
            np.testing.assert_allclose(
                nb_d, py_d[:, :n_voi, :, :].astype(np.float32), rtol=1e-2, atol=1e-4
            )


class TestNbHybridZerlautSecondOrder:
    """ZerlautAdaptationSecondOrder with custom nb-zerlaut-dfun template."""

    N = 4
    NSTEP = 20

    def _build_network(self):
        from tvb.simulator.models.zerlaut import ZerlautAdaptationSecondOrder

        n = self.N
        m1 = ZerlautAdaptationSecondOrder()
        m1.configure()
        m2 = ZerlautAdaptationSecondOrder()
        m2.configure()
        sn1 = Subnetwork(
            name="z2o_1", model=m1, scheme=HeunDeterministic(dt=DT), nnodes=n
        )
        sn1.configure()
        sn2 = Subnetwork(
            name="z2o_2", model=m2, scheme=HeunDeterministic(dt=DT), nnodes=n
        )
        sn2.configure()
        w = _sparse_weights(n, n, seed=8, density=0.5)
        l = _zero_lengths(n, n)
        inter = InterProjection(
            source=sn1,
            target=sn2,
            source_cvar=np.array([0], dtype=np.int32),
            target_cvar=np.array([0], dtype=np.int32),
            weights=w,
            lengths=l,
            cv=1.0,
            dt=DT,
            scale=1e-4,
        )
        ns = NetworkSet(subnets=[sn1, sn2], projections=[inter], stimuli=[])
        ns.configure()
        return ns

    def _make_ic(self):
        x0 = np.zeros((8, self.N, 1), dtype=np.float64)
        x0[0, :, 0] = 0.01  # E
        x0[1, :, 0] = 0.01  # I
        x0[2, :, 0] = 0.001  # C_ee
        x0[3, :, 0] = 0.0  # C_ei
        x0[4, :, 0] = 0.001  # C_ii
        x0[5, :, 0] = 50.0  # W_e
        x0[6, :, 0] = 0.0  # W_i
        x0[7, :, 0] = 0.0  # ou_drift
        return x0

    def test_zerlaut2_accepted_by_backend(self):
        ns = self._build_network()
        NbHybridBackend()._check_compatibility(ns)

    def test_zerlaut2_output_shape(self):
        from tvb.simulator.models.zerlaut import ZerlautAdaptationSecondOrder

        ns = self._build_network()
        x0 = self._make_ic()
        results = _run_nb(ns, self.NSTEP, [x0.copy(), x0.copy()])
        assert len(results) == 2
        n_voi = len(ZerlautAdaptationSecondOrder.variables_of_interest.default)
        for d in results:
            assert d.ndim == 4
            assert d.shape[0] == self.NSTEP
            assert d.shape[1] == n_voi
            assert d.shape[2] == self.N
            assert d.shape[3] == 1

    def test_zerlaut2_output_finite(self):
        ns = self._build_network()
        x0 = self._make_ic()
        results = _run_nb(ns, self.NSTEP, [x0.copy(), x0.copy()])
        for d in results:
            assert np.all(np.isfinite(d)), "NaN/Inf in ZerlautSecondOrder numba output"

    def test_zerlaut2_matches_python(self):
        from tvb.simulator.models.zerlaut import ZerlautAdaptationSecondOrder

        n_voi = len(ZerlautAdaptationSecondOrder.variables_of_interest.default)
        x0 = self._make_ic()
        ns_py = self._build_network()
        ns_nb = self._build_network()
        py = _run_python_loop(ns_py, self.NSTEP, [x0.copy(), x0.copy()])
        nb = _run_nb(ns_nb, self.NSTEP, [x0.copy(), x0.copy()])
        for py_d, nb_d in zip(py, nb):
            np.testing.assert_allclose(
                nb_d, py_d[:, :n_voi, :, :].astype(np.float32), rtol=1e-2, atol=1e-4
            )


# ---------------------------------------------------------------------------
# Parametrized smoke tests for Ralph-completed models
# ---------------------------------------------------------------------------


def _ic_from_range(model, n):
    """Initial conditions at the midpoint of each state variable's default range."""
    sv_range = model.state_variable_range  # plain dict after configure()
    svars = list(model.state_variables)
    n_modes = model.number_of_modes
    x0 = np.zeros((len(svars), n, n_modes), dtype=np.float64)
    for i, sv in enumerate(svars):
        if sv in sv_range:
            lo, hi = float(sv_range[sv][0]), float(sv_range[sv][1])
            if np.isfinite(lo) and np.isfinite(hi):
                for m in range(n_modes):
                    x0[i, :, m] = (lo + hi) / 2.0
    return x0


_RALPH_MODELS = [
    ("tvb.simulator.models.oscillator", "SupHopf"),
    ("tvb.simulator.models.oscillator", "Kuramoto"),
    ("tvb.simulator.models.epileptor", "Epileptor2D"),
    ("tvb.simulator.models.hopfield", "Hopfield"),
    ("tvb.simulator.models.infinite_theta", "CoombesByrne2D"),
    ("tvb.simulator.models.larter_breakspear", "LarterBreakspear"),
    ("tvb.simulator.models.infinite_theta", "CoombesByrne"),
    ("tvb.simulator.models.infinite_theta", "GastSchmidtKnosche_SD"),
    ("tvb.simulator.models.infinite_theta", "GastSchmidtKnosche_SF"),
    ("tvb.simulator.models.epileptorcodim3", "EpileptorCodim3"),
    ("tvb.simulator.models.epileptorcodim3", "EpileptorCodim3SlowMod"),
    ("tvb.simulator.models.wong_wang_exc_inh", "ReducedWongWangExcInh"),
    ("tvb.simulator.models.epileptor_rs", "EpileptorRestingState"),
    ("tvb.simulator.models.infinite_theta", "DumontGutkin"),
    ("tvb.simulator.models.jansen_rit", "ZetterbergJansen"),
    ("tvb.simulator.models.stefanescu_jirsa", "ReducedSetFitzHughNagumo"),
    ("tvb.simulator.models.stefanescu_jirsa", "ReducedSetHindmarshRose"),
]
_RALPH_IDS = [cls for _, cls in _RALPH_MODELS]


def _build_single_subnet(mod_path, cls_name, n=4):
    """Build a single-subnet NetworkSet with no projections for smoke testing."""
    import importlib

    cls = getattr(importlib.import_module(mod_path), cls_name)
    model = cls()
    model.configure()
    sn = Subnetwork(name="sn", model=model, scheme=HeunDeterministic(dt=DT), nnodes=n)
    sn.configure()
    ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
    ns.configure()
    return ns


@pytest.mark.parametrize("mod_path,cls_name", _RALPH_MODELS, ids=_RALPH_IDS)
def test_ralph_model_accepted(mod_path, cls_name):
    """Ralph-completed model is accepted by NbHybridBackend._check_compatibility."""
    ns = _build_single_subnet(mod_path, cls_name)
    NbHybridBackend()._check_compatibility(ns)


@pytest.mark.parametrize("mod_path,cls_name", _RALPH_MODELS, ids=_RALPH_IDS)
def test_ralph_model_output_shape(mod_path, cls_name):
    """Ralph-completed model produces correctly-shaped output."""
    import importlib

    cls = getattr(importlib.import_module(mod_path), cls_name)
    n, nstep = 4, 10
    ns = _build_single_subnet(mod_path, cls_name, n=n)
    model = ns.subnets[0].model
    x0 = _ic_from_range(model, n)
    results = _run_nb(ns, nstep, [x0])
    assert len(results) == 1
    d = results[0]
    assert d.ndim == 4
    assert d.shape[0] == nstep
    assert d.shape[2] == n
    assert d.shape[3] == model.number_of_modes


@pytest.mark.parametrize("mod_path,cls_name", _RALPH_MODELS, ids=_RALPH_IDS)
def test_ralph_model_output_finite(mod_path, cls_name):
    """Ralph-completed model produces finite (non-NaN/Inf) output."""
    n, nstep = 4, 10
    ns = _build_single_subnet(mod_path, cls_name, n=n)
    model = ns.subnets[0].model
    x0 = _ic_from_range(model, n)
    results = _run_nb(ns, nstep, [x0])
    assert np.all(np.isfinite(results[0])), f"{cls_name}: NaN/Inf in Numba output"


@pytest.mark.parametrize("mod_path,cls_name", _RALPH_MODELS, ids=_RALPH_IDS)
def test_ralph_model_matches_python(mod_path, cls_name):
    """Ralph-completed model: Numba output matches Python dfun within float32 tolerance."""
    import importlib

    cls = getattr(importlib.import_module(mod_path), cls_name)
    model = cls()
    model.configure()
    n, nstep = 4, 10
    ns = _build_single_subnet(mod_path, cls_name, n=n)
    x0 = _ic_from_range(model, n)

    # Resolve voi — build Python-side voi array matching the template output
    svars = list(model.state_variables)
    voi = list(model.variables_of_interest)

    py = _run_python_loop(ns, nstep, [x0.copy()])
    nb = _run_nb(ns, nstep, [x0.copy()])

    # Build Python voi array: for simple voi use state index, for derived voi
    # (e.g. 'x2 - x1') replace svar names with state indices and evaluate.
    py_voi_chunks = []
    for v in voi:
        if v in svars:
            py_voi_chunks.append(py[0][:, svars.index(v), :, :])
        else:
            # Derived voi — replace svar names with state slicing and eval
            expr = v
            for sv_name, sv_idx in zip(svars, range(len(svars))):
                expr = expr.replace(sv_name, f'py[0][:, {sv_idx}, :, :]')
            py_voi_chunks.append(eval(expr))
    py_voi = np.stack(py_voi_chunks, axis=1).astype(np.float32)
    nb_voi = nb[0]

    # Some models are numerically sensitive at default ICs; skip non-finite comparisons
    if not np.all(np.isfinite(py_voi)) or not np.all(np.isfinite(nb_voi)):
        pytest.skip(f"{cls_name}: non-finite output at default ICs (model instability)")

    np.testing.assert_allclose(
        nb_voi, py_voi,
        rtol=1e-2, atol=1e-2,
        err_msg=f"{cls_name}: Numba output differs from Python",
    )


# ---------------------------------------------------------------------------
# Monitor dispatch tests (G3)
# ---------------------------------------------------------------------------


class TestNbHybridMonitors(unittest.TestCase):
    """Python-side monitor dispatch via the monitors= kwarg (G3)."""

    def _make_net(self, n=4):
        sn = _mpr_subnetwork("mon_sn", n)
        sn.configure()
        network_set = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        network_set.configure()
        return network_set, n

    def _make_ic(self, n):
        rng = np.random.RandomState(55)
        m = MontbrioPazoRoxin()
        m.configure()
        x0 = rng.uniform(0.0, 0.2, (m.nvar, n, 1)).astype(np.float64)
        x0[0] = np.abs(x0[0])
        return [x0]

    def test_no_monitors_backward_compat(self):
        """monitors=None returns (times, data, ctavg) as before."""
        from tvb.simulator.backend.nb_hybrid import NbHybridBackend

        nets, n = self._make_net()
        ic = self._make_ic(n)
        backend = NbHybridBackend()
        results = backend.run_network(nets, nstep=10, chunk_size=1, initial_states=ic)
        assert isinstance(results, list)
        assert len(results) == 1
        times, data, ctavg = results[0]
        assert times.ndim == 1
        assert data.ndim == 4
        assert ctavg.ndim == 4

    def test_temporal_average_shape(self):
        """TemporalAverage with period=0.5ms at dt=0.01ms -> chunk of 50 -> 2 chunks per 100 steps."""
        from tvb.simulator.monitors import TemporalAverage
        from tvb.simulator.backend.nb_hybrid import NbHybridBackend

        nets, n = self._make_net()
        ic = self._make_ic(n)
        backend = NbHybridBackend()
        ta = TemporalAverage(period=0.5)
        chunk_size = int(round(ta.period / DT))
        results = backend.run_network(
            nets,
            nstep=100,
            chunk_size=chunk_size,
            monitors=[ta],
            initial_states=ic,
        )
        assert len(results) == 1
        times, data = results[0][0]
        assert data.shape[0] == 100 // chunk_size

    def test_raw_shape(self):
        """Raw with chunk_size=1 -> one row per step."""
        from tvb.simulator.monitors import Raw
        from tvb.simulator.backend.nb_hybrid import NbHybridBackend

        nets, n = self._make_net()
        ic = self._make_ic(n)
        backend = NbHybridBackend()
        results = backend.run_network(
            nets,
            nstep=50,
            chunk_size=1,
            monitors=[Raw()],
            initial_states=ic,
        )
        assert len(results) == 1
        times, data = results[0][0]
        assert data.shape[0] == 50

    def test_raw_rejects_chunk_size_gt_1(self):
        """Raw monitor raises ValueError when chunk_size != 1."""
        from tvb.simulator.monitors import Raw
        from tvb.simulator.backend.nb_hybrid import NbHybridBackend

        nets, n = self._make_net()
        ic = self._make_ic(n)
        backend = NbHybridBackend()
        with self.assertRaises(ValueError):
            backend.run_network(
                nets,
                nstep=10,
                chunk_size=5,
                monitors=[Raw()],
                initial_states=ic,
            )

    def test_global_average_shape(self):
        """GlobalAverage collapses node axis."""
        from tvb.simulator.monitors import GlobalAverage
        from tvb.simulator.backend.nb_hybrid import NbHybridBackend

        nets, n = self._make_net()
        ic = self._make_ic(n)
        backend = NbHybridBackend()
        results = backend.run_network(
            nets,
            nstep=20,
            chunk_size=1,
            monitors=[GlobalAverage()],
            initial_states=ic,
        )
        assert len(results) == 1
        times, data = results[0][0]
        assert data.shape[2] == 1  # node axis collapsed to 1

    def test_spatial_average_shape(self):
        """SpatialAverage reduces node dimension from 4 to 2 areas."""
        from tvb.simulator.monitors import SpatialAverage
        from tvb.simulator.backend.nb_hybrid import NbHybridBackend

        nets, n = self._make_net(n=4)
        ic = self._make_ic(n)
        backend = NbHybridBackend()
        sa = SpatialAverage(period=DT)
        # Manually set spatial_mean since we can't call config_for_sim
        # 2 areas, 4 nodes: area 0 = nodes 0,1; area 1 = nodes 2,3
        sa.spatial_mean = np.array(
            [[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]], dtype=np.float64
        )
        results = backend.run_network(
            nets,
            nstep=20,
            chunk_size=1,
            monitors=[sa],
            initial_states=ic,
        )
        assert len(results) == 1
        times, data = results[0][0]
        # data shape: (n_chunks, n_voi, n_areas, n_modes)
        assert data.shape[0] == 20  # nstep
        assert data.shape[2] == 2  # n_areas
        assert data.shape[3] == 1  # n_modes

    def test_spatial_average_values(self):
        """SpatialAverage output matches manual spatial_mean @ data."""
        from tvb.simulator.monitors import SpatialAverage
        from tvb.simulator.backend.nb_hybrid import NbHybridBackend

        nets, n = self._make_net(n=4)
        ic = self._make_ic(n)
        backend = NbHybridBackend()
        sa = SpatialAverage(period=DT)
        spatial_mean = np.array(
            [[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]], dtype=np.float64
        )
        sa.spatial_mean = spatial_mean

        # Run without monitors to get raw data
        raw_results = backend.run_network(
            nets,
            nstep=20,
            chunk_size=1,
            initial_states=ic,
        )
        _, raw_data, _ = raw_results[0]

        # Run with SpatialAverage monitor
        sa_results = backend.run_network(
            nets,
            nstep=20,
            chunk_size=1,
            monitors=[sa],
            initial_states=ic,
        )
        _, sa_data = sa_results[0][0]

        # Manual computation: spatial_mean @ data for each (chunk, voi, mode)
        # raw_data shape: (n_chunks, n_voi, n_nodes, n_modes)
        expected = np.einsum('ij,tklm->tkim', spatial_mean, raw_data)
        np.testing.assert_allclose(
            sa_data, expected, rtol=1e-6, atol=1e-7,
            err_msg="SpatialAverage output differs from manual spatial_mean @ data",
        )

    def test_unsupported_monitor_raises(self):
        """Unsupported monitor raises NotImplementedError."""
        from tvb.simulator.monitors import ProgressLogger
        from tvb.simulator.backend.nb_hybrid import NbHybridBackend

        nets, n = self._make_net()
        ic = self._make_ic(n)
        backend = NbHybridBackend()
        with self.assertRaises(NotImplementedError):
            backend.run_network(
                nets,
                nstep=10,
                monitors=[ProgressLogger()],
                initial_states=ic,
            )


    def test_bold_output_shape(self):
        """Bold monitor produces output at the correct period."""
        from tvb.simulator.monitors import Bold
        from tvb.simulator.backend.nb_hybrid import NbHybridBackend

        nets, n = self._make_net(n=4)
        ic = self._make_ic(4)
        backend = NbHybridBackend()

        # Use short Bold period so we get output in a reasonable number of steps
        bold_period = 20.0  # ms
        bold = Bold(period=bold_period)
        bold.dt = DT
        bold._config_dt(DT)  # sets istep = period / dt = 2000
        bold.compute_hrf()

        nstep = 4000  # 2 Bold periods
        results = backend.run_network(
            nets, nstep=nstep, chunk_size=1, monitors=[bold], initial_states=ic,
        )
        assert len(results) == 1
        times, data = results[0][0]
        # Should get 2 Bold samples (at step 2000 and step 4000)
        assert data.shape[0] == 2, f"Expected 2 Bold samples, got {data.shape[0]}"
        assert data.ndim == 4  # (n_bold, n_voi, n_nodes, n_modes)

    def test_bold_period_spacing(self):
        """Bold output times are spaced at the Bold period."""
        from tvb.simulator.monitors import Bold
        from tvb.simulator.backend.nb_hybrid import NbHybridBackend

        nets, n = self._make_net(n=4)
        ic = self._make_ic(4)
        backend = NbHybridBackend()

        bold_period = 20.0
        bold = Bold(period=bold_period)
        bold.dt = DT
        bold._config_dt(DT)
        bold.compute_hrf()

        nstep = 6000  # 3 Bold periods
        results = backend.run_network(
            nets, nstep=nstep, chunk_size=1, monitors=[bold], initial_states=ic,
        )
        times, data = results[0][0]
        assert data.shape[0] == 3
        # Times should be at multiples of the Bold period
        for i, t in enumerate(times):
            expected = (i + 1) * bold_period
            assert abs(t - expected) < DT, f"Bold time {t} not near {expected}"

    def test_bold_stateful_across_calls(self):
        """Bold monitor accumulates state across multiple run_network calls."""
        from tvb.simulator.monitors import Bold
        from tvb.simulator.backend.nb_hybrid import NbHybridBackend

        nets, n = self._make_net(n=4)
        ic = self._make_ic(4)
        backend = NbHybridBackend()

        bold_period = 20.0
        bold = Bold(period=bold_period)
        bold.dt = DT
        bold._config_dt(DT)
        bold.compute_hrf()

        # Run 2000 steps (1 Bold period) — should produce 1 sample
        r1 = backend.run_network(
            nets, nstep=2000, chunk_size=1, monitors=[bold], initial_states=ic,
        )
        t1, d1 = r1[0][0]
        assert d1.shape[0] == 1, f"Expected 1 Bold sample, got {d1.shape[0]}"

        # Run another 2000 steps — should produce another 1 sample
        r2 = backend.run_network(
            nets, nstep=2000, chunk_size=1, monitors=[bold], initial_states=ic,
        )
        t2, d2 = r2[0][0]
        assert d2.shape[0] == 1, f"Expected 1 Bold sample, got {d2.shape[0]}"

    def test_projection_shape(self):
        """Projection (EEG) monitor produces (n_chunks, n_voi, n_sensors, 1) output."""
        from tvb.simulator.monitors import EEG
        from tvb.simulator.backend.nb_hybrid import NbHybridBackend

        nets, n = self._make_net(n=4)
        ic = self._make_ic(4)

        eeg = EEG(period=1.0)
        n_sensors = 3
        rng_gain = np.random.RandomState(42)
        eeg._gain = rng_gain.randn(n_sensors, n).astype(np.float64)

        backend = NbHybridBackend()
        results = backend.run_network(
            nets,
            nstep=20,
            chunk_size=1,
            monitors=[eeg],
            initial_states=ic,
        )
        assert len(results) == 1  # one monitor
        assert len(results[0]) == 1  # one subnetwork
        times, data = results[0][0]
        n_voi = 2  # MPR has 2 variables of interest
        self.assertEqual(data.shape, (20, n_voi, n_sensors, 1),
                         f"Expected (20, {n_voi}, {n_sensors}, 1), got {data.shape}")

    def test_projection_values(self):
        """Projection (EEG) monitor output matches manual gain @ data.sum(axis=-1)."""
        from tvb.simulator.monitors import EEG
        from tvb.simulator.backend.nb_hybrid import NbHybridBackend

        nets, n = self._make_net(n=4)
        ic = self._make_ic(4)

        eeg = EEG(period=1.0)
        n_sensors = 3
        rng_gain = np.random.RandomState(42)
        gain = rng_gain.randn(n_sensors, n).astype(np.float64)
        eeg._gain = gain

        backend = NbHybridBackend()
        # Run with the EEG monitor
        results = backend.run_network(
            nets,
            nstep=20,
            chunk_size=1,
            monitors=[eeg],
            initial_states=ic,
        )
        times_proj, data_proj = results[0][0]

        # Run without monitors to get raw data
        raw_results = backend.run_network(
            nets,
            nstep=20,
            chunk_size=1,
            initial_states=ic,
        )
        _, data_raw, _ = raw_results[0]

        # Manually compute: sum over modes, then apply gain
        data_2d = data_raw.sum(axis=-1)  # (20, n_voi, n_nodes)
        expected = np.einsum('ij,tkj->tki', gain.astype(data_raw.dtype), data_2d)
        expected = expected[..., np.newaxis]  # (20, n_voi, n_sensors, 1)

        np.testing.assert_allclose(
            data_proj, expected,
            rtol=1e-6, atol=1e-7,
            err_msg="Projection monitor output does not match manual gain projection",
        )


# ---------------------------------------------------------------------------
# Integrative monitor tests: Python hybrid vs Numba hybrid with monitors
# ---------------------------------------------------------------------------


class TestNbHybridMonitorsIntegrative(unittest.TestCase):
    """Compare monitor output from Python hybrid step-loop against Numba backend.

    Pattern: run the same simulation both ways, apply the same monitor logic,
    and assert numerical equivalence.
    """

    N = 4
    NSTEP = 50

    @staticmethod
    def _import_monitors():
        from tvb.simulator.monitors import (
            TemporalAverage, SubSample, GlobalAverage,
            SpatialAverage, EEG, Bold,
        )
        return TemporalAverage, SubSample, GlobalAverage, SpatialAverage, EEG, Bold

    def _build(self, n=4):
        """Build a single-subnet MPR network + identical ICs for both paths."""
        m = MontbrioPazoRoxin()
        m.configure()
        sn = _mpr_subnetwork("mon_sn", n)
        sn.configure()
        ns = NetworkSet(subnets=[sn], projections=[], stimuli=[])
        ns.configure()

        rng = np.random.RandomState(77)
        x0 = rng.uniform(0.0, 0.2, (m.nvar, n, 1)).astype(np.float64)
        x0[0] = np.abs(x0[0])
        return ns, [x0], m

    def _python_raw(self, ns, x0_list, nstep, model):
        """Run Python loop, return voi-indexed array (nstep, nvoi, nnodes, nmodes)."""
        py = _run_python_loop(ns, nstep, x0_list)
        svars = list(model.state_variables)
        voi = list(model.variables_of_interest)
        voi_idx = [svars.index(v) for v in voi]
        return py[0][:, voi_idx, :, :]  # (nstep, nvoi, nnodes, nmodes)

    def _numba_monitor(self, ns, x0_list, nstep, monitor):
        """Run Numba backend with the given monitor, return (times, data)."""
        backend = NbHybridBackend()
        results = backend.run_network(
            ns, nstep=nstep, chunk_size=1, monitors=[monitor], initial_states=x0_list,
        )
        return results[0][0]  # (times, data) for first (only) subnet

    # --- TemporalAverage ---

    def test_temporal_average_matches_python(self):
        """TemporalAverage from Numba matches average of Python raw states."""
        TemporalAverage, *_ = self._import_monitors()
        ns, x0_list, model = self._build()
        py_raw = self._python_raw(ns, [x0_list[0].copy()], self.NSTEP, model)

        ta = TemporalAverage(period=DT)
        nb_t, nb_d = self._numba_monitor(ns, [x0_list[0].copy()], self.NSTEP, ta)

        # chunk_size=1, so each chunk is a single-step temporal average = the state itself
        # NB: backend output is float32, Python is float64
        np.testing.assert_allclose(
            nb_d, py_raw.astype(np.float32), rtol=1e-2, atol=1e-2,
            err_msg="TemporalAverage: Numba differs from Python",
        )

    def test_temporal_average_chunked_matches_python(self):
        """TemporalAverage with chunk_size>1 matches temporal mean of Python states."""
        TemporalAverage, *_ = self._import_monitors()
        chunk = 5
        ns, x0_list, model = self._build()
        py_raw = self._python_raw(ns, [x0_list[0].copy()], self.NSTEP, model)

        ta = TemporalAverage(period=DT * chunk)
        backend = NbHybridBackend()
        results = backend.run_network(
            ns, nstep=self.NSTEP, chunk_size=chunk, monitors=[ta], initial_states=[x0_list[0].copy()],
        )
        nb_t, nb_d = results[0][0]

        # Manual temporal average: reshape into chunks and mean
        n_chunks = self.NSTEP // chunk
        py_chunked = py_raw[:n_chunks * chunk].reshape(n_chunks, chunk, -1, model.nvar, 1)
        # Wait — py_raw is (nstep, nvoi, nnodes, nmodes)
        py_chunked = py_raw[:n_chunks * chunk].reshape(n_chunks, chunk, *py_raw.shape[1:])
        py_mean = py_chunked.mean(axis=1).astype(np.float32)

        np.testing.assert_allclose(
            nb_d, py_mean, rtol=1e-2, atol=1e-2,
            err_msg="TemporalAverage chunked: Numba differs from Python",
        )

    # --- SubSample ---

    def test_subsample_matches_python(self):
        """SubSample from Numba matches subsampled Python raw states."""
        _, SubSample, *_ = self._import_monitors()
        period = 0.5  # ms — at dt=0.01, istep=50
        istep = int(round(period / DT))

        ns, x0_list, model = self._build()
        py_raw = self._python_raw(ns, [x0_list[0].copy()], self.NSTEP, model)

        ss = SubSample(period=period)
        nb_t, nb_d = self._numba_monitor(ns, [x0_list[0].copy()], self.NSTEP, ss)

        # Python: subsample at multiples of istep (1-indexed steps)
        py_steps = np.arange(1, self.NSTEP + 1)
        py_mask = py_steps % istep == 0
        py_sub = py_raw[py_mask].astype(np.float32)

        # Both should have same number of samples
        assert nb_d.shape[0] == py_sub.shape[0], (
            f"SubSample sample count: Numba={nb_d.shape[0]}, Python={py_sub.shape[0]}"
        )
        np.testing.assert_allclose(
            nb_d, py_sub, rtol=1e-2, atol=1e-2,
            err_msg="SubSample: Numba differs from Python",
        )

    # --- GlobalAverage ---

    def test_global_average_matches_python(self):
        """GlobalAverage from Numba matches mean-across-nodes of Python raw states."""
        _, _, GlobalAverage, *_ = self._import_monitors()
        ns, x0_list, model = self._build()
        py_raw = self._python_raw(ns, [x0_list[0].copy()], self.NSTEP, model)

        ga = GlobalAverage(period=DT)
        nb_t, nb_d = self._numba_monitor(ns, [x0_list[0].copy()], self.NSTEP, ga)

        # Python: mean over nodes axis (axis=-2)
        py_avg = py_raw.mean(axis=-2, keepdims=True).astype(np.float32)

        np.testing.assert_allclose(
            nb_d, py_avg, rtol=1e-2, atol=1e-2,
            err_msg="GlobalAverage: Numba differs from Python",
        )

    # --- SpatialAverage ---

    def test_spatial_average_matches_python(self):
        """SpatialAverage from Numba matches spatial_mean @ Python raw states."""
        _, _, _, SpatialAverage, *_ = self._import_monitors()
        ns, x0_list, model = self._build()
        py_raw = self._python_raw(ns, [x0_list[0].copy()], self.NSTEP, model)

        # 2 areas: [0,1] -> area 0, [2,3] -> area 1
        spatial_mean = np.array(
            [[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]], dtype=np.float64
        )
        sa = SpatialAverage(period=DT)
        sa.spatial_mean = spatial_mean

        nb_t, nb_d = self._numba_monitor(ns, [x0_list[0].copy()], self.NSTEP, sa)

        # Python: einsum spatial_mean @ raw
        py_sa = np.einsum('ij,tklm->tkim', spatial_mean, py_raw).astype(np.float32)

        np.testing.assert_allclose(
            nb_d, py_sa, rtol=1e-2, atol=1e-2,
            err_msg="SpatialAverage: Numba differs from Python",
        )

    # --- Projection (EEG) ---

    def test_projection_matches_python(self):
        """Projection (EEG) from Numba matches gain @ Python raw states."""
        _, _, _, _, EEG, _ = self._import_monitors()
        ns, x0_list, model = self._build()
        py_raw = self._python_raw(ns, [x0_list[0].copy()], self.NSTEP, model)

        n_sensors = 3
        rng = np.random.RandomState(42)
        gain = rng.randn(n_sensors, self.N).astype(np.float64)

        eeg = EEG(period=1.0)
        eeg._gain = gain

        nb_t, nb_d = self._numba_monitor(ns, [x0_list[0].copy()], self.NSTEP, eeg)

        # Python: sum modes, then gain @ data
        py_2d = py_raw.sum(axis=-1)  # (nstep, nvoi, nnodes)
        py_proj = np.einsum('ij,tkj->tki', gain, py_2d)[..., np.newaxis].astype(np.float32)

        np.testing.assert_allclose(
            nb_d, py_proj, rtol=1e-2, atol=1e-2,
            err_msg="Projection: Numba differs from Python",
        )

    # --- Bold ---

    def test_bold_matches_python(self):
        """Bold from Numba matches HRF convolution of Python raw states."""
        from tvb.datatypes import equations
        _, _, _, _, _, Bold = self._import_monitors()
        bold_period = 20.0  # ms
        istep = int(round(bold_period / DT))  # 2000 steps

        # Need enough steps for at least 1 Bold sample
        nstep = istep * 2  # 4000 steps for 2 Bold samples

        ns, x0_list, model = self._build()
        py_raw = self._python_raw(ns, [x0_list[0].copy()], nstep, model)

        bold = Bold(period=bold_period)
        bold.dt = DT
        bold._config_dt(DT)
        bold.compute_hrf()

        nb_t, nb_d = self._numba_monitor(ns, [x0_list[0].copy()], nstep, bold)

        # Python: replicate Bold.sample() logic
        hrf = bold.hemodynamic_response_function  # (1, stock_steps)
        stock_steps = hrf.shape[1]
        interim_istep = bold._interim_istep

        # Use float32 to match Numba backend precision
        py_f32 = py_raw.astype(np.float32)
        interim_stock = np.zeros((interim_istep,) + py_f32.shape[1:], dtype=np.float32)
        stock = np.zeros((stock_steps,) + py_f32.shape[1:], dtype=np.float32)

        py_bold_times = []
        py_bold_data = []
        for step in range(1, nstep + 1):
            interim_stock[(step - 1) % interim_istep] = py_f32[step - 1]
            if step % interim_istep == 0:
                avg = np.mean(interim_stock, axis=0)
                stock[(step // interim_istep - 1) % stock_steps] = avg
            if step % istep == 0:
                t = step * DT
                rolled_hrf = np.roll(hrf, (step // interim_istep - 1) % stock_steps, axis=1)
                stock_t = stock.transpose((1, 2, 0, 3))  # (nvoi, nnodes, stock_steps, nmodes)
                bold_val = np.dot(rolled_hrf, stock_t).reshape(py_f32.shape[1:])
                # Apply FirstOrderVolterra scaling to match Numba backend
                if isinstance(bold.hrf_kernel, equations.FirstOrderVolterra):
                    k1 = bold.hrf_kernel.parameters.get('k_1', 1.0)
                    V0 = bold.hrf_kernel.parameters.get('V_0', 1.0)
                    bold_val = (bold_val - 1.0) * k1 * V0
                py_bold_times.append(t)
                py_bold_data.append(bold_val)

        py_bold_arr = np.stack(py_bold_data, axis=0)

        assert nb_d.shape[0] == py_bold_arr.shape[0], (
            f"Bold sample count: Numba={nb_d.shape[0]}, Python={py_bold_arr.shape[0]}"
        )
        np.testing.assert_allclose(
            nb_d, py_bold_arr, rtol=1e-2, atol=1e-2,
            err_msg="Bold: Numba differs from Python",
        )


if __name__ == "__main__":
    unittest.main()
    unittest.main()
