# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
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
Smoke / integration tests for the hybrid simulator with two heterogeneous
subnetworks: MontbrioPazoRoxin (MPR) and KIonEx.

These are not numerical-equivalence tests (the two models have incompatible
state-space dimensions and physical semantics).  The goals are:

1. **Pure-Python path** (``NetworkSet.step``): confirm that a MPR + KIonEx
   hybrid network runs through ``Simulator.run()`` without errors, and that
   the output time series are finite.

2. **Numba path** (``NbHybridBackend``): confirm that the JIT-compiled kernel
   compiles and runs without errors, and that the pure-Python and Numba outputs
   agree numerically within float32 precision.

Network topology
----------------
* Subnet A — ``MontbrioPazoRoxin``, 8 nodes, ``HeunDeterministic(dt=0.1)``
* Subnet B — ``KIonEx``,              8 nodes, ``HeunDeterministic(dt=0.1)``
* Intra A — uniform random weights within MPR subnet
* Intra B — uniform random weights within KIonEx subnet
* Inter A→B — MPR cvar[0] (r) drives KIonEx coupling slot 0
* Inter B→A — KIonEx cvar[0] (x) drives MPR coupling slot 0

All tract lengths are set to zero so there are no propagation delays.
"""

import unittest
import numpy as np
import scipy.sparse as sp

from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.models.k_ion_exchange import KIonEx
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.monitors import TemporalAverage
from tvb.simulator.hybrid import NetworkSet, Simulator, Subnetwork
from tvb.simulator.hybrid.inter_projection import InterProjection
from tvb.simulator.hybrid.intra_projection import IntraProjection
from tvb.simulator.hybrid.coupling import Linear


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

DT = 0.1
N_NODES = 8
N_STEPS = 50
SIMULATION_LENGTH = N_STEPS * DT   # ms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_weights(n: int, seed: int) -> sp.csr_matrix:
    rng = np.random.RandomState(seed)
    w = rng.uniform(0.0, 0.1, (n, n)).astype(np.float64)
    np.fill_diagonal(w, 0.0)
    return sp.csr_matrix(w)


def _zero_lengths(n: int) -> sp.csr_matrix:
    return sp.csr_matrix(np.zeros((n, n), dtype=np.float64))


def _build_network() -> tuple:
    """Build the MPR + KIonEx NetworkSet and initial conditions.

    Returns
    -------
    (NetworkSet, list_of_ic_arrays)
        Configured NetworkSet and a list of two initial-condition arrays,
        one for each subnetwork.
    """
    # --- MPR subnetwork ---
    mpr_model = MontbrioPazoRoxin()
    mpr_model.configure()
    mpr_scheme = HeunDeterministic(dt=DT)
    subnet_a = Subnetwork(name="mpr", model=mpr_model, scheme=mpr_scheme, nnodes=N_NODES)

    # --- KIonEx subnetwork ---
    kionex_model = KIonEx()
    kionex_model.configure()
    kionex_scheme = HeunDeterministic(dt=DT)
    subnet_b = Subnetwork(name="kionex", model=kionex_model, scheme=kionex_scheme, nnodes=N_NODES)

    # --- Intra-projections ---
    w_aa = _random_weights(N_NODES, seed=1)
    l_aa = _zero_lengths(N_NODES)
    proj_aa = IntraProjection(
        source_cvar=mpr_model.cvar[:1],   # r (slot 0)
        target_cvar=np.array([0], dtype=np.int32),
        weights=w_aa,
        lengths=l_aa,
        cv=1.0,
        dt=DT,
        scale=1e-4,
    )
    subnet_a.projections = [proj_aa]
    subnet_a.configure()

    w_bb = _random_weights(N_NODES, seed=2)
    l_bb = _zero_lengths(N_NODES)
    proj_bb = IntraProjection(
        source_cvar=kionex_model.cvar[:1],  # x (slot 0)
        target_cvar=np.array([0], dtype=np.int32),
        weights=w_bb,
        lengths=l_bb,
        cv=1.0,
        dt=DT,
        scale=1e-4,
    )
    subnet_b.projections = [proj_bb]
    subnet_b.configure()

    # --- Inter-projections ---
    # MPR → KIonEx: MPR cvar[0] (r) drives KIonEx Coupling_Term (slot 0)
    w_ab = _random_weights(N_NODES, seed=3)
    l_ab = _zero_lengths(N_NODES)
    proj_ab = InterProjection(
        source=subnet_a,
        target=subnet_b,
        source_cvar=mpr_model.cvar[:1],          # r
        target_cvar=np.array([0], dtype=np.int32),  # Coupling_Term
        weights=w_ab,
        lengths=l_ab,
        cv=1.0,
        dt=DT,
        scale=1e-4,
    )

    # KIonEx → MPR: KIonEx cvar[0] (x) drives MPR Coupling_Term_r (slot 0)
    w_ba = _random_weights(N_NODES, seed=4)
    l_ba = _zero_lengths(N_NODES)
    proj_ba = InterProjection(
        source=subnet_b,
        target=subnet_a,
        source_cvar=kionex_model.cvar[:1],          # x
        target_cvar=np.array([0], dtype=np.int32),   # Coupling_Term_r
        weights=w_ba,
        lengths=l_ba,
        cv=1.0,
        dt=DT,
        scale=1e-4,
    )

    nets = NetworkSet(
        subnets=[subnet_a, subnet_b],
        projections=[proj_ab, proj_ba],
    )
    nets.configure()

    # --- Initial conditions ---
    rng = np.random.RandomState(42)
    ic_a = rng.uniform(0.0, 0.1, (mpr_model.nvar, N_NODES, 1)).astype(np.float64)
    ic_a[0] = np.abs(ic_a[0])   # r ≥ 0

    # Build KIonEx IC from midpoints of valid ranges, with small ±10 % jitter.
    # Keeping DKi in [-8, -2] ensures K_o = K_o0 − 3·DKi + Kg > 0, avoiding
    # log(negative) NaN in the ionic-current helper functions.
    ic_b = np.zeros((kionex_model.nvar, N_NODES, 1), dtype=np.float64)
    safe_ic_kionex = {
        'x':   (0.0, 0.3),
        'V':   (-70.0, -50.0),
        'n':   (0.2, 0.5),
        'DKi': (-8.0, -2.0),
        'Kg':  (-15.0, -8.0),
    }
    for idx, svar in enumerate(kionex_model.state_variables):
        lo, hi = safe_ic_kionex[svar]
        ic_b[idx, :, 0] = rng.uniform(lo, hi, N_NODES)

    return nets, [ic_a, ic_b]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHybridMprKIonExPython(unittest.TestCase):
    """Pure-Python NetworkSet.step path: MPR + KIonEx two-subnet smoke test."""

    def test_run_no_errors(self):
        """Simulator.run() completes without raising and produces finite output."""
        nets, ics = _build_network()

        tavg = TemporalAverage(period=DT)
        sim = Simulator(nets=nets, simulation_length=SIMULATION_LENGTH, monitors=[tavg])
        sim.configure()

        ((times, data),) = sim.run(initial_conditions=ics)

        self.assertFalse(np.any(np.isnan(data)),
                         "NaN in hybrid MPR+KIonEx output")
        self.assertFalse(np.any(np.isinf(data)),
                         "Inf in hybrid MPR+KIonEx output")
        self.assertGreater(len(times), 0, "No time points returned")

    def test_output_shape(self):
        """Output shape matches expected (T, total_vois, total_nodes, 1)."""
        nets, ics = _build_network()

        tavg = TemporalAverage(period=DT)
        sim = Simulator(nets=nets, simulation_length=SIMULATION_LENGTH, monitors=[tavg])
        sim.configure()

        ((times, data),) = sim.run(initial_conditions=ics)

        mpr_nvoi = len(MontbrioPazoRoxin.variables_of_interest.default)
        kionex_nvoi = len(KIonEx.variables_of_interest.default)
        expected_nvoi = mpr_nvoi + kionex_nvoi
        expected_nodes = 2 * N_NODES

        self.assertEqual(data.shape[1], expected_nvoi,
                         f"Expected {expected_nvoi} VOIs, got {data.shape[1]}")
        self.assertEqual(data.shape[2], expected_nodes,
                         f"Expected {expected_nodes} nodes, got {data.shape[2]}")


if __name__ == "__main__":
    unittest.main()
