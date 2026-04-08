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
Validate that the hybrid simulator reproduces classic TVB output when the same
model is partitioned across two subnetworks with different per-subnet parameter
values.

**Setup**: the 76-region default TVB connectivity is split into two contiguous
groups (nodes ``0:38`` → subnet A, nodes ``38:76`` → subnet B).  Each subnet
uses the same model class but with distinct scalar parameter values, matching
the per-region parameter array that the classic simulator assembles from those
same values.  All tract lengths are zeroed to eliminate delay discrepancies
(see :mod:`~tvb.tests.library.simulator.hybrid.test_validation_single_model`
for the rationale).

**Coupling algebra**: because the global weight matrix block-decomposes as
``W = [[W_AA, W_AB], [W_BA, W_BB]]``, the total coupling at every node is
identical between classic and hybrid when each block is assigned to the
corresponding intra- or inter-projection.  With ``node_indices`` set on both
subnetworks the hybrid simulator's global monitor emits output in original
connectome order, making direct element-wise comparison to the classic output
valid.

**Comparison**: time arrays and every state-variable trajectory are checked
with ``rtol=1e-4, atol=1e-5`` via
:meth:`~tvb.tests.library.simulator.hybrid.test_validation_base.ValidationTestBase.assert_simulation_equivalent`.
"""

import numpy as np
from scipy import sparse as sp

from tvb.datatypes.connectivity import Connectivity
from tvb.simulator import simulator as classic_simulator
from tvb.simulator import coupling, integrators, monitors
from tvb.simulator.models import Generic2dOscillator, JansenRit
from tvb.simulator.hybrid import (
    NetworkSet,
    Simulator,
    Subnetwork,
    projection_utils,
)

from tvb.tests.library.simulator.hybrid.test_validation_base import ValidationTestBase


class TestValidationTwoSubnetworks(ValidationTestBase):
    """Cross-validate hybrid two-subnet simulations against classic TVB.

    Both test cases follow the same pattern:

    1. Build and run the classic simulator with zeroed tract lengths, a
       per-region parameter array, and a fixed random seed.
    2. Capture the initial state produced by ``configure()``.
    3. Partition nodes into two groups (``0:38`` / ``38:76``), build two
       ``Subnetwork`` instances each with the matching scalar parameter value
       and ``node_indices`` set, wire them with intra- and inter-projections
       covering all four weight-matrix blocks, and run the hybrid simulator
       with the same initial state.
    4. Assert time arrays and all state-variable trajectories match.
    """

    # Shared simulation parameters
    _DT = 0.1
    _LENGTH = 100.0
    _COUPLING_STRENGTH = 0.004
    _N = 76        # total nodes in default connectivity
    _NA = 38       # nodes in subnet A (indices 0..37)

    def _build_classic_sim(self, model):
        """Configure and run a classic simulator, return (t, y, init_state)."""
        conn = Connectivity.from_file()
        conn.configure()
        conn.tract_lengths[:] = 0.0
        conn.configure()

        np.random.seed(42)

        sim = classic_simulator.Simulator(
            connectivity=conn,
            model=model,
            coupling=coupling.Linear(a=np.array([self._COUPLING_STRENGTH])),
            integrator=integrators.HeunDeterministic(dt=self._DT),
            monitors=(monitors.TemporalAverage(period=1.0),),
        )
        sim.configure()
        init_state = sim.current_state.copy()
        ((t, y),) = sim.run(simulation_length=self._LENGTH)
        return t, y, init_state, conn

    def _build_hybrid_sim(self, model_a, model_b, ia, ib, conn, init_state):
        """Build and run a hybrid simulator, return (t, y)."""
        subnet_a = Subnetwork(
            name="subnet_a",
            model=model_a,
            scheme=integrators.HeunDeterministic(dt=self._DT),
            nnodes=len(ia),
            node_indices=ia,
        ).configure()

        subnet_b = Subnetwork(
            name="subnet_b",
            model=model_b,
            scheme=integrators.HeunDeterministic(dt=self._DT),
            nnodes=len(ib),
            node_indices=ib,
        ).configure()

        # source_cvar: state-variable indices used to *read* from the history buffer.
        # target_cvar: 0-based coupling-slot indices used to *write* into the
        #              coupling array (shape = (len(cvar), nnodes, modes)).
        # For models where cvar starts at 0 (e.g. G2d: cvar=[0]) the two are
        # identical.  For JansenRit (cvar=[1,2]) source=[1,2] but target=[0,1].
        cvar_src_a = model_a.cvar
        cvar_tgt_a = np.arange(len(model_a.cvar))
        cvar_src_b = model_b.cvar
        cvar_tgt_b = np.arange(len(model_b.cvar))

        # Intra-subnet projections (diagonal blocks: W_AA, W_BB)
        sub_aa = projection_utils.extract_connectivity_subset(conn, ia, ia)
        proj_aa = projection_utils.create_intra_projection(
            subnet=subnet_a,
            source_cvar=cvar_src_a,
            target_cvar=cvar_tgt_a,
            weights=sub_aa["weights"],
            lengths=sub_aa["lengths"],
            scale=self._COUPLING_STRENGTH,
            dt=self._DT,
        )
        subnet_a.projections = [proj_aa]
        subnet_a.configure()

        sub_bb = projection_utils.extract_connectivity_subset(conn, ib, ib)
        proj_bb = projection_utils.create_intra_projection(
            subnet=subnet_b,
            source_cvar=cvar_src_b,
            target_cvar=cvar_tgt_b,
            weights=sub_bb["weights"],
            lengths=sub_bb["lengths"],
            scale=self._COUPLING_STRENGTH,
            dt=self._DT,
        )
        subnet_b.projections = [proj_bb]
        subnet_b.configure()

        # Inter-subnet projections (off-diagonal blocks: W_AB, W_BA)
        sub_ab = projection_utils.extract_connectivity_subset(conn, ia, ib)
        proj_ab = projection_utils.create_inter_projection(
            source_subnet=subnet_a,
            target_subnet=subnet_b,
            source_cvar=cvar_src_a,
            target_cvar=cvar_tgt_b,
            weights=sub_ab["weights"],
            lengths=sub_ab["lengths"],
            scale=self._COUPLING_STRENGTH,
            dt=self._DT,
        )

        sub_ba = projection_utils.extract_connectivity_subset(conn, ib, ia)
        proj_ba = projection_utils.create_inter_projection(
            source_subnet=subnet_b,
            target_subnet=subnet_a,
            source_cvar=cvar_src_b,
            target_cvar=cvar_tgt_a,
            weights=sub_ba["weights"],
            lengths=sub_ba["lengths"],
            scale=self._COUPLING_STRENGTH,
            dt=self._DT,
        )

        nets = NetworkSet(
            subnets=[subnet_a, subnet_b],
            projections=[proj_ab, proj_ba],
        )

        tavg = monitors.TemporalAverage(period=1.0)
        sim = Simulator(nets=nets, simulation_length=self._LENGTH, monitors=[tavg])
        sim.configure()

        # Split init_state (shape: nvar, 76, modes) along the node axis
        ic_a = init_state[:, ia, :]
        ic_b = init_state[:, ib, :]
        ((t, y),) = sim.run(initial_conditions=[ic_a, ic_b])
        return t, y

    def test_two_subnetwork_generic2d(self):
        """Generic2dOscillator: two subnets with different ``tau`` values match classic.

        Subnet A uses ``tau=1.0`` (default), subnet B uses ``tau=2.0``.  The
        classic simulator receives a 76-element ``tau`` array with those values
        assigned to nodes ``0:38`` and ``38:76`` respectively.  With zero
        delays and the 38+38 contiguous partition the coupling algebra is
        algebraically identical across classic and hybrid.
        """
        ia = np.arange(self._NA)
        ib = np.arange(self._NA, self._N)

        tau_a, tau_b = 1.0, 2.0
        model_classic = Generic2dOscillator(
            tau=np.r_[np.full(self._NA, tau_a), np.full(self._NA, tau_b)]
        )

        t_classic, y_classic, init_state, conn = self._build_classic_sim(model_classic)

        n_vois = len(Generic2dOscillator.variables_of_interest.default)
        # Confirm classic output shape
        self.assert_equal((int(self._LENGTH), n_vois, self._N, 1), y_classic.shape)

        model_a = Generic2dOscillator(tau=np.array([tau_a]))
        model_b = Generic2dOscillator(tau=np.array([tau_b]))

        t_hybrid, y_hybrid = self._build_hybrid_sim(model_a, model_b, ia, ib, conn, init_state)

        # Merged mode: output shape matches classic exactly
        self.assert_equal(y_classic.shape, y_hybrid.shape)

        np.testing.assert_allclose(t_classic, t_hybrid, rtol=1e-6, atol=1e-8)
        self.assert_simulation_equivalent(y_classic, y_hybrid)

    def test_two_subnetwork_jansen_rit(self):
        """JansenRit: two subnets with different ``A`` values match classic.

        Subnet A uses ``A=3.25`` (default), subnet B uses ``A=4.0``.  The
        classic simulator receives a 76-element ``A`` array.  The coupling
        variables are the full ``JansenRit.cvar`` (``[y1, y2]``) for all four
        projections.

        JansenRit has 4 variables of interest (``y0``–``y3``); with both
        subnets sharing the same VOI count the merged-mode output has shape
        ``(T, 4, 76, 1)``, matching classic.
        """
        ia = np.arange(self._NA)
        ib = np.arange(self._NA, self._N)

        A_a, A_b = 3.25, 4.0
        model_classic = JansenRit(
            A=np.r_[np.full(self._NA, A_a), np.full(self._NA, A_b)]
        )

        t_classic, y_classic, init_state, conn = self._build_classic_sim(model_classic)

        n_vois = len(JansenRit.variables_of_interest.default)
        self.assert_equal((int(self._LENGTH), n_vois, self._N, 1), y_classic.shape)

        model_a = JansenRit(A=np.array([A_a]))
        model_b = JansenRit(A=np.array([A_b]))

        t_hybrid, y_hybrid = self._build_hybrid_sim(model_a, model_b, ia, ib, conn, init_state)

        self.assert_equal(y_classic.shape, y_hybrid.shape)

        np.testing.assert_allclose(t_classic, t_hybrid, rtol=1e-6, atol=1e-8)
        self.assert_simulation_equivalent(y_classic, y_hybrid)
