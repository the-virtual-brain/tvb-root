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
Validate that the hybrid simulator reproduces classic TVB output for single-model scenarios.

For each tested model the same initial conditions and integration parameters
are applied to both the classic :class:`~tvb.simulator.simulator.Simulator`
and the hybrid :class:`~tvb.simulator.hybrid.Simulator`.  Outputs are
compared element-wise (``rtol=1e-4, atol=1e-5``) and via summary statistics.

**Why tract lengths are zeroed out**: the classic simulator initialises its
delay buffer to zeros, and the hybrid simulator's history buffer is likewise
zero-initialised.  Any non-zero delay would cause a transient divergence in
the early simulation steps because the two implementations step through their
respective buffers with different indexing strategies.  Setting all tract
lengths to zero eliminates propagation delays entirely so that both
simulators read the same ``x(t)`` value on every step, making the comparison
valid from the very first integration step.
"""

import numpy as np
from scipy import sparse as sp
from tvb.tests.library.simulator.hybrid.test_validation_base import ValidationTestBase
from tvb.simulator import simulator as classic_simulator
from tvb.simulator import coupling, integrators, monitors
from tvb.simulator.models import Generic2dOscillator, JansenRit, ReducedWongWang
from tvb.simulator.hybrid import Subnetwork, NetworkSet, Simulator, projection_utils
from tvb.datatypes.connectivity import Connectivity


class TestValidationSingleModel(ValidationTestBase):
    """Cross-validate the hybrid simulator against classic TVB for single-model networks.

    Each test follows the same pattern:

    1. Build and run the classic simulator with zeroed tract lengths and a
       fixed random seed (``np.random.seed(42)``).
    2. Capture the initial state produced by the classic simulator's
       ``configure()`` call.
    3. Build and run the hybrid simulator with identical parameters and the
       same initial state.
    4. Assert time arrays and all state-variable trajectories match to within
       ``rtol=1e-4, atol=1e-5``.
    """

    def test_generic_2d_oscillator(self):
        """Generic2dOscillator: hybrid and classic trajectories must match with zero delays.

        Uses a linear intra-projection on state variable ``V`` with coupling
        strength ``a=0.004``.  Tract lengths are zeroed to remove delay
        discrepancies.  Simulation length: 100 ms at dt=0.1 ms.
        """
        dt = 0.1
        simulation_length = 100.0
        coupling_strength = 0.004

        conn_classic = Connectivity.from_file()
        conn_classic.configure()

        # Zero out tract lengths to eliminate delay-related differences
        conn_classic.tract_lengths[:] = 0.0
        conn_classic.configure()
        np.random.seed(42)

        model_classic = Generic2dOscillator()
        coupling_classic = coupling.Linear(a=np.array([coupling_strength]))
        integrator_classic = integrators.HeunDeterministic(dt=dt)
        monitor_classic = monitors.TemporalAverage(period=1.0)

        sim_classic = classic_simulator.Simulator()
        sim_classic.connectivity = conn_classic
        sim_classic.model = model_classic
        sim_classic.coupling = coupling_classic
        sim_classic.integrator = integrator_classic
        sim_classic.monitors = (monitor_classic,)
        sim_classic.configure()

        # Capture initial state before running
        init_state = sim_classic.current_state.copy()

        ((t_classic, y_classic),) = sim_classic.run(simulation_length=simulation_length)

        scheme = integrators.HeunDeterministic(dt=dt)
        subnet = Subnetwork(
            name="all_regions",
            model=Generic2dOscillator(),
            scheme=scheme,
            nnodes=conn_classic.number_of_regions,
        )

        weights_sp = sp.csr_matrix(conn_classic.weights)

        proj = projection_utils.create_intra_projection(
            subnet=subnet,
            source_cvar="V",
            target_cvar="V",
            weights=weights_sp,
            scale=coupling_strength,
            dt=scheme.dt,
        )

        subnet.projections = [proj]
        subnet.configure()

        nets = NetworkSet(subnets=[subnet])
        monitor_hybrid = monitors.TemporalAverage(period=1.0)
        sim_hybrid = Simulator(
            nets=nets, simulation_length=simulation_length, monitors=[monitor_hybrid]
        )
        sim_hybrid.configure()

        # Use same initial conditions as classic simulator
        ((t_hybrid, y_hybrid),) = sim_hybrid.run(initial_conditions=[init_state])

        assert t_classic.shape == t_hybrid.shape
        assert y_classic.shape == y_hybrid.shape

        np.testing.assert_allclose(t_classic, t_hybrid, rtol=1e-6, atol=1e-8)

        for i in range(y_classic.shape[1]):
            np.testing.assert_allclose(
                y_classic[:, i, :, :],
                y_hybrid[:, i, :, :],
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"Mismatch in state variable {i}",
            )

        self.assert_statistics_equivalent(y_classic, y_hybrid, places=4)

    def test_reduced_wong_wang(self):
        """ReducedWongWang: hybrid and classic trajectories must match with zero delays.

        Uses a linear intra-projection on the synaptic gating variable ``S``
        with coupling strength ``a=0.004``.  Tract lengths are zeroed.
        Simulation length: 100 ms at dt=0.1 ms.
        """
        dt = 0.1
        simulation_length = 100.0
        coupling_strength = 0.004

        conn_classic = Connectivity.from_file()
        conn_classic.configure()

        # Zero out tract lengths to eliminate delay-related differences
        conn_classic.tract_lengths[:] = 0.0
        conn_classic.configure()
        np.random.seed(42)

        model_classic = ReducedWongWang()
        coupling_classic = coupling.Linear(a=np.array([coupling_strength]))
        integrator_classic = integrators.HeunDeterministic(dt=dt)
        monitor_classic = monitors.TemporalAverage(period=1.0)

        sim_classic = classic_simulator.Simulator()
        sim_classic.connectivity = conn_classic
        sim_classic.model = model_classic
        sim_classic.coupling = coupling_classic
        sim_classic.integrator = integrator_classic
        sim_classic.monitors = (monitor_classic,)
        sim_classic.configure()

        # Capture initial state before running
        init_state = sim_classic.current_state.copy()

        ((t_classic, y_classic),) = sim_classic.run(simulation_length=simulation_length)

        scheme = integrators.HeunDeterministic(dt=dt)
        subnet = Subnetwork(
            name="all_regions",
            model=ReducedWongWang(),
            scheme=scheme,
            nnodes=conn_classic.number_of_regions,
        )

        weights_sp = sp.csr_matrix(conn_classic.weights)

        proj = projection_utils.create_intra_projection(
            subnet=subnet,
            source_cvar="S",
            target_cvar="S",
            weights=weights_sp,
            scale=coupling_strength,
            dt=scheme.dt,
        )

        subnet.projections = [proj]
        subnet.configure()

        nets = NetworkSet(subnets=[subnet])
        monitor_hybrid = monitors.TemporalAverage(period=1.0)
        sim_hybrid = Simulator(
            nets=nets, simulation_length=simulation_length, monitors=[monitor_hybrid]
        )
        sim_hybrid.configure()

        # Use same initial conditions as classic simulator
        ((t_hybrid, y_hybrid),) = sim_hybrid.run(initial_conditions=[init_state])

        assert t_classic.shape == t_hybrid.shape
        assert y_classic.shape == y_hybrid.shape

        np.testing.assert_allclose(t_classic, t_hybrid, rtol=1e-6, atol=1e-8)

        for i in range(y_classic.shape[1]):
            np.testing.assert_allclose(
                y_classic[:, i, :, :],
                y_hybrid[:, i, :, :],
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"Mismatch in state variable {i}",
            )

        self.assert_statistics_equivalent(y_classic, y_hybrid, places=4)
