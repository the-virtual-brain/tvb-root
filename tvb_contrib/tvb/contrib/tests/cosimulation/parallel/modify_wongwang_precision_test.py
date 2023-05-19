# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

"""
.. moduleauthor:: Lionel Kusch <lkusch@thevirtualbrain.org>
.. moduleauthor:: Dionysios Perdikis <dionperd@gmail.com>
"""

import numpy as np

import tvb.simulator.lab as lab
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.contrib.tests.cosimulation.synchronization_time_set import SYNCHRONIZATION_TIME, adjust_connectivity_delays
from tvb.contrib.tests.cosimulation.parallel.ReducedWongWang import ReducedWongWangProxy
from tvb.contrib.cosimulation.cosim_monitors import RawCosim
from tvb.contrib.cosimulation.cosimulator import CoSimulator


class TestModifyWongWang(BaseTestCase):
    """
     test for compare the version in tvb and the modified version in different condition
    """

    _simulation_length = 3.0

    @staticmethod
    def _reference_simulation(simulation_length, model_class=lab.models.ReducedWongWang, simulator=lab.simulator.Simulator, init=None):
        # reference simulation
        np.random.seed(42)
        if init is None:
            init = np.random.random_sample((385, 1, 76, 1))
        np.random.seed(42)
        model = model_class(tau_s=np.random.rand(76))
        connectivity = lab.connectivity.Connectivity().from_file()
        connectivity.speed = np.array([4.0])
        connectivity = adjust_connectivity_delays(connectivity)
        coupling = lab.coupling.Linear(a=np.array(0.0154))
        integrator = lab.integrators.HeunDeterministic(dt=0.1, bounded_state_variable_indices=np.array([0]),
                                                       state_variable_boundaries=np.array([[0.0, 1.0]]))
        monitors = lab.monitors.Raw(period=0.1, variables_of_interest=np.array(0, dtype=np.int_))
        # Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
        sim = simulator(model=model,
                        connectivity=connectivity,
                        coupling=coupling,
                        integrator=integrator,
                        monitors=(monitors,),
                        initial_conditions=init,
                        simulation_length=simulation_length
                        )
        sim.configure()
        result_all = sim.run()
        result = result_all[0][1][0][0]
        return connectivity, coupling, integrator, monitors, sim, result, result_all

    def test_with_no_cosimulation(self):
        connectivity, coupling, integrator, monitors, sim, result, result_all = \
            self._reference_simulation(self._simulation_length)
        np.random.seed(42)
        init = np.concatenate((np.random.random_sample((385, 1, 76, 1)),
                               np.random.random_sample((385, 1, 76, 1))), axis=1)
        np.random.seed(42)
        result_2 = self._reference_simulation(self._simulation_length,model_class=ReducedWongWangProxy,
                                              simulator=CoSimulator, init=init)[5]
        np.testing.assert_array_equal(result, result_2)

    def test_precision_with_proxy(self):
        connectivity, coupling, integrator, monitors, sim, result, result_all = \
            self._reference_simulation(self._simulation_length)
        # New simulator with proxy
        np.random.seed(42)
        init = np.concatenate((np.random.random_sample((385, 1, 76, 1)),
                               np.random.random_sample((385, 1, 76, 1))), axis=1)
        np.random.seed(42)
        model_1 = ReducedWongWangProxy(tau_s=np.random.rand(76))
        synchronization_time = SYNCHRONIZATION_TIME
        # Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
        sim_1 = CoSimulator(
                            voi=np.array([0]),
                            synchronization_time=synchronization_time,
                            cosim_monitors=(RawCosim(),),
                            proxy_inds=np.asarray([0], dtype=np.int_),
                            model=model_1,
                            connectivity=connectivity,
                            coupling=coupling,
                            integrator=integrator,
                            monitors=(monitors,),
                            initial_conditions=init,
        )
        sim_1.configure()

        sim_to_sync_time = int(self._simulation_length / synchronization_time)
        sync_steps = int(synchronization_time / integrator.dt)

        result_1_all = [np.empty((0,)), np.empty((sync_steps, 2, 76, 1))]
        sim_1.run() # run the first steps because the history is delayed

        for j in range(0, sim_to_sync_time):
            result_1_all_step = sim_1.run(
                cosim_updates=[np.array([result_all[0][0][(sync_steps * j) + i] for i in range(sync_steps)]),
                            np.array([result_all[0][1][(sync_steps * j) + i][0][0]
                                      for i in range(sync_steps)]).reshape((sync_steps, 1, 1, 1))])
            result_1_all[0] = np.concatenate((result_1_all[0], result_1_all_step[0][0]))
            result_1_all[1] = np.concatenate((result_1_all[1], result_1_all_step[0][1]))

        for i in range(int(self._simulation_length/integrator.dt)):
            np.testing.assert_array_equal(result_all[0][1][i][0][1:],result_1_all[1][i+sync_steps, 0, 1:])
            np.testing.assert_array_equal(result_all[0][1][i][0][:1], result_1_all[1][i+sync_steps, 0, :1])