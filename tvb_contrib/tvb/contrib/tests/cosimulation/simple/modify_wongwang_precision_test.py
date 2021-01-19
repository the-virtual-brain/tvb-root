# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)

"""
.. moduleauthor:: Lionel Kusch <lkusch@thevirtualbrain.org>
.. moduleauthor:: Dionysios Perdikis <dionperd@gmail.com>
"""

import numpy as np

import tvb.simulator.lab as lab
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.contrib.tests.cosimulation.parallel.ReducedWongWang import ReducedWongWangProxy

from tvb.contrib.cosimulation.cosim_monitors import RawCosim
from tvb.contrib.cosimulation.cosimulator import CoSimulator


class TestModifyWongWang(BaseTestCase):
    """
    Initialisation of the test for the reference simulation
    """
    @staticmethod
    def _reference_simulation():
        # reference simulation
        np.random.seed(42)
        init = np.concatenate((np.random.random_sample((385, 1, 76, 1)),
                               np.random.random_sample((385, 1, 76, 1))), axis=1)
        np.random.seed(42)
        model = ReducedWongWangProxy(tau_s=np.random.rand(76))
        connectivity = lab.connectivity.Connectivity().from_file()
        connectivity.speed = np.array([4.0])
        connectivity.configure()
        coupling = lab.coupling.Linear(a=np.array(0.0154))
        integrator = lab.integrators.HeunDeterministic(dt=0.1, bounded_state_variable_indices=np.array([0]),
                                                       state_variable_boundaries=np.array([[0.0, 1.0]]))
        monitors = lab.monitors.Raw(period=0.1, variables_of_interest=np.array(0, dtype=np.int))
        # Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
        sim = lab.simulator.Simulator(model=model,
                                      connectivity=connectivity,
                                      coupling=coupling,
                                      integrator=integrator,
                                      monitors=(monitors,),
                                      initial_conditions=init,
                                      )
        sim.configure()
        result_all = sim.run(simulation_length=10.0)
        result = result_all[0][1][:,:,0,0]
        return connectivity, coupling, integrator, monitors, sim, result, result_all

class TestModifyWongWangRate(TestModifyWongWang):
    """
    Test to compare the version in TVB and the modified version
    """
    def test_without_proxy(self):
        connectivity, coupling, integrator, monitors, sim, result, result_all = self._reference_simulation()
        # The modify model without proxy
        np.random.seed(42)
        init = np.concatenate((np.random.random_sample((385, 1, 76, 1)),
                               np.random.random_sample((385, 1, 76, 1))), axis=1)
        np.random.seed(42)
        model = ReducedWongWangProxy(tau_s=np.random.rand(76))
        # Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
        sim_2 = CoSimulator(
            voi=np.array([0]),
            synchronization_time=1.0,
            cosim_monitors=(RawCosim(),),
            proxy_inds=np.asarray([], dtype=np.int),
            model=model,
            connectivity=connectivity,
            coupling=coupling,
            integrator=integrator,
            monitors=(monitors,),
            initial_conditions=init,
        )
        sim_2.configure()
        result_2_init = sim_2.run()[0][1][:,:,0,0] # run the first steps because the history is delayed
        for i in range(10):
            result_2 = sim_2.run()[0][1][:,:,0,0]
            diff = result[i*10:(i+1)*10] - result_2
            assert np.sum(diff) == 0.0

    def test_with_proxy(self):
        connectivity, coupling, integrator, monitors, sim, result, result_all = self._reference_simulation()
        # The modify model without proxy
        np.random.seed(42)
        init = np.concatenate((np.random.random_sample((385, 1, 76, 1)),
                               np.random.random_sample((385, 1, 76, 1))), axis=1)
        np.random.seed(42)
        id_proxy = range(11)
        model = ReducedWongWangProxy(tau_s=np.random.rand(76))
        # Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
        sim_3 = CoSimulator(
            voi=np.array([0]),
            synchronization_time=1.,
            cosim_monitors=(RawCosim(),),
            proxy_inds=np.asarray(id_proxy, dtype=np.int),
            model=model,
            connectivity=connectivity,
            coupling=coupling,
            integrator=integrator,
            monitors=(monitors,),
            initial_conditions=init,
        )
        sim_3.configure()
        sim_3.run() # run the first steps because the history is delayed
        result_3_all = [np.empty((0,)), np.empty((10, 2, 76, 1))]
        for j in range(0,10):
            result_3_all_step = sim_3.run()
            result_3_all[0] = np.concatenate((result_3_all[0], result_3_all_step[0][0]))
            result_3_all[1] = np.concatenate((result_3_all[1], result_3_all_step[0][1]))

        # The begging is good for rate and S
        for i in range(np.min(sim_3.connectivity.idelays[np.nonzero(sim_3.connectivity.idelays)]) + 1):
            diff = result_all[0][1][i][0][len(id_proxy):] - result_3_all[1][i+10, 0, len(id_proxy):]
            diff_2 = result_all[0][1][i][0][:len(id_proxy)] - result_3_all[1][i+10, 0, :len(id_proxy)]
            assert np.sum(diff) == 0.0 and np.sum(np.isnan(diff_2)) == len(id_proxy)
        for i in range(np.min(sim_3.connectivity.idelays[np.nonzero(sim_3.connectivity.idelays)]) + 1,100):
            diff = result_all[0][1][i][0][len(id_proxy):] - result_3_all[1][i + 10, 0, len(id_proxy):]
            diff_2 = result_all[0][1][i][0][:len(id_proxy)] - result_3_all[1][i + 10, 0, :len(id_proxy)]
            assert np.sum(diff) != 0.0 and np.sum(np.isnan(diff_2)) == len(id_proxy)
        # after the delayed impact the simulation, This create some difference for rate and S
        for i in range(np.min(sim_3.connectivity.idelays[np.nonzero(sim_3.connectivity.idelays)]) + 1):
            diff = result_all[0][1][i][1][len(id_proxy):] - result_3_all[1][i+10, 1, len(id_proxy):]
            diff_2 = result_all[0][1][i][1][:len(id_proxy)] - result_3_all[1][i+10, 1, :len(id_proxy)]
            assert np.sum(diff) == 0.0 and np.sum(np.isnan(diff_2)) == len(id_proxy)
        for i in range(np.min(sim_3.connectivity.idelays[np.nonzero(sim_3.connectivity.idelays)]) + 1,100):
            diff = result_all[0][1][i][1][len(id_proxy):] - result_3_all[1][i + 10, 1, len(id_proxy):]
            diff_2 = result_all[0][1][i][1][:len(id_proxy)] - result_3_all[1][i + 10, 1, :len(id_proxy)]
            assert np.sum(diff) != 0.0 and np.sum(np.isnan(diff_2)) == len(id_proxy)

    def test_with_proxy_bad_input(self):
        connectivity, coupling, integrator, monitors, sim, result, result_all = self._reference_simulation()
        # The modify model without proxy
        np.random.seed(42)
        init = np.concatenate((np.random.random_sample((385, 1, 76, 1)),
                               np.random.random_sample((385, 1, 76, 1))), axis=1)
        np.random.seed(42)
        id_proxy = range(11)
        model = ReducedWongWangProxy(tau_s=np.random.rand(76))
        # Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
        sim_4 = CoSimulator(
            voi=np.array([0]),
            synchronization_time=1.,
            cosim_monitors=(RawCosim(),),
            proxy_inds=np.asarray(id_proxy, dtype=np.int),
            model=model,
            connectivity=connectivity,
            coupling=coupling,
            integrator=integrator,
            monitors=(monitors,),
            initial_conditions=init,
        )
        sim_4.configure()
        sim_4.run() # run the first steps because the history is delayed
        result_4_all = [np.empty((0,)), np.empty((10, 2, 76, 1))]
        for j in range(0,10):
            result_4_all_step = sim_4.run(
                cosim_updates=[np.array([result_all[0][0][(10 * j) + i] for i in range(10)]),
                               np.ones((10, 1, len(id_proxy), 1)) * 0.7])
            result_4_all[0] = np.concatenate((result_4_all[0], result_4_all_step[0][0]))
            result_4_all[1] = np.concatenate((result_4_all[1], result_4_all_step[0][1]))

        # The beggining is good for rate and S
        for i in range(np.min(sim_4.connectivity.idelays[np.nonzero(sim_4.connectivity.idelays)])+1):
            diff = result_all[0][1][i][0][len(id_proxy):] - result_4_all[1][i+10, 0, len(id_proxy):]
            diff_2 = result_all[0][1][i][0][:len(id_proxy)] - result_4_all[1][i+10, 0, :len(id_proxy)]
            assert np.sum(diff, where=np.logical_not(np.isnan(diff))) == 0.0 and \
                   np.sum(diff_2, where=np.logical_not(np.isnan(diff_2))) != 0.0
        for i in range(np.min(sim_4.connectivity.idelays[np.nonzero(sim_4.connectivity.idelays)])+1,100):
            diff = result_all[0][1][i][0][len(id_proxy):] - result_4_all[1][i+10, 0, len(id_proxy):]
            diff_2 = result_all[0][1][i][0][:len(id_proxy)] - result_4_all[1][i+10, 0, :len(id_proxy)]
            assert np.sum(diff, where=np.logical_not(np.isnan(diff))) != 0.0 and \
                   np.sum(diff_2, where=np.logical_not(np.isnan(diff_2))) != 0.0
        # after the delayed impact the simulation, This create some difference for rate and S
        for i in range(np.min(sim_4.connectivity.idelays[np.nonzero(sim_4.connectivity.idelays)])+1):
            diff = result_all[0][1][i][1][len(id_proxy):] - result_4_all[1][i+10, 1, len(id_proxy):]
            diff_2 = result_all[0][1][i][1][:len(id_proxy)] - result_4_all[1][i+10, 1, :len(id_proxy)]
            assert np.sum(diff, where=np.logical_not(np.isnan(diff))) == 0.0 and \
                   np.sum(np.isnan(diff_2)) == len(id_proxy)
        for i in range(np.min(sim_4.connectivity.idelays[np.nonzero(sim_4.connectivity.idelays)])+1,100):
            diff = result_all[0][1][i][1][len(id_proxy):] - result_4_all[1][i+10, 1, len(id_proxy):]
            diff_2 = result_all[0][1][i][1][:len(id_proxy)] - result_4_all[1][i+10, 1, :len(id_proxy)]
            assert np.sum(diff, where=np.logical_not(np.isnan(diff))) != 0.0 and \
                   np.sum(np.isnan(diff_2)) == len(id_proxy)

    def test_with_proxy_right_input(self):
        connectivity, coupling, integrator, monitors, sim, result, result_all = self._reference_simulation()
        # The modify model without proxy
        np.random.seed(42)
        init = np.concatenate((np.random.random_sample((385, 1, 76, 1)),
                               np.random.random_sample((385, 1, 76, 1))), axis=1)
        np.random.seed(42)
        id_proxy = range(11)
        model = ReducedWongWangProxy(tau_s=np.random.rand(76))
        # Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
        sim_5 = CoSimulator(
            voi = np.array([0]),
            synchronization_time=1.,
            cosim_monitors=(RawCosim(),),
            proxy_inds=np.asarray(id_proxy, dtype=np.int),
            model=model,
            connectivity=connectivity,
            coupling=coupling,
            integrator=integrator,
            monitors=(monitors,),
            initial_conditions=init,
        )
        sim_5.configure()
        sim_5.run() # run the first steps because the history is delayed
        result_5_all = [np.empty((0,)), np.empty((10, 2, 76, 1))]
        for j in range(0,10):
            result_5_all_step = sim_5.run(
                cosim_updates=[np.array([result_all[0][0][(10 * j) + i] for i in range(10)]),
                               np.array([result_all[0][1][(10 * j) + i][0][id_proxy]
                                         for i in range(10)]).reshape((10, 1, len(id_proxy), 1))])
            result_5_all[0] = np.concatenate((result_5_all[0], result_5_all_step[0][0]))
            result_5_all[1] = np.concatenate((result_5_all[1], result_5_all_step[0][1]))
        # test for rate and after for S
        for i in range(100):
            diff = result_all[0][1][i][0][len(id_proxy):] - result_5_all[1][i+10, 0, len(id_proxy):]
            diff_2 = result_all[0][1][i][0][:len(id_proxy)] - result_5_all[1][i+10, 0, :len(id_proxy)]
            assert np.sum(diff, where=np.logical_not(np.isnan(diff))) == 0.0 and \
                   np.sum(diff_2, where=np.logical_not(np.isnan(diff_2))) == 0.0
        for i in range(100):
            diff = result_all[0][1][i][1][len(id_proxy):] - result_5_all[1][i+10, 1, len(id_proxy):]
            diff_2 = result_all[0][1][i][1][:len(id_proxy)] - result_5_all[1][i+10, 1, :len(id_proxy)]
            assert np.sum(diff, where=np.logical_not(np.isnan(diff))) == 0.0 and \
                   np.sum(np.isnan(diff_2)) == len(id_proxy)
