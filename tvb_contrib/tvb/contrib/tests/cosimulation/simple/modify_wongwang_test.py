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
import pytest
import operator
import tvb.simulator.lab as lab
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.contrib.tests.cosimulation.synchronization_time_set import SYNCHRONIZATION_TIME, adjust_connectivity_delays
from tvb.contrib.tests.cosimulation.parallel.ReducedWongWang import ReducedWongWangProxy
from tvb.contrib.cosimulation.cosim_monitors import RawCosim, CosimCoupling
from tvb.contrib.cosimulation.cosimulator import CoSimulator

SIMULATION_LENGTH = 3.0


class TestModifyWongWang(BaseTestCase):
    """
    Test to compare the version in tvb and the modified version
    """

    @staticmethod
    def _reference_simulation(model_class=lab.models.ReducedWongWang, simulator=lab.simulator.Simulator, init=None):
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
                        simulation_length=SIMULATION_LENGTH,
                        )
        sim.configure()
        result_all = sim.run()
        result = result_all[0][1][:, 0, 0, 0]
        return connectivity, coupling, integrator, monitors, sim, result, result_all


class TestModifyWongWangSimple(TestModifyWongWang):

    def test_with_no_cosimulation(self):
        connectivity, coupling, integrator, monitors, sim, result, result_all = self._reference_simulation()
        np.random.seed(42)
        init = np.concatenate((np.random.random_sample((385, 1, 76, 1)),
                               np.random.random_sample((385, 1, 76, 1))), axis=1)
        np.random.seed(42)
        result_2 = self._reference_simulation(model_class=ReducedWongWangProxy, simulator=CoSimulator,
                                              init=init)[5]
        np.testing.assert_array_equal(result, result_2)

    def test_with_proxy(self):
        connectivity, coupling, integrator, monitors, sim, result, result_all = self._reference_simulation()
        # The modify model without proxy
        np.random.seed(42)
        init = np.concatenate((np.random.random_sample((385, 1, 76, 1)),
                               np.random.random_sample((385, 1, 76, 1))), axis=1)
        np.random.seed(42)
        id_proxy = range(11)
        model = ReducedWongWangProxy(tau_s=np.random.rand(76))
        synchronization_time = SYNCHRONIZATION_TIME
        # Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
        sim_3 = CoSimulator(
            voi=np.array([0]),
            synchronization_time=synchronization_time,
            cosim_monitors=(RawCosim(),),
            proxy_inds=np.asarray(id_proxy, dtype=np.int_),
            model=model,
            connectivity=connectivity,
            coupling=coupling,
            integrator=integrator,
            monitors=(monitors,),
            initial_conditions=init,
        )
        sim_3.configure()
        sim_3.run()  # run the first steps because the history is delayed

        sim_to_sync_time = int(SIMULATION_LENGTH / synchronization_time)
        sync_steps = int(synchronization_time / integrator.dt)

        result_3_all = [np.empty((0,)), np.empty((sync_steps, 2, 76, 1))]

        for j in range(0, sim_to_sync_time):
            result_3_all_step = sim_3.run()
            result_3_all[0] = np.concatenate((result_3_all[0], result_3_all_step[0][0]))
            result_3_all[1] = np.concatenate((result_3_all[1], result_3_all_step[0][1]))

        # The beginning is good for rate and S
        for i in range(np.min(sim_3.connectivity.idelays[np.nonzero(sim_3.connectivity.idelays)]) + 1):
            np.testing.assert_array_equal(result_all[0][1][i][0][len(id_proxy):],
                                          result_3_all[1][i + sync_steps, 0, len(id_proxy):])
            np.testing.assert_array_equal(result_all[0][1][i][0][:len(id_proxy)] * np.NAN,
                                          result_3_all[1][i + sync_steps, 0, :len(id_proxy)])
        # After the delays impact the simulation, there is some difference for S
        idelays = np.copy(sim_3.connectivity.idelays)
        idelays = idelays[len(id_proxy):, :len(id_proxy)]
        min_delay = idelays[np.nonzero(idelays)].min()
        for i in range(min_delay + 1, int(SIMULATION_LENGTH / integrator.dt)):
            diff = result_all[0][1][i][0][len(id_proxy):] - result_3_all[1][i + sync_steps, 0, len(id_proxy):]
            assert np.isnan(diff.sum())
            np.testing.assert_array_equal(result_all[0][1][i][0][:len(id_proxy)] * np.NAN,
                                          result_3_all[1][i + sync_steps, 0, :len(id_proxy)])

    def test_with_proxy_bad_input(self):
        connectivity, coupling, integrator, monitors, sim, result, result_all = self._reference_simulation()
        # The modify model without proxy
        np.random.seed(42)
        init = np.concatenate((np.random.random_sample((385, 1, 76, 1)),
                               np.random.random_sample((385, 1, 76, 1))), axis=1)
        np.random.seed(42)
        id_proxy = range(11)
        model = ReducedWongWangProxy(tau_s=np.random.rand(76))
        synchronization_time = SYNCHRONIZATION_TIME
        # Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
        sim_4 = CoSimulator(
            voi=np.array([0]),
            synchronization_time=synchronization_time,
            cosim_monitors=(RawCosim(),),
            proxy_inds=np.asarray(id_proxy, dtype=np.int_),
            model=model,
            connectivity=connectivity,
            coupling=coupling,
            integrator=integrator,
            monitors=(monitors,),
            initial_conditions=init,
        )
        sim_4.configure()
        sim_4.run()  # run the first steps because the history is delayed

        sim_to_sync_time = int(SIMULATION_LENGTH / synchronization_time)
        sync_steps = int(synchronization_time / integrator.dt)

        result_4_all = [np.empty((0,)), np.empty((sync_steps, 2, 76, 1))]

        for j in range(0, sim_to_sync_time):
            result_4_all_step = sim_4.run(
                cosim_updates=[np.array([result_all[0][0][(sync_steps * j) + i] for i in range(sync_steps)]),
                               np.ones((sync_steps, 1, len(id_proxy), 1)) * 0.7])
            result_4_all[0] = np.concatenate((result_4_all[0], result_4_all_step[0][0]))
            result_4_all[1] = np.concatenate((result_4_all[1], result_4_all_step[0][1]))

        # The beginning is good for rate and S
        for i in range(np.min(sim_4.connectivity.idelays[np.nonzero(sim_4.connectivity.idelays)]) + 1):
            np.testing.assert_array_equal(result_all[0][1][i][0][len(id_proxy):],
                                          result_4_all[1][i + sync_steps, 0, len(id_proxy):])
            np.testing.assert_array_compare(operator.__ne__,
                                            result_all[0][1][i][0][:len(id_proxy)],
                                            result_4_all[1][i + sync_steps, 0, :len(id_proxy)])
        # After the delays impact the simulation, there is some difference for for rate and S
        idelays = np.copy(sim_4.connectivity.idelays)
        idelays = idelays[len(id_proxy):, :len(id_proxy)]
        min_delay = idelays[np.nonzero(idelays)].min()
        for i in range(min_delay + 1, int(SIMULATION_LENGTH / integrator.dt)):
            diff = result_all[0][1][i][0][len(id_proxy):] - result_4_all[1][i + sync_steps, 0, len(id_proxy):]
            assert np.sum(diff) != 0.0  # TODO: Find out why it fails for the first two iterations!
            np.testing.assert_array_compare(operator.__ne__,
                                            result_all[0][1][i][0][:len(id_proxy)],
                                            result_4_all[1][i + sync_steps, 0, :len(id_proxy)])

    def test_with_proxy_right_input(self):
        connectivity, coupling, integrator, monitors, sim, result, result_all = self._reference_simulation()
        # The modify model without proxy
        np.random.seed(42)
        init = np.concatenate((np.random.random_sample((385, 1, 76, 1)),
                               np.random.random_sample((385, 1, 76, 1))), axis=1)
        np.random.seed(42)
        id_proxy = range(11)
        model = ReducedWongWangProxy(tau_s=np.random.rand(76))
        synchronization_time = SYNCHRONIZATION_TIME
        # Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
        sim_5 = CoSimulator(
            voi=np.array([0]),
            synchronization_time=synchronization_time,
            cosim_monitors=(RawCosim(),),
            proxy_inds=np.asarray(id_proxy, dtype=np.int_),
            model=model,
            connectivity=connectivity,
            coupling=coupling,
            integrator=integrator,
            monitors=(monitors,),
            initial_conditions=init,
        )
        sim_5.configure()
        sim_5.run()  # run the first steps because the history is delayed

        sim_to_sync_time = int(SIMULATION_LENGTH / synchronization_time)
        sync_steps = int(synchronization_time / integrator.dt)

        result_5_all = [np.empty((0,)), np.empty((sync_steps, 2, 76, 1))]

        for j in range(0, sim_to_sync_time):
            result_5_all_step = sim_5.run(
                cosim_updates=[np.array([result_all[0][0][(sync_steps * j) + i] for i in range(sync_steps)]),
                               np.array([result_all[0][1][(sync_steps * j) + i][0][id_proxy]
                                         for i in range(sync_steps)]).reshape((sync_steps, 1, len(id_proxy), 1))])
            result_5_all[0] = np.concatenate((result_5_all[0], result_5_all_step[0][0]))
            result_5_all[1] = np.concatenate((result_5_all[1], result_5_all_step[0][1]))

        for i in range(int(SIMULATION_LENGTH / integrator.dt)):
            np.testing.assert_array_equal(result_all[0][1][i][0][len(id_proxy):],
                                          result_5_all[1][i + sync_steps, 0, len(id_proxy):])
            np.testing.assert_array_equal(result_all[0][1][i][0][:len(id_proxy)],
                                          result_5_all[1][i + sync_steps, 0, :len(id_proxy)])

    def test_without_proxy_coupling(self):
        connectivity, coupling, integrator, monitors, sim, result, result_all = self._reference_simulation()
        # The modify model without proxy
        np.random.seed(42)
        init = np.concatenate((np.random.random_sample((385, 1, 76, 1)),
                               np.random.random_sample((385, 1, 76, 1))), axis=1)
        model = ReducedWongWangProxy(tau_s=np.random.rand(76))
        synchronization_time = SYNCHRONIZATION_TIME
        # Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
        sim_6 = CoSimulator(
            voi=np.array([0]),
            synchronization_time=synchronization_time,
            cosim_monitors=(CosimCoupling(coupling=coupling),),
            proxy_inds=np.asarray([0], dtype=np.int_),
            model=model,
            connectivity=connectivity,
            coupling=coupling,
            integrator=integrator,
            monitors=(monitors,),
            initial_conditions=init,
        )
        sim_6.configure()

        sim_to_sync_time = int(SIMULATION_LENGTH / synchronization_time)
        sync_steps = int(synchronization_time / integrator.dt)

        with pytest.raises(ValueError):
            coupling_future = sim_6.loop_cosim_monitor_output(sync_steps, 1)

        coupling_future = sim_6.loop_cosim_monitor_output()

        for i in range(sim_to_sync_time):
            result_2 = sim_6.run(
                cosim_updates=[
                    np.arange(i * synchronization_time, (i + 1) * synchronization_time, 0.1) - synchronization_time,
                    np.zeros((int(synchronization_time / 0.1), 1, 1, 1))])[0][1][:, 0, 0, 0]
            np.testing.assert_array_equal(result[i * sync_steps:(i + 1) * sync_steps] * 0.0, result_2)
            assert np.sum(np.isnan(sim_6.loop_cosim_monitor_output()[0][1])) == 9
