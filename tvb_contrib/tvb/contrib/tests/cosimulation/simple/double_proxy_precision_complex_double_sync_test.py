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

from copy import deepcopy
import numpy as np
import operator

from tvb.tests.library.base_testcase import BaseTestCase
from tvb.contrib.tests.cosimulation.parallel.function_tvb import TvbSim, tvb_simulation, CoSimulator


class TvbSimDoubleSync(TvbSim):

    def __init__(self, weight, delay, id_proxy, resolution_simulation, synchronization_time,
                 initial_condition=None, relative_output_time_steps=0):
        """
        initialise the simulator
        :param weight: weight on the connection
        :param delay: delay of the connections
        :param id_proxy: the id of the proxy
        :param resolution_simulation: the resolution of the simulation
        :param initial_condition: initial condition for S and H
        :param relative_output_time_steps: number of time steps in the past to sample for TVB monitors
        """
        super(TvbSimDoubleSync, self).__init__(
            weight, delay, id_proxy, resolution_simulation, synchronization_time, initial_condition)
        if isinstance(self.sim, CoSimulator):
            self.sim.relative_output_time_steps = relative_output_time_steps

    def __call__(self, time, proxy_data=None, rate_data=None, rate=False):
        """
        Run simulation for t biological
        :param time: the time of the simulation
        :param proxy_data: the firing rate for the next steps for the proxy
        :return:
            the result of time, the firing rate and the state of the network
        Notice that this function reverses the order of integration and input transformation for co-simulation.
        """
        # Simulate for t + Tsync, no matter if there is any input or not
        time, s_out, rates = tvb_simulation(time, self.sim, proxy_data)
        proxy_data = None
        if rate_data is not None:
            # if there is data from the other cosimulator towards the transformer
            # ...transform them:
            proxy_data = self.transform_rate_to_s(rate_data)

        if rate:
            return time, s_out, rates, proxy_data
        else:
            return time, s_out


class TestDoubleProxyPrecisionComplexDoubleSync(BaseTestCase):
    """
    Test the transmission of information between two models with proxy in most complex cases
    """

    def test_double_proxy_precision_complex_double_sync(self):
        weight = np.array([[5, 2, 4, 0], [8, 5, 4, 1], [6, 1, 7, 9], [10, 0, 5, 6]])
        delay = np.array([[7, 8, 5, 2], [10, 3, 7, 9], [4, 3, 2, 8], [9, 10, 11, 5]])
        max = np.int_(np.max(delay) * 10 + 1)
        init_value = np.array([[[0.9, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]]] * max)
        initial_condition = init_value.reshape((max, 2, weight.shape[0], 1))
        resolution_simulation = 0.1
        synchronization_time = np.min(delay) / 2.0  # = 1.0 = 10 integration time steps in this example
        relative_output_time_steps = int(np.round(synchronization_time/resolution_simulation))
        proxy_id_1 = [1]
        proxy_id_2 = [0, 2]

        # full simulation
        np.random.seed(42)
        sim_ref = TvbSimDoubleSync(weight, delay, [], resolution_simulation,
                                   synchronization_time, initial_condition=initial_condition)
        time_ref, s_ref, result_ref, _ = sim_ref(synchronization_time, rate=True)

        # simulation with one proxy
        np.random.seed(42)
        sim_1 = TvbSimDoubleSync(weight, delay, proxy_id_1, resolution_simulation,
                                 synchronization_time, initial_condition=initial_condition,
                                relative_output_time_steps=relative_output_time_steps)
        time, s_1, rate_1, proxy_data_1 = sim_1(synchronization_time, rate=True)

        # simulation_2 with one proxy
        np.random.seed(42)
        sim_2 = TvbSimDoubleSync(weight, delay, proxy_id_2, resolution_simulation,
                                 synchronization_time, initial_condition=initial_condition,
                                 relative_output_time_steps=relative_output_time_steps)
        time, s_2, rate_2, proxy_data_2 = sim_2(synchronization_time, rate=True)

        # COMPARE PROXY 1
        np.testing.assert_array_equal(np.squeeze(result_ref[:, proxy_id_2, :], axis=2),
                                      np.squeeze(rate_1[0][:, proxy_id_2, :], axis=2))
        np.testing.assert_array_equal(np.squeeze(s_ref[:, proxy_id_2, :], axis=2),
                                      np.squeeze(s_1[0][:, proxy_id_2, :], axis=2))

        # COMPARE PROXY 2
        np.testing.assert_array_equal(np.squeeze(result_ref[:, proxy_id_1, :], axis=2),
                                      np.squeeze(rate_2[0][:, proxy_id_1, :], axis=2))
        np.testing.assert_array_equal(np.squeeze(s_ref[:, proxy_id_1, :], axis=2),
                                      np.squeeze(s_2[0][:, proxy_id_1, :], axis=2))

        for i in range(0, 2000):

            input_time = time_ref - synchronization_time
            input_rate1 = rate_2[0][:, proxy_id_1, :][:, :, 0]
            input_rate2 = rate_1[0][:, proxy_id_2, :][:, :, 0]
            input_proxy_data1 = proxy_data_1
            if proxy_data_1 is not None:
                input_proxy_data1 = [input_time, input_proxy_data1]
            input_proxy_data2 = proxy_data_2
            if input_proxy_data2 is not None:
                input_proxy_data2 = [input_time, input_proxy_data2]

            time, s_1, rate_1, proxy_data_1 = \
                sim_1(synchronization_time, rate_data=input_rate1, proxy_data=input_proxy_data1, rate=True)

            # compare with Raw monitor delayed by 2*synchronization_time
            if i>1:
                np.testing.assert_array_equal(result_ref0[:, proxy_id_2, :], rate_1[1][:, proxy_id_2, :])
                np.testing.assert_array_equal(result_ref0[:, proxy_id_1, :] * np.NAN, rate_1[1][:, proxy_id_1, :])
                np.testing.assert_array_equal(s_ref0, s_1[1])

            time, s_2, rate_2, proxy_data_2 = \
                sim_2(synchronization_time, rate_data=input_rate2, proxy_data=input_proxy_data2, rate=True)

            # compare with Raw monitor delayed by 2*synchronization_time
            if i>1:
                np.testing.assert_array_equal(result_ref0[:, proxy_id_1, :], rate_2[1][:, proxy_id_1, :])
                np.testing.assert_array_equal(result_ref0[:, proxy_id_2, :] * np.NAN, rate_2[1][:, proxy_id_2, :])
                np.testing.assert_array_equal(s_ref0, s_2[1])

            s_ref0 = deepcopy(s_ref)
            result_ref0 = deepcopy(result_ref)
            time_ref, s_ref, result_ref, _ = sim_ref(synchronization_time, rate=True)

            # COMPARE PROXY 1
            np.testing.assert_array_equal(
                np.squeeze(result_ref[:, proxy_id_2, :], axis=2),
                np.squeeze(rate_1[0][:, proxy_id_2, :], axis=2))
            np.testing.assert_array_equal(
                np.squeeze(s_ref[:, proxy_id_2, :], axis=2),
                np.squeeze(s_1[0][:, proxy_id_2, :], axis=2))
            # COMPARE PROXY 2
            np.testing.assert_array_equal(
                np.squeeze(result_ref[:, proxy_id_1, :], axis=2),
                np.squeeze(rate_2[0][:, proxy_id_1, :], axis=2))
            np.testing.assert_array_equal(
                np.squeeze(s_ref[:, proxy_id_1, :], axis=2),
                np.squeeze(s_2[0][:, proxy_id_1, :], axis=2))
