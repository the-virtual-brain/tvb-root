# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
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
#
#
from tvb.tests.library.simulator.test_monitor_cosimulation.co_simulation_paralelle.function_tvb import TvbSim
from tvb.tests.library.base_testcase import BaseTestCase
import numpy as np
import numpy.random as rgn


class TestDoubleProxyPrecisionComplexDelayUpdate(BaseTestCase):
    def test_double_precision_complex_delay_update(self):
        weight = np.array([[5, 2, 4, 0], [8, 5, 4, 1], [6, 1, 7, 9], [10, 0, 5, 6]])
        delay = np.array([[7, 8, 5, 1], [9, 3, 7, 9], [4, 3, 2, 8], [9, 10, 11, 5]])
        resolution_simulation = 0.1
        time_synchronize = np.min(delay)
        proxy_id_1 = [1]
        proxy_id_2 = [0, 2]

        # full simulation
        rgn.seed(42)
        sim_ref = TvbSim(weight, delay, [], resolution_simulation, time_synchronize)
        time, result_ref = sim_ref(time_synchronize)
        delay_input_1 = [time, result_ref[:, proxy_id_1][:, :, 0]]

        # simulation with one proxy
        rgn.seed(42)
        sim_1 = TvbSim(weight, delay, proxy_id_1, resolution_simulation, time_synchronize)
        time, result_1 = sim_1(time_synchronize, delay_input_1)
        delay_input_2 = [time, result_ref[:, proxy_id_2][:, :, 0]]

        # simulation_2 with one proxy
        rgn.seed(42)
        sim_2 = TvbSim(weight, delay, proxy_id_2, resolution_simulation, time_synchronize)
        time, result_2 = sim_2(time_synchronize, delay_input_2)

        # COMPARE PROXY 1
        diff_1 = np.where(np.squeeze(result_ref, axis=2)[0] != np.squeeze(result_1, axis=2)[0])
        assert diff_1[0].size == 0
        # COMPARE PROXY 2
        diff_2 = np.where(np.squeeze(result_ref, axis=2)[0] != np.squeeze(result_2, axis=2)[0])
        assert diff_2[0].size == 0

        for i in range(0, 100):
            time, result_ref = sim_ref(time_synchronize)
            delay_input_1 = [time, result_ref[:, proxy_id_1][:, :, 0]]
            time, result_1 = sim_1(time_synchronize, delay_input_1)
            delay_input_2 = [time, result_ref[:, proxy_id_2][:, :, 0]]
            time, result_2 = sim_2(time_synchronize, delay_input_2)

            # COMPARE PROXY 1
            diff_1 = np.where(np.squeeze(result_ref, axis=2)[0] != np.squeeze(result_1, axis=2)[0])
            assert diff_1[0].size == 0
            # COMPARE PROXY 2
            diff_2 = np.where(np.squeeze(result_ref, axis=2)[0] != np.squeeze(result_2, axis=2)[0])
            assert diff_2[0].size == 0
