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
from tvb.tests.cosimulation.co_simulation_paralelle.function_tvb import TvbSim
from tvb.tests.library.base_testcase import BaseTestCase
import numpy as np
import numpy.random as rgn


class TestPrecisionMultipleDelayUpdate(BaseTestCase):
    """
    compare the result between simulation with 1-3 proxy and without proxy and different delay
    """
    def test_precision_multiple_delay_update(self):
        weight = np.array([[5, 2, 4, 0], [8, 5, 4, 1], [6, 1, 7, 9], [10, 0, 5, 6]])
        delay = np.array([[0.1, 0.2, 0.5, 0.8], [0.3, 0.5, 0.5, 0.4], [0.6, 0.7, 0.1, 0.2], [0.3, 0.4, 0.5, 1.2]]) * 10
        max = np.int(np.max(delay)*10+1)
        init_value = np.array([[[0.1,0.0], [0.1,0.0], [0.2,0.0], [0.2,0.0]]] * max)
        initial_condition = init_value.reshape((max, 2, weight.shape[0], 1))
        resolution_simulation = 0.1
        time_synchronize = 0.1 * 10
        proxy_id = [0, 1, 2]
        no_proxy = [3]

        # simulation with one or more proxy
        rgn.seed(42)
        sim = TvbSim(weight, delay, proxy_id, resolution_simulation, time_synchronize,
                     initial_condition=initial_condition)
        time, result = sim(time_synchronize)

        # full simulation
        rgn.seed(42)
        sim_ref = TvbSim(weight, delay, [], resolution_simulation, time_synchronize,
                         initial_condition=initial_condition)
        time, result_ref = sim_ref(time_synchronize)

        # compare with the co-sim monitor raw
        diff = np.where(np.squeeze(result_ref[:,no_proxy,:], axis=1)[0] != np.squeeze(result[0][:,no_proxy,:], axis=1)[0])
        assert diff[0].size == 0

        for i in range(0, 10000):
            delai_input = [time, result_ref[:, proxy_id][:, :, 0]]
            time, result = sim(time_synchronize, delai_input)

            # compare with raw monitor delayed of time_synchronize
            diff_1 = np.where(result_ref != result[1])
            assert diff_1[0].size ==0

            time, result_ref = sim_ref(time_synchronize)

            # compare with the co-sim monitor raw
            diff = np.where(result_ref[:,no_proxy,:] != result[0][:,no_proxy,:])
            assert diff[0].size == 0
