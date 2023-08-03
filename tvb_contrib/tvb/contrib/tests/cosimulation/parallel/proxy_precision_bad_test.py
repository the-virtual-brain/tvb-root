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
import operator

from tvb.tests.library.base_testcase import BaseTestCase
from tvb.contrib.tests.cosimulation.parallel.function_tvb import TvbSim


# this test is based on the proxy_precision.py
class TestPrecisionBad(BaseTestCase):
    """
    Compare the result between simulation with one proxy and without proxy but whith the wrong value at the beginning
    """

    def test_precision_bad(self):
        weight = np.array([[2, 8], [3, 5]])
        delay = 100.0
        delays = np.array([[delay, delay], [delay, delay]])
        max = np.int_(np.max(delay)*10+1)
        init_value = np.array([[0.9,0.0], [0.9,0.0]]*max)
        resolution_simulation = 0.1
        synchronization_time = 0.1 * 10.0
        nb_init = (int(delay / resolution_simulation)) + 1
        initial_condition = np.array(init_value * nb_init).reshape((nb_init, 2, weight.shape[0], 1))
        proxy_id = [0]
        no_proxy = [1]

        # simulation with one proxy
        np.random.seed(42)
        sim = TvbSim(weight, delays, proxy_id, resolution_simulation, synchronization_time,
                     initial_condition=initial_condition)
        time, result = sim(synchronization_time)

        # full simulation
        np.random.seed(42)
        sim_ref = TvbSim(weight, delays, [], resolution_simulation, synchronization_time,
                         initial_condition=initial_condition)
        time, result_ref = sim_ref(synchronization_time)

        # the results are different because the data of the proxy is wrong
        np.testing.assert_array_equal(result_ref[:, no_proxy, :], result[0][:, no_proxy, :])

        # the first part of the result are correct because the wrong result are delayed
        for i in range(0, 99):
            time, result = sim(synchronization_time, [time, np.zeros_like(result_ref[:, proxy_id][:, :, 0])])

            # compare with Raw monitor delayed by synchronization_time
            np.testing.assert_array_equal(result_ref[:, no_proxy, :], result[1][:, no_proxy, :])
            np.testing.assert_array_compare(operator.__ne__, result_ref[:, proxy_id, :], result[1][:, proxy_id, :])

            time, result_ref = sim_ref(synchronization_time)
            # compare with the CosimMonitor RawCosim
            np.testing.assert_array_equal(result_ref[:, no_proxy, :], result[0][:, no_proxy, :])

        # the result become of different value when the delayed result is computed
        for i in range(100, 1000):
            time, result = sim(synchronization_time, [time, np.zeros_like(result_ref[:, proxy_id][:, :, 0])])

            # compare with Raw monitor delayed by synchronization_time
            if i == 100: # as precedent
                np.testing.assert_array_equal(result_ref[:, no_proxy, :], result[1][:, no_proxy, :])
                np.testing.assert_array_compare(operator.__ne__, result_ref[:, proxy_id, :], result[1][:, proxy_id, :])
            elif i == 101: # the first one is the same
                np.testing.assert_array_equal(result_ref[:1, no_proxy, :], result[1][:1, no_proxy, :])
                np.testing.assert_array_compare(operator.__ne__, result_ref[1:, proxy_id, :], result[1][1:, proxy_id, :])
                np.testing.assert_array_compare(operator.__ne__, result_ref[1:, :, :], result[1][1:, :, :])
            else:
                np.testing.assert_array_compare(operator.__ne__,result_ref, result[1])

            time, result_ref = sim_ref(synchronization_time)
            # compare with the CosimMonitor RawCosim
            if i == 100: # the first one is the same
                np.testing.assert_array_equal(result_ref[:1, no_proxy, :], result[0][:1, no_proxy, :])
                np.testing.assert_array_compare(operator.__ne__,result_ref[1:, no_proxy, :], result[0][1:, no_proxy, :])
            else:
                np.testing.assert_array_compare(operator.__ne__,result_ref[:, no_proxy, :], result[0][:, no_proxy, :])
