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

from tvb.tests.library.base_testcase import BaseTestCase
from tvb.contrib.tests.cosimulation.parallel.function_tvb import TvbSim


class TestProxyPrecisionDelay(BaseTestCase):
    """
    compare the result between simulation with one proxy and without proxy and different delay
    """

    def test_precision_delay(self):
        weight = np.array([[2, 8, 10], [0.2, 0.5, 0.1], [3, 0.6, 1]])
        delay = np.array([[0.6, 0.5, 1.0], [0.7, 0.8, 3.0], [1.0, 0.5, 0.7]])
        max = np.int_(np.max(delay) * 10 + 1)
        init_value = np.array([[[0.1, 0.0], [0.1, 0.0], [0.2, 0.0]]] * max)
        initial_condition = init_value.reshape((max, 2, weight.shape[0], 1))
        resolution_simulation = 0.1
        synchronization_time = 0.1 * 4
        proxy_id = [0]
        no_proxy = [1, 2]

        # simulation with one proxy
        np.random.seed(42)
        sim = TvbSim(weight, delay, proxy_id, resolution_simulation,
                     synchronization_time, initial_condition=initial_condition)
        time, s, result = sim(synchronization_time, rate=True)

        # full simulation
        np.random.seed(42)
        sim_ref = TvbSim(weight, delay, [], resolution_simulation,
                         synchronization_time, initial_condition=initial_condition)
        time, s_ref, result_ref = sim_ref(synchronization_time, rate=True)

        # compare with the CosimMonitor RawCosim
        np.testing.assert_array_equal(np.squeeze(result_ref[:, no_proxy, :], axis=2),
                                      np.squeeze(result[0][:, no_proxy, :], axis=2))
        np.testing.assert_array_equal(np.squeeze(s_ref[:, no_proxy, :], axis=2),
                                      np.squeeze(s[0][:, no_proxy, :], axis=2))

        for i in range(0, 1000):
            time, s, result = sim(synchronization_time,
                                  rate_data=[time, result_ref[:, proxy_id][:, :, 0]], rate=True)

            # compare with RawDelayed monitor, delayed by synchronization_time
            np.testing.assert_array_equal(result_ref[:, no_proxy, :], result[1][:, no_proxy, :])
            np.testing.assert_array_equal(result_ref[:, proxy_id, :] * np.NAN, result[1][:, proxy_id, :])
            np.testing.assert_array_equal(s_ref, s[1])

            time, s_ref, result_ref = sim_ref(synchronization_time, rate=True)

            # compare with the CosimMonitor RawCosim
            np.testing.assert_array_equal(result_ref[:, no_proxy, :], result[0][:, no_proxy, :])
            np.testing.assert_array_equal(s_ref[:, no_proxy, :], s[0][:, no_proxy, :])
