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


class TestProxyPrecision(BaseTestCase):
    """
    Compare the result between simulation with one proxy and without proxy
    """

    def test_precision(self):
        weight = np.array([[2, 8], [3, 5]])
        delay = 1.5
        delays = np.array([[delay, delay], [delay, delay]])
        init_value = [[0.9,0.0], [0.9,0.0]]
        resolution_simulation = 0.1
        synchronization_time = resolution_simulation * 10.0
        nb_init = (int(delay / resolution_simulation)) + 1
        initial_condition = np.array(init_value * nb_init).reshape((nb_init, 2, weight.shape[0], 1))
        proxy_id = [0]
        no_proxy = [1]

        # simulation with one proxy
        np.random.seed(42)
        sim = TvbSim(weight, delays, proxy_id, resolution_simulation, synchronization_time,
                     initial_condition=initial_condition)
        time, s, result = sim(synchronization_time, rate=True)

        # full simulation
        np.random.seed(42)
        sim_ref = TvbSim(weight, delays, [], resolution_simulation, synchronization_time,
                         initial_condition=initial_condition)
        time_ref, s_ref, result_ref = sim_ref(synchronization_time, rate=True)

        # compare with the Cosim Monitor RawCosim
        np.testing.assert_array_equal(result_ref[:,no_proxy,:], result[0][:,no_proxy,:])
        np.testing.assert_array_equal(s_ref[:,no_proxy,:], s[0][:,no_proxy,:])

        for i in range(0, 1000):
            time, s, result = sim(synchronization_time,
                                  rate_data=[time_ref, result_ref[:, proxy_id][:, :, 0]], rate=True)

            # compare with monitor delayed by synchronization_time
            np.testing.assert_array_equal(result_ref[:, no_proxy, :], result[1][:, no_proxy, :])
            np.testing.assert_array_equal(result_ref[:, proxy_id, :]*np.NAN, result[1][:, proxy_id, :])
            np.testing.assert_array_equal(s_ref, s[1])

            time_ref, s_ref, result_ref = sim_ref(synchronization_time, rate=True)

            # compare with the Cosim Monitor RawCosim
            np.testing.assert_array_equal(result_ref[:, no_proxy, :], result[0][:, no_proxy, :])
            np.testing.assert_array_equal(s_ref[:, no_proxy, :], s[0][:, no_proxy, :])
