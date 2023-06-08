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


class TestDoubleProxyPrecisionComplex(BaseTestCase):
    """
    Test the transmission of information between two models with proxy in most complex cases
    """

    def test_double_proxy_precision_complex(self):
        weight = np.array([[5, 2, 4, 0], [8, 5, 4, 1], [6, 1, 7, 9], [10, 0, 5, 6]])
        delay = np.array([[7, 8, 5, 1], [10, 3, 7, 9], [4, 3, 2, 8], [9, 10, 11, 5]])
        max = np.int_(np.max(delay) * 10 + 1)
        init_value = np.array([[[0.9, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]]] * max)
        initial_condition = init_value.reshape((max, 2, weight.shape[0], 1))
        resolution_simulation = 0.1
        synchronization_time = np.min(delay)
        proxy_id_1 = [1]
        proxy_id_2 = [0, 2]

        # simulation_2 with one proxy
        np.random.seed(42)
        sim_2 = TvbSim(weight, delay, proxy_id_2, resolution_simulation,
                       synchronization_time, initial_condition=initial_condition)
        time, s_2, result_2 = sim_2(synchronization_time, rate=True)

        # simulation with one proxy
        np.random.seed(42)
        sim_1 = TvbSim(weight, delay, proxy_id_1, resolution_simulation,
                       synchronization_time, initial_condition=initial_condition)
        time, s_1, result_1 = sim_1(synchronization_time, rate=True)

        # full simulation
        np.random.seed(42)
        sim_ref = TvbSim(weight, delay, [], resolution_simulation,
                         synchronization_time, initial_condition=initial_condition)
        time_ref, s_ref, result_ref = sim_ref(synchronization_time, rate=True)

        # COMPARE PROXY 1
        np.testing.assert_array_equal(np.squeeze(result_ref[:, proxy_id_2, :], axis=2),
                                      np.squeeze(result_1[0][:, proxy_id_2, :], axis=2))
        np.testing.assert_array_equal(np.squeeze(s_ref[:, proxy_id_2, :], axis=2),
                                      np.squeeze(s_1[0][:, proxy_id_2, :], axis=2))
        # COMPARE PROXY 2
        np.testing.assert_array_equal(np.squeeze(result_ref[:, proxy_id_1, :], axis=2),
                                      np.squeeze(result_2[0][:, proxy_id_1, :], axis=2))
        np.testing.assert_array_equal(np.squeeze(s_ref[:, proxy_id_1, :], axis=2),
                                      np.squeeze(s_2[0][:, proxy_id_1, :], axis=2))

        for i in range(0, 1000):
            time, s_2, result_2 = sim_2(synchronization_time,
                                        rate_data=[time, result_1[0][:, proxy_id_2][:, :, 0]], rate=True)

            # compare with Raw monitor delayed by synchronization_time
            np.testing.assert_array_equal(result_ref[:, proxy_id_1, :], result_2[1][:, proxy_id_1, :])
            np.testing.assert_array_equal(result_ref[:, proxy_id_2, :] * np.NAN, result_2[1][:, proxy_id_2, :])
            np.testing.assert_array_equal(s_ref, s_2[1])

            time, s_1, result_1 = sim_1(synchronization_time,
                                        rate_data=[time_ref, result_ref[:, proxy_id_1][:, :, 0]], rate=True)

            # compare with Raw monitor delayed by synchronization_time
            np.testing.assert_array_equal(result_ref[:, proxy_id_2, :], result_1[1][:, proxy_id_2, :])
            np.testing.assert_array_equal(result_ref[:, proxy_id_1, :] * np.NAN, result_1[1][:, proxy_id_1, :])
            np.testing.assert_array_equal(s_ref, s_1[1])

            time_ref, s_ref, result_ref = sim_ref(synchronization_time, rate=True)
            # COMPARE PROXY 1
            np.testing.assert_array_equal(
                np.squeeze(result_ref[:, proxy_id_2, :], axis=2),
                np.squeeze(result_1[0][:, proxy_id_2, :], axis=2))
            np.testing.assert_array_equal(
                np.squeeze(s_ref[:, proxy_id_2, :], axis=2),
                np.squeeze(s_1[0][:, proxy_id_2, :], axis=2))
            # COMPARE PROXY 2
            np.testing.assert_array_equal(
                np.squeeze(result_ref[:, proxy_id_1, :], axis=2),
                np.squeeze(result_2[0][:, proxy_id_1, :], axis=2))
            np.testing.assert_array_equal(
                np.squeeze(s_ref[:, proxy_id_1, :], axis=2),
                np.squeeze(s_2[0][:, proxy_id_1, :], axis=2))
