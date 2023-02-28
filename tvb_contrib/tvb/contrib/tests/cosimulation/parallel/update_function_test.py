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

from tvb.tests.library.base_testcase import BaseTestCase
from tvb.contrib.tests.cosimulation.parallel.function_tvb import TvbSim


class TestUpdateModel(BaseTestCase):
    """
    test the function of function_tvb
    """
    def test_update_model(self):
        weight = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        delay = np.array([[1.5, 1.5, 1.5, 1.5], [1.5, 1.5, 1.5, 1.5], [1.5, 1.5, 1.5, 1.5], [1.5, 1.5, 1.5, 1.5]])
        resolution_simulation = 0.1
        synchronization_time = 1.0
        proxy_id = [0, 1]
        firing_rate = np.array([[20.0, 10.0]]) * 10 ** -3  # units time in tvb is ms so the rate is in KHz

        # Test the the update function
        sim = TvbSim(weight, delay, proxy_id, resolution_simulation, synchronization_time)
        for i in range(0, 100):
            time, result = sim(synchronization_time,
                               [np.arange(i * synchronization_time, (i + 1) * synchronization_time,
                                          resolution_simulation)-synchronization_time,
                                np.repeat(firing_rate.reshape(1, 2),
                                          int(synchronization_time / resolution_simulation), axis=0)])
        assert True

        # Test a fail function due to the time of simulation too long
        with pytest.raises(ValueError):
            sim(synchronization_time,[np.arange(100 * synchronization_time, 102 * synchronization_time,
                                          resolution_simulation),
                                np.repeat(firing_rate.reshape(1, 2),
                                          int(synchronization_time / resolution_simulation)*2, axis=0)] )
        # Test a fail function due to the resoulation time is not good
        with pytest.raises(ValueError):
            sim(synchronization_time,[np.arange(100 * synchronization_time, 101 * synchronization_time,
                                            resolution_simulation*2),
                                  np.repeat(firing_rate.reshape(1, 2),
                                            int(synchronization_time / resolution_simulation)*2, axis=0)] )
