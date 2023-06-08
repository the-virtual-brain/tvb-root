# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
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
#
#
"""
Created on Mar 20, 2013

.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import numpy
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.datatypes import volumes


class TestVolumes(BaseTestCase):
    """
    Tests the defaults for `tvb.datatypes.volumes` module.
    """

    def test_volume(self):
        dt = volumes.Volume(origin=numpy.array([]), voxel_size=numpy.array([]))
        summary_info = dt.summary_info()
        assert summary_info['Origin'] is not None
        assert summary_info['Voxel size'] is not None
        assert summary_info['Volume type'] == 'Volume'
        assert summary_info['Units'] == 'mm'
        assert dt.origin is not None
        assert dt.voxel_size is not None
        assert dt.voxel_unit == 'mm'
