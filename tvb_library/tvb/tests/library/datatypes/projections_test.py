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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import numpy
from tvb.datatypes.sensors import SensorsEEG, SensorsMEG
from tvb.datatypes.surfaces import CorticalSurface
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.datatypes import projections


class TestPatterns(BaseTestCase):
    """
    Tests the defaults for `tvb.datatypes.projections` module.
    """

    def test_projectionmatrix(self):
        dt = projections.ProjectionMatrix(projection_type=str(""), sources=CorticalSurface(), projection_data=numpy.array([]))
        assert dt.sources is not None
        assert dt.sensors is None
        assert dt.projection_data is not None

    def test_projection_surface_eeg(self):
        dt = projections.ProjectionSurfaceEEG(sensors=SensorsEEG(),projection_data=numpy.array([]), sources=CorticalSurface())
        assert dt.sources is not None
        assert dt.skin_air is None
        assert dt.skull_skin is None
        assert dt.sensors is not None
        assert dt.projection_data is not None

    def test_projection_surface_meg(self):
        dt = projections.ProjectionSurfaceMEG(sensors=SensorsMEG(),projection_data=numpy.array([]), sources=CorticalSurface())
        assert dt.sources is not None
        assert dt.skin_air is None
        assert dt.skull_skin is None
        assert dt.sensors is not None
        assert dt.projection_data is not None
