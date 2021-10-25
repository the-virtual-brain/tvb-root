# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
"""
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import numpy

from tvb.datatypes.equations import DoubleGaussian, Gaussian, DiscreteEquation
from tvb.datatypes.volumes import Volume
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.datatypes import patterns, equations, connectivity, surfaces


class TestPatterns(BaseTestCase):
    """
    Tests the defaults for `tvb.datatypes.patterns` module.
    """

    def test_spatialpattern(self):
        dt = patterns.SpatialPattern()
        dt.spatial = DoubleGaussian()
        dt.configure_space(numpy.arange(100).reshape((10, 10)))
        dt.configure()
        summary = dt.summary_info()
        assert summary['Type'] == 'SpatialPattern'
        assert dt.space.shape == (10, 10)
        assert isinstance(dt.spatial, DoubleGaussian)
        assert dt.spatial_pattern.shape, (10, 1)

    def test_spatiotemporalpattern(self):
        dt = patterns.SpatioTemporalPattern()
        dt.spatial = DoubleGaussian()
        dt.temporal = Gaussian()
        dt.configure_space(numpy.arange(100).reshape((10, 10)))
        dt.configure()
        summary = dt.summary_info()
        assert summary['Type'] == 'SpatioTemporalPattern'
        assert dt.space.shape == (10, 10)
        assert isinstance(dt.spatial, DoubleGaussian)
        assert dt.spatial_pattern.shape == (10, 1)
        assert isinstance(dt.temporal, Gaussian)
        assert dt.temporal_pattern is None
        assert dt.time is None

    def test_stimuliregion(self):
        conn = connectivity.Connectivity.from_file()
        conn.configure()
        dt = patterns.StimuliRegion()
        dt.connectivity = conn
        dt.spatial = DiscreteEquation()
        dt.temporal = Gaussian()
        dt.weight = numpy.array([0 for _ in range(conn.number_of_regions)])
        dt.configure_space()
        assert dt.summary_info()['Type'] == 'StimuliRegion'
        assert dt.connectivity is not None
        assert dt.space.shape == (76, 1)
        assert dt.spatial_pattern.shape == (76, 1)
        assert isinstance(dt.temporal, Gaussian)
        assert dt.temporal_pattern is None
        assert dt.time is None

    def test_stimulisurface(self):
        srf = surfaces.CorticalSurface.from_file()
        srf.configure()
        dt = patterns.StimuliSurface()
        dt.surface = srf
        dt.spatial = DoubleGaussian()
        dt.temporal = Gaussian()
        dt.focal_points_triangles = numpy.array([0, 1, 2])
        dt.configure()
        dt.configure_space()
        summary = dt.summary_info()
        assert summary['Type'] == "StimuliSurface"
        assert dt.space.shape == (16384, 3)
        assert isinstance(dt.spatial, DoubleGaussian)
        assert dt.spatial_pattern.shape == (16384, 1)
        assert dt.surface is not None
        assert isinstance(dt.temporal, Gaussian)
        assert dt.temporal_pattern is None
        assert dt.time is None

    def test_spatialpatternvolume(self):
        dt = patterns.SpatialPatternVolume(spatial=Gaussian(),
                                           volume=Volume(origin=numpy.array([]), voxel_size=numpy.array([])),
                                           focal_points_volume=numpy.array([1]))
        assert dt.space is None
        assert dt.spatial is not None
        assert dt.spatial_pattern is None
        assert dt.volume is not None
        assert dt.focal_points_volume is not None
