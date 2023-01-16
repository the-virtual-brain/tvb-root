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
import sys
import numpy
import pytest
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.surfaces import CorticalSurface, SurfaceTypesEnum
from tvb.datatypes.cortex import Cortex
from tvb.datatypes.local_connectivity import LocalConnectivity
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes import surfaces
from tvb.tests.library.simulator.simulator_test import Simulator

class TestSurfaces(BaseTestCase):
    """
    Tests the defaults for `tvb.datatypes.surfaces` module.
    """

    def test_surface(self):
        dt = surfaces.Surface(valid_for_simulations=True)
        dt.vertices = numpy.array(list(range(30))).reshape(10, 3).astype(numpy.float64)
        dt.triangles = numpy.array(list(range(9))).reshape(3, 3)
        dt.triangle_normals = numpy.array(list(range(9))).reshape(3, 3)
        dt.vertex_normals = numpy.array(list(range(30))).reshape(10, 3).astype('f')

        dt.configure()

        summary_info = dt.summary_info()
        assert summary_info['Number of edges'] == 9
        assert summary_info['Number of triangles'] == 3
        assert summary_info['Number of vertices'] == 10
        assert summary_info['Surface type'] == 'Surface'
        assert len(dt.vertex_neighbours) == 10
        assert isinstance(dt.vertex_neighbours[0], frozenset)
        assert len(dt.vertex_triangles) == 10
        assert isinstance(dt.vertex_triangles[0], frozenset)
        assert len(dt.nth_ring(0)) == 0
        assert dt.triangle_areas.shape == (3, 1)
        assert dt.triangle_angles.shape == (3, 3)
        assert len(dt.edges) == 9
        assert len(dt.edge_triangles) == 9
        assert [] != dt.validate_topology_for_simulations().warnings
        assert dt.vertices.shape == (10, 3)
        assert dt.vertex_normals.shape == (10, 3)
        assert dt.triangles.shape == (3, 3)

    def test_cortical_surface(self):
        dt = surfaces.CorticalSurface().from_file()
        dt.__setattr__('valid_for_simulations', True)
        assert isinstance(dt, surfaces.CorticalSurface)
        dt.configure()
        summary_info = dt.summary_info()
        assert summary_info['Number of edges'] == 49140
        assert summary_info['Number of triangles'] == 32760
        assert summary_info['Number of vertices'] == 16384
        assert dt.surface_type == SurfaceTypesEnum.CORTICAL_SURFACE.value
        assert len(dt.vertex_neighbours) == 16384
        assert isinstance(dt.vertex_neighbours[0], frozenset)
        assert len(dt.vertex_triangles) == 16384
        assert isinstance(dt.vertex_triangles[0], frozenset)
        assert len(dt.nth_ring(0)) == 17
        assert dt.triangle_areas.shape == (32760, 1)
        assert dt.triangle_angles.shape == (32760, 3)
        assert len(dt.edges) == 49140
        assert abs(dt.edge_mean_length - 3.97605292887) < 0.00000001
        assert abs(dt.edge_min_length - 0.6638) < 0.0001
        assert abs(dt.edge_max_length - 7.7567) < 0.0001
        assert len(dt.edge_triangles) == 49140
        assert [] == dt.validate_topology_for_simulations().warnings
        assert dt.vertices.shape == (16384, 3)
        assert dt.vertex_normals.shape == (16384, 3)
        assert dt.triangles.shape == (32760, 3)
        topologicals = dt.compute_topological_constants()
        assert 4 == topologicals[0]
        assert all([a.size == 0 for a in topologicals[1:]])

    def test_cortical_topology_pyramid(self):
        dt = surfaces.Surface()
        dt.vertices = numpy.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(numpy.float64)
        dt.triangles = numpy.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        dt.configure()

        euler, isolated, pinched_off, holes = dt.compute_topological_constants()
        assert 2 == euler
        assert 0 == isolated.size
        assert 0 == pinched_off.size
        assert 0 == holes.size

    def test_cortical_topology_isolated_vertex(self):
        dt = surfaces.Surface()
        dt.vertices = numpy.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 2]]).astype(numpy.float64)
        dt.triangles = numpy.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        dt.triangle_normals = numpy.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        dt.vertex_normals = numpy.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 2]]).astype(numpy.float64)
        dt.configure()

        euler, isolated, pinched_off, holes = dt.compute_topological_constants()
        assert 3 == euler
        assert 1 == isolated.size
        assert 0 == pinched_off.size
        assert 0 == holes.size

    def test_cortical_topology_pinched(self):
        dt = surfaces.Surface()
        dt.vertices = numpy.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(numpy.float64)
        dt.triangles = numpy.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3], [1, 2, 3]])
        dt.configure()

        euler, isolated, pinched_off, holes = dt.compute_topological_constants()
        assert 3 == euler
        assert 0 == isolated.size
        assert 3 == pinched_off.size
        assert 0 == holes.size

    def test_cortical_topology_hole(self):
        dt = surfaces.Surface()
        dt.vertices = numpy.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(numpy.float64)
        dt.triangles = numpy.array([[0, 2, 1], [0, 1, 3], [0, 3, 2]])
        dt.configure()

        euler, isolated, pinched_off, holes = dt.compute_topological_constants()
        assert 1 == euler
        assert 3 == isolated.size
        assert 0 == pinched_off.size
        assert 3 == holes.size

    def test_skinair(self):
        dt = surfaces.SkinAir.from_file()
        assert isinstance(dt, surfaces.SkinAir)
        assert dt.vertices.shape == (4096, 3)
        assert dt.vertex_normals.shape == (4096, 3)
        assert dt.triangles.shape == (8188, 3)

    def test_brainskull(self):
        dt = surfaces.BrainSkull.from_file()
        assert isinstance(dt, surfaces.BrainSkull)
        assert dt.vertices.shape == (4096, 3)
        assert dt.vertex_normals.shape == (4096, 3)
        assert dt.triangles.shape == (8188, 3)

    def test_skullskin(self):
        dt = surfaces.SkullSkin.from_file()
        assert isinstance(dt, surfaces.SkullSkin)
        assert dt.vertices.shape == (4096, 3)
        assert dt.vertex_normals.shape == (4096, 3)
        assert dt.triangles.shape == (8188, 3)

    def test_eegcap(self):
        dt = surfaces.EEGCap.from_file()
        assert isinstance(dt, surfaces.EEGCap)
        assert dt.vertices.shape == (1082, 3)
        assert dt.vertex_normals.shape == (1082, 3)
        assert dt.triangles.shape == (2160, 3)

    def test_facesurface(self):
        dt = surfaces.FaceSurface.from_file()
        assert isinstance(dt, surfaces.FaceSurface)
        assert dt.vertices.shape == (8614, 3)
        assert dt.vertex_normals.shape == (0,)
        assert dt.triangles.shape == (17224, 3)

    def test_regionmapping(self):
        dt = RegionMapping.from_file()
        assert isinstance(dt, RegionMapping)
        assert dt.array_data.shape == (16384,)

    def test_localconnectivity_empty(self):
        dt = LocalConnectivity(surface=CorticalSurface())
        assert dt.surface is not None

    @pytest.mark.skipif(sys.maxsize <= 2147483647, reason="Cannot deal with local connectivity on a 32-bit machine.")
    def test_cortexdata(self):
        dt = Cortex.from_file(local_connectivity_file="local_connectivity_16384.mat")
        dt.region_mapping_data.connectivity = Connectivity.from_file()
        assert isinstance(dt, Cortex)
        assert dt.region_mapping is not None

        dt.configure()
        assert dt.vertices.shape == (16384, 3)
        assert dt.vertex_normals.shape == (16384, 3)
        assert dt.triangles.shape == (32760, 3)

    def test_cortex_reg_map_without_subcorticals(self):
        dt = Cortex.from_file()
        dt.region_mapping_data.connectivity = Connectivity.from_file()
        self.add_subcorticals_to_conn(dt.region_mapping_data.connectivity)
        dt.region_mapping_data.connectivity.configure()

        assert isinstance(dt, Cortex)
        assert dt.region_mapping is not None
        assert numpy.unique(dt.region_mapping).size == dt.region_mapping_data.connectivity.number_of_regions

    @pytest.mark.skipif(sys.maxsize <= 2147483647, reason="Cannot deal with local connectivity on a 32-bit machine.")
    def test_init_conds(self):
        import numpy as np
        from tvb.datatypes.cortex import Cortex
        sim = Simulator()
        ctx = Cortex.from_file()
        sim.configure(
            surface_sim=True,
            initial_conditions=np.random.rand(1,2,ctx.vertices.shape[0],1))
