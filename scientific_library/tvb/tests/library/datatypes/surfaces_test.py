# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
from tvb.datatypes.cortex import Cortex
from tvb.datatypes.local_connectivity import LocalConnectivity
from tvb.datatypes.region_mapping import RegionMapping

if __name__ == "__main__":
    from tvb.tests.library import setup_test_console_env
    setup_test_console_env()

import unittest
import sys
import numpy
from tvb.datatypes import surfaces
from tvb.tests.library.base_testcase import BaseTestCase



class SurfacesTest(BaseTestCase):
    """
    Tests the defaults for `tvb.datatypes.surfaces` module.
    """

    def test_surface(self):
        dt = surfaces.Surface()
        dt.vertices = numpy.array(range(30)).reshape(10, 3).astype(numpy.float64)
        dt.triangles = numpy.array(range(9)).reshape(3, 3)
        dt.configure()
        summary_info = dt.summary_info
        self.assertEqual(summary_info['Number of edges'], 9)
        self.assertEqual(summary_info['Number of triangles'], 3)
        self.assertEqual(summary_info['Number of vertices'], 10)
        self.assertEqual(summary_info['Surface type'], 'Surface')
        self.assertEqual(len(dt.vertex_neighbours), 10)
        self.assertTrue(isinstance(dt.vertex_neighbours[0], frozenset))
        self.assertEqual(len(dt.vertex_triangles), 10)
        self.assertTrue(isinstance(dt.vertex_triangles[0], frozenset))
        self.assertEqual(len(dt.nth_ring(0)), 0)
        self.assertEqual(dt.triangle_areas.shape, (3, 1))
        self.assertEqual(dt.triangle_angles.shape, (3, 3))
        self.assertEqual(len(dt.edges), 9)
        self.assertEqual(len(dt.edge_triangles), 9)
        self.assertNotEqual([], dt.validate_topology_for_simulations().warnings)
        self.assertEqual(dt.get_data_shape('vertices'), (10, 3))
        self.assertEqual(dt.get_data_shape('vertex_normals'), (10, 3))
        self.assertEqual(dt.get_data_shape('triangles'), (3, 3))


    def test_cortical_surface(self):
        dt = surfaces.CorticalSurface(load_default=True)
        self.assertTrue(isinstance(dt, surfaces.CorticalSurface))
        dt.configure()
        summary_info = dt.summary_info
        self.assertEqual(summary_info['Number of edges'], 49140)
        self.assertEqual(summary_info['Number of triangles'], 32760)
        self.assertEqual(summary_info['Number of vertices'], 16384)
        self.assertEqual(dt.surface_type, surfaces.CORTICAL)
        self.assertEqual(len(dt.vertex_neighbours), 16384)
        self.assertTrue(isinstance(dt.vertex_neighbours[0], frozenset))
        self.assertEqual(len(dt.vertex_triangles), 16384)
        self.assertTrue(isinstance(dt.vertex_triangles[0], frozenset))
        self.assertEqual(len(dt.nth_ring(0)), 17)
        self.assertEqual(dt.triangle_areas.shape, (32760, 1))
        self.assertEqual(dt.triangle_angles.shape, (32760, 3))
        self.assertEqual(len(dt.edges), 49140)
        self.assertTrue(abs(dt.edge_length_mean - 3.97605292887) < 0.00000001)
        self.assertTrue(abs(dt.edge_length_min - 0.6638) < 0.0001)
        self.assertTrue(abs(dt.edge_length_max - 7.7567) < 0.0001)
        self.assertEqual(len(dt.edge_triangles), 49140)
        self.assertEqual([], dt.validate_topology_for_simulations().warnings)
        self.assertEqual(dt.get_data_shape('vertices'), (16384, 3))
        self.assertEqual(dt.get_data_shape('vertex_normals'), (16384, 3))
        self.assertEqual(dt.get_data_shape('triangles'), (32760, 3))
        topologicals = dt.compute_topological_constants()
        self.assertEqual(4, topologicals[0])
        self.assertTrue(all([a.size == 0 for a in topologicals[1:]]))


    def test_cortical_topology_pyramid(self):
        dt = surfaces.Surface()
        dt.vertices = numpy.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(numpy.float64)
        dt.triangles = numpy.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        dt.configure()

        euler, isolated, pinched_off, holes = dt.compute_topological_constants()
        self.assertEqual(2, euler)
        self.assertEqual(0, isolated.size)
        self.assertEqual(0, pinched_off.size)
        self.assertEqual(0, holes.size)


    def test_cortical_topology_isolated_vertex(self):
        dt = surfaces.Surface()
        dt.vertices = numpy.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 2]]).astype(numpy.float64)
        dt.triangles = numpy.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        dt.configure()

        euler, isolated, pinched_off, holes = dt.compute_topological_constants()
        self.assertEqual(3, euler)
        self.assertEqual(1, isolated.size)
        self.assertEqual(0, pinched_off.size)
        self.assertEqual(0, holes.size)


    def test_cortical_topology_pinched(self):
        dt = surfaces.Surface()
        dt.vertices = numpy.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(numpy.float64)
        dt.triangles = numpy.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3], [1, 2, 3]])
        dt.configure()

        euler, isolated, pinched_off, holes = dt.compute_topological_constants()
        self.assertEqual(3, euler)
        self.assertEqual(0, isolated.size)
        self.assertEqual(3, pinched_off.size)
        self.assertEqual(0, holes.size)


    def test_cortical_topology_hole(self):
        dt = surfaces.Surface()
        dt.vertices = numpy.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(numpy.float64)
        dt.triangles = numpy.array([[0, 2, 1], [0, 1, 3], [0, 3, 2]])
        dt.configure()

        euler, isolated, pinched_off, holes = dt.compute_topological_constants()
        self.assertEqual(1, euler)
        self.assertEqual(3, isolated.size)
        self.assertEqual(0, pinched_off.size)
        self.assertEqual(3, holes.size)


    def test_skinair(self):
        dt = surfaces.SkinAir(load_default=True)
        self.assertTrue(isinstance(dt, surfaces.SkinAir))
        self.assertEqual(dt.get_data_shape('vertices'), (4096, 3))
        self.assertEqual(dt.get_data_shape('vertex_normals'), (4096, 3))
        self.assertEqual(dt.get_data_shape('triangles'), (8188, 3))


    def test_brainskull(self):
        dt = surfaces.BrainSkull(load_default=True)
        self.assertTrue(isinstance(dt, surfaces.BrainSkull))
        self.assertEqual(dt.get_data_shape('vertices'), (4096, 3))
        self.assertEqual(dt.get_data_shape('vertex_normals'), (4096, 3))
        self.assertEqual(dt.get_data_shape('triangles'), (8188, 3))


    def test_skullskin(self):
        dt = surfaces.SkullSkin(load_default=True)
        self.assertTrue(isinstance(dt, surfaces.SkullSkin))
        self.assertEqual(dt.get_data_shape('vertices'), (4096, 3))
        self.assertEqual(dt.get_data_shape('vertex_normals'), (4096, 3))
        self.assertEqual(dt.get_data_shape('triangles'), (8188, 3))


    def test_eegcap(self):
        dt = surfaces.EEGCap(load_default=True)
        self.assertTrue(isinstance(dt, surfaces.EEGCap))
        self.assertEqual(dt.get_data_shape('vertices'), (1082, 3))
        self.assertEqual(dt.get_data_shape('vertex_normals'), (1082, 3))
        self.assertEqual(dt.get_data_shape('triangles'), (2160, 3))


    def test_facesurface(self):
        dt = surfaces.FaceSurface(load_default=True)
        self.assertTrue(isinstance(dt, surfaces.FaceSurface))
        self.assertEqual(dt.get_data_shape('vertices'), (8614, 3))
        self.assertEqual(dt.get_data_shape('vertex_normals'), (0,))
        self.assertEqual(dt.get_data_shape('triangles'), (17224, 3))


    def test_regionmapping(self):
        dt = RegionMapping(load_default=True)
        self.assertTrue(isinstance(dt, RegionMapping))
        self.assertEqual(dt.shape, (16384,))


    def test_localconnectivity_empty(self):
        dt = LocalConnectivity()
        self.assertTrue(dt.surface is None)


    @unittest.skipIf(sys.maxsize <= 2147483647, "Cannot deal with local connectivity on a 32-bit machine.")
    def test_cortexdata(self):

        dt = Cortex(load_default=True)
        self.assertTrue(isinstance(dt, Cortex))
        self.assertTrue(dt.region_mapping is not None)
        ## Initialize Local Connectivity, to avoid long computation time.
        dt.local_connectivity = LocalConnectivity(load_default=True)

        dt.configure()
        summary_info = dt.summary_info
        self.assertTrue(abs(summary_info['Region area, maximum (mm:math:`^2`)'] - 9333.39) < 0.01)
        self.assertTrue(abs(summary_info['Region area, mean (mm:math:`^2`)'] - 3038.51) < 0.01)
        self.assertTrue(abs(summary_info['Region area, minimum (mm:math:`^2`)'] - 540.90) < 0.01)
        self.assertEqual(dt.get_data_shape('vertices'), (16384, 3))
        self.assertEqual(dt.get_data_shape('vertex_normals'), (16384, 3))
        self.assertEqual(dt.get_data_shape('triangles'), (32760, 3))



def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(SurfacesTest))
    return test_suite



if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE) 
    
    