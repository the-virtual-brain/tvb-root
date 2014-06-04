# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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

if __name__ == "__main__":
    from tvb.tests.library import setup_test_console_env
    setup_test_console_env()

try:
    import unittest2 as unittest
except Exception:
    import unittest

import sys
import numpy
import tvb.datatypes.surfaces_data as surfaces_data
import tvb.datatypes.surfaces as surfaces
from tvb.tests.library.base_testcase import BaseTestCase



class SurfacesTest(BaseTestCase):
    """
    Tests the defaults for `tvb.datatypes.surfaces` module.
    """

    def test_surface(self):
        dt = surfaces.Surface()
        dt.vertices = numpy.array(range(30)).reshape(10, 3)
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
        self.assertFalse(dt.check()[0])
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
        self.assertEqual(dt.surface_type, surfaces_data.CORTICAL)
        self.assertEqual(len(dt.vertex_neighbours), 16384)
        self.assertTrue(isinstance(dt.vertex_neighbours[0], frozenset))
        self.assertEqual(len(dt.vertex_triangles), 16384)
        self.assertTrue(isinstance(dt.vertex_triangles[0], frozenset))
        self.assertEqual(len(dt.nth_ring(0)), 17)
        self.assertEqual(dt.triangle_areas.shape, (32760, 1))
        self.assertEqual(dt.triangle_angles.shape, (32760, 3))
        self.assertEqual(len(dt.edges), 49140)
        self.assertTrue(abs(dt.edge_length_mean - 3.97605292887) < 0.00000001)
        self.assertTrue(abs(dt.edge_length_min - 0.663807567201) < 0.00000001)
        self.assertTrue(abs(dt.edge_length_max - 7.75671853782) < 0.00000001)
        self.assertEqual(len(dt.edge_triangles), 49140)
        self.assertEqual(dt.check(), (True, 4, [], [], [], ""))
        self.assertEqual(dt.get_data_shape('vertices'), (16384, 3))
        self.assertEqual(dt.get_data_shape('vertex_normals'), (16384, 3))
        self.assertEqual(dt.get_data_shape('triangles'), (32760, 3))


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
        self.assertEqual(dt.get_data_shape('vertices'), (4096, 3))
        self.assertEqual(dt.get_data_shape('vertex_normals'), (4096, 3))
        self.assertEqual(dt.get_data_shape('triangles'), (7062, 3))


    def test_facesurface(self):
        dt = surfaces.FaceSurface(load_default=True)
        self.assertTrue(isinstance(dt, surfaces.FaceSurface))
        self.assertEqual(dt.get_data_shape('vertices'), (35613, 3))
        self.assertEqual(dt.get_data_shape('vertex_normals'), (35613, 3))
        self.assertEqual(dt.get_data_shape('triangles'), (10452, 3))


    def test_regionmapping(self):
        dt = surfaces.RegionMapping(load_default=True)
        self.assertTrue(isinstance(dt, surfaces.RegionMapping))
        self.assertEqual(dt.shape, (16384,))


    def test_localconnectivity_empty(self):
        dt = surfaces.LocalConnectivity()
        self.assertTrue(dt.surface is None)


    @unittest.skipIf(sys.maxsize <= 2147483647, "Cannot deal with local connectivity on a 32-bit machine.")
    def test_cortexdata(self):

        dt = surfaces.Cortex(load_default=True)
        self.assertTrue(isinstance(dt, surfaces.Cortex))
        self.assertTrue(dt.region_mapping is not None)
        ## Initialize Local Connectivity, to avoid long computation time.
        dt.local_connectivity = surfaces.LocalConnectivity(load_default=True)

        dt.configure()
        summary_info = dt.summary_info
        self.assertTrue(abs(summary_info['Region area, maximum (mm:math:`^2`)'] - 9119.4540365252615) < 0.00000001)
        self.assertTrue(abs(summary_info['Region area, mean (mm:math:`^2`)'] - 3366.2542250541251) < 0.00000001)
        self.assertTrue(abs(summary_info['Region area, minimum (mm:math:`^2`)'] - 366.48271886512993) < 0.00000001)
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
    
    