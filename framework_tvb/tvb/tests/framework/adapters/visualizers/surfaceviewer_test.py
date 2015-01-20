# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import unittest
from tvb.adapters.visualizers.surface_view import SurfaceViewer, RegionMappingViewer
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.datatypes.surfaces import CorticalSurface, RegionMapping
from tvb.tests.framework.core.test_factory import TestFactory
from tvb.tests.framework.core.base_testcase import TransactionalTestCase


class SurfaceViewersTest(TransactionalTestCase):
    """
    Unit-tests for Surface & RegionMapping viewers.
    """

    EXPECTED_KEYS = {'urlVertices': None, 'urlTriangles': None, 'urlLines': None, 'urlNormals': None,
                     'urlRegionMap': None, 'biHemispheric': False, 'hemisphereChunkMask': None,
                     'noOfMeasurePoints': 74, 'urlMeasurePoints': None, 'boundaryURL': None, 'minMeasure': 0,
                     'maxMeasure': 74, 'clientMeasureUrl': None}

    def setUp(self):
        """
        Sets up the environment for running the tests;
        creates a test user, a test project, a connectivity and a surface;
        imports a CFF data-set
        """
        self.test_user = TestFactory.create_user()
        self.test_project = TestFactory.import_default_project(self.test_user)

        self.surface = TestFactory.get_entity(self.test_project, CorticalSurface())
        self.assertTrue(self.surface is not None)

        self.region_mapping = TestFactory.get_entity(self.test_project, RegionMapping())
        self.assertTrue(self.region_mapping is not None)


    def tearDown(self):
        """
        Clean-up tests data
        """
        FilesHelper().remove_project_structure(self.test_project.name)


    def test_launch_surface(self):
        """
        Check that all required keys are present in output from SurfaceViewer launch.
        """
        viewer = SurfaceViewer()
        viewer.current_project_id = self.test_project.id
        result = viewer.launch(self.surface, self.region_mapping)

        self.assert_compliant_dictionary(self.EXPECTED_KEYS, result)


    def test_launch_region(self):
        """
        Check that all required keys are present in output from RegionMappingViewer launch.
        """
        viewer = RegionMappingViewer()
        viewer.current_project_id = self.test_project.id
        result = viewer.launch(self.region_mapping)

        self.assert_compliant_dictionary(self.EXPECTED_KEYS, result)



def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(SurfaceViewersTest))
    return test_suite


if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)