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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import unittest
from tvb.tests.framework.interfaces.web.controllers.base_controller_test import BaseTransactionalControllerTest
from tvb.tests.framework.datatypes.datatypes_factory import DatatypesFactory
from tvb.interfaces.web.controllers.common import get_from_session
from tvb.interfaces.web.controllers.spatial.surface_stimulus_controller import SurfaceStimulusController
from tvb.interfaces.web.controllers.spatial.surface_stimulus_controller import KEY_SURFACE_CONTEXT
from tvb.core.entities.transient.context_stimulus import SURFACE_PARAMETER



class SurfaceStimulusControllerTest(BaseTransactionalControllerTest):
    """ Unit tests for SurfaceStimulusController """
    
    def setUp(self):
        self.init()
        self.surface_s_c = SurfaceStimulusController()


    def tearDown(self):
        """ Cleans the testing environment """
        self.cleanup()


    def test_step_1(self):
        self.surface_s_c.step_1_submit(1, 1)
        result_dict = self.surface_s_c.step_1()
        expected_keys = ['temporalPlotInputList', 'temporalFieldsPrefixes', 'temporalEquationViewerUrl',
                         'spatialPlotInputList', 'spatialFieldsPrefixes', 'spatialEquationViewerUrl',
                         'selectedFocalPoints', 'mainContent', 'existentEntitiesInputList']
        map(lambda x: self.assertTrue(x in result_dict), expected_keys)
        self.assertEqual(result_dict['mainContent'], 'spatial/stimulus_surface_step1_main')
        self.assertEqual(result_dict['next_step_url'], '/spatial/stimulus/surface/step_1_submit')
        
     
    def test_step_2(self):
        _, surface = DatatypesFactory().create_surface()
        self.surface_s_c.step_1_submit(1, 1)
        context = get_from_session(KEY_SURFACE_CONTEXT)
        context.equation_kwargs[SURFACE_PARAMETER] = surface.gid
        result_dict = self.surface_s_c.step_2()
        expected_keys = ['urlVerticesPick', 'urlVertices', 'urlTrianglesPick', 'urlTriangles',
                         'urlNormalsPick', 'urlNormals', 'surfaceGID', 'mainContent', 
                         'loadExistentEntityUrl', 'existentEntitiesInputList', 'definedFocalPoints']
        map(lambda x: self.assertTrue(x in result_dict), expected_keys)
        self.assertEqual(result_dict['next_step_url'], '/spatial/stimulus/surface/step_2_submit')
        self.assertEqual(result_dict['mainContent'], 'spatial/stimulus_surface_step2_main')
        self.assertEqual(result_dict['loadExistentEntityUrl'], '/spatial/stimulus/surface/load_surface_stimulus')


def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(SurfaceStimulusControllerTest))
    return test_suite


if __name__ == "__main__":
    #So you can run tests individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)