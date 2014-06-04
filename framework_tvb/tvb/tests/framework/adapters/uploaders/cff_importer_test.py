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
import os
import unittest
import tvb_data.cff as dataset
from tvb.core.services.flow_service import FlowService
from tvb.core.entities.storage import dao
from tvb.tests.framework.core.test_factory import TestFactory
from tvb.tests.framework.core.base_testcase import TransactionalTestCase


class CFFUploadTest(TransactionalTestCase):
    """
    Unit-tests for CFF-importer.
    """
    INVALID_CFF = ''
    VALID_CFF = os.path.join(os.path.dirname(dataset.__file__), 'dataset_74.cff')
    
    def setUp(self):
        """
        Reset the database before each test.
        """
        self.test_user = TestFactory.create_user('CFF_User')
        self.test_project = TestFactory.create_project(self.test_user, "CFF_Project")
        
    
    def test_happy_flow_import(self):
        """
        Test that importing a CFF generates at least one DataType in DB.
        """
        all_dt = self.get_all_datatypes()
        self.assertEqual(0, len(all_dt))
        TestFactory.import_cff(cff_path=self.VALID_CFF)
        all_dt = self.get_all_datatypes()
        self.assertTrue(0 < len(all_dt))
    
        
    def test_full_import(self):
        """
        Test that importing a CFF generates at least one DataType in DB.
        """
        all_dt = self.get_all_datatypes()
        self.assertEqual(0, len(all_dt))
        TestFactory.import_cff(cff_path=self.VALID_CFF, test_user=self.test_user, test_project=self.test_project)
        flow_service = FlowService()
        ### Check that at one Connectivity was persisted
        count = flow_service.get_available_datatypes(self.test_project.id, 'tvb.datatypes.connectivity.Connectivity')[1]
        self.assertEquals(count, 1)
        ### Check that at one RegionMapping was persisted
        count = flow_service.get_available_datatypes(self.test_project.id, 'tvb.datatypes.surfaces.RegionMapping')[1]
        self.assertEquals(count, 1)
        ### Check that at one LocalConnectivity was persisted
        gids = flow_service.get_available_datatypes(self.test_project.id, 'tvb.datatypes.surfaces.LocalConnectivity')[0]
        self.assertEquals(len(gids), 1)
        connectivity = dao.get_datatype_by_gid(gids[0][2])
        metadata = connectivity.get_metadata()
        self.assertEqual(metadata['Cutoff'], '40.0')
        self.assertEqual(metadata['Equation'], 'null')
        self.assertFalse(metadata['Invalid'])
        self.assertFalse(metadata['Is_nan'])
        self.assertEqual(metadata['Type'], 'LocalConnectivity')
        ### Check that at 2 Surfaces were persisted
        count = flow_service.get_available_datatypes(self.test_project.id, 'tvb.datatypes.surfaces_data.SurfaceData')[1]
        self.assertEquals(count, 2)
        
     
     
     
def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(CFFUploadTest))
    return test_suite


if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)
    
    
    
       
        
        
        
    