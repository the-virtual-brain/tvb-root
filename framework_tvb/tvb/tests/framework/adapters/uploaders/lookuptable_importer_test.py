# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
import tvb_data.tables as dataset
from tvb.adapters.uploaders.lookup_table_importer import LookupTableImporter
from tvb.datatypes.lookup_tables import NerfTable, PsiTable
from tvb.tests.framework.core.test_factory import TestFactory
from tvb.tests.framework.core.base_testcase import TransactionalTestCase


class LookupTableImporterTest(TransactionalTestCase):
    """
    Unit-tests for CFF-importer.
    """
    
    def setUp(self):
        """
        Reset the database before each test.
        """
        self.test_user = TestFactory.create_user('Tables_User')
        self.test_project = TestFactory.create_project(self.test_user, "Tables_Project")


    def tearDown(self):
        """
        Remove files left after tests.
        """
        for file_ in os.listdir("."):
            if os.path.isfile(file_) and (file_.startswith("NerfTable_") or
                                          file_.startswith("PsiTable_")) and file_.endswith(".h5"):
                os.remove(file_)


    def test_psi_table_import(self):
        """
        Test that importing a CFF generates one DataType.
        """
        zip_path = os.path.join(os.path.abspath(os.path.dirname(dataset.__file__)), 'psi.npz')

        importer = LookupTableImporter()
        result = importer.launch(zip_path, LookupTableImporter.PSI_TABLE)[0]

        self.assertTrue(isinstance(result, PsiTable))
        self.assertIsNotNone(result.data)
        self.assertEqual(0.0, result.xmin)
        self.assertEqual(0.3, result.xmax)
        
        
    def test_nerf_table_import(self):
        """
        Test that importing a CFF generates a valid DataType.
        """
        zip_path = os.path.join(os.path.abspath(os.path.dirname(dataset.__file__)), 'nerf_int.npz')

        importer = LookupTableImporter()
        result = importer.launch(zip_path, LookupTableImporter.NERF_TABLE)[0]

        self.assertTrue(isinstance(result, NerfTable))
        self.assertIsNotNone(result.data)
        self.assertEqual(20, result.xmax)
        self.assertEqual(-10, result.xmin)

    
        
def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(LookupTableImporterTest))
    return test_suite


if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)
    
    
