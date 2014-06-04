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
module docstring
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

import unittest
import os.path
import tvb_data
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.services.exceptions import OperationException
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.graph import ConnectivityMeasure
from tvb.tests.framework.core.test_factory import TestFactory
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.core.entities.storage import dao
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.services.flow_service import FlowService
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.tests.framework.adapters.uploaders import test_data


class ConnectivityMeasureImporterTest(TransactionalTestCase):
    """
    Unit-tests for ConnectivityMeasureImporter
    """

    def setUp(self):
        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_66.zip')
        self.test_user = TestFactory.create_user('Test_User')
        self.test_project = TestFactory.create_project(self.test_user, "Test_Project")
        TestFactory.import_zip_connectivity(self.test_user,self.test_project, "John", zip_path)
        self.connectivity = TestFactory.get_entity(self.test_project, Connectivity())


    def tearDown(self):
        FilesHelper().remove_project_structure(self.test_project.name)


    def _import(self, import_file_name):
        ### Retrieve Adapter instance
        group = dao.find_group('tvb.adapters.uploaders.connectivity_measure_importer', 'ConnectivityMeasureImporter')
        importer = ABCAdapter.build_adapter(group)
        path = os.path.join(os.path.dirname(test_data.__file__), import_file_name)

        args = {'data_file': path,
                'connectivity' : self.connectivity.gid,
                DataTypeMetaData.KEY_SUBJECT: "John"
        }

        ### Launch import Operation
        FlowService().fire_operation(importer, self.test_user, self.test_project.id, **args)


    def test_happy_flow(self):
        self.assertEqual(0, TestFactory.get_entity_count(self.test_project, ConnectivityMeasure()))
        self._import('mantini_networks.mat')
        self.assertEqual(6, TestFactory.get_entity_count(self.test_project, ConnectivityMeasure()))

    def test_connectivity_mismatch(self):
        self.assertRaises(OperationException, self._import, 'mantini_networks_33.mat')



def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(ConnectivityMeasureImporterTest))
    return test_suite


if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)