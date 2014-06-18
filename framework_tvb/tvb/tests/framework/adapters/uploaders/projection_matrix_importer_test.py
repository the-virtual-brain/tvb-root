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
import tvb_data.projectionMatrix as dataset
import tvb_data.sensors as sensors_dataset
from tvb.datatypes.projections import ProjectionRegionEEG, ProjectionSurfaceEEG
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.sensors import SensorsEEG
from tvb.datatypes.surfaces import CorticalSurface
from tvb.core.entities.storage import dao
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.services.flow_service import FlowService
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.tests.framework.core.test_factory import TestFactory
from tvb.tests.framework.core.base_testcase import TransactionalTestCase


class ProjectionMatrixTest(TransactionalTestCase):
    """
    Unit-tests for CFF-importer.
    """
    
    def setUp(self):
        """
        Reset the database before each test.
        """
        self.test_user = TestFactory.create_user('CFF_User')
        self.test_project = TestFactory.create_project(self.test_user, "CFF_Project")
        TestFactory.import_cff(test_user=self.test_user, test_project=self.test_project)
        zip_path = os.path.join(os.path.dirname(sensors_dataset.__file__), 
                                'EEG_unit_vectors_BrainProducts_62.txt.bz2')
        TestFactory.import_sensors(self.test_user, self.test_project, zip_path, 'EEG Sensors')
        self.connectivity = TestFactory.get_entity(self.test_project, Connectivity())
        self.assertTrue(self.connectivity is not None)
        self.surface = TestFactory.get_entity(self.test_project, CorticalSurface())
        self.assertTrue(self.surface is not None)
        self.sensors = TestFactory.get_entity(self.test_project, SensorsEEG())
        self.assertTrue(self.sensors is not None)
        
    
    def test_happy_flow_region_import(self):
        """
        Verifies the happy flow for importing a region.
        """
        dt_count_before = TestFactory.get_entity_count(self.test_project, ProjectionRegionEEG())
        group = dao.find_group('tvb.adapters.uploaders.projection_matrix_importer', 
                               'ProjectionMatrixRegionEEGImporter')
        importer = ABCAdapter.build_adapter(group)
        zip_path = os.path.join(os.path.abspath(os.path.dirname(dataset.__file__)), 
                                'region_conn_74_eeg_1020_62.mat')
        args = {'projection_file': zip_path, 'dataset_name': 'ProjectionMatrix',
                'connectivity': self.connectivity.gid, 'sensors': self.sensors.gid,
                DataTypeMetaData.KEY_SUBJECT: DataTypeMetaData.DEFAULT_SUBJECT}
        FlowService().fire_operation(importer, self.test_user, self.test_project.id, **args)
        dt_count_after = TestFactory.get_entity_count(self.test_project, ProjectionRegionEEG())
        self.assertTrue(dt_count_after == dt_count_before + 1)
        
        
    def test_happy_flow_surface_import(self):
        """
        Verifies the happy flow for importing a surface.
        """
        dt_count_before = TestFactory.get_entity_count(self.test_project, ProjectionSurfaceEEG())
        group = dao.find_group('tvb.adapters.uploaders.projection_matrix_importer', 
                               'ProjectionMatrixSurfaceEEGImporter')
        importer = ABCAdapter.build_adapter(group)

        zip_path = os.path.join(os.path.abspath(os.path.dirname(dataset.__file__)), 
                                'surface_reg_13_eeg_62.mat')
        args = {'projection_file': zip_path, 'dataset_name': 'ProjectionMatrix',
                'connectivity': self.connectivity.gid, 'sensors': self.sensors.gid,
                'surface': self.surface.gid,
                DataTypeMetaData.KEY_SUBJECT: DataTypeMetaData.DEFAULT_SUBJECT}
        FlowService().fire_operation(importer, self.test_user, self.test_project.id, **args)
        dt_count_after = TestFactory.get_entity_count(self.test_project, ProjectionRegionEEG())
        self.assertTrue(dt_count_after == dt_count_before + 1)
    
        
def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(ProjectionMatrixTest))
    return test_suite


if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)
    
    
    
       
        
        
        
    