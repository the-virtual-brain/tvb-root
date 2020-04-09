# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""
import os
import tvb_data
import shutil
import pytest
import numpy
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.basic.profile import TvbProfile
from tvb.core.entities.storage import dao
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.services.import_service import ImportService
from tvb.core.services.flow_service import FlowService
from tvb.core.services.project_service import ProjectService
from tvb.core.services.operation_service import OperationService
from tvb.core.services.exceptions import ProjectImportException
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.adapters.exporters.export_manager import ExportManager
from tvb.datatypes.time_series import TimeSeries
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.adapters.storeadapter import StoreAdapter



class TestImportService(TransactionalTestCase):
    """
    This class contains tests for the tvb.core.services.import_service module.
    """  
    
    def transactional_setup_method(self):
        """
        Reset the database before each test.
        """
        self.import_service = ImportService()
        self.flow_service = FlowService()
        self.project_service = ProjectService()
        
        self.test_user = TestFactory.create_user()
        self.test_project = TestFactory.create_project(self.test_user, name="GeneratedProject", description="test_desc")
        self.operation = TestFactory.create_operation(test_user=self.test_user, test_project=self.test_project)
        self.adapter_instance = TestFactory.create_adapter()
        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_66.zip')
        TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path)
        self.zip_path = None 
        

    def transactional_teardown_method(self):
        """
        Reset the database when test is done.
        """
        ### Delete TEMP folder
        if os.path.exists(TvbProfile.current.TVB_TEMP_FOLDER):
            shutil.rmtree(TvbProfile.current.TVB_TEMP_FOLDER)
        
        ### Delete folder where data was exported
        if self.zip_path and os.path.exists(self.zip_path):
            shutil.rmtree(os.path.split(self.zip_path)[0])
            
        self.delete_project_folders()

            
    def test_import_export(self):
        """
        Test the import/export mechanism for a project structure.
        The project contains the following data types: Connectivity, Surface, MappedArray and ValueWrapper.
        """
        result = self.get_all_datatypes()
        expected_results = {}
        for one_data in result:
            expected_results[one_data.gid] = (one_data.module, one_data.type)
        
        #create an array mapped in DB
        data = {'param_1': 'some value'}
        OperationService().initiate_prelaunch(self.operation, self.adapter_instance, {}, **data)
        inserted = self.flow_service.get_available_datatypes(self.test_project.id,
                                                             "tvb.datatypes.arrays.MappedArray")[1]
        assert 1 == inserted, "Problems when inserting data"
        
        #create a value wrapper
        value_wrapper = self._create_value_wrapper()
        count_operations = dao.get_filtered_operations(self.test_project.id, None, is_count=True)
        assert 2 == count_operations, "Invalid ops number before export!"

        # Export project as ZIP
        self.zip_path = ExportManager().export_project(self.test_project)
        assert self.zip_path is not None, "Exported file is none"
        
        # Remove the original project
        self.project_service.remove_project(self.test_project.id)
        result, lng_ = self.project_service.retrieve_projects_for_user(self.test_user.id)
        assert 0 == len(result), "Project Not removed!"
        assert 0 == lng_, "Project Not removed!"
        
        # Now try to import again project
        self.import_service.import_project_structure(self.zip_path, self.test_user.id)
        result = self.project_service.retrieve_projects_for_user(self.test_user.id)[0]
        assert len(result) == 1, "There should be only one project."
        assert result[0].name == "GeneratedProject", "The project name is not correct."
        assert result[0].description == "test_desc", "The project description is not correct."
        self.test_project = result[0]
        
        count_operations = dao.get_filtered_operations(self.test_project.id, None, is_count=True)
        
        #1 op. - import cff; 2 op. - save the array wrapper;
        assert 2 == count_operations, "Invalid ops number after export and import !"
        for gid in expected_results:
            datatype = dao.get_datatype_by_gid(gid)
            assert datatype.module == expected_results[gid][0], 'DataTypes not imported correctly'
            assert datatype.type == expected_results[gid][1], 'DataTypes not imported correctly'
        #check the value wrapper
        new_val = self.flow_service.get_available_datatypes(self.test_project.id, 
                                                            "tvb.datatypes.mapped_values.ValueWrapper")[0]
        assert 1 == len(new_val), "One !=" + str(len(new_val))
        new_val = ABCAdapter.load_entity_by_gid(new_val[0][2])
        assert value_wrapper.data_value == new_val.data_value, "Data value incorrect"
        assert value_wrapper.data_type == new_val.data_type, "Data type incorrect"
        assert value_wrapper.data_name == new_val.data_name, "Data name incorrect"
        

    def test_import_export_existing(self):
        """
        Test the import/export mechanism for a project structure.
        The project contains the following data types: Connectivity, Surface, MappedArray and ValueWrapper.
        """
        count_operations = dao.get_filtered_operations(self.test_project.id, None, is_count=True)
        assert 2 == count_operations, "Invalid ops before export!"

        self.zip_path = ExportManager().export_project(self.test_project)
        assert self.zip_path is not None, "Exported file is none"

        with pytest.raises(ProjectImportException):
            self.import_service.import_project_structure(self.zip_path, self.test_user.id)


    def _create_timeseries(self):
        """Launch adapter to persist a TimeSeries entity"""
        activity_data = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        time_data = numpy.array([1, 2, 3])
        storage_path = FilesHelper().get_project_folder(self.test_project)
        time_series = TimeSeries(time_files=None, activity_files=None, 
                                 max_chunk=10, maxes=None, mins=None, data_shape=numpy.shape(activity_data), 
                                 storage_path=storage_path, label_y="Time", time_data=time_data, data_name='TestSeries',
                                 activity_data=activity_data, sample_period=10.0)
        self._store_entity(time_series, "TimeSeries", "tvb.datatypes.time_series")
        timeseries_count = self.flow_service.get_available_datatypes(self.test_project.id,
                                                                     "tvb.datatypes.time_series.TimeSeries")[1]
        assert timeseries_count == 1, "Should be only one TimeSeries"


    def _create_value_wrapper(self):
        """Persist ValueWrapper"""
        value_ = ValueWrapper(data_value=5.0, data_name="my_value")
        self._store_entity(value_, "ValueWrapper", "tvb.datatypes.mapped_values")
        valuew = self.flow_service.get_available_datatypes(self.test_project.id,
                                                           "tvb.datatypes.mapped_values.ValueWrapper")[0]
        assert len(valuew) == 1, "Should be only one value wrapper"
        return ABCAdapter.load_entity_by_gid(valuew[0][2])


    def _store_entity(self, entity, type_, module):
        """Launch adapter to store a create a persistent DataType."""
        entity.type = type_
        entity.module = module
        entity.subject = "John Doe"
        entity.state = "RAW_STATE"
        entity.set_operation_id(self.operation.id)
        adapter_instance = StoreAdapter([entity])
        OperationService().initiate_prelaunch(self.operation, adapter_instance, {})



