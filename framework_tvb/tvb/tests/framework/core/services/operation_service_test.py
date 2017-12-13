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

import pytest
import json
import numpy
from tvb.tests.framework.core.base_testcase import BaseTestCase
from tvb.basic.profile import TvbProfile
from tvb.core.utils import string2array
from tvb.core.entities import model
from tvb.core.entities.storage import dao
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.services.operation_service import OperationService
from tvb.core.services.project_service import initialize_storage, ProjectService
from tvb.core.services.flow_service import FlowService
from tvb.tests.framework.datatypes.datatype1 import Datatype1
from tvb.tests.framework.datatypes.datatype2 import Datatype2
from tvb.tests.framework.adapters.ndimensionarrayadapter import NDimensionArrayAdapter
from tvb.tests.framework.core.factory import TestFactory
from tvb.core.adapters.exceptions import NoMemoryAvailableException



class TestOperationService(BaseTestCase):
    """
    Test class for the introspection module. Some tests from here do async launches. For those
    cases Transactional tests won't work.
    TODO: this is still to be refactored, for being huge, with duplicates and many irrelevant checks
    """


    def setup_method(self):
        """
        Reset the database before each test.
        """
        self.clean_database()
        initialize_storage()
        self.test_user = TestFactory.create_user()
        self.test_project = TestFactory.create_project(self.test_user)
        self.operation_service = OperationService()
        self.backup_hdd_size = TvbProfile.current.MAX_DISK_SPACE


    def teardown_method(self):
        """
        Reset the database when test is done.
        """
        TvbProfile.current.MAX_DISK_SPACE = self.backup_hdd_size
        self.clean_database()


    def _assert_no_dt2(self):
        count = dao.count_datatypes(self.test_project.id, Datatype2)
        assert 0 == count


    def _assert_stored_dt2(self, expected_cnt=1):
        count = dao.count_datatypes(self.test_project.id, Datatype2)
        assert expected_cnt == count
        datatype = dao.try_load_last_entity_of_type(self.test_project.id, Datatype2)
        assert datatype.subject == DataTypeMetaData.DEFAULT_SUBJECT, "Wrong data stored."
        return datatype


    def test_datatypes_groups(self):
        """
        Tests if the dataType group is set correct on the dataTypes resulted from the same operation group.
        """
        flow_service = FlowService()

        all_operations = dao.get_filtered_operations(self.test_project.id, None)
        assert len(all_operations) == 0, "There should be no operation"

        adapter_instance = TestFactory.create_adapter('tvb.tests.framework.adapters.testadapter3', 'TestAdapter3')
        data = {model.RANGE_PARAMETER_1: 'param_5', 'param_5': [1, 2]}
        ## Create Group of operations
        flow_service.fire_operation(adapter_instance, self.test_user, self.test_project.id, **data)

        all_operations = dao.get_filtered_operations(self.test_project.id, None)
        assert len(all_operations) == 1, "Expected one operation group"
        assert all_operations[0][2] == 2, "Expected 2 operations in group"

        operation_group_id = all_operations[0][3]
        assert operation_group_id != None, "The operation should be part of a group."

        self.operation_service.stop_operation(all_operations[0][0])
        self.operation_service.stop_operation(all_operations[0][1])
        ## Make sure operations are executed
        self.operation_service.launch_operation(all_operations[0][0], False)
        self.operation_service.launch_operation(all_operations[0][1], False)

        resulted_datatypes = dao.get_datatype_in_group(operation_group_id=operation_group_id)
        assert len(resulted_datatypes) >= 2, "Expected at least 2, but: " + str(len(resulted_datatypes))

        dt = dao.get_datatype_by_id(resulted_datatypes[0].id)
        datatype_group = dao.get_datatypegroup_by_op_group_id(operation_group_id)
        assert dt.fk_datatype_group == datatype_group.id, "DataTypeGroup is incorrect"


    def test_initiate_operation(self):
        """
        Test the actual operation flow by executing a test adapter.
        """
        module = "tvb.tests.framework.adapters.testadapter1"
        class_name = "TestAdapter1"
        adapter = TestFactory.create_adapter(module, class_name)
        output = adapter.get_output()
        output_type = output[0].__name__
        data = {"test1_val1": 5, "test1_val2": 5}
        tmp_folder = FilesHelper().get_project_folder(self.test_project, "TEMP")
        res = self.operation_service.initiate_operation(self.test_user, self.test_project.id, adapter,
                                                        tmp_folder, **data)
        assert res.index("has finished.") > 10, "Operation didn't finish"
        group = dao.get_algorithm_by_module(module, class_name)
        assert group.module == 'tvb.tests.framework.adapters.testadapter1', "Wrong data stored."
        assert group.classname == 'TestAdapter1', "Wrong data stored."
        dts, count = dao.get_values_of_datatype(self.test_project.id, Datatype1)
        assert count == 1
        assert len(dts) == 1
        datatype = dao.get_datatype_by_id(dts[0][0])
        assert datatype.subject == DataTypeMetaData.DEFAULT_SUBJECT, "Wrong data stored."
        assert datatype.type == output_type, "Wrong data stored."


    def test_delete_dt_free_HDD_space(self):
        """
        Launch two operations and give enough available space for user so that both should finish.
        """
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.testadapter3", "TestAdapterHDDRequired")
        data = {"test": 100}
        TvbProfile.current.MAX_DISK_SPACE = float(adapter.get_required_disk_size(**data))
        tmp_folder = FilesHelper().get_project_folder(self.test_project, "TEMP")

        self._assert_no_dt2()
        self.operation_service.initiate_operation(self.test_user, self.test_project.id, adapter, tmp_folder, **data)
        datatype = self._assert_stored_dt2()

        # Now free some space and relaunch
        ProjectService().remove_datatype(self.test_project.id, datatype.gid)
        self._assert_no_dt2()
        self.operation_service.initiate_operation(self.test_user, self.test_project.id, adapter, tmp_folder, **data)
        self._assert_stored_dt2()


    def test_launch_two_ops_HDD_with_space(self):
        """
        Launch two operations and give enough available space for user so that both should finish.
        """
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.testadapter3", "TestAdapterHDDRequired")
        data = {"test": 100}
        TvbProfile.current.MAX_DISK_SPACE = 2 * float(adapter.get_required_disk_size(**data))
        tmp_folder = FilesHelper().get_project_folder(self.test_project, "TEMP")

        self.operation_service.initiate_operation(self.test_user, self.test_project.id, adapter, tmp_folder, **data)
        datatype = self._assert_stored_dt2()

        #Now update the maximum disk size to be the size of the previously resulted datatypes (transform from kB to MB)
        #plus what is estimated to be required from the next one (transform from B to MB)
        TvbProfile.current.MAX_DISK_SPACE = float(datatype.disk_size) + float(adapter.get_required_disk_size(**data))

        self.operation_service.initiate_operation(self.test_user, self.test_project.id, adapter, tmp_folder, **data)
        self._assert_stored_dt2(2)


    def test_launch_two_ops_HDD_full_space(self):
        """
        Launch two operations and give available space for user so that the first should finish,
        but after the update to the user hdd size the second should not.
        """
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.testadapter3", "TestAdapterHDDRequired")
        data = {"test": 100}

        TvbProfile.current.MAX_DISK_SPACE = (1 + float(adapter.get_required_disk_size(**data)))
        tmp_folder = FilesHelper().get_project_folder(self.test_project, "TEMP")
        self.operation_service.initiate_operation(self.test_user, self.test_project.id, adapter, tmp_folder, **data)

        datatype = self._assert_stored_dt2()
        #Now update the maximum disk size to be less than size of the previously resulted datatypes (transform kB to MB)
        #plus what is estimated to be required from the next one (transform from B to MB)
        TvbProfile.current.MAX_DISK_SPACE = float(datatype.disk_size - 1) + \
                                            float(adapter.get_required_disk_size(**data) - 1)

        with pytest.raises(NoMemoryAvailableException):
            self.operation_service.initiate_operation(self.test_user,self.test_project.id, adapter, tmp_folder, **data)
        self._assert_stored_dt2()


    def test_launch_operation_HDD_with_space(self):
        """
        Test the actual operation flow by executing a test adapter.
        """
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.testadapter3", "TestAdapterHDDRequired")
        data = {"test": 100}

        TvbProfile.current.MAX_DISK_SPACE = float(adapter.get_required_disk_size(**data))
        tmp_folder = FilesHelper().get_project_folder(self.test_project, "TEMP")
        self.operation_service.initiate_operation(self.test_user, self.test_project.id, adapter, tmp_folder, **data)
        self._assert_stored_dt2()


    def test_launch_operation_HDD_with_space_started_ops(self):
        """
        Test the actual operation flow by executing a test adapter.
        """
        space_taken_by_started = 100
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.testadapter3", "TestAdapterHDDRequired")
        started_operation = model.Operation(self.test_user.id, self.test_project.id, adapter.stored_adapter.id, "",
                                            status=model.STATUS_STARTED, estimated_disk_size=space_taken_by_started)
        dao.store_entity(started_operation)
        data = {"test": 100}
        TvbProfile.current.MAX_DISK_SPACE = float(adapter.get_required_disk_size(**data) + space_taken_by_started)
        tmp_folder = FilesHelper().get_project_folder(self.test_project, "TEMP")
        self.operation_service.initiate_operation(self.test_user, self.test_project.id, adapter, tmp_folder, **data)
        self._assert_stored_dt2()


    def test_launch_operation_HDD_full_space(self):
        """
        Test the actual operation flow by executing a test adapter.
        """
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.testadapter3", "TestAdapterHDDRequired")
        data = {"test": 100}
        TvbProfile.current.MAX_DISK_SPACE = float(adapter.get_required_disk_size(**data) - 1)
        tmp_folder = FilesHelper().get_project_folder(self.test_project, "TEMP")
        with pytest.raises(NoMemoryAvailableException):
            self.operation_service.initiate_operation(self.test_user, self.test_project.id, adapter, tmp_folder, **data)
        self._assert_no_dt2()


    def test_launch_operation_HDD_full_space_started_ops(self):
        """
        Test the actual operation flow by executing a test adapter.
        """
        space_taken_by_started = 100
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.testadapter3", "TestAdapterHDDRequired")
        started_operation = model.Operation(self.test_user.id, self.test_project.id, adapter.stored_adapter.id, "",
                                            status=model.STATUS_STARTED, estimated_disk_size=space_taken_by_started)
        dao.store_entity(started_operation)
        data = {"test": 100}
        TvbProfile.current.MAX_DISK_SPACE = float(adapter.get_required_disk_size(**data) + space_taken_by_started - 1)
        tmp_folder = FilesHelper().get_project_folder(self.test_project, "TEMP")
        with pytest.raises(NoMemoryAvailableException):
            self.operation_service.initiate_operation(self.test_user,self.test_project.id, adapter, tmp_folder, **data)
        self._assert_no_dt2()


    def test_stop_operation(self):
        """
        Test that an operation is successfully stopped.
        """
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.testadapter2", "TestAdapter2")
        data = {"test": 5}
        algo = adapter.stored_adapter
        algo_category = dao.get_category_by_id(algo.fk_category)
        operations, _ = self.operation_service.prepare_operations(self.test_user.id, self.test_project.id, algo,
                                                                  algo_category, {}, **data)
        self.operation_service._send_to_cluster(operations, adapter)
        self.operation_service.stop_operation(operations[0].id)
        operation = dao.get_operation_by_id(operations[0].id)
        assert operation.status, model.STATUS_CANCELED == "Operation should have been canceled!"


    def test_stop_operation_finished(self):
        """
        Test that an operation that is already finished is not changed by the stop operation.
        """
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.testadapter1", "TestAdapter1")
        data = {"test1_val1": 5, 'test1_val2': 5}
        algo = adapter.stored_adapter
        algo_category = dao.get_category_by_id(algo.fk_category)
        operations, _ = self.operation_service.prepare_operations(self.test_user.id, self.test_project.id, algo,
                                                                  algo_category, {}, **data)
        self.operation_service._send_to_cluster(operations, adapter)
        operation = dao.get_operation_by_id(operations[0].id)
        operation.status = model.STATUS_FINISHED
        dao.store_entity(operation)
        self.operation_service.stop_operation(operations[0].id)
        operation = dao.get_operation_by_id(operations[0].id)
        assert operation.status, model.STATUS_FINISHED == "Operation shouldn't have been canceled!"


    def test_array_from_string(self):
        """
        Simple test for parse array on 1d, 2d and 3d array.
        """
        row = {'description': 'test.',
               'default': 'None',
               'required': True,
               'label': 'test: ',
               'attributes': None,
               'elementType': 'float',
               'type': 'array',
               'options': None,
               'name': 'test'}
        input_data_string = '[ [1 2 3] [4 5 6]]'
        output = string2array(input_data_string, ' ', row['elementType'])
        assert output.shape, (2, 3) == "Dimensions not properly parsed"
        for i in output[0]:
            assert i in [1, 2, 3]
        for i in output[1]:
            assert i in [4, 5, 6]
        input_data_string = '[1, 2, 3, 4, 5, 6]'
        output = string2array(input_data_string, ',', row['elementType'])
        assert output.shape == (6,), "Dimensions not properly parsed"
        for i in output:
            assert i in [1, 2, 3, 4, 5, 6]
        input_data_string = '[ [ [1,1], [2, 2] ], [ [3 ,3], [4,4] ] ]'
        output = string2array(input_data_string, ',', row['elementType'])
        assert output.shape == (2, 2, 2), "Wrong dimensions."
        for i in output[0][0]:
            assert i == 1
        for i in output[0][1]:
            assert i == 2
        for i in output[1][0]:
            assert i == 3
        for i in output[1][1]:
            assert i == 4
        row = {'description': 'test.',
               'default': 'None',
               'required': True,
               'label': 'test: ',
               'attributes': None,
               'elementType': 'str',
               'type': 'array',
               'options': None,
               'name': 'test'}
        input_data_string = '[1, 2, 3, 4, 5, 6]'
        output = string2array(input_data_string, ',', row['elementType'])
        for i in output:
            assert i in [1, 2, 3, 4, 5, 6]


    def test_wrong_array_from_string(self):
        """Test that parsing an array from string is throwing the expected 
        exception when wrong input string"""
        row = {'description': 'test.',
               'default': 'None',
               'required': True,
               'label': 'test: ',
               'attributes': None,
               'elementType': 'float',
               'type': 'array',
               'options': None,
               'name': 'test'}
        input_data_string = '[ [1,2 3] [4,5,6]]'
        with pytest.raises(ValueError):
            string2array(input_data_string, ',', row['elementType'])
        input_data_string = '[ [1,2,wrong], [4, 5, 6]]'
        with pytest.raises(ValueError):
            string2array(input_data_string, ',', row['elementType'])
        row = {'description': 'test.',
               'default': 'None',
               'required': True,
               'label': 'test: ',
               'attributes': None,
               'elementType': 'str',
               'type': 'array',
               'options': None,
               'name': 'test'}
        output = string2array(input_data_string, ',', row['elementType'])
        assert output.shape == (2, 3)
        assert output[0][2] == 'wrong', 'String data not converted properly'
        input_data_string = '[ [1,2 3] [4,5,6]]'
        output = string2array(input_data_string, ',', row['elementType'])
        assert output[0][1] == '2 3'


    def test_reduce_dimension_component(self):
        """
         This method tests if the data passed to the launch method of
         the NDimensionArrayAdapter adapter is correct. The passed data should be a list
         of arrays with one dimension.
        """
        inserted_count = FlowService().get_available_datatypes(self.test_project.id,
                                                               "tvb.datatypes.arrays.MappedArray")[1]
        assert inserted_count == 0, "Expected to find no data."
        #create an operation
        algorithm_id = FlowService().get_algorithm_by_module_and_class('tvb.tests.framework.adapters.ndimensionarrayadapter',
                                                                       'NDimensionArrayAdapter').id
        operation = model.Operation(self.test_user.id, self.test_project.id, algorithm_id, 'test params',
                                    meta=json.dumps({DataTypeMetaData.KEY_STATE: "RAW_DATA"}),
                                    status=model.STATUS_FINISHED)
        operation = dao.store_entity(operation)
        #save the array wrapper in DB
        adapter_instance = NDimensionArrayAdapter()
        PARAMS = {}
        self.operation_service.initiate_prelaunch(operation, adapter_instance, {}, **PARAMS)
        inserted_data = FlowService().get_available_datatypes(self.test_project.id,
                                                              "tvb.datatypes.arrays.MappedArray")[0]
        assert len(inserted_data) == 1, "Problems when inserting data"
        gid = inserted_data[0][2]
        entity = dao.get_datatype_by_gid(gid)
        #from the 3D array do not select any array
        PARAMS = {"python_method": "reduce_dimension", "input_data": gid,
                  "input_data_dimensions_0": "requiredDim_1",
                  "input_data_dimensions_1": "",
                  "input_data_dimensions_2": ""}
        try:
            self.operation_service.initiate_prelaunch(operation, adapter_instance, {}, **PARAMS)
            raise AssertionError("Test should not pass. The resulted array should be a 1D array.")
        except Exception:
            # OK, do nothing; we were expecting to produce a 1D array
            pass
        #from the 3D array select only a 1D array
        first_dim = [gid + '_1_0', 'requiredDim_1']
        PARAMS = {"python_method": "reduce_dimension", "input_data": gid,
                  "input_data_dimensions_0": first_dim,
                  "input_data_dimensions_1": gid + "_2_1"}
        self.operation_service.initiate_prelaunch(operation, adapter_instance, {}, **PARAMS)
        expected_result = entity.array_data[:, 0, 1]
        actual_result = adapter_instance.launch_param
        assert len(actual_result) == len(expected_result), "Not the same size for results!"
        assert numpy.equal(actual_result, expected_result).all()

        #from the 3D array select a 2D array
        first_dim = [gid + '_1_0', gid + '_1_1', 'requiredDim_2']
        PARAMS = {"python_method": "reduce_dimension", "input_data": gid,
                  "input_data_dimensions_0": first_dim,
                  "input_data_dimensions_1": gid + "_2_1"}
        self.operation_service.initiate_prelaunch(operation, adapter_instance, {}, **PARAMS)
        expected_result = entity.array_data[slice(0, None), [0, 1], 1]
        actual_result = adapter_instance.launch_param
        assert len(actual_result) == len(expected_result), "Not the same size for results!"
        assert numpy.equal(actual_result, expected_result).all()

        #from 3D array select 1D array by applying SUM function on the first
        #dimension and average function on the second dimension
        PARAMS = {"python_method": "reduce_dimension", "input_data": gid,
                  "input_data_dimensions_0": ["requiredDim_1", "func_sum"],
                  "input_data_dimensions_1": "func_average",
                  "input_data_dimensions_2": ""}
        self.operation_service.initiate_prelaunch(operation, adapter_instance, {}, **PARAMS)
        aux = numpy.sum(entity.array_data, axis=0)
        expected_result = numpy.average(aux, axis=0)
        actual_result = adapter_instance.launch_param
        assert len(actual_result) == len(expected_result), "Not the same size of results!"
        assert numpy.equal(actual_result, expected_result).all()

        #from 3D array select a 2D array and apply op. on the second dimension
        PARAMS = {"python_method": "reduce_dimension", "input_data": gid,
                  "input_data_dimensions_0": ["requiredDim_2", "func_sum",
                                              "expected_shape_x,512", "operations_x,&gt;"],
                  "input_data_dimensions_1": "",
                  "input_data_dimensions_2": ""}
        try:
            self.operation_service.initiate_prelaunch(operation, adapter_instance, {}, **PARAMS)
            raise AssertionError("Test should not pass! The second dimension of the array should be >512.")
        except Exception:
            # OK, do nothing;
            pass


    