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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import json
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities import model
from tvb.core.entities.storage import dao
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.entities.transient.filtering import StaticFiltersFactory
from tvb.core.services.project_service import ProjectService
from tvb.core.services.flow_service import FlowService
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.core.services.project_service_test import TestProjectService
from tvb.tests.framework.core.services.flow_service_test import TEST_ADAPTER_VALID_MODULE, TEST_ADAPTER_VALID_CLASS



class TestProjectStructure(TransactionalTestCase):
    """
    Test ProjectService methods (part related to Project Data Structure).
    """

    def transactional_setup_method(self):
        """
        Prepare before each test.
        """
        self.project_service = ProjectService()
        self.flow_service = FlowService()
        self.structure_helper = FilesHelper()

        self.test_user = TestFactory.create_user()
        self.test_project = TestFactory.create_project(self.test_user, "ProjectStructure")

        self.relevant_filter = StaticFiltersFactory.build_datatype_filters(single_filter=StaticFiltersFactory.RELEVANT_VIEW)
        self.full_filter = StaticFiltersFactory.build_datatype_filters(single_filter=StaticFiltersFactory.FULL_VIEW)

    
    def transactional_teardown_method(self):
        """
        Clear project folders after testing
        """
        self.delete_project_folders()


    def test_set_operation_visibility(self):
        """
        Check if the visibility for an operation is set correct.
        """
        self.__init_algorithmn()
        op1 = model.Operation(self.test_user.id, self.test_project.id, self.algo_inst.id, "")
        op1 = dao.store_entity(op1)
        assert op1.visible, "The operation should be visible."
        self.project_service.set_operation_and_group_visibility(op1.gid, False)
        updated_op = dao.get_operation_by_id(op1.id)
        assert not updated_op.visible, "The operation should not be visible."


    def test_set_op_and_group_visibility(self):
        """
        When changing the visibility for an operation that belongs to an operation group, we
        should also change the visibility for the entire group of operations.
        """
        _, group_id = TestFactory.create_group(self.test_user, subject="test-subject-1")
        list_of_operations = dao.get_operations_in_group(group_id)
        for operation in list_of_operations:
            assert operation.visible, "The operation should be visible."
        self.project_service.set_operation_and_group_visibility(list_of_operations[0].gid, False)
        operations = dao.get_operations_in_group(group_id)
        for operation in operations:
            assert not operation.visible, "The operation should not be visible."


    def test_set_op_group_visibility(self):
        """
        Tests if the visibility for an operation group is set correct.
        """
        _, group_id = TestFactory.create_group(self.test_user, subject="test-subject-1")
        list_of_operations = dao.get_operations_in_group(group_id)
        for operation in list_of_operations:
            assert operation.visible, "The operation should be visible."
        op_group = dao.get_operationgroup_by_id(group_id)
        self.project_service.set_operation_and_group_visibility(op_group.gid, False, True)
        operations = dao.get_operations_in_group(group_id)
        for operation in operations:
            assert not operation.visible, "The operation should not be visible."


    def test_is_upload_operation(self):
        """
        Tests that upload and non-upload algorithms are created and run accordingly
        """
        self.__init_algorithmn()
        upload_algo = self._create_algo_for_upload()
        op1 = model.Operation(self.test_user.id, self.test_project.id, self.algo_inst.id, "")
        op2 = model.Operation(self.test_user.id, self.test_project.id, upload_algo.id, "")
        operations = dao.store_entities([op1, op2])
        is_upload_operation = self.project_service.is_upload_operation(operations[0].gid)
        assert not is_upload_operation, "The operation is not an upload operation."
        is_upload_operation = self.project_service.is_upload_operation(operations[1].gid)
        assert is_upload_operation, "The operation is an upload operation."


    def test_get_upload_operations(self):
        """
        Test get_all when filter is for Upload category.
        """
        self.__init_algorithmn()
        upload_algo = self._create_algo_for_upload()

        project = model.Project("test_proj_2", self.test_user.id, "desc")
        project = dao.store_entity(project)

        op1 = model.Operation(self.test_user.id, self.test_project.id, self.algo_inst.id, "")
        op2 = model.Operation(self.test_user.id, project.id, upload_algo.id, "", status=model.STATUS_FINISHED)
        op3 = model.Operation(self.test_user.id, self.test_project.id, upload_algo.id, "")
        op4 = model.Operation(self.test_user.id, self.test_project.id, upload_algo.id, "", status=model.STATUS_FINISHED)
        op5 = model.Operation(self.test_user.id, self.test_project.id, upload_algo.id, "", status=model.STATUS_FINISHED)
        operations = dao.store_entities([op1, op2, op3, op4, op5])

        upload_operations = self.project_service.get_all_operations_for_uploaders(self.test_project.id)
        assert 2 == len(upload_operations), "Wrong number of upload operations."
        upload_ids = [operation.id for operation in upload_operations]
        for i in [3, 4]:
            assert operations[i].id in upload_ids,\
                            "The operation should be an upload operation."
        for i in [0, 1, 2]:                    
            assert not operations[i].id in upload_ids,\
                             "The operation should not be an upload operation."


    def test_is_datatype_group(self):
        """
        Tests if a datatype is group.
        """
        _, dt_group_id, first_dt, _ = self._create_datatype_group()
        dt_group = dao.get_generic_entity(model.DataTypeGroup, dt_group_id)[0]
        is_dt_group = self.project_service.is_datatype_group(dt_group.gid)
        assert is_dt_group, "The datatype should be a datatype group."
        is_dt_group = self.project_service.is_datatype_group(first_dt.gid)
        assert not is_dt_group, "The datatype should not be a datatype group."


    def test_count_datatypes_in_group(self):
        """ Test that counting dataTypes is correct. Happy flow."""
        _, dt_group_id, first_dt, _ = self._create_datatype_group()
        count = dao.count_datatypes_in_group(dt_group_id)
        assert count == 2
        count = dao.count_datatypes_in_group(first_dt.id)
        assert count == 0, "There should be no dataType."


    def test_set_datatype_visibility(self):
        """
        Check if the visibility for a datatype is set correct.
        """
        #it's a list of 3 elem.
        mapped_arrays = self._create_mapped_arrays(self.test_project.id)
        for mapped_array in mapped_arrays:
            is_visible = dao.get_datatype_by_id(mapped_array[0]).visible
            assert is_visible, "The data type should be visible."

        self.project_service.set_datatype_visibility(mapped_arrays[0][2], False)
        for i in range(len(mapped_arrays)):
            is_visible = dao.get_datatype_by_id(mapped_arrays[i][0]).visible
            if not i:
                assert not is_visible, "The data type should not be visible."
            else:
                assert is_visible, "The data type should be visible."


    def test_set_visibility_for_dt_in_group(self):
        """
        Check if the visibility for a datatype from a datatype group is set correct.
        """
        _, dt_group_id, first_dt, second_dt = self._create_datatype_group()
        assert first_dt.visible, "The data type should be visible."
        assert second_dt.visible, "The data type should be visible."
        self.project_service.set_datatype_visibility(first_dt.gid, False)

        db_dt_group = self.project_service.get_datatype_by_id(dt_group_id)
        db_first_dt = self.project_service.get_datatype_by_id(first_dt.id)
        db_second_dt = self.project_service.get_datatype_by_id(second_dt.id)

        assert not db_dt_group.visible, "The data type should be visible."
        assert not db_first_dt.visible, "The data type should not be visible."
        assert not db_second_dt.visible, "The data type should be visible."


    def test_set_visibility_for_group(self):
        """
        Check if the visibility for a datatype group is set correct.
        """
        _, dt_group_id, first_dt, second_dt = self._create_datatype_group()
        dt_group = dao.get_generic_entity(model.DataTypeGroup, dt_group_id)[0]

        assert dt_group.visible, "The data type group should be visible."
        assert first_dt.visible, "The data type should be visible."
        assert second_dt.visible, "The data type should be visible."
        self.project_service.set_datatype_visibility(dt_group.gid, False)

        updated_dt_group = self.project_service.get_datatype_by_id(dt_group_id)
        updated_first_dt = self.project_service.get_datatype_by_id(first_dt.id)
        updated_second_dt = self.project_service.get_datatype_by_id(second_dt.id)

        assert not updated_dt_group.visible, "The data type group should be visible."
        assert not updated_first_dt.visible, "The data type should be visible."
        assert not updated_second_dt.visible, "The data type should be visible."


    def test_getdatatypes_from_dtgroup(self):
        """
        Validate that we can retrieve all DTs from a DT_Group
        """
        _, dt_group_id, first_dt, second_dt = self._create_datatype_group()
        datatypes = self.project_service.get_datatypes_from_datatype_group(dt_group_id)
        assert len(datatypes) == 2, "There should be 2 datatypes into the datatype group."
        expected_dict = {first_dt.id: first_dt, second_dt.id: second_dt}
        actual_dict = {datatypes[0].id: datatypes[0], datatypes[1].id: datatypes[1]}

        for key in expected_dict.keys():
            expected = expected_dict[key]
            actual = actual_dict[key]
            assert expected.id == actual.id, "Not the same id."
            assert expected.gid == actual.gid, "Not the same gid."
            assert expected.type == actual.type, "Not the same type."
            assert expected.subject == actual.subject, "Not the same subject."
            assert expected.state == actual.state, "Not the same state."
            assert expected.visible == actual.visible, "The datatype visibility is not correct."
            assert expected.module == actual.module, "Not the same module."
            assert expected.user_tag_1 == actual.user_tag_1, "Not the same user_tag_1."
            assert expected.invalid == actual.invalid, "The invalid field value is not correct."
            assert expected.is_nan == actual.is_nan, "The is_nan field value is not correct."


    def test_get_operations_for_dt(self):
        """
        Tests method get_operations_for_datatype.
        Verifies result dictionary has the correct values
        """
        created_ops, datatype_gid = self._create_operations_with_inputs()
        operations = self.project_service.get_operations_for_datatype(datatype_gid, self.relevant_filter)
        assert len(operations) == 2
        assert created_ops[0].id in [operations[0].id, operations[1].id], "Retrieved wrong operations."
        assert created_ops[2].id in [operations[0].id, operations[1].id], "Retrieved wrong operations."

        operations = self.project_service.get_operations_for_datatype(datatype_gid, self.full_filter)
        assert len(operations) == 4
        ids = [operations[0].id, operations[1].id, operations[2].id, operations[3].id]
        for i in range(4):
            assert created_ops[i].id in ids, "Retrieved wrong operations."

        operations = self.project_service.get_operations_for_datatype(datatype_gid, self.relevant_filter, True)
        assert len(operations) == 1
        assert created_ops[4].id == operations[0].id, "Incorrect number of operations."

        operations = self.project_service.get_operations_for_datatype(datatype_gid, self.full_filter, True)
        assert len(operations) == 2
        assert created_ops[4].id in [operations[0].id, operations[1].id], "Retrieved wrong operations."
        assert created_ops[5].id in [operations[0].id, operations[1].id], "Retrieved wrong operations."


    def test_get_operations_for_dt_group(self):
        """
        Tests method get_operations_for_datatype_group.
        Verifies filters' influence over results is as expected
        """
        created_ops, dt_group_id = self._create_operations_with_inputs(True)

        ops = self.project_service.get_operations_for_datatype_group(dt_group_id, self.relevant_filter)
        assert len(ops) == 2
        assert created_ops[0].id in [ops[0].id, ops[1].id], "Retrieved wrong operations."
        assert created_ops[2].id in [ops[0].id, ops[1].id], "Retrieved wrong operations."

        ops = self.project_service.get_operations_for_datatype_group(dt_group_id, self.full_filter)
        assert len(ops) == 4, "Incorrect number of operations."
        ids = [ops[0].id, ops[1].id, ops[2].id, ops[3].id]
        for i in range(4):
            assert created_ops[i].id in ids, "Retrieved wrong operations."

        ops = self.project_service.get_operations_for_datatype_group(dt_group_id, self.relevant_filter, True)
        assert len(ops) == 1
        assert created_ops[4].id == ops[0].id, "Incorrect number of operations."

        ops = self.project_service.get_operations_for_datatype_group(dt_group_id, self.full_filter, True)
        assert len(ops), 2
        assert created_ops[4].id in [ops[0].id, ops[1].id], "Retrieved wrong operations."
        assert created_ops[5].id in [ops[0].id, ops[1].id], "Retrieved wrong operations."


    def test_get_inputs_for_operation(self):
        """
        Tests method get_datatype_and_datatypegroup_inputs_for_operation.
        Verifies filters' influence over results is as expected
        """
        algo = dao.get_algorithm_by_module('tvb.tests.framework.adapters.testadapter3', 'TestAdapter3')

        array_wrappers = self._create_mapped_arrays(self.test_project.id)
        ids = []
        for datatype in array_wrappers:
            ids.append(datatype[0])

        datatype = dao.get_datatype_by_id(ids[0])
        datatype.visible = False
        dao.store_entity(datatype)

        parameters = json.dumps({"param_5": "1", "param_1": array_wrappers[0][2],
                                 "param_2": array_wrappers[1][2], "param_3": array_wrappers[2][2], "param_6": "0"})
        operation = model.Operation(self.test_user.id, self.test_project.id, algo.id, parameters)
        operation = dao.store_entity(operation)

        inputs = self.project_service.get_datatype_and_datatypegroup_inputs_for_operation(operation.gid,
                                                                                          self.relevant_filter)
        assert len(inputs) == 2
        assert ids[1] in [inputs[0].id, inputs[1].id], "Retrieved wrong dataType."
        assert ids[2] in [inputs[0].id, inputs[1].id], "Retrieved wrong dataType."
        assert not ids[0] in [inputs[0].id, inputs[1].id], "Retrieved wrong dataType."

        inputs = self.project_service.get_datatype_and_datatypegroup_inputs_for_operation(operation.gid,
                                                                                          self.full_filter)
        assert len(inputs) == 3, "Incorrect number of operations."
        assert ids[0] in [inputs[0].id, inputs[1].id, inputs[2].id], "Retrieved wrong dataType."
        assert ids[1] in [inputs[0].id, inputs[1].id, inputs[2].id], "Retrieved wrong dataType."
        assert ids[2] in [inputs[0].id, inputs[1].id, inputs[2].id], "Retrieved wrong dataType."

        project, dt_group_id, first_dt, _ = self._create_datatype_group()
        first_dt.visible = False
        dao.store_entity(first_dt)
        parameters = json.dumps({"other_param": "_", "param_1": first_dt.gid})
        operation = model.Operation(self.test_user.id, project.id, algo.id, parameters)
        operation = dao.store_entity(operation)

        inputs = self.project_service.get_datatype_and_datatypegroup_inputs_for_operation(operation.gid,
                                                                                          self.relevant_filter)
        assert len(inputs) == 0, "Incorrect number of dataTypes."
        inputs = self.project_service.get_datatype_and_datatypegroup_inputs_for_operation(operation.gid,
                                                                                          self.full_filter)
        assert len(inputs) == 1, "Incorrect number of dataTypes."
        assert inputs[0].id == dt_group_id, "Wrong dataType."
        assert inputs[0].id != first_dt.id, "Wrong dataType."


    def test_get_inputs_for_op_group(self):
        """
        Tests method get_datatypes_inputs_for_operation_group.
        The DataType inputs will be from a DataType group.
        """
        project, dt_group_id, first_dt, second_dt = self._create_datatype_group()
        first_dt.visible = False
        dao.store_entity(first_dt)
        second_dt.visible = False
        dao.store_entity(second_dt)

        op_group = model.OperationGroup(project.id, "group", "range1[1..2]")
        op_group = dao.store_entity(op_group)
        params_1 = json.dumps({"param_5": "1", "param_1": first_dt.gid, "param_6": "2"})
        params_2 = json.dumps({"param_5": "1", "param_4": second_dt.gid, "param_6": "5"})

        algo = dao.get_algorithm_by_module('tvb.tests.framework.adapters.testadapter3', 'TestAdapter3')

        op1 = model.Operation(self.test_user.id, project.id, algo.id, params_1, op_group_id=op_group.id)
        op2 = model.Operation(self.test_user.id, project.id, algo.id, params_2, op_group_id=op_group.id)
        dao.store_entities([op1, op2])

        inputs = self.project_service.get_datatypes_inputs_for_operation_group(op_group.id, self.relevant_filter)
        assert len(inputs) == 0

        inputs = self.project_service.get_datatypes_inputs_for_operation_group(op_group.id, self.full_filter)
        assert len(inputs) == 1, "Incorrect number of dataTypes."
        assert not first_dt.id == inputs[0].id, "Retrieved wrong dataType."
        assert not second_dt.id == inputs[0].id, "Retrieved wrong dataType."
        assert dt_group_id == inputs[0].id, "Retrieved wrong dataType."

        first_dt.visible = True
        dao.store_entity(first_dt)

        inputs = self.project_service.get_datatypes_inputs_for_operation_group(op_group.id, self.relevant_filter)
        assert len(inputs) == 1, "Incorrect number of dataTypes."
        assert not first_dt.id == inputs[0].id, "Retrieved wrong dataType."
        assert not second_dt.id == inputs[0].id, "Retrieved wrong dataType."
        assert dt_group_id == inputs[0].id, "Retrieved wrong dataType."

        inputs = self.project_service.get_datatypes_inputs_for_operation_group(op_group.id, self.full_filter)
        assert len(inputs) == 1, "Incorrect number of dataTypes."
        assert not first_dt.id == inputs[0].id, "Retrieved wrong dataType."
        assert not second_dt.id == inputs[0].id, "Retrieved wrong dataType."
        assert dt_group_id == inputs[0].id, "Retrieved wrong dataType."


    def test_get_inputs_for_op_group_simple_inputs(self):
        """
        Tests method get_datatypes_inputs_for_operation_group.
        The dataType inputs will not be part of a dataType group.
        """
        #it's a list of 3 elem.
        array_wrappers = self._create_mapped_arrays(self.test_project.id)
        array_wrapper_ids = []
        for datatype in array_wrappers:
            array_wrapper_ids.append(datatype[0])

        datatype = dao.get_datatype_by_id(array_wrapper_ids[0])
        datatype.visible = False
        dao.store_entity(datatype)

        op_group = model.OperationGroup(self.test_project.id, "group", "range1[1..2]")
        op_group = dao.store_entity(op_group)
        params_1 = json.dumps({"param_5": "2", "param_1": array_wrappers[0][2],
                               "param_2": array_wrappers[1][2], "param_6": "7"})
        params_2 = json.dumps({"param_5": "5", "param_3": array_wrappers[2][2],
                               "param_2": array_wrappers[1][2], "param_6": "6"})

        algo = dao.get_algorithm_by_module('tvb.tests.framework.adapters.testadapter3', 'TestAdapter3')

        op1 = model.Operation(self.test_user.id, self.test_project.id, algo.id, params_1, op_group_id=op_group.id)
        op2 = model.Operation(self.test_user.id, self.test_project.id, algo.id, params_2, op_group_id=op_group.id)
        dao.store_entities([op1, op2])

        inputs = self.project_service.get_datatypes_inputs_for_operation_group(op_group.id, self.relevant_filter)
        assert len(inputs) == 2
        assert not array_wrapper_ids[0] in [inputs[0].id, inputs[1].id], "Retrieved wrong dataType."
        assert array_wrapper_ids[1] in [inputs[0].id, inputs[1].id], "Retrieved wrong dataType."
        assert array_wrapper_ids[2] in [inputs[0].id, inputs[1].id], "Retrieved wrong dataType."

        inputs = self.project_service.get_datatypes_inputs_for_operation_group(op_group.id, self.full_filter)
        assert len(inputs) == 3, "Incorrect number of dataTypes."
        assert array_wrapper_ids[0] in [inputs[0].id, inputs[1].id, inputs[2].id]
        assert array_wrapper_ids[1] in [inputs[0].id, inputs[1].id, inputs[2].id]
        assert array_wrapper_ids[2] in [inputs[0].id, inputs[1].id, inputs[2].id]


    def test_remove_datatype(self):
        """
        Tests the deletion of a datatype.
        """
        #it's a list of 3 elem.
        array_wrappers = self._create_mapped_arrays(self.test_project.id)
        dt_list = []
        for array_wrapper in array_wrappers:
            dt_list.append(dao.get_datatype_by_id(array_wrapper[0]))

        self.project_service.remove_datatype(self.test_project.id, dt_list[0].gid)
        self._check_if_datatype_was_removed(dt_list[0])


    def test_remove_datatype_from_group(self):
        """
        Tests the deletion of a datatype group.
        """
        project, dt_group_id, first_dt, second_dt = self._create_datatype_group()
        datatype_group = dao.get_generic_entity(model.DataTypeGroup, dt_group_id)[0]

        self.project_service.remove_datatype(project.id, first_dt.gid)
        self._check_if_datatype_was_removed(first_dt)
        self._check_if_datatype_was_removed(second_dt)
        self._check_if_datatype_was_removed(datatype_group)
        self._check_datatype_group_removed(dt_group_id, datatype_group.fk_operation_group)


    def test_remove_datatype_group(self):
        """
        Tests the deletion of a datatype group.
        """
        project, dt_group_id, first_dt, second_dt = self._create_datatype_group()
        datatype_group = dao.get_generic_entity(model.DataTypeGroup, dt_group_id)[0]

        self.project_service.remove_datatype(project.id, datatype_group.gid)
        self._check_if_datatype_was_removed(first_dt)
        self._check_if_datatype_was_removed(second_dt)
        self._check_if_datatype_was_removed(datatype_group)
        self._check_datatype_group_removed(dt_group_id, datatype_group.fk_operation_group)


    def _create_mapped_arrays(self, project_id):
        """
        :param project_id: the project in which the arrays are created
        :return: a list of dummy `MappedArray`
        """
        count = self.flow_service.get_available_datatypes(project_id, "tvb.datatypes.arrays.MappedArray")[1]
        assert count == 0
        
        group = dao.get_algorithm_by_module('tvb.tests.framework.adapters.ndimensionarrayadapter', 'NDimensionArrayAdapter')
        adapter_instance = ABCAdapter.build_adapter(group)
        data = {'param_1': 'some value'}
        #create 3 data types
        self.flow_service.fire_operation(adapter_instance, self.test_user, project_id, **data)
        count = self.flow_service.get_available_datatypes(project_id, "tvb.datatypes.arrays.MappedArray")[1]
        assert count == 1
        
        self.flow_service.fire_operation(adapter_instance, self.test_user, project_id, **data)
        count = self.flow_service.get_available_datatypes(project_id, "tvb.datatypes.arrays.MappedArray")[1]
        assert count == 2
        
        self.flow_service.fire_operation(adapter_instance, self.test_user, project_id, **data)
        array_wrappers, count = self.flow_service.get_available_datatypes(project_id,
                                                                          "tvb.datatypes.arrays.MappedArray")
        assert count == 3

        return array_wrappers


    def _create_operation(self, project_id, algorithm_id):
        """
        dummy operation
        :param project_id: the project in which the operation is created
        :param algorithm_id: the algorithm to be run for the operation
        :return: a dummy `Operation` with the given specifications
        """
        algorithm = dao.get_algorithm_by_id(algorithm_id)
        meta = {DataTypeMetaData.KEY_SUBJECT: "John Doe",
                DataTypeMetaData.KEY_STATE: "RAW_DATA"}
        operation = model.Operation(self.test_user.id, project_id, algorithm.id, 'test params',
                                    meta=json.dumps(meta), status=model.STATUS_FINISHED)
        return dao.store_entity(operation)


    def _create_datatype_group(self):
        """
        Creates a project, one DataTypeGroup with 2 DataTypes into the new group.
        """
        test_project = TestFactory.create_project(self.test_user, "NewProject")

        all_operations = dao.get_filtered_operations(test_project.id, None, is_count=True)
        assert 0 == all_operations, "There should be no operation."
        
        datatypes, op_group_id = TestFactory.create_group(self.test_user, test_project)
        dt_group = dao.get_datatypegroup_by_op_group_id(op_group_id)

        return test_project, dt_group.id, datatypes[0], datatypes[1]



    def _create_operations_with_inputs(self, is_group_parent=False):
        """
        Method used for creating a complex tree of operations.

        If 'if_group_parent' is True then a new group will be created and one of its entries it will be used as
        input for the returned operations.
        """
        group_dts, root_op_group_id = TestFactory.create_group(self.test_user, self.test_project)
        if is_group_parent:
            datatype_gid = group_dts[0].gid
        else:
            datatype_gid = TestProjectService._create_value_wrapper(self.test_user, self.test_project)[1]

        parameters = json.dumps({"param_name": datatype_gid})

        ops = []
        for i in range(4):
            ops.append(TestFactory.create_operation(test_user=self.test_user, test_project=self.test_project))
            if i in [1, 3]:
                ops[i].visible = False
            ops[i].parameters = parameters
            ops[i] = dao.store_entity(ops[i])
            
        #groups
        _, ops_group = TestFactory.create_group(self.test_user, self.test_project)
        ops_group = dao.get_operations_in_group(ops_group)
        assert 2 == len(ops_group)
        ops_group[0].parameters = parameters
        ops_group[0] = dao.store_entity(ops_group[0])
        ops_group[1].visible = False
        ops_group[1].parameters = parameters
        ops_group[1] = dao.store_entity(ops_group[1])

        ops.extend(ops_group)
        if is_group_parent:
            dt_group = dao.get_datatypegroup_by_op_group_id(root_op_group_id)
            return ops, dt_group.id
        return ops, datatype_gid


    def _check_if_datatype_was_removed(self, datatype):
        """
        Check if a certain datatype was removed.
        """
        try:
            dao.get_datatype_by_id(datatype.id)
            raise AssertionError("The datatype was not deleted.")
        except Exception:
            pass
        try:
            dao.get_operation_by_id(datatype.fk_from_operation)
            raise AssertionError("The operation was not deleted.")
        except Exception:
            pass


    def _check_datatype_group_removed(self, datatype_group_id, operation_groupp_id):
        """
        Checks if the DataTypeGroup and OperationGroup was removed.
        """
        try:
            dao.get_generic_entity(model.DataTypeGroup, datatype_group_id)
            raise AssertionError("The DataTypeGroup entity was not removed.")
        except Exception:
            pass

        try:
            dao.get_operationgroup_by_id(operation_groupp_id)
            raise AssertionError("The OperationGroup entity was not removed.")
        except Exception:
            pass


    def __init_algorithmn(self):
        """
        Insert some starting data in the database.
        """
        categ1 = model.AlgorithmCategory('one', True)
        self.categ1 = dao.store_entity(categ1)
        ad = model.Algorithm(TEST_ADAPTER_VALID_MODULE, TEST_ADAPTER_VALID_CLASS, categ1.id)
        self.algo_inst = dao.store_entity(ad)

    @staticmethod
    def _create_algo_for_upload():
        """ Creates a fake algorithm for an upload category. """
        category = dao.store_entity(model.AlgorithmCategory("upload_category", rawinput=True))
        return dao.store_entity(model.Algorithm("module", "classname", category.id))
