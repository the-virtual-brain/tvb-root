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
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""
import json
import os
import numpy
import pytest
import tvb_data
from tvb.adapters.analyzers.bct_adapters import BaseBCTModel
from tvb.adapters.analyzers.bct_clustering_adapters import TransitivityBinaryDirected
from tvb.adapters.datatypes.db.mapped_value import DatatypeMeasureIndex
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.load import get_filtered_datatypes
from tvb.core.entities.model.model_operation import OperationGroup, Operation
from tvb.core.neocom import h5
from tvb.core.entities.model.model_operation import *
from tvb.core.entities.model.model_datatype import *
from tvb.core.entities.storage import dao
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.entities.filters.factory import StaticFiltersFactory
from tvb.core.services.operation_service import OperationService
from tvb.core.services.project_service import ProjectService
from tvb.datatypes.graph import ConnectivityMeasure
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.core.services.algorithm_service_test import TEST_ADAPTER_VALID_MODULE, TEST_ADAPTER_VALID_CLASS


class TestProjectStructure(TransactionalTestCase):
    """
    Test ProjectService methods (part related to Project Data Structure).
    """

    def transactional_setup_method(self):
        """
        Prepare before each test.
        """
        self.project_service = ProjectService()

        self.test_user = TestFactory.create_user()
        self.test_project = TestFactory.create_project(self.test_user, "ProjectStructure")

        self.relevant_filter = StaticFiltersFactory.build_datatype_filters(
            single_filter=StaticFiltersFactory.RELEVANT_VIEW)
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
        op1 = Operation(self.test_user.id, self.test_project.id, self.algo_inst.id, "")
        op1 = dao.store_entity(op1)
        assert op1.visible, "The operation should be visible."
        self.project_service.set_operation_and_group_visibility(op1.gid, False)
        updated_op = dao.get_operation_by_id(op1.id)
        assert not updated_op.visible, "The operation should not be visible."

    def test_set_op_and_group_visibility(self, datatype_group_factory):
        """
        When changing the visibility for an operation that belongs to an operation group, we
        should also change the visibility for the entire group of operations.
        """
        group = datatype_group_factory()
        list_of_operations = dao.get_operations_in_group(group.id)
        for operation in list_of_operations:
            assert operation.visible, "The operation should be visible."
        self.project_service.set_operation_and_group_visibility(list_of_operations[0].gid, False)
        operations = dao.get_operations_in_group(group.id)
        for operation in operations:
            assert not operation.visible, "The operation should not be visible."

    def test_set_op_group_visibility(self, datatype_group_factory):
        """
        Tests if the visibility for an operation group is set correct.
        """
        group = datatype_group_factory()
        list_of_operations = dao.get_operations_in_group(group.id)
        for operation in list_of_operations:
            assert operation.visible, "The operation should be visible."
        op_group = dao.get_operationgroup_by_id(group.id)
        self.project_service.set_operation_and_group_visibility(op_group.gid, False, True)
        operations = dao.get_operations_in_group(group.id)
        for operation in operations:
            assert not operation.visible, "The operation should not be visible."

    def test_is_upload_operation(self):
        """
        Tests that upload and non-upload algorithms are created and run accordingly
        """
        self.__init_algorithmn()
        upload_algo = self._create_algo_for_upload()
        op1 = Operation(self.test_user.id, self.test_project.id, self.algo_inst.id, "")
        op2 = Operation(self.test_user.id, self.test_project.id, upload_algo.id, "")
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

        project = Project("test_proj_2", self.test_user.id, "desc")
        project = dao.store_entity(project)

        op1 = Operation(self.test_user.id, self.test_project.id, self.algo_inst.id, "")
        op2 = Operation(self.test_user.id, project.id, upload_algo.id, "", status=STATUS_FINISHED)
        op3 = Operation(self.test_user.id, self.test_project.id, upload_algo.id, "")
        op4 = Operation(self.test_user.id, self.test_project.id, upload_algo.id, "", status=STATUS_FINISHED)
        op5 = Operation(self.test_user.id, self.test_project.id, upload_algo.id, "", status=STATUS_FINISHED)
        operations = dao.store_entities([op1, op2, op3, op4, op5])

        upload_operations = self.project_service.get_all_operations_for_uploaders(self.test_project.id)
        assert 2 == len(upload_operations), "Wrong number of upload operations."
        upload_ids = [operation.id for operation in upload_operations]
        for i in [3, 4]:
            assert operations[i].id in upload_ids, \
                "The operation should be an upload operation."
        for i in [0, 1, 2]:
            assert not operations[i].id in upload_ids, \
                "The operation should not be an upload operation."

    def test_is_datatype_group(self, datatype_group_factory):
        """
        Tests if a datatype is group.
        """
        group = datatype_group_factory()
        dt_group = dao.get_generic_entity(DataTypeGroup, group.id)[0]
        is_dt_group = self.project_service.is_datatype_group(dt_group.gid)
        assert is_dt_group, "The datatype should be a datatype group."
        datatypes = dao.get_datatypes_from_datatype_group(dt_group.id)
        is_dt_group = self.project_service.is_datatype_group(datatypes[0].gid)
        assert not is_dt_group, "The datatype should not be a datatype group."

    def test_count_datatypes_in_group(self, datatype_group_factory):
        """ Test that counting dataTypes is correct. Happy flow."""
        group = datatype_group_factory()
        count = dao.count_datatypes_in_group(group.id)
        assert count == 10
        datatypes = dao.get_datatypes_from_datatype_group(group.id)
        count = dao.count_datatypes_in_group(datatypes[0].id)
        assert count == 0, "There should be no dataType."

    def test_set_datatype_visibility(self, dummy_datatype_index_factory):
        """
        Check if the visibility for a datatype is set correct.
        """
        # it's a list of 3 elem.
        dummy_dt_index = dummy_datatype_index_factory()
        is_visible = dummy_dt_index.visible
        assert is_visible, "The data type should be visible."

        self.project_service.set_datatype_visibility(dummy_dt_index.gid, False)
        is_visible = dao.get_datatype_by_id(dummy_dt_index.id).visible
        assert not is_visible, "The data type should not be visible."

    def test_set_visibility_for_dt_in_group(self, datatype_group_factory):
        """
        Check if the visibility for a datatype from a datatype group is set correct.
        """
        group = datatype_group_factory()
        datatypes = dao.get_datatypes_from_datatype_group(group.id)
        assert datatypes[0].visible, "The data type should be visible."
        assert datatypes[1].visible, "The data type should be visible."
        self.project_service.set_datatype_visibility(datatypes[0].gid, False)

        db_dt_group = self.project_service.get_datatype_by_id(group.id)
        db_first_dt = self.project_service.get_datatype_by_id(datatypes[0].id)
        db_second_dt = self.project_service.get_datatype_by_id(datatypes[1].id)

        assert not db_dt_group.visible, "The data type should be visible."
        assert not db_first_dt.visible, "The data type should not be visible."
        assert not db_second_dt.visible, "The data type should be visible."

    def test_set_visibility_for_group(self, datatype_group_factory):
        """
        Check if the visibility for a datatype group is set correct.
        """
        group = datatype_group_factory()
        dt_group = dao.get_generic_entity(DataTypeGroup, group.id)[0]
        datatypes = dao.get_datatypes_from_datatype_group(dt_group.id)

        assert dt_group.visible, "The data type group should be visible."
        assert datatypes[0].visible, "The data type should be visible."
        assert datatypes[1].visible, "The data type should be visible."
        self.project_service.set_datatype_visibility(dt_group.gid, False)

        updated_dt_group = self.project_service.get_datatype_by_id(dt_group.id)
        updated_first_dt = self.project_service.get_datatype_by_id(datatypes[0].id)
        updated_second_dt = self.project_service.get_datatype_by_id(datatypes[1].id)

        assert not updated_dt_group.visible, "The data type group should be visible."
        assert not updated_first_dt.visible, "The data type should be visible."
        assert not updated_second_dt.visible, "The data type should be visible."

    def test_getdatatypes_from_dtgroup(self, datatype_group_factory):
        """
        Validate that we can retrieve all DTs from a DT_Group
        """
        group = datatype_group_factory()
        exp_datatypes = dao.get_datatypes_from_datatype_group(group.id)
        datatypes = self.project_service.get_datatypes_from_datatype_group(group.id)
        assert len(datatypes) == 10, "There should be 10 datatypes into the datatype group."
        expected_dict = {exp_datatypes[0].id: exp_datatypes[0], exp_datatypes[1].id: exp_datatypes[1]}
        actual_dict = {datatypes[0].id: datatypes[0], datatypes[1].id: datatypes[1]}

        for key in expected_dict:
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

    def test_get_operations_for_dt(self, datatype_group_factory):
        """
        Tests method get_operations_for_datatype.
        Verifies result dictionary has the correct values
        """
        group = datatype_group_factory()
        created_ops, datatype_gid = self._create_operations_with_inputs(group)
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

    def test_get_operations_for_dt_group(self, datatype_group_factory):
        """
        Tests method get_operations_for_datatype_group.
        Verifies filters' influence over results is as expected
        """
        group = datatype_group_factory()
        created_ops, dt_group_id = self._create_operations_with_inputs(group, True)

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
        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_66.zip')
        conn = TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path)
        view_model = BaseBCTModel()
        view_model.connectivity = conn.gid
        adapter = ABCAdapter.build_adapter_from_class(TransitivityBinaryDirected)
        result = OperationService().fire_operation(adapter, self.test_user, self.test_project.id,
                                                   view_model=view_model)

        conn.visible = False
        dao.store_entity(conn)
        operation = dao.get_operation_by_id(result[0].id)

        inputs = self.project_service.get_datatype_and_datatypegroup_inputs_for_operation(operation.gid,
                                                                                          self.relevant_filter)
        assert len(inputs) == 0

        inputs = self.project_service.get_datatype_and_datatypegroup_inputs_for_operation(operation.gid,
                                                                                          self.full_filter)
        assert len(inputs) == 1, "Incorrect number of inputs."
        assert conn.id == inputs[0].id, "Retrieved wrong input dataType."

    def test_get_inputs_for_group(self, datatype_group_factory, test_adapter_factory):
        """
        Tests method get_datatypes_inputs_for_operation_group.
        """
        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_66.zip')
        conn = TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path)

        group = OperationGroup(self.test_project.id, "group", "range1[1..2]")
        group = dao.store_entity(group)

        view_model = BaseBCTModel()
        view_model.connectivity = conn.gid
        adapter1 = ABCAdapter.build_adapter_from_class(TransitivityBinaryDirected)

        algorithm1 = adapter1.stored_adapter
        algorithm2 = adapter1.stored_adapter

        conn.visible = False
        dao.store_entity(conn)

        operation1 = Operation(self.test_user.id, self.test_project.id, algorithm1.id,
                              json.dumps({'gid': view_model.gid.hex}), op_group_id=group.id)
        operation1.fk_operation_grup = group.id

        operation2 = Operation(self.test_user.id, self.test_project.id, algorithm2.id,
                              json.dumps({'gid': view_model.gid.hex}), op_group_id=group.id)
        operation2.fk_operation_grup = group.id

        dao.store_entities([operation1, operation2])

        OperationService()._store_view_model(operation1, dao.get_project_by_id(self.test_project.id), view_model)
        OperationService()._store_view_model(operation2, dao.get_project_by_id(self.test_project.id), view_model)

        inputs = self.project_service.get_datatypes_inputs_for_operation_group(group.id, self.relevant_filter)
        assert len(inputs) == 0

        inputs = self.project_service.get_datatypes_inputs_for_operation_group(group.id, self.full_filter)
        assert len(inputs) == 1, "Incorrect number of dataTypes."
        assert group.id == inputs[0].id, "Retrieved wrong dataType."

        conn.visible = True
        dao.store_entity(conn)

        inputs = self.project_service.get_datatypes_inputs_for_operation_group(group.id, self.relevant_filter)
        assert len(inputs) == 1, "Incorrect number of dataTypes."

        inputs = self.project_service.get_datatypes_inputs_for_operation_group(group.id, self.full_filter)
        assert len(inputs) == 1, "Incorrect number of dataTypes."
        assert group.id == inputs[0].id, "Retrieved wrong dataType."

    def test_remove_datatype(self, array_factory):
        """
        Tests the deletion of a datatype.
        """
        # it's a list of 3 elem.
        array_wrappers = array_factory(self.test_project)
        dt_list = []
        for array_wrapper in array_wrappers:
            dt_list.append(dao.get_datatype_by_id(array_wrapper[0]))

        self.project_service.remove_datatype(self.test_project.id, dt_list[0].gid)
        self._check_if_datatype_was_removed(dt_list[0])

    def test_remove_datatype_from_group(self, datatype_group_factory, project_factory, user_factory):
        """
        Tests the deletion of a datatype group.
        """
        user = user_factory()
        project = project_factory(user)
        group = datatype_group_factory(project=project)

        datatype_group = dao.get_generic_entity(DataTypeGroup, group.id)[0]
        datatypes = dao.get_datatypes_from_datatype_group(group.id)
        datatype_measure = dao.get_generic_entity(DatatypeMeasureIndex, datatypes[0].gid, "fk_source_gid")[0]

        # When trying to delete one entity in a group the entire group will be removed
        #  First remove the DTMeasures, to avoid FK failures
        self.project_service.remove_datatype(project.id, datatype_measure.gid)
        self.project_service.remove_datatype(project.id, datatypes[0].gid)
        self._check_if_datatype_was_removed(datatypes[0])
        self._check_if_datatype_was_removed(datatypes[1])
        self._check_if_datatype_was_removed(datatype_group)
        self._check_if_datatype_was_removed(datatype_measure)
        self._check_datatype_group_removed(group.id, datatype_group.fk_operation_group)

    def test_remove_datatype_group(self, datatype_group_factory, project_factory, user_factory):
        """
        Tests the deletion of a datatype group.
        """
        user = user_factory()
        project = project_factory(user)
        group = datatype_group_factory(project=project)

        datatype_groups = self.get_all_entities(DataTypeGroup)
        datatypes = dao.get_datatypes_from_datatype_group(group.id)
        assert 2 == len(datatype_groups)

        self.project_service.remove_datatype(project.id, datatype_groups[1].gid)
        self.project_service.remove_datatype(project.id, datatype_groups[0].gid)
        self._check_if_datatype_was_removed(datatypes[0])
        self._check_if_datatype_was_removed(datatypes[1])
        self._check_if_datatype_was_removed(datatype_groups[0])
        self._check_if_datatype_was_removed(datatype_groups[1])
        self._check_datatype_group_removed(group.id, datatype_groups[0].fk_operation_group)

    @pytest.fixture()
    def array_factory(self, operation_factory, connectivity_index_factory):
        def _create_measure(conn, op, op_dir, project_id):
            conn_measure = ConnectivityMeasure()
            conn_measure.connectivity = h5.load_from_index(conn)
            conn_measure.array_data = numpy.array(conn.number_of_regions)

            conn_measure_db = h5.store_complete(conn_measure, op_dir)
            conn_measure_db.fk_from_operation = op.id
            dao.store_entity(conn_measure_db)

            count = dao.count_datatypes(project_id, DataTypeMatrix)
            return count

        def build(project):
            count = dao.count_datatypes(project.id, DataTypeMatrix)
            assert count == 0

            op = operation_factory(test_project=project)
            conn = connectivity_index_factory(op=op)
            storage_path = FilesHelper().get_project_folder(op.project, str(op.id))

            count = _create_measure(conn, op, storage_path, project.id)
            assert count == 1

            count = _create_measure(conn, op, storage_path, project.id)
            assert count == 2

            count = _create_measure(conn, op, storage_path, project.id)
            assert count == 3

            return get_filtered_datatypes(project.id, DataTypeMatrix)[0]

        return build

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
        operation = Operation(self.test_user.id, project_id, algorithm.id, 'test params',
                              meta=json.dumps(meta), status=STATUS_FINISHED)
        return dao.store_entity(operation)

    def _create_operations_with_inputs(self, datatype_group, is_group_parent=False):
        """
        Method used for creating a complex tree of operations.

        If 'if_group_parent' is True then a new group will be created and one of its entries it will be used as
        input for the returned operations.
        """
        group_dts = dao.get_datatypes_from_datatype_group(datatype_group.id)
        if is_group_parent:
            datatype_gid = group_dts[0].gid
        else:
            datatype_gid = TestFactory.create_value_wrapper(self.test_user, self.test_project)[1]

        parameters = json.dumps({"param_name": datatype_gid})

        ops = []
        for i in range(4):
            ops.append(TestFactory.create_operation(test_user=self.test_user, test_project=self.test_project))
            if i in [1, 3]:
                ops[i].visible = False
            ops[i].parameters = parameters
            ops[i] = dao.store_entity(ops[i])

        # groups
        ops_group = dao.get_operations_in_group(datatype_group.fk_from_operation)
        assert 10 == len(ops_group)
        ops_group[0].parameters = parameters
        ops_group[0] = dao.store_entity(ops_group[0])
        ops_group[1].visible = False
        ops_group[1].parameters = parameters
        ops_group[1] = dao.store_entity(ops_group[1])

        ops.extend(ops_group)
        if is_group_parent:
            dt_group = dao.get_datatypegroup_by_op_group_id(datatype_group.id)
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
            dao.get_generic_entity(DataTypeGroup, datatype_group_id)
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
        categ1 = AlgorithmCategory('one', True)
        self.categ1 = dao.store_entity(categ1)
        ad = Algorithm(TEST_ADAPTER_VALID_MODULE, TEST_ADAPTER_VALID_CLASS, categ1.id)
        self.algo_inst = dao.store_entity(ad)

    @staticmethod
    def _create_algo_for_upload():
        """ Creates a fake algorithm for an upload category. """
        category = dao.store_entity(AlgorithmCategory("upload_category", rawinput=True))
        return dao.store_entity(Algorithm("module", "classname", category.id))
