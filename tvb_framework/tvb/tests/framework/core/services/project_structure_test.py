# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#
"""
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import pytest

from tvb.core.entities.filters.factory import StaticFiltersFactory
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.entities.load import get_filtered_datatypes
from tvb.core.entities.model.model_datatype import *
from tvb.core.entities.model.model_operation import *
from tvb.core.entities.storage import dao
from tvb.core.services.project_service import ProjectService
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
        op1 = Operation(None, self.test_user.id, self.test_project.id, self.algo_inst.id)
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
        group, _ = datatype_group_factory()
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
        group, _ = datatype_group_factory()
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
        op1 = Operation(None, self.test_user.id, self.test_project.id, self.algo_inst.id)
        op2 = Operation(None, self.test_user.id, self.test_project.id, upload_algo.id)
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

        op1 = Operation(None, self.test_user.id, self.test_project.id, self.algo_inst.id)
        op2 = Operation(None, self.test_user.id, project.id, upload_algo.id, status=STATUS_FINISHED)
        op3 = Operation(None, self.test_user.id, self.test_project.id, upload_algo.id)
        op4 = Operation(None, self.test_user.id, self.test_project.id, upload_algo.id, status=STATUS_FINISHED)
        op5 = Operation(None, self.test_user.id, self.test_project.id, upload_algo.id, status=STATUS_FINISHED)
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
        group, _ = datatype_group_factory()
        dt_group = dao.get_generic_entity(DataTypeGroup, group.id)[0]
        is_dt_group = self.project_service.is_datatype_group(dt_group.gid)
        assert is_dt_group, "The datatype should be a datatype group."
        datatypes = dao.get_datatypes_from_datatype_group(dt_group.id)
        is_dt_group = self.project_service.is_datatype_group(datatypes[0].gid)
        assert not is_dt_group, "The datatype should not be a datatype group."

    def test_count_datatypes_in_group(self, datatype_group_factory):
        """ Test that counting dataTypes is correct. Happy flow."""
        group, _ = datatype_group_factory()
        count = dao.count_datatypes_in_group(group.id)
        assert count == group.count_results
        assert count == 6
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
        group, _ = datatype_group_factory()
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
        group, _ = datatype_group_factory()
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
        group, _ = datatype_group_factory()
        exp_datatypes = dao.get_datatypes_from_datatype_group(group.id)
        datatypes = self.project_service.get_datatypes_from_datatype_group(group.id)
        assert len(datatypes) == group.count_results, "There should be 10 datatypes into the datatype group."
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

    def test_remove_datatype_group(self, datatype_group_factory, project_factory, user_factory):
        """
        Tests the deletion of a datatype group.
        """
        user = user_factory()
        project = project_factory(user)
        group, _ = datatype_group_factory(project=project)

        datatype_groups = self.get_all_entities(DataTypeGroup)
        datatypes = dao.get_datatypes_from_datatype_group(group.id)
        assert 2 == len(datatype_groups)

        self.project_service.remove_datatype(project.id, datatype_groups[1].gid, skip_validation=True)
        self.project_service.remove_datatype(project.id, datatype_groups[0].gid, skip_validation=True)
        self._check_if_datatype_was_removed(datatypes[0])
        self._check_if_datatype_was_removed(datatypes[1])
        self._check_if_datatype_was_removed(datatype_groups[0])
        self._check_if_datatype_was_removed(datatype_groups[1])
        self._check_datatype_group_removed(group.id, datatype_groups[0].fk_operation_group)

    def test_get_data_in_project(self, project_factory, user_factory, connectivity_index_factory):
        user = user_factory()
        project = project_factory(user)
        connectivity_index_factory()

        conn_not_visible = connectivity_index_factory()
        conn_not_visible.visible = False
        dao.store_entity(conn_not_visible)

        datatypes = dao.get_data_in_project(project.id)
        assert len(datatypes) == 2

        visibility_filter = FilterChain(fields=[FilterChain.datatype + '.visible'], operations=['=='], values=[True])
        datatypes = dao.get_data_in_project(project.id, visibility_filter=visibility_filter)
        assert len(datatypes) == 1

    @pytest.fixture()
    def array_factory(self, operation_factory, connectivity_index_factory, connectivity_measure_index_factory):
        def _create_measure(conn, op, project):
            connectivity_measure_index_factory(conn, op, project)

            count = dao.count_datatypes(project.id, DataTypeMatrix)
            return count

        def build(project):
            count = dao.count_datatypes(project.id, DataTypeMatrix)
            assert count == 0

            op = operation_factory(test_project=project)
            conn = connectivity_index_factory(op=op)

            count = _create_measure(conn, op, project)
            assert count == 1

            count = _create_measure(conn, op, project)
            assert count == 2

            count = _create_measure(conn, op, project)
            assert count == 3

            return get_filtered_datatypes(project.id, DataTypeMatrix)[0]

        return build

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
