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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import pytest
from tvb.basic.profile import TvbProfile
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.adapters.exceptions import NoMemoryAvailableException
from tvb.core.entities.model import model_burst, model_operation
from tvb.core.entities.storage import dao
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.services.operation_service import OperationService
from tvb.core.services.project_service import initialize_storage, ProjectService
from tvb.tests.framework.adapters.testadapter2 import TestAdapter2
from tvb.tests.framework.adapters.testadapter3 import TestAdapter3, TestAdapterHDDRequired, TestAdapterHDDRequiredForm
from tvb.tests.framework.core.base_testcase import BaseTestCase
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.datatypes.dummy_datatype_index import DummyDataTypeIndex


class TestOperationService(BaseTestCase):
    """
    Test class for the introspection module. Some tests from here do async launches. For those
    cases Transactional tests won't work.
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

    def _assert_no_ddti(self):
        count = dao.count_datatypes(self.test_project.id, DummyDataTypeIndex)
        assert 0 == count

    def _assert_stored_ddti(self, expected_cnt=1):
        count = dao.count_datatypes(self.test_project.id, DummyDataTypeIndex)
        assert expected_cnt == count
        datatype = dao.try_load_last_entity_of_type(self.test_project.id, DummyDataTypeIndex)
        assert datatype.subject == DataTypeMetaData.DEFAULT_SUBJECT, "Wrong data stored."
        return datatype

    def test_datatypes_groups(self, test_adapter_factory):
        """
        Tests if the dataType group is set correct on the dataTypes resulted from the same operation group.
        """
        # TODO: re-write this to use groups correctly
        all_operations = dao.get_filtered_operations(self.test_project.id, None)
        assert len(all_operations) == 0, "There should be no operation"

        algo = test_adapter_factory(TestAdapter3)
        adapter_instance = ABCAdapter.build_adapter(algo)
        data = {model_burst.RANGE_PARAMETER_1: 'param_5', 'param_5': [1, 2]}
        ## Create Group of operations
        OperationService().fire_operation(adapter_instance, self.test_user, self.test_project.id)

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

    def test_initiate_operation(self, test_adapter_factory):
        """
        Test the actual operation flow by executing a test adapter.
        """
        module = "tvb.tests.framework.adapters.testadapter1"
        class_name = "TestAdapter1"
        test_adapter_factory()
        adapter = TestFactory.create_adapter(module, class_name)
        output = adapter.get_output()
        output_type = output[0].__name__
        tmp_folder = FilesHelper().get_project_folder(self.test_project, "TEMP")

        view_model = adapter.get_view_model()()
        view_model.test1_val1 = 5
        view_model.test1_val2 = 5
        self.operation_service.initiate_operation(self.test_user, self.test_project, adapter,
                                                  tmp_folder, model_view=view_model)

        group = dao.get_algorithm_by_module(module, class_name)
        assert group.module == 'tvb.tests.framework.adapters.testadapter1', "Wrong data stored."
        assert group.classname == 'TestAdapter1', "Wrong data stored."
        dts, count = dao.get_values_of_datatype(self.test_project.id, DummyDataTypeIndex)
        assert count == 1
        assert len(dts) == 1
        datatype = dao.get_datatype_by_id(dts[0][0])
        assert datatype.subject == DataTypeMetaData.DEFAULT_SUBJECT, "Wrong data stored."
        assert datatype.type == output_type, "Wrong data stored."

    def test_delete_dt_free_hdd_space(self, test_adapter_factory, operation_factory):
        """
        Launch two operations and give enough available space for user so that both should finish.
        """
        test_adapter_factory(adapter_class=TestAdapterHDDRequired)
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.testadapter3", "TestAdapterHDDRequired")
        view_model = adapter.get_view_model()()
        TvbProfile.current.MAX_DISK_SPACE = float(adapter.get_required_disk_size(view_model))
        tmp_folder = FilesHelper().get_project_folder(self.test_project, "TEMP")

        self._assert_no_ddti()
        self.operation_service.initiate_operation(self.test_user, self.test_project, adapter, tmp_folder,
                                                  model_view=view_model)
        datatype = self._assert_stored_ddti()

        # Now free some space and relaunch
        ProjectService().remove_datatype(self.test_project.id, datatype.gid)
        self._assert_no_ddti()
        self.operation_service.initiate_operation(self.test_user, self.test_project, adapter, tmp_folder,
                                                  model_view=view_model)
        self._assert_stored_ddti()

    def test_launch_two_ops_hdd_with_space(self, test_adapter_factory):
        """
        Launch two operations and give enough available space for user so that both should finish.
        """
        test_adapter_factory(adapter_class=TestAdapterHDDRequired)
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.testadapter3", "TestAdapterHDDRequired")
        view_model = adapter.get_view_model()()
        TvbProfile.current.MAX_DISK_SPACE = 2 * float(adapter.get_required_disk_size(view_model))
        tmp_folder = FilesHelper().get_project_folder(self.test_project, "TEMP")

        self.operation_service.initiate_operation(self.test_user, self.test_project, adapter, tmp_folder,
                                                  model_view=view_model)
        datatype = self._assert_stored_ddti()

        # Now update the maximum disk size to be the size of the previously resulted datatypes (transform from kB to MB)
        # plus what is estimated to be required from the next one (transform from B to MB)
        TvbProfile.current.MAX_DISK_SPACE = float(datatype.disk_size) + float(
            adapter.get_required_disk_size(view_model))

        self.operation_service.initiate_operation(self.test_user, self.test_project, adapter, tmp_folder,
                                                  model_view=view_model)
        self._assert_stored_ddti(2)

    def test_launch_two_ops_hdd_full_space(self):
        """
        Launch two operations and give available space for user so that the first should finish,
        but after the update to the user hdd size the second should not.
        """
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.testadapter3", "TestAdapterHDDRequired")
        view_model = adapter.get_view_model()()

        TvbProfile.current.MAX_DISK_SPACE = (1 + float(adapter.get_required_disk_size(view_model)))
        tmp_folder = FilesHelper().get_project_folder(self.test_project, "TEMP")
        self.operation_service.initiate_operation(self.test_user, self.test_project, adapter, tmp_folder,
                                                  model_view=view_model)

        datatype = self._assert_stored_ddti()
        # Now update the maximum disk size to be less than size of the previously resulted datatypes (transform kB to MB)
        # plus what is estimated to be required from the next one (transform from B to MB)
        TvbProfile.current.MAX_DISK_SPACE = float(datatype.disk_size - 1) + \
                                            float(adapter.get_required_disk_size(view_model) - 1)

        with pytest.raises(NoMemoryAvailableException):
            self.operation_service.initiate_operation(self.test_user, self.test_project, adapter, tmp_folder,
                                                      model_view=view_model)
        self._assert_stored_ddti()

    def test_launch_operation_hdd_with_space(self):
        """
        Test the actual operation flow by executing a test adapter.
        """
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.testadapter3", "TestAdapterHDDRequired")
        view_model = adapter.get_view_model()()

        TvbProfile.current.MAX_DISK_SPACE = float(adapter.get_required_disk_size(view_model))
        tmp_folder = FilesHelper().get_project_folder(self.test_project, "TEMP")
        self.operation_service.initiate_operation(self.test_user, self.test_project, adapter, tmp_folder,
                                                  model_view=view_model)
        self._assert_stored_ddti()

    def test_launch_operation_hdd_with_space_started_ops(self, test_adapter_factory):
        """
        Test the actual operation flow by executing a test adapter.
        """
        test_adapter_factory(adapter_class=TestAdapterHDDRequired)

        space_taken_by_started = 100
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.testadapter3", "TestAdapterHDDRequired")
        form = TestAdapterHDDRequiredForm()
        adapter.submit_form(form)
        started_operation = model_operation.Operation(self.test_user.id, self.test_project.id,
                                                      adapter.stored_adapter.id, "",
                                                      status=model_operation.STATUS_STARTED,
                                                      estimated_disk_size=space_taken_by_started)
        view_model = adapter.get_view_model()()

        dao.store_entity(started_operation)
        TvbProfile.current.MAX_DISK_SPACE = float(adapter.get_required_disk_size(view_model) + space_taken_by_started)
        tmp_folder = FilesHelper().get_project_folder(self.test_project, "TEMP")
        self.operation_service.initiate_operation(self.test_user, self.test_project, adapter, tmp_folder,
                                                  model_view=view_model)
        self._assert_stored_ddti()

    def test_launch_operation_hdd_full_space(self, test_adapter_factory):
        """
        Test the actual operation flow by executing a test adapter.
        """
        test_adapter_factory(adapter_class=TestAdapterHDDRequired)

        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.testadapter3", "TestAdapterHDDRequired")
        form = TestAdapterHDDRequiredForm()
        adapter.submit_form(form)
        view_model = adapter.get_view_model()()

        TvbProfile.current.MAX_DISK_SPACE = float(adapter.get_required_disk_size(view_model) - 1)
        tmp_folder = FilesHelper().get_project_folder(self.test_project, "TEMP")
        with pytest.raises(NoMemoryAvailableException):
            self.operation_service.initiate_operation(self.test_user, self.test_project, adapter, tmp_folder,
                                                      model_view=view_model)
        self._assert_no_ddti()

    def test_launch_operation_hdd_full_space_started_ops(self, test_adapter_factory):
        """
        Test the actual operation flow by executing a test adapter.
        """
        test_adapter_factory(adapter_class=TestAdapterHDDRequired)

        space_taken_by_started = 100
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.testadapter3", "TestAdapterHDDRequired")
        form = TestAdapterHDDRequiredForm()
        adapter.submit_form(form)
        started_operation = model_operation.Operation(self.test_user.id, self.test_project.id,
                                                      adapter.stored_adapter.id, "",
                                                      status=model_operation.STATUS_STARTED,
                                                      estimated_disk_size=space_taken_by_started)
        view_model = adapter.get_view_model()()

        dao.store_entity(started_operation)
        TvbProfile.current.MAX_DISK_SPACE = float(
            adapter.get_required_disk_size(view_model) + space_taken_by_started - 1)
        tmp_folder = FilesHelper().get_project_folder(self.test_project, "TEMP")
        with pytest.raises(NoMemoryAvailableException):
            self.operation_service.initiate_operation(self.test_user, self.test_project, adapter, tmp_folder,
                                                      model_view=view_model)
        self._assert_no_ddti()

    def test_stop_operation(self, test_adapter_factory):
        """
        Test that an operation is successfully stopped.
        """
        test_adapter_factory(adapter_class=TestAdapter2)
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.testadapter2", "TestAdapter2")
        view_model = adapter.get_view_model()()
        view_model.test = 5
        algo = adapter.stored_adapter
        algo_category = dao.get_category_by_id(algo.fk_category)
        operations, _ = self.operation_service.prepare_operations(self.test_user.id, self.test_project, algo,
                                                                  algo_category, {},
                                                                  view_model=view_model)
        self.operation_service._send_to_cluster(operations, adapter)
        self.operation_service.stop_operation(operations[0].id)
        operation = dao.get_operation_by_id(operations[0].id)
        assert operation.status, model_operation.STATUS_CANCELED == "Operation should have been canceled!"

    def test_stop_operation_finished(self, test_adapter_factory):
        """
        Test that an operation that is already finished is not changed by the stop operation.
        """
        test_adapter_factory()
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.testadapter1", "TestAdapter1")
        view_model = adapter.get_view_model()()
        view_model.test1_val1 = 5
        view_model.test1_val2 = 5
        algo = adapter.stored_adapter
        algo_category = dao.get_category_by_id(algo.fk_category)
        operations, _ = self.operation_service.prepare_operations(self.test_user.id, self.test_project, algo,
                                                                  algo_category, {}, view_model=view_model)
        self.operation_service._send_to_cluster(operations, adapter)
        operation = dao.get_operation_by_id(operations[0].id)
        operation.status = model_operation.STATUS_FINISHED
        dao.store_entity(operation)
        self.operation_service.stop_operation(operations[0].id)
        operation = dao.get_operation_by_id(operations[0].id)
        assert operation.status, model_operation.STATUS_FINISHED == "Operation shouldn't have been canceled!"

    def test_fire_operation(self):
        """
        Test preparation of an adapter and launch mechanism.
        """
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.testadapter1", "TestAdapter1")
        test_user = TestFactory.create_user(username="test_user_fire_sim")
        test_project = TestFactory.create_project(admin=test_user, name="test_project_fire_sim")

        result = OperationService().fire_operation(adapter, test_user, test_project.id,
                                                   view_model=adapter.get_view_model()())
        assert result.endswith("has finished."), "Operation fail"
