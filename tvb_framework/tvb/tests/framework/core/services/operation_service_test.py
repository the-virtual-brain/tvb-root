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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import pytest
import uuid

from tvb.basic.profile import TvbProfile
from tvb.core.adapters.exceptions import NoMemoryAvailableException
from tvb.core.entities.model import model_operation
from tvb.core.entities.storage import dao
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.services.operation_service import OperationService
from tvb.core.services.project_service import initialize_storage, ProjectService
from tvb.storage.storage_interface import StorageInterface
from tvb.tests.framework.adapters.dummy_adapter2 import DummyAdapter2
from tvb.tests.framework.adapters.dummy_adapter3 import *
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

    def test_datatypes_groups(self, test_adapter_factory, datatype_group_factory):
        """
        Tests if the dataType group is set correct on the dataTypes resulted from the same operation group.
        """
        all_operations = dao.get_filtered_operations(self.test_project.id, None)
        assert len(all_operations) == 0, "There should be no operation"

        dt_group, _ = datatype_group_factory(project=self.test_project)
        model = DummyModel()
        test_adapter_factory()
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.dummy_adapter1", "DummyAdapter1")

        operations = dao.get_operations_in_group(dt_group.id)

        for op in operations:
            model.gid = uuid.uuid4()
            op_path = StorageInterface().get_project_folder(self.test_project.name, str(op.id))
            op.view_model_gid = model.gid.hex
            op.algorithm = adapter.stored_adapter
            h5.store_view_model(model, op_path)
            dao.store_entity(op)

        all_operations = dao.get_filtered_operations(self.test_project.id, None)
        assert len(all_operations) == 2, "Expected two operation groups"
        assert all_operations[0][2] == 6, "Expected 6 operations in one group"

        operation_group_id = all_operations[0][3]
        assert operation_group_id != None, "The operation should be part of a group."

        self.operation_service.stop_operation(all_operations[1][0])
        self.operation_service.stop_operation(all_operations[1][1])
        # Make sure operations are executed
        self.operation_service.launch_operation(all_operations[1][0], False)
        self.operation_service.launch_operation(all_operations[1][1], False)

        resulted_datatypes = dao.get_datatype_in_group(operation_group_id=operation_group_id)
        assert len(resulted_datatypes) >= 2, "Expected at least 2, but: " + str(len(resulted_datatypes))

        dt = dao.get_datatype_by_id(resulted_datatypes[0].id)
        datatype_group = dao.get_datatypegroup_by_op_group_id(operation_group_id)
        assert dt.fk_datatype_group == datatype_group.id, "DataTypeGroup is incorrect"

    def test_initiate_operation(self, test_adapter_factory):
        """
        Test the actual operation flow by executing a test adapter.
        """
        test_adapter_factory()
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.dummy_adapter1", "DummyAdapter1")
        view_model = DummyModel()
        view_model.test1_val1 = 5
        view_model.test1_val2 = 5
        adapter.generic_attributes.subject = "Test4242"

        self.operation_service.initiate_operation(self.test_user, self.test_project, adapter,
                                                  model_view=view_model)

        dts, count = dao.get_values_of_datatype(self.test_project.id, DummyDataTypeIndex)
        assert count == 1
        assert len(dts) == 1
        datatype = dao.get_datatype_by_id(dts[0][0])
        assert datatype.subject == "Test4242", "Wrong data stored."
        assert datatype.type == adapter.get_output()[0].__name__, "Wrong data stored."

    def test_delete_dt_free_hdd_space(self, test_adapter_factory, operation_factory):
        """
        Launch two operations and give enough available space for user so that both should finish.
        """
        test_adapter_factory(adapter_class=DummyAdapterHDDRequired)
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.dummy_adapter3", "DummyAdapterHDDRequired")
        view_model = adapter.get_view_model()()
        TvbProfile.current.MAX_DISK_SPACE = float(adapter.get_required_disk_size(view_model))

        self._assert_no_ddti()
        self.operation_service.initiate_operation(self.test_user, self.test_project, adapter,
                                                  model_view=view_model)
        datatype = self._assert_stored_ddti()

        # Now free some space and relaunch
        ProjectService().remove_datatype(self.test_project.id, datatype.gid)
        self._assert_no_ddti()
        self.operation_service.initiate_operation(self.test_user, self.test_project, adapter,
                                                  model_view=view_model)
        self._assert_stored_ddti()

    def test_launch_two_ops_hdd_with_space(self, test_adapter_factory):
        """
        Launch two operations and give enough available space for user so that both should finish.
        """
        test_adapter_factory(adapter_class=DummyAdapterHDDRequired)
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.dummy_adapter3", "DummyAdapterHDDRequired")
        view_model = adapter.get_view_model()()
        TvbProfile.current.MAX_DISK_SPACE = 2 * float(adapter.get_required_disk_size(view_model))

        self.operation_service.initiate_operation(self.test_user, self.test_project, adapter,
                                                  model_view=view_model)
        datatype = self._assert_stored_ddti()

        # Now update the maximum disk size to be the size of the previously resulted datatypes (transform from kB to MB)
        # plus what is estimated to be required from the next one (transform from B to MB)
        TvbProfile.current.MAX_DISK_SPACE = float(datatype.disk_size) + float(
            adapter.get_required_disk_size(view_model))

        self.operation_service.initiate_operation(self.test_user, self.test_project, adapter,
                                                  model_view=view_model)
        self._assert_stored_ddti(2)

    def test_launch_two_ops_hdd_full_space(self):
        """
        Launch two operations and give available space for user so that the first should finish,
        but after the update to the user hdd size the second should not.
        """
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.dummy_adapter3", "DummyAdapterHDDRequired")
        view_model = adapter.get_view_model()()

        TvbProfile.current.MAX_DISK_SPACE = (1 + float(adapter.get_required_disk_size(view_model)))
        self.operation_service.initiate_operation(self.test_user, self.test_project, adapter,
                                                  model_view=view_model)

        datatype = self._assert_stored_ddti()
        # Now update the maximum disk size to be less than size of the previously resulted dts (transform kB to MB)
        # plus what is estimated to be required from the next one (transform from B to MB)
        TvbProfile.current.MAX_DISK_SPACE = float(datatype.disk_size - 1) + \
                                            float(adapter.get_required_disk_size(view_model) - 1)

        with pytest.raises(NoMemoryAvailableException):
            self.operation_service.initiate_operation(self.test_user, self.test_project, adapter,
                                                      model_view=view_model)
        self._assert_stored_ddti()

    def test_launch_operation_hdd_with_space(self):
        """
        Test the actual operation flow by executing a test adapter.
        """
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.dummy_adapter3", "DummyAdapterHDDRequired")
        view_model = adapter.get_view_model()()

        TvbProfile.current.MAX_DISK_SPACE = float(adapter.get_required_disk_size(view_model))
        self.operation_service.initiate_operation(self.test_user, self.test_project, adapter,
                                                  model_view=view_model)
        self._assert_stored_ddti()

    def test_launch_operation_hdd_with_space_started_ops(self, test_adapter_factory):
        """
        Test the actual operation flow by executing a test adapter.
        """
        test_adapter_factory(adapter_class=DummyAdapterHDDRequired)
        space_taken_by_started = 100
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.dummy_adapter3", "DummyAdapterHDDRequired")
        form = DummyAdapterHDDRequiredForm()
        adapter.submit_form(form)
        started_operation = model_operation.Operation(None, self.test_user.id, self.test_project.id,
                                                      adapter.stored_adapter.id,
                                                      status=model_operation.STATUS_STARTED,
                                                      estimated_disk_size=space_taken_by_started)
        view_model = adapter.get_view_model()()
        dao.store_entity(started_operation)
        TvbProfile.current.MAX_DISK_SPACE = float(adapter.get_required_disk_size(view_model) + space_taken_by_started)

        self.operation_service.initiate_operation(self.test_user, self.test_project, adapter,
                                                  model_view=view_model)
        self._assert_stored_ddti()

    def test_launch_operation_hdd_full_space(self, test_adapter_factory):
        """
        Test the actual operation flow by executing a test adapter.
        """
        test_adapter_factory(adapter_class=DummyAdapterHDDRequired)
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.dummy_adapter3", "DummyAdapterHDDRequired")
        form = DummyAdapterHDDRequiredForm()
        adapter.submit_form(form)
        view_model = adapter.get_view_model()()
        TvbProfile.current.MAX_DISK_SPACE = float(adapter.get_required_disk_size(view_model) - 1)

        with pytest.raises(NoMemoryAvailableException):
            self.operation_service.initiate_operation(self.test_user, self.test_project, adapter,
                                                      model_view=view_model)
        self._assert_no_ddti()

    def test_launch_operation_hdd_full_space_started_ops(self, test_adapter_factory):
        """
        Test the actual operation flow by executing a test adapter.
        """
        test_adapter_factory(adapter_class=DummyAdapterHDDRequired)
        space_taken_by_started = 100
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.dummy_adapter3", "DummyAdapterHDDRequired")
        form = DummyAdapterHDDRequiredForm()
        adapter.submit_form(form)
        started_operation = model_operation.Operation(None, self.test_user.id, self.test_project.id,
                                                      adapter.stored_adapter.id,
                                                      status=model_operation.STATUS_STARTED,
                                                      estimated_disk_size=space_taken_by_started)
        view_model = adapter.get_view_model()()
        dao.store_entity(started_operation)
        TvbProfile.current.MAX_DISK_SPACE = float(adapter.get_required_disk_size(view_model) +
                                                  space_taken_by_started - 1)

        with pytest.raises(NoMemoryAvailableException):
            self.operation_service.initiate_operation(self.test_user, self.test_project, adapter,
                                                      model_view=view_model)
        self._assert_no_ddti()

    def test_stop_operation(self, test_adapter_factory):
        """
        Test that an operation is successfully stopped.
        """
        test_adapter_factory(adapter_class=DummyAdapter2)
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.dummy_adapter2", "DummyAdapter2")
        view_model = adapter.get_view_model()()
        view_model.test = 5
        algo = adapter.stored_adapter
        operation = self.operation_service.prepare_operation(self.test_user.id, self.test_project, algo,
                                                             view_model=view_model)

        self.operation_service._send_to_cluster(operation, adapter)
        self.operation_service.stop_operation(operation)

        operation = dao.get_operation_by_id(operation.id)
        assert operation.status, model_operation.STATUS_CANCELED == "Operation should have been canceled!"

    def test_stop_operation_finished(self, test_adapter_factory):
        """
        Test that an operation that is already finished is not changed by the stop operation.
        """
        test_adapter_factory()
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.dummy_adapter1", "DummyAdapter1")
        view_model = adapter.get_view_model()()
        view_model.test1_val1 = 5
        view_model.test1_val2 = 5
        algo = adapter.stored_adapter
        operation = self.operation_service.prepare_operation(self.test_user.id, self.test_project, algo,
                                                             view_model=view_model)
        self.operation_service._send_to_cluster(operation, adapter)
        operation = dao.get_operation_by_id(operation.id)
        operation.status = model_operation.STATUS_FINISHED
        dao.store_entity(operation)
        self.operation_service.stop_operation(operation.id)
        operation = dao.get_operation_by_id(operation.id)
        assert operation.status, model_operation.STATUS_FINISHED == "Operation shouldn't have been canceled!"

    def test_fire_operation(self):
        """
        Test preparation of an adapter and launch mechanism.
        """
        adapter = TestFactory.create_adapter("tvb.tests.framework.adapters.dummy_adapter1", "DummyAdapter1")
        test_user = TestFactory.create_user(username="test_user_fire_sim")
        test_project = TestFactory.create_project(admin=test_user, name="test_project_fire_sim")

        result = OperationService().fire_operation(adapter, test_user, test_project.id,
                                                   view_model=adapter.get_view_model()())
        assert result.endswith("has finished."), "Operation fail"
