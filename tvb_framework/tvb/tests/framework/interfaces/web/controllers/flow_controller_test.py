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

"""
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""
from time import sleep
import cherrypy

from tvb.basic.profile import TvbProfile
from tvb.config.algorithm_categories import CreateAlgorithmCategoryConfig
from tvb.core.entities.model.model_operation import STATUS_PENDING
from tvb.core.entities.storage import dao
from tvb.core.services.backend_clients.standalone_client import LOCKS_QUEUE, StandAloneClient
from tvb.core.services.operation_service import OperationService
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.flow_controller import FlowController
from tvb.interfaces.web.controllers.simulator.simulator_controller import SimulatorController
from tvb.tests.framework.adapters.dummy_adapter1 import DummyAdapter1Form, DummyModel
from tvb.tests.framework.core.factory import TestFactory, STATUS_CANCELED, STATUS_STARTED
from tvb.tests.framework.interfaces.web.controllers.base_controller_test import BaseControllersTest


class TestFlowController(BaseControllersTest):
    """ Unit tests for FlowController """

    def setup_method(self):
        """
        Sets up the environment for testing;
        creates a `FlowController`
        """
        self.init()
        self.flow_c = FlowController()
        self.burst_c = SimulatorController()
        self.operation_service = OperationService()

    def teardown_method(self):
        """ Cleans up the testing environment """
        self.cleanup()
        self.clean_database()

    def test_context_selected(self):
        """
        Remove the project from CherryPy session and check that you are redirected to projects page.
        """
        del cherrypy.session[common.KEY_PROJECT]
        self._expect_redirect('/project/viewall', self.flow_c.step_analyzers)

    def test_valid_step(self):
        """
        For all algorithm categories check that a submenu is generated and the result
        page has it's title given by category name.
        """
        result_dict = self.flow_c.step_analyzers()
        assert common.KEY_SUBMENU_LIST in result_dict, \
            "Expect to have a submenu with available algorithms for category."
        assert result_dict["section_name"] == 'analyze'

    def test_step_connectivity(self):
        """
        Check that the correct section name and connectivity sub-menu are returned for the connectivity step.
        """
        result_dict = self.flow_c.step_connectivity()
        assert result_dict['section_name'] == 'connectivity'
        assert result_dict['submenu_list'] == self.flow_c.connectivity_submenu

    def test_default(self):
        """
        Test default method from step controllers. Check that the submit link is ok, that a mainContent
        is present in result dict and that the isAdapter flag is set to true.
        """
        cherrypy.request.method = "GET"
        categories = dao.get_algorithm_categories()
        for categ in categories:
            # Ignore creators, as those won't go through this flow
            if categ.displayname in [CreateAlgorithmCategoryConfig.category_name]:
                continue
            algo_groups = dao.get_adapters_from_categories([categ.id])
            for algo in algo_groups:
                result_dict = self.flow_c.default(categ.id, algo.id)
                assert result_dict[common.KEY_SUBMIT_LINK] == '/flow/%i/%i' % (categ.id, algo.id)
                assert 'mainContent' in result_dict
                assert result_dict['isAdapter']

    def test_default_cancel(self):
        """
        On cancel we should get a redirect to the back page link.
        """
        cherrypy.request.method = "POST"
        categories = dao.get_algorithm_categories()
        algo_groups = dao.get_adapters_from_categories([categories[0].id])
        self._expect_redirect('/project/viewoperations/%i' % self.test_project.id, self.flow_c.default,
                              categories[0].id, algo_groups[0].id, cancel=True, back_page='operations')

    def test_default_invalid_key(self):
        """
        Pass invalid keys for adapter and step and check you get redirect to tvb entry
        page with error set.
        """
        self._expect_redirect('/tvb?error=True', self.flow_c.default, 'invalid', 'invalid')

    def test_read_datatype_attribute(self, dummy_datatype_index_factory):
        """
        Read an attribute from a datatype.
        """
        dt = dummy_datatype_index_factory(row1='This is stored data')
        dt.subject = "test_subject"
        dt.state = "RAW_STATE"

        returned_data = self.flow_c.read_datatype_attribute(dt.gid, "row1")
        assert returned_data == '"This is stored data"'

    def test_read_datatype_attribute_method_call(self, dummy_datatype_index_factory):
        """
        Call method on given datatype.
        """
        dt = dummy_datatype_index_factory(row1='This is stored data')
        args = {'length': 101}
        returned_data = self.flow_c.read_datatype_attribute(dt.gid, 'return_test_data', **args)
        assert returned_data.replace('"', '') == " ".join(str(x) for x in range(101))

    def test_get_simple_adapter_interface(self, test_adapter_factory):
        algo = test_adapter_factory()
        form = DummyAdapter1Form()
        adapter = TestFactory.create_adapter('tvb.tests.framework.adapters.dummy_adapter1', 'DummyAdapter1')
        adapter.submit_form(form)
        result = self.flow_c.get_simple_adapter_interface(algo.id)
        expected_interface = adapter.get_form()
        found_form = result['adapter_form']['adapter_form']
        assert isinstance(result['adapter_form'], dict)
        assert isinstance(found_form, DummyAdapter1Form)
        assert found_form.test1_val1.value == expected_interface.test1_val1.value
        assert found_form.test1_val2.value == expected_interface.test1_val2.value

    def test_stop_burst_operation(self, simulation_launch):
        operation = simulation_launch(self.test_user, self.test_project, 1000)
        assert not operation.has_finished
        sleep(5)
        self.flow_c.cancel_or_remove_operation(operation.id, 0, False)
        operation = dao.get_operation_by_id(operation.id)
        assert operation.status == STATUS_CANCELED

    def test_stop_burst_operation_group(self, simulation_launch):
        first_op = simulation_launch(self.test_user, self.test_project, 1000, True)
        operations_group_id = first_op.fk_operation_group
        assert not first_op.has_finished
        sleep(5)
        self.flow_c.cancel_or_remove_operation(operations_group_id, 1, False)
        operations = dao.get_operations_in_group(operations_group_id)
        for operation in operations:
            operation = dao.get_operation_by_id(operation.id)
            assert operation.status == STATUS_CANCELED

    def test_remove_burst_operation(self, simulation_launch):
        operation = simulation_launch(self.test_user, self.test_project, 1000)
        assert not operation.has_finished
        self.flow_c.cancel_or_remove_operation(operation.id, 0, True)
        operation = dao.try_get_operation_by_id(operation.id)
        assert operation is None

    def test_launch_multiple_operations(self, simulation_launch):
        assert TvbProfile.current.MAX_THREADS_NUMBER == LOCKS_QUEUE.qsize(), "Queue wasn't correctly initialized"
        # Launch more operations that can be executed in parallel
        operations = []
        for i in range(TvbProfile.current.MAX_THREADS_NUMBER + 2):
            operations.append(simulation_launch(self.test_user, self.test_project, 1000))
        # Wait until queue is actually full of Started operations
        preparing_operations = True
        while preparing_operations:
            op_ready = True
            for i in range(TvbProfile.current.MAX_THREADS_NUMBER):
                op = dao.get_operation_by_id(operations[i].id)
                if op.status == STATUS_PENDING:
                    op_ready = False
                    sleep(0.3)
                    break
            if op_ready:
                preparing_operations = False
        # All started except for the last, as the queue is not enough to launch all
        for i, operation in enumerate(operations):
            op = dao.get_operation_by_id(operation.id)
            if i >= TvbProfile.current.MAX_THREADS_NUMBER:
                assert op.status == STATUS_PENDING
            else:
                assert op.status == STATUS_STARTED
        # wait while queue is still full
        while LOCKS_QUEUE.qsize() == 0:
            sleep(0.3)
        # Make sure the last operation is being launched ...
        op = operations[len(operations) - 1]
        while dao.get_operation_by_id(op.id).status == STATUS_PENDING:
            # Simulate the background job launching next round of operations
            StandAloneClient.process_queued_operations()
            sleep(0.3)
        op = dao.get_operation_by_id(op.id)
        assert op.status == STATUS_STARTED

    def test_remove_burst_operation_group(self, simulation_launch):
        first_op = simulation_launch(self.test_user, self.test_project, 1000, True)
        operations_group_id = first_op.fk_operation_group
        assert not first_op.has_finished
        self.flow_c.cancel_or_remove_operation(operations_group_id, 1, True)
        operations = dao.get_operations_in_group(operations_group_id)
        for operation in operations:
            operation = dao.try_get_operation_by_id(operation.id)
            assert operation is None

    def _asynch_launch_simple_op(self):
        adapter = TestFactory.create_adapter('tvb.tests.framework.adapters.dummy_adapter1', 'DummyAdapter1')
        view_model = DummyModel()
        view_model.test1_val1 = 5
        view_model.test1_val2 = 6
        algo = adapter.stored_adapter
        operation = self.operation_service.prepare_operation(self.test_user.id, self.test_project, algo,
                                                             view_model=view_model)
        self.operation_service._send_to_cluster(operation, adapter)
        return operation

    def test_stop_operation(self):
        operation = self._asynch_launch_simple_op()
        operation = dao.get_operation_by_id(operation.id)
        assert not operation.has_finished
        self.flow_c.cancel_or_remove_operation(operation.id, 0, False)
        operation = dao.get_operation_by_id(operation.id)
        assert operation.status == STATUS_CANCELED

    def test_stop_operations_group(self, test_adapter_factory, datatype_group_factory):
        group, _ = datatype_group_factory(status=STATUS_STARTED, store_vm=True)
        operations = dao.get_operations_in_group(group.fk_from_operation)
        operation_group_id = 0
        for operation in operations:
            operation = dao.get_operation_by_id(operation.id)
            assert not operation.has_finished
            operation_group_id = operation.fk_operation_group
        self.flow_c.cancel_or_remove_operation(operation_group_id, 1, False)
        for operation in operations:
            operation = dao.get_operation_by_id(operation.id)
            assert operation.status == STATUS_CANCELED
