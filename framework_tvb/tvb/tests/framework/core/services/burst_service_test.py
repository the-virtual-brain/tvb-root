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
.. moduleauthor:: bogdan.neacsa <bogdan.neacsa@codemart.ro>
"""

import copy
import pytest
import numpy
import json
from time import sleep
from tvb.tests.framework.core.base_testcase import BaseTestCase
from tvb.core.adapters.input_tree import InputTreeManager
from tvb.config import SIMULATOR_MODULE, SIMULATOR_CLASS
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.mapped_values import DatatypeMeasure
from tvb.datatypes.time_series import TimeSeriesRegion
from tvb.datatypes.simulation_state import SimulationState
from tvb.core.entities import model
from tvb.core.entities.model import BurstConfiguration
from tvb.core.entities.storage import dao
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.transient.burst_configuration_entities import WorkflowStepConfiguration as wf_cfg
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.services.burst_service import BurstService
from tvb.core.services.flow_service import FlowService
from tvb.core.services.workflow_service import WorkflowService
from tvb.core.services.project_service import ProjectService
from tvb.core.services.operation_service import OperationService
from tvb.core.services.exceptions import InvalidPortletConfiguration
from tvb.core.portlets.xml_reader import KEY_DYNAMIC
from tvb.core.portlets.portlet_configurer import ADAPTER_PREFIX_ROOT
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.datatypes.datatype1 import Datatype1
from tvb.tests.framework.datatypes.datatype2 import Datatype2
from tvb.tests.framework.datatypes import datatypes_factory
from tvb.tests.framework.adapters.storeadapter import StoreAdapter
from tvb.tests.framework.adapters.simulator.simulator_adapter_test import SIMULATOR_PARAMETERS



class TestBurstService(BaseTestCase):
    """
    Test the service layer for BURST PAGE. We can't have this transactional since
    we launch operations in different threads and the transactional operator only rolls back 
    sessions bounded to the current thread transaction.
    """
    PORTLET_ID = "TA1TA2"
    ## This should not be present in portlets.xml
    INVALID_PORTLET_ID = "this_is_not_a_non_existent_test_portlet_ID"

    burst_service = BurstService()
    flow_service = FlowService()
    operation_service = OperationService()
    workflow_service = WorkflowService()
    sim_algorithm = flow_service.get_algorithm_by_module_and_class(SIMULATOR_MODULE, SIMULATOR_CLASS)
    local_simulation_params = copy.deepcopy(SIMULATOR_PARAMETERS)


    def setup_method(self):
        """
        Sets up the environment for running the tests;
        cleans the database before testing and saves config file;
        creates a test user, a test project;
        creates burst, flow, operation and workflow services

        """
        self.clean_database()
        self.test_user = TestFactory.create_user()
        self.test_project = TestFactory.create_project(self.test_user)


    def teardown_method(self):
        """
        Remove project folders and clean up database.
        """
        FilesHelper().remove_project_structure(self.test_project.name)
        self.clean_database()

    def test_new_portlet_configuration(self):
        """
        Test that the correct portlet configuration is generated for the test portlet.
        """
        # Passing an invalid portlet ID should fail and raise an InvalidPortletConfiguration exception.
        with pytest.raises(InvalidPortletConfiguration):
            self.burst_service.new_portlet_configuration(-1)

        # Now the happy flow
        test_portlet = dao.get_portlet_by_identifier(self.PORTLET_ID)
        portlet_configuration = self.burst_service.new_portlet_configuration(test_portlet.id)
        analyzers = portlet_configuration.analyzers
        assert len(
            analyzers) == 1, "Portlet configuration not build properly. " \
                             "Portlet's analyzers list has unexpected number of elements."
        assert analyzers[0].dynamic_param == {u'test_dt_input': {wf_cfg.DATATYPE_INDEX_KEY: 0,
                                                                 wf_cfg.STEP_INDEX_KEY: 0}}, \
            "Dynamic parameters not loaded properly"
        visualizer = portlet_configuration.visualizer
        assert visualizer.dynamic_param == {}, "Dynamic parameters not loaded properly"
        assert visualizer.static_param == {u'test2': u'0'}, 'Static parameters not loaded properly'


    def test_build_portlet_interface(self):
        """
        Test that the portlet interface is build properly, splitted by steps and prefixed.
        """
        test_portlet = dao.get_portlet_by_identifier(self.PORTLET_ID)
        portlet_configuration = self.burst_service.new_portlet_configuration(test_portlet.id)
        actual_interface = self.burst_service.build_portlet_interface(portlet_configuration, self.test_project.id)
        #The expected portlet steps and interface in correspondace to the xml declaration
        #from tvb.tests.framework/core/portlets/test_portlet.xml
        expected_steps = [{'ui_name': 'TestAdapterDatatypeInput'},
                          {'ui_name': 'TestAdapter2'}]
        expected_interface = [{ABCAdapter.KEY_DEFAULT: 'step_0[0]', ABCAdapter.KEY_DISABLED: True,
                               KEY_DYNAMIC: True, ABCAdapter.KEY_NAME: ADAPTER_PREFIX_ROOT + '0test_dt_input'},
                              {ABCAdapter.KEY_DEFAULT: '0', ABCAdapter.KEY_DISABLED: False,
                               KEY_DYNAMIC: False, ABCAdapter.KEY_NAME: ADAPTER_PREFIX_ROOT + '1test2'}]
        for idx, entry in enumerate(expected_steps):
            step = actual_interface[idx]
            for key in entry:
                assert entry.get(key) == getattr(step, key)
            for key in expected_interface[idx]:
                assert expected_interface[idx].get(key, False) == step.interface[0].get(key, False)


    def test_build_portlet_interface_invalid(self):
        """
        Test that a proper exception is raised in case an invalid portlet configuration is provided.
        """
        test_portlet = dao.get_portlet_by_identifier(self.PORTLET_ID)
        portlet_configuration = self.burst_service.new_portlet_configuration(test_portlet.id)
        portlet_configuration.portlet_id = "this-is-invalid"
        with pytest.raises(InvalidPortletConfiguration):
            self.burst_service.build_portlet_interface(portlet_configuration, self.test_project.id)


    def test_update_portlet_config(self):
        """
        Test if a portlet configuration parameters are updated accordingly with a set
        of overwrites that would normally come from UI. Make sure to restart only if 
        analyzer parameters change.
        """


        def __update_params(declared_overwrites, expected_result):
            """
            Do the update and check that we get indeed the expected_result.
            :param declared_overwrites: a input dictionary in the form {'$$name$$' : '$$value$$'}. Make
                sure $$name$$ has the prefix that is added in case of portlet parameters,
                namely ADAPTER_PREFIX_ROOT + step_index + actual_name
            :param expected_result: boolean which should represent if we need or not to restart. (Was a
                visualizer parameter change or an analyzer one)
            """
            result = self.burst_service.update_portlet_configuration(portlet_configuration, declared_overwrites)
            assert expected_result == result, "After update expected %s as 'need_restart' but got %s." \
                                              "" % (expected_result, result)


        test_portlet = dao.get_portlet_by_identifier(self.PORTLET_ID)
        portlet_configuration = self.burst_service.new_portlet_configuration(test_portlet.id)
        previous_entry = portlet_configuration.analyzers[0].static_param['test_non_dt_input']
        declared_overwrites = {ADAPTER_PREFIX_ROOT + '0test_non_dt_input': previous_entry}
        __update_params(declared_overwrites, False)
        declared_overwrites = {ADAPTER_PREFIX_ROOT + '1test2': 'new_value'}
        __update_params(declared_overwrites, False)
        declared_overwrites = {ADAPTER_PREFIX_ROOT + '0test_non_dt_input': '1'}
        __update_params(declared_overwrites, True)


    def test_update_portlet_config_invalid_data(self):
        """
        Trying an update on a portlet configuration with invalid data
        should not change the configuration instance in any way.
        """
        test_portlet = dao.get_portlet_by_identifier(self.PORTLET_ID)
        portlet_configuration = self.burst_service.new_portlet_configuration(test_portlet.id)

        invalid_overwrites = {'this_is_not_a_valid_key': 'for_test_portlet_update'}
        before_update = copy.deepcopy(portlet_configuration)
        self.burst_service.update_portlet_configuration(portlet_configuration, invalid_overwrites)
        assert set(dir(before_update)) == set(dir(portlet_configuration))
        #An update with invalid input data should have no effect on the configuration, but attributes changed
        for key in portlet_configuration.__dict__.keys():
            if hasattr(getattr(portlet_configuration, key), '__call__'):
                assert getattr(before_update, key) == getattr(portlet_configuration, key),\
                                 "The value of attribute %s changed by a update with invalid data "\
                                 "when it shouldn't have." % key


    def test_clone_burst_configuration(self):
        """
        Test that all the major attributes are the same after a clone burst but the
        id of the cloned one is None.
        """
        first_burst = TestFactory.store_burst(self.test_project.id)
        cloned_burst = first_burst.clone()
        self._compare_bursts(first_burst, cloned_burst)
        assert first_burst.selected_tab == cloned_burst.selected_tab, "Selected tabs not equal for bursts."
        assert len(first_burst.tabs) == len(cloned_burst.tabs), "Tabs not equal for bursts."
        assert cloned_burst.id is None, 'id should be none for cloned entry.'


    def test_store_burst_config(self):
        """
        Test that a burst entity is properly stored in db.
        """
        burst_config = TestFactory.store_burst(self.test_project.id)
        assert burst_config.id is not None, 'Burst was not stored properly.'
        stored_entity = dao.get_burst_by_id(burst_config.id)
        assert stored_entity is not None, 'Burst was not stored properly.'
        self._compare_bursts(burst_config, stored_entity)


    def _compare_bursts(self, first_burst, second_burst):
        """
        Compare that all important attributes are the same between two bursts. (name, project id and status)
        """
        assert first_burst.name == second_burst.name, "Names not equal for bursts."
        assert first_burst.fk_project == second_burst.fk_project, "Projects not equal for bursts."
        assert first_burst.status == second_burst.status, "Statuses not equal for bursts."


    def test_getavailablebursts_none(self):
        """
        Test that an empty list is returned if no data is available in db.
        """
        bursts = self.burst_service.get_available_bursts(self.test_project.id)
        assert bursts == [], "Unexpected result returned : %s" % (bursts,)


    def test_get_available_bursts_happy(self):
        """
        Test that all the correct burst are returned for the given project.
        """
        project = model.Project("second_test_proj", self.test_user.id, "description")
        second_project = dao.store_entity(project)
        test_project_bursts = [TestFactory.store_burst(self.test_project.id).id for _ in range(4)]
        second_project_bursts = [TestFactory.store_burst(second_project.id).id for _ in range(3)]
        returned_test_project_bursts = [burst.id for burst in
                                        self.burst_service.get_available_bursts(self.test_project.id)]
        returned_second_project_bursts = [burst.id for burst in
                                          self.burst_service.get_available_bursts(second_project.id)]
        assert len(test_project_bursts) == len(returned_test_project_bursts),\
                         "Incorrect bursts retrieved for project %s." % self.test_project
        assert len(second_project_bursts) == len(returned_second_project_bursts),\
                         "Incorrect bursts retrieved for project %s." % second_project
        assert set(second_project_bursts) == set(returned_second_project_bursts),\
                         "Incorrect bursts retrieved for project %s." % second_project
        assert set(test_project_bursts) == set(returned_test_project_bursts),\
                         "Incorrect bursts retrieved for project %s." % self.test_project


    def test_select_simulator_inputs(self):
        """
        Test that given a dictionary of selected inputs as it would arrive from UI, only
        the selected simulator inputs are kept.
        """
        simulator_input_tree = self.flow_service.prepare_adapter(self.test_project.id, self.sim_algorithm)
        child_parameter = ''
        checked_parameters = {simulator_input_tree[0][ABCAdapter.KEY_NAME]: {model.KEY_PARAMETER_CHECKED: True,
                                                                             model.KEY_SAVED_VALUE: 'new_value'},
                              simulator_input_tree[1][ABCAdapter.KEY_NAME]: {model.KEY_PARAMETER_CHECKED: True,
                                                                             model.KEY_SAVED_VALUE: 'new_value'}}
        #Look for a entry from a subtree to add to the selected simulator inputs
        for idx, entry in enumerate(simulator_input_tree):
            found_it = False
            if idx not in (0, 1) and entry.get(ABCAdapter.KEY_OPTIONS, False):
                for option in entry[ABCAdapter.KEY_OPTIONS]:
                    if option[ABCAdapter.KEY_VALUE] == entry[ABCAdapter.KEY_DEFAULT]:
                        if option[ABCAdapter.KEY_ATTRIBUTES]:
                            child_parameter = option[ABCAdapter.KEY_ATTRIBUTES][0][ABCAdapter.KEY_NAME]
                            checked_parameters[entry[ABCAdapter.KEY_NAME]] = {model.KEY_PARAMETER_CHECKED: False,
                                                                              model.KEY_SAVED_VALUE: entry[
                                                                                  ABCAdapter.KEY_DEFAULT]}
                            checked_parameters[child_parameter] = {model.KEY_PARAMETER_CHECKED: True,
                                                                   model.KEY_SAVED_VALUE: 'new_value'}
                            found_it = True
                            break
            if found_it:
                break
        assert child_parameter != '', "Could not find any sub-tree entry in simulator interface."
        subtree = InputTreeManager.select_simulator_inputs(simulator_input_tree, checked_parameters)
        #After the select method we expect only the checked parameters entries to remain with
        #the new values updated accordingly.
        expected_outputs = [{ABCAdapter.KEY_NAME: simulator_input_tree[0][ABCAdapter.KEY_NAME],
                             ABCAdapter.KEY_DEFAULT: 'new_value'},
                            {ABCAdapter.KEY_NAME: simulator_input_tree[1][ABCAdapter.KEY_NAME],
                             ABCAdapter.KEY_DEFAULT: 'new_value'},
                            {ABCAdapter.KEY_NAME: child_parameter,
                             ABCAdapter.KEY_DEFAULT: 'new_value'}]
        assert len(expected_outputs) == len(subtree),\
                         "Some entries that should not have been displayed still are."
        for idx, entry in enumerate(expected_outputs):
            assert expected_outputs[idx][ABCAdapter.KEY_NAME] == subtree[idx][ABCAdapter.KEY_NAME]
            assert expected_outputs[idx][ABCAdapter.KEY_DEFAULT] == subtree[idx][ABCAdapter.KEY_DEFAULT],\
                             'Default value not update properly.'


    def test_rename_burst(self):
        """
        Test that renaming of a burst functions properly.
        """
        burst_config = TestFactory.store_burst(self.test_project.id)
        self.burst_service.rename_burst(burst_config.id, "new_burst_name")
        loaded_burst = dao.get_burst_by_id(burst_config.id)
        assert loaded_burst.name == "new_burst_name", "Burst was not renamed properly."


    def test_load_burst(self):
        """ 
        Test that the load burst works properly. NOTE: this method is also tested
        in the actual burst launch tests. This is just basic test to verify that the simulator
        interface is loaded properly.
        """
        burst_config = TestFactory.store_burst(self.test_project.id)
        loaded_burst = self.burst_service.load_burst(burst_config.id)[0]
        assert loaded_burst.simulator_configuration == {}, "No simulator configuration should have been loaded"
        assert burst_config.fk_project == loaded_burst.fk_project, "Loaded burst different from original one."
        burst_config = TestFactory.store_burst(self.test_project.id, simulator_config={"test": "test"})
        loaded_burst, _ = self.burst_service.load_burst(burst_config.id)
        assert loaded_burst.simulator_configuration == {"test": "test"}, "different burst loaded"
        assert burst_config.fk_project == loaded_burst.fk_project, "Loaded burst different from original one."


    def test_remove_burst(self):
        """
        Test the remove burst method added to burst_service.
        """
        loaded_burst, _ = self._prepare_and_launch_sync_burst()
        self.burst_service.cancel_or_remove_burst(loaded_burst.id)
        self._check_burst_removed()


    def test_branch_burst(self):
        """
        Test the branching of an existing burst.
        """
        burst_config = self._prepare_and_launch_async_burst(wait_to_finish=60)
        burst_config.prepare_after_load()

        launch_params = self._prepare_simulation_params(4)
        burst_config.update_simulator_configuration(launch_params)

        burst_id, _ = self.burst_service.launch_burst(burst_config, 0, self.sim_algorithm.id,
                                                      self.test_user.id, "branch")
        burst_config = dao.get_burst_by_id(burst_id)
        self._wait_for_burst(burst_config)

        ts_regions = self.count_all_entities(TimeSeriesRegion)
        sim_states = self.count_all_entities(SimulationState)
        assert 2 == ts_regions, "An operation group should have been created for each step."
        assert 2 == sim_states, "An dataType group should have been created for each step."


    def test_remove_group_burst(self):
        """
        Same remove burst but for a burst that contains group of workflows launched as
        it would be from a Parameter Space Exploration. Check that the workflows are also
        deleted with the burst.
        """
        burst_config = self._prepare_and_launch_async_burst(length=1, is_range=True, nr_ops=4, wait_to_finish=60)

        launched_workflows = dao.get_workflows_for_burst(burst_config.id, is_count=True)
        assert 4 == launched_workflows, "4 workflows should have been launched due to group parameter."

        got_deleted = self.burst_service.cancel_or_remove_burst(burst_config.id)
        assert got_deleted, "Burst should be deleted"

        launched_workflows = dao.get_workflows_for_burst(burst_config.id, is_count=True)
        assert 0 == launched_workflows, "No workflows should remain after delete."

        burst_config = dao.get_burst_by_id(burst_config.id)
        assert burst_config is None, "Removing a canceled burst should delete it from db."


    def test_remove_started_burst(self):
        """
        Try removing a started burst, which should result in it getting canceled.
        """
        burst_entity = self._prepare_and_launch_async_burst(length=20000)
        assert BurstConfiguration.BURST_RUNNING == burst_entity.status,\
                         'A 20000 length simulation should still be started immediately after launch.'
        got_deleted = self.burst_service.cancel_or_remove_burst(burst_entity.id)
        assert not got_deleted, "Burst should be cancelled before deleted."
        burst_entity = dao.get_burst_by_id(burst_entity.id)
        assert BurstConfiguration.BURST_CANCELED == burst_entity.status,\
                         'Deleting a running burst should just cancel it first.'
        got_deleted = self.burst_service.cancel_or_remove_burst(burst_entity.id)
        assert got_deleted, "Burst should be deleted if status is cancelled."
        burst_entity = dao.get_burst_by_id(burst_entity.id)
        assert burst_entity is None, "Removing a canceled burst should delete it from db."


    def test_burst_delete_with_project(self):
        """
        Test that on removal of a project all burst related data is cleared.
        """
        self._prepare_and_launch_sync_burst()
        ProjectService().remove_project(self.test_project.id)
        self._check_burst_removed()


    def test_sync_burst_launch(self):
        """
        A full test for launching a burst. 
        First create the workflow steps and launch the burst.
        Then check that only operation created is for the first adapter from the portlet. The
        second should be viewed as a visualizer.
        After that load the burst and check that the visualizer and analyzer are loaded in the
        corresponding tab and that all the parameters are still the same. Finally check that burst
        status updates corresponding to final operation status.
        """
        loaded_burst, workflow_step_list = self._prepare_and_launch_sync_burst()
        finished, started, error, _, _ = dao.get_operation_numbers(self.test_project.id)
        assert finished == 1, "One operations should have been generated for this burst."
        assert started == 0, "No operations should remain started since workflow was launched synchronous."
        assert error == 0, "No operations should return error status."
        assert loaded_burst.tabs[0].portlets[0] is not None, "Portlet not loaded from config!"
        portlet_config = loaded_burst.tabs[0].portlets[0]
        analyzers = portlet_config.analyzers
        assert len(analyzers) == 0, "Only have 'simulator' and a visualizer. No analyzers should be loaded."
        visualizer = portlet_config.visualizer
        assert visualizer is not None, "Visualizer should not be none."
        assert visualizer.fk_algorithm == workflow_step_list[0].fk_algorithm,\
                         "Different ids after burst load for visualizer."
        assert visualizer.static_param == workflow_step_list[0].static_param,\
                         "Different static params after burst load for visualizer."
        assert visualizer.dynamic_param == workflow_step_list[0].dynamic_param,\
                         "Different static params after burst load for visualizer."


    def test_launch_burst(self):
        """
        Test the launch burst method from burst service.
        """
        first_step_algo = self.flow_service.get_algorithm_by_module_and_class(
            'tvb.tests.framework.adapters.testadapter1', 'TestAdapter1')
        adapter_interface = self.flow_service.prepare_adapter(self.test_project.id, first_step_algo)
        ui_submited_simulator_iface_replica = {}
        kwargs_replica = {}
        for entry in adapter_interface:
            ui_submited_simulator_iface_replica[entry[ABCAdapter.KEY_NAME]] = {model.KEY_PARAMETER_CHECKED: True,
                                                                               model.KEY_SAVED_VALUE: entry[
                                                                                   ABCAdapter.KEY_DEFAULT]}
            kwargs_replica[entry[ABCAdapter.KEY_NAME]] = entry[ABCAdapter.KEY_DEFAULT]
        burst_config = self.burst_service.new_burst_configuration(self.test_project.id)
        burst_config.simulator_configuration = ui_submited_simulator_iface_replica
        test_portlet = dao.get_portlet_by_identifier(self.PORTLET_ID)
        tab_config = {test_portlet.id: [(0, 0), (0, 1), (1, 0)]}
        self._add_portlets_to_burst(burst_config, tab_config)
        burst_config.update_simulator_configuration(kwargs_replica)
        burst_id, _ = self.burst_service.launch_burst(burst_config, 0, first_step_algo.id, self.test_user.id)
        burst_config = dao.get_burst_by_id(burst_id)
        assert burst_config.status in (BurstConfiguration.BURST_FINISHED, BurstConfiguration.BURST_RUNNING),\
                        "Burst not launched successfully!"
        # Wait maximum x seconds for burst to finish
        self._wait_for_burst(burst_config)



    def test_load_group_burst(self):
        """
        Launch a group adapter and load it afterwards and check that a group_id is properly loaded.
        """
        launch_params = self._prepare_simulation_params(1, True, 3)

        burst_config = self.burst_service.new_burst_configuration(self.test_project.id)
        burst_config.update_simulator_configuration(launch_params)
        burst_id, _ = self.burst_service.launch_burst(burst_config, 0, self.sim_algorithm.id, self.test_user.id)
        burst_config = dao.get_burst_by_id(burst_id)
        # Wait maximum x seconds for burst to finish
        self._wait_for_burst(burst_config)

        launched_workflows = dao.get_workflows_for_burst(burst_id, is_count=True)
        assert 3 == launched_workflows, "3 workflows should have been launched due to group parameter."

        group_id = self.burst_service.load_burst(burst_id)[1]
        assert group_id >= 0, "Should be part of group."
        datatype_measures = self.count_all_entities(DatatypeMeasure)
        assert 3 == datatype_measures


    def test_launch_burst_invalid_simulator_parameters(self):
        """
        Test that burst is marked as error if invalid data is passed to the first step.
        """
        algo_id = self.flow_service.get_algorithm_by_module_and_class('tvb.tests.framework.adapters.testadapter1',
                                                                      'TestAdapter1').id
        #Passing invalid kwargs to the 'simulator' component
        burst_config = self.burst_service.new_burst_configuration(self.test_project.id)
        kwargs_replica = {'test1_val1_invalid': '0', 'test1_val2': '0'}
        burst_config.update_simulator_configuration(kwargs_replica)
        burst_id, _ = self.burst_service.launch_burst(burst_config, 0, algo_id, self.test_user.id)
        burst_config = dao.get_burst_by_id(burst_id)
        #Wait maximum x seconds for burst to finish
        self._wait_for_burst(burst_config, error_expected=True)



    def test_launch_burst_invalid_simulator_data(self):
        """
        Test that burst is marked as error if invalid data is passed to the first step.
        """
        algo_id = self.flow_service.get_algorithm_by_module_and_class('tvb.tests.framework.adapters.testadapter1',
                                                                      'TestAdapter1').id
        #Adapter tries to do an int(test1_val1) so this should fail
        burst_config = self.burst_service.new_burst_configuration(self.test_project.id)
        kwargs_replica = {'test1_val1': 'asa', 'test1_val2': '0'}
        burst_config.update_simulator_configuration(kwargs_replica)
        burst_id, _ = self.burst_service.launch_burst(burst_config, 0, algo_id, self.test_user.id)
        burst_config = dao.get_burst_by_id(burst_id)
        #Wait maximum x seconds for burst to finish
        self._wait_for_burst(burst_config, error_expected=True)


    def test_launch_burst_invalid_portlet_analyzer_data(self):
        """
        Test that burst is marked as error if invalid data is passed to the first step.
        """
        algo_id = self.flow_service.get_algorithm_by_module_and_class('tvb.tests.framework.adapters.testadapter1',
                                                                      'TestAdapter1').id
        #Adapter tries to do an int(test1_val1) and int(test1_val2) so this should be valid
        burst_config = self.burst_service.new_burst_configuration(self.test_project.id)
        kwargs_replica = {'test1_val1': '1', 'test1_val2': '0'}
        burst_config.update_simulator_configuration(kwargs_replica)

        test_portlet = dao.get_portlet_by_identifier(self.PORTLET_ID)
        portlet_configuration = self.burst_service.new_portlet_configuration(test_portlet.id)
        #Portlet analyzer tries to do int(input) which should fail
        declared_overwrites = {ADAPTER_PREFIX_ROOT + '0test_non_dt_input': 'asa'}
        self.burst_service.update_portlet_configuration(portlet_configuration, declared_overwrites)
        burst_config.tabs[0].portlets[0] = portlet_configuration

        burst_id, _ = self.burst_service.launch_burst(burst_config, 0, algo_id, self.test_user.id)
        burst_config = dao.get_burst_by_id(burst_id)
        #Wait maximum x seconds for burst to finish
        burst_config = self._wait_for_burst(burst_config, error_expected=True)

        burst_wf = dao.get_workflows_for_burst(burst_config.id)[0]
        wf_steps = dao.get_workflow_steps(burst_wf.id)
        assert len(wf_steps) == 2,\
                        "Should have exactly 2 wf steps. One for 'simulation' one for portlet analyze operation."
        simulator_op = dao.get_operation_by_id(wf_steps[0].fk_operation)
        assert model.STATUS_FINISHED == simulator_op.status,\
                         "First operation should be simulator which should have 'finished' status."
        portlet_analyze_op = dao.get_operation_by_id(wf_steps[1].fk_operation)
        assert portlet_analyze_op.status == model.STATUS_ERROR,\
                         "Second operation should be portlet analyze step which should have 'error' status."


    def test_launch_group_burst_happy_flow(self):
        """
        Happy flow of launching a burst with a range parameter. Expect to get both and operation
        group and a DataType group for the results of the simulations and for the metric steps.
        """
        burst_config = self._prepare_and_launch_async_burst(length=1, is_range=True, nr_ops=4, wait_to_finish=120)
        if burst_config.status != BurstConfiguration.BURST_FINISHED:
            self.burst_service.stop_burst(burst_config)
            raise AssertionError("Burst should have finished successfully.")

        op_groups = self.count_all_entities(model.OperationGroup)
        dt_groups = self.get_all_entities(model.DataTypeGroup)
        assert 2 == op_groups, "An operation group should have been created for each step."
        assert len(dt_groups) == 2, "An dataType group should have been created for each step."
        for datatype in dt_groups:
            assert 4 == datatype.count_results, "Should have 4 datatypes in group"


    def test_launch_group_burst_no_metric(self):
        """
        Test the launch burst method from burst service. Try to launch a burst with test adapter which has
        no metrics associated. This should fail.
        """
        burst_config = self.burst_service.new_burst_configuration(self.test_project.id)

        algo_id = self.flow_service.get_algorithm_by_module_and_class('tvb.tests.framework.adapters.testadapter1',
                                                                      'TestAdapter1').id
        kwargs_replica = {'test1_val1': '[0, 1, 2]', 'test1_val2': '0', model.RANGE_PARAMETER_1: 'test1_val1'}
        test_portlet = dao.get_portlet_by_identifier(self.PORTLET_ID)
        tab_config = {test_portlet.id: [(0, 0), (0, 1), (1, 0)]}
        self._add_portlets_to_burst(burst_config, tab_config)
        burst_config.update_simulator_configuration(kwargs_replica)
        burst_id, _ = self.burst_service.launch_burst(burst_config, 0, algo_id, self.test_user.id)
        burst_config = dao.get_burst_by_id(burst_id)
        # Wait maximum x seconds for burst to finish
        self._wait_for_burst(burst_config, error_expected=True)

        launched_workflows = dao.get_workflows_for_burst(burst_id, is_count=True)
        assert 3 == launched_workflows, "3 workflows should have been launched due to group parameter."

        op_groups = self.count_all_entities(model.OperationGroup)
        dt_groups = self.count_all_entities(model.DataTypeGroup)
        assert 5 == op_groups, "An operation group should have been created for each step."
        assert 5 == dt_groups, "An dataType group should have been created for each step."


    def test_load_tab_configuration(self):
        """
        Create a burst with some predefined portlets in some known positions. Check that the
        load_tab_configuration method does what it is expected, and we get the portlets in the
        corresponding tab positions.
        """
        burst_config = self.burst_service.new_burst_configuration(self.test_project.id)
        SIMULATOR_MODULE = 'tvb.tests.framework.adapters.testadapter1'
        SIMULATOR_CLASS = 'TestAdapter1'
        algo_id = self.flow_service.get_algorithm_by_module_and_class(SIMULATOR_MODULE, SIMULATOR_CLASS).id
        kwargs_replica = {'test1_val1': '0', 'test1_val2': '0'}
        test_portlet = dao.get_portlet_by_identifier(self.PORTLET_ID)
        # Add test_portlet to positions (0,0), (0,1) and (1,0)
        tab_config = {test_portlet.id: [(0, 0), (0, 1), (1, 0)]}
        self._add_portlets_to_burst(burst_config, tab_config)
        burst_config.update_simulator_configuration(kwargs_replica)
        burst_id, _ = self.burst_service.launch_burst(burst_config, 0, algo_id, self.test_user.id)
        burst_config = dao.get_burst_by_id(burst_id)
        burst_config = self._wait_for_burst(burst_config)
        burst_wf = dao.get_workflows_for_burst(burst_config.id)[0]
        wf_step = dao.get_workflow_steps(burst_wf.id)[0]
        burst_config.prepare_after_load()
        for tab in burst_config.tabs:
            for portlet in tab.portlets:
                assert portlet is None, "Before loading the tab configuration all portlets should be none."
        burst_config = self.burst_service.load_tab_configuration(burst_config, wf_step.fk_operation)
        for tab_idx, tab in enumerate(burst_config.tabs):
            for portlet_idx, portlet in enumerate(tab.portlets):
                if (tab_idx == 0 and portlet_idx in [0, 1]) or (tab_idx == 1 and portlet_idx == 0):
                    assert portlet is not None, "portlet gonfiguration not set"
                    assert test_portlet.id == portlet.portlet_id, "Unexpected portlet entity loaded."
                else:
                    assert portlet is None, "Before loading the tab configuration all portlets should be none"


    def _wait_for_burst(self, burst_config, error_expected=False, timeout=500):
        """
        Method that just waits until a burst configuration is finished or a maximum timeout is reached.

        :param burst_config: the burst configuration that should be waited on
        :param timeout: the maximum number of seconds to wait after the burst
        """
        waited = 0
        while burst_config.status == BurstConfiguration.BURST_RUNNING and waited <= timeout:
            sleep(0.5)
            waited += 0.5
            burst_config = dao.get_burst_by_id(burst_config.id)

        if waited > timeout:
            self.burst_service.stop_burst(burst_config)
            raise AssertionError("Timed out waiting for simulations to finish. We will cancel it")

        if error_expected and burst_config.status != BurstConfiguration.BURST_ERROR:
            self.burst_service.stop_burst(burst_config)
            raise AssertionError("Burst should have failed due to invalid input data.")

        if (not error_expected) and burst_config.status != BurstConfiguration.BURST_FINISHED:
            msg = "Burst status should have been FINISH. Instead got %s %s" % (burst_config.status,
                                                                               burst_config.error_message)
            self.burst_service.stop_burst(burst_config)
            raise AssertionError(msg)

        return burst_config


    def _prepare_and_launch_async_burst(self, length=4, is_range=False, nr_ops=0, wait_to_finish=0):
        """
        Launch an asynchronous burst with a simulation having all the default parameters, only the length received as
        a parameters. This is launched with actual simulator and not with a dummy test adapter as replacement.
        :param length: the length of the simulation in milliseconds. This is also used in case we need
            a group burst, in which case we will have `nr_ops` simulations with lengths starting from 
            `length` to `length + nr_ops` milliseconds
        :param is_range: a boolean which switches between a group burst and a non group burst.
            !! even if `is_range` is `True` you still need a non-zero positive `nr_ops` to have an actual group burst
        :param nr_ops: the number of operations in the group burst
        """
        launch_params = self._prepare_simulation_params(length, is_range, nr_ops)

        burst_config = self.burst_service.new_burst_configuration(self.test_project.id)
        burst_config.update_simulator_configuration(launch_params)
        burst_id = self.burst_service.launch_burst(burst_config, 0, self.sim_algorithm.id, self.test_user.id)[0]
        burst_config = dao.get_burst_by_id(burst_id)

        __timeout = 15
        __waited = 0
        # Wait a maximum of 15 seconds for the burst launch to be performed
        while dao.get_workflows_for_burst(burst_config.id, is_count=True) == 0 and __waited < __timeout:
            sleep(0.5)
            __waited += 0.5

        if wait_to_finish:
            burst_config = self._wait_for_burst(burst_config, timeout=wait_to_finish)
        return burst_config


    def _prepare_and_launch_sync_burst(self):
        """
        Private method to launch a dummy burst. Return the burst loaded after the launch finished
        as well as the workflow steps that initially formed the burst.
        NOTE: the burst launched by this method is a `dummy` one, meaning we do not use an actual
        simulation, but instead test adapters.
        """
        burst_config = TestFactory.store_burst(self.test_project.id)

        workflow_step_list = []
        test_portlet = dao.get_portlet_by_identifier(self.PORTLET_ID)

        stored_dt = datatypes_factory.DatatypesFactory()._store_datatype(Datatype1())
        first_step_algorithm = self.flow_service.get_algorithm_by_module_and_class(
            "tvb.tests.framework.adapters.testadapter1", "TestAdapterDatatypeInput")
        metadata = {DataTypeMetaData.KEY_BURST: burst_config.id}
        kwargs = {"test_dt_input": stored_dt.gid, 'test_non_dt_input': '0'}
        operations, group = self.operation_service.prepare_operations(self.test_user.id, self.test_project.id,
                                                                      first_step_algorithm,
                                                                      first_step_algorithm.algorithm_category,
                                                                      metadata, **kwargs)
        view_step = TestFactory.create_workflow_step("tvb.tests.framework.adapters.testadapter2", "TestAdapter2",
                                                     {"test2": 2}, {}, 0, 0, 0, 0, is_view_step=True)
        view_step.fk_portlet = test_portlet.id
        workflow_step_list.append(view_step)

        workflows = self.workflow_service.create_and_store_workflow(self.test_project.id, burst_config.id, 0,
                                                                    first_step_algorithm.id, operations)
        self.operation_service.prepare_operations_for_workflowsteps(workflow_step_list, workflows, self.test_user.id,
                                                                    burst_config.id, self.test_project.id, group,
                                                                    operations)
        ### Now fire the workflow and also update and store the burst configuration ##
        self.operation_service.launch_operation(operations[0].id, False)
        loaded_burst, _ = self.burst_service.load_burst(burst_config.id)
        import_operation = dao.get_operation_by_id(stored_dt.fk_from_operation)
        dao.remove_entity(import_operation.__class__, import_operation.id)
        dao.remove_datatype(stored_dt.gid)
        return loaded_burst, workflow_step_list


    def _check_burst_removed(self):
        """
        Test that a burst was properly removed. This means checking that the burst entity,
        any workflow steps and any datatypes resulted from the burst are also removed.
        """
        remaining_bursts = dao.get_bursts_for_project(self.test_project.id)
        assert 0 == len(remaining_bursts), "Burst was not deleted"
        ops_number = dao.get_operation_numbers(self.test_project.id)[0]
        assert 0 == ops_number, "Operations were not deleted."
        datatypes = dao.get_datatypes_in_project(self.test_project.id)
        assert 0 == len(datatypes)

        wf_steps = self.count_all_entities(model.WorkflowStep)
        datatype1_stored = self.count_all_entities(Datatype1)
        datatype2_stored = self.count_all_entities(Datatype2)
        assert 0 == wf_steps, "Workflow steps were not deleted."
        assert 0 == datatype1_stored, "Specific datatype entries for DataType1 were not deleted."
        assert 0 == datatype2_stored, "Specific datatype entries for DataType2 were not deleted."


    def _add_portlets_to_burst(self, burst_config, portlet_dict):
        """
        Adds portlets to a burst config in certain tab position as received
        from a properly syntaxed list of dictionaries.
        :param burst_config: the burst configuration to which the portlet will be added
        :param portlet_dict: a list of dictionaries in the form
                { 'portlet_id' : [(tab_idx, idx_in_tab), (tab_idx1, idx_in_tab2), ...]
        NOTE: This will overwrite any portlets that are added to the burst in any of the positions
        received in parameter `portlet_dict`
        """
        for prt_id in portlet_dict:
            positions = portlet_dict[prt_id]
            for pos in positions:
                burst_config.tabs[pos[0]].portlets[pos[1]] = self.burst_service.new_portlet_configuration(
                    prt_id, pos[0], pos[1])


    def _prepare_simulation_params(self, length, is_range=False, no_ops=0):

        connectivity = self._burst_create_connectivity()

        launch_params = self.local_simulation_params
        launch_params['connectivity'] = connectivity.gid
        if is_range:
            launch_params['simulation_length'] = str(range(length, length + no_ops))
            launch_params[model.RANGE_PARAMETER_1] = 'simulation_length'
        else:
            launch_params['simulation_length'] = str(length)
            launch_params[model.RANGE_PARAMETER_1] = None

        return launch_params


    def _burst_create_connectivity(self):
        """
        Create a connectivity that will be used in "non-dummy" burst launches (with the actual simulator).
        """
        meta = {DataTypeMetaData.KEY_SUBJECT: "John Doe", DataTypeMetaData.KEY_STATE: "RAW_DATA"}

        self.operation = model.Operation(self.test_user.id, self.test_project.id, self.sim_algorithm.id,
                                         json.dumps(''), meta=json.dumps(meta), status=model.STATUS_STARTED)
        self.operation = dao.store_entity(self.operation)
        storage_path = FilesHelper().get_project_folder(self.test_project, str(self.operation.id))
        connectivity = Connectivity(storage_path=storage_path)
        connectivity.weights = numpy.ones((74, 74))
        connectivity.centres = numpy.ones((74, 3))
        adapter_instance = StoreAdapter([connectivity])
        self.operation_service.initiate_prelaunch(self.operation, adapter_instance, {})
        return connectivity
