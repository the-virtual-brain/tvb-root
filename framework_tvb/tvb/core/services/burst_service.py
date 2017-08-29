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
.. moduleauthor:: lia.domide <lia.domide@codemart.ro>
.. moduleauthor:: bogdan.neacsa <bogdan.neacsa@codemart.ro>
"""

import json
import threading
from datetime import datetime
from types import IntType
from tvb.config import MEASURE_METRICS_MODULE, MEASURE_METRICS_CLASS, DEFAULT_PORTLETS
from tvb.config import SIMULATION_DATATYPE_MODULE, SIMULATION_DATATYPE_CLASS
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.input_tree import KEY_TYPE, TYPE_SELECT, KEY_NAME, InputTreeManager
import tvb.core.entities.model as model
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.entities.transient.burst_configuration_entities import PortletConfiguration, WorkflowStepConfiguration
from tvb.core.entities.storage import dao, transactional
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.core.services.operation_service import OperationService
from tvb.core.services.flow_service import FlowService
from tvb.core.services.workflow_service import WorkflowService
from tvb.core.services.project_service import ProjectService
from tvb.core.services.exceptions import RemoveDataTypeException, InvalidPortletConfiguration, BurstServiceException
from tvb.core.portlets.portlet_configurer import PortletConfigurer
from tvb.core.utils import format_timedelta, format_bytes_human


MAX_BURSTS_DISPLAYED = 50
LAUNCH_NEW = 'new'
LAUNCH_BRANCH = 'branch'


class BurstService(object):
    """
    Service layer for Burst related entities.
    """

    def __init__(self):
        self.operation_service = OperationService()
        self.workflow_service = WorkflowService()
        self.logger = get_logger(self.__class__.__module__)
        self.cache_portlet_configurators = {}


    def build_portlet_interface(self, portlet_configuration, project_id):
        """
        From a portlet_id and a project_id, first build the portlet
        entity then get it's configurable interface. 
        
        :param portlet_configuration: a portlet configuration entity. It holds at the
            least the portlet_id, and in case any default parameters were saved
            they can be rebuilt from the analyzers // visualizer parameters
        :param project_id: the id of the current project   
            
        :returns: the portlet interface will be of the following form::
            [{'interface': adapter_interface, 
            'prefix': prefix_for_parameter_names, 
            'subalg': {algorithm_field_name: default_algorithm_value},
            'algo_group': algorithm_group,
            'alg_ui_name': displayname},
            ......]
            A list of dictionaries for each adapter that makes up the portlet.
            
        """
        portlet_configurer = self._get_portlet_configurer(portlet_configuration.portlet_id)
        portlet_interface = portlet_configurer.get_configurable_interface()

        for adapter_conf in portlet_interface:
            interface = adapter_conf.interface
            itree_mngr = InputTreeManager()
            interface = itree_mngr.fill_input_tree_with_options(interface, project_id,
                                                                adapter_conf.stored_adapter.fk_category)
            adapter_conf.interface = itree_mngr.prepare_param_names(interface)

        portlet_configurer.update_default_values(portlet_interface, portlet_configuration)
        portlet_configurer.prefix_adapters_parameters(portlet_interface)

        return portlet_interface


    def _get_portlet_configurer(self, portlet_id):

        if portlet_id not in self.cache_portlet_configurators:

            portlet_entity = dao.get_portlet_by_id(portlet_id)
            if portlet_entity is None:
                raise InvalidPortletConfiguration("No portlet entity located in database with id=%s. " % portlet_id)

            self.cache_portlet_configurators[portlet_id] = PortletConfigurer(portlet_entity)
            self.logger.debug("Recently parsed portlet XML:" + str([portlet_entity]))

        return self.cache_portlet_configurators[portlet_id]


    def update_portlet_configuration(self, portlet_configuration, submited_parameters):
        """
        :param portlet_configuration: the portlet configuration that needs to be updated
        :param submited_parameters: a list of parameters as submitted from the UI. This 
            is a dictionary in the form : 
            {'dynamic' : {name:value pairs}, 'static' : {name:value pairs}}
            
        All names are prefixed with adapter specific generated prefix.
        """
        portlet_configurer = self._get_portlet_configurer(portlet_configuration.portlet_id)
        return portlet_configurer.update_portlet_configuration(portlet_configuration, submited_parameters)


    def new_burst_configuration(self, project_id):
        """
        Return a new burst configuration entity with all the default values.
        """
        burst_configuration = model.BurstConfiguration(project_id)
        burst_configuration.selected_tab = 0

        # Now set the default portlets for the specified burst configuration.
        # The default portlets are specified in the __init__.py script from tvb root.
        for tab_idx, value in DEFAULT_PORTLETS.items():
            for sel_idx, portlet_identifier in value.items():
                portlet = BurstService.get_portlet_by_identifier(portlet_identifier)
                if portlet is not None:
                    portlet_configuration = self.new_portlet_configuration(portlet.id, tab_idx, sel_idx,
                                                                           portlet.algorithm_identifier)
                    burst_configuration.set_portlet(tab_idx, sel_idx, portlet_configuration)

        return burst_configuration


    @staticmethod
    def _store_burst_config(burst_config):
        """
        Store a burst configuration entity.
        """
        burst_config.prepare_before_save()
        saved_entity = dao.store_entity(burst_config)
        return saved_entity.id


    @staticmethod
    def get_available_bursts(project_id):
        """
        Return all the burst for the current project.
        """
        bursts = dao.get_bursts_for_project(project_id, page_size=MAX_BURSTS_DISPLAYED) or []
        for burst in bursts:
            burst.prepare_after_load()
        return bursts


    @staticmethod
    def populate_burst_disk_usage(bursts):
        """
        Adds a disk_usage field to each burst object.
        The disk usage is computed as the sum of the datatypes generated by a burst
        """
        sizes = dao.compute_bursts_disk_size([b.id for b in bursts])
        for b in bursts:
            b.disk_size = format_bytes_human(sizes[b.id])


    @staticmethod
    def rename_burst(burst_id, new_name):
        """
        Rename the burst given by burst_id, setting it's new name to
        burst_name.
        """
        burst = dao.get_burst_by_id(burst_id)
        burst.name = new_name
        dao.store_entity(burst)


    def load_burst(self, burst_id):
        """
        :param burst_id: the id of the burst that should be loaded
        
        Having this input the method should:
        
            - load the entity from the DB
            - get all the workflow steps for the saved burst id
            - go trough the visualization workflow steps to create the tab 
                configuration of the burst using the tab_index and index_in_tab 
                fields saved on each workflow_step
                
        """
        burst = dao.get_burst_by_id(burst_id)
        burst.prepare_after_load()
        burst.reset_tabs()
        burst_workflows = dao.get_workflows_for_burst(burst.id)

        group_gid = None
        if len(burst_workflows) == 1:
            # A simple burst with no range parameters
            burst = self.__populate_tabs_from_workflow(burst, burst_workflows[0])
        elif len(burst_workflows) > 1:
            # A burst workflow with a range of values, created multiple workflows and need
            # to launch parameter space exploration with the resulted group
            self.__populate_tabs_from_workflow(burst, burst_workflows[0])
            executed_steps = dao.get_workflow_steps(burst_workflows[0].id)

            operation = dao.get_operation_by_id(executed_steps[0].fk_operation)
            if operation.operation_group:
                workflow_group = dao.get_datatypegroup_by_op_group_id(operation.operation_group.id)
                group_gid = workflow_group.gid
        return burst, group_gid

    @staticmethod
    def __populate_tabs_from_workflow(burst_entity, workflow):
        """
        Given a burst and a workflow populate the tabs of the burst with the PortletConfigurations
        generated from the steps of the workflow.
        """
        visualizers = dao.get_visualization_steps(workflow.id)
        for entry in visualizers:
            ## For each visualize step, also load all of the analyze steps.
            portlet_cfg = PortletConfiguration(entry.fk_portlet)
            portlet_cfg.set_visualizer(entry)
            analyzers = dao.get_workflow_steps_for_position(entry.fk_workflow, entry.tab_index, entry.index_in_tab)
            portlet_cfg.set_analyzers(analyzers)
            burst_entity.tabs[entry.tab_index].portlets[entry.index_in_tab] = portlet_cfg
        return burst_entity

    def load_tab_configuration(self, burst_entity, op_id):
        """
        Given a burst entity and an operation id, find the workflow to which the op_id
        belongs and the load the burst_entity's tab configuration with those workflow steps.
        """
        originating_workflow = dao.get_workflow_for_operation_id(op_id)
        burst_entity = self.__populate_tabs_from_workflow(burst_entity, originating_workflow)
        return burst_entity


    def new_portlet_configuration(self, portlet_id, tab_nr=-1, index_in_tab=-1, portlet_name='Default'):
        """
        Return a new portlet configuration entity with default parameters.
        
        :param portlet_id: the id of the portlet for which a configuration will be stored
        :param tab_nr: the index of the currently selected tab
        :param index_in_tab: the index from the currently selected tab
        """
        portlet_configurer = self._get_portlet_configurer(portlet_id)
        configuration = portlet_configurer.create_new_portlet_configuration(portlet_name)
        for wf_step in configuration.analyzers:
            wf_step.tab_index = tab_nr
            wf_step.index_in_tab = index_in_tab
        configuration.visualizer.tab_index = tab_nr
        configuration.visualizer.index_in_tab = index_in_tab
        return configuration


    @staticmethod
    def get_available_portlets():
        """
        :returns: a list of all the available portlet entites
        """
        return dao.get_available_portlets()

    @staticmethod
    def get_portlet_by_id(portlet_id):
        """
        :returns: the portlet entity with the id =@portlet_id
        """
        return dao.get_portlet_by_id(portlet_id)

    @staticmethod
    def get_portlet_by_identifier(portlet_identifier):
        """
        :returns: the portlet entity with the algorithm identifier =@portlet_identifier
        """
        return dao.get_portlet_by_identifier(portlet_identifier)


    def launch_burst(self, burst_configuration, simulator_index, simulator_id, user_id, launch_mode=LAUNCH_NEW):
        """
        Given a burst configuration and all the necessary data do the actual launch.
        
        :param burst_configuration: BurstConfiguration   
        :param simulator_index: the position within the workflows step list that the simulator will take. This is needed
            so that the rest of the portlet workflow steps know what steps do their dynamic parameters come from.
        :param simulator_id: the id of the simulator adapter as stored in the DB. It's needed to load the simulator algo
            group and category that are then passed to the launcher's prepare_operation method.
        :param user_id: the id of the user that launched this burst
        :param launch_mode: new/branch/continue
        """
        ## 1. Prepare BurstConfiguration entity
        if launch_mode == LAUNCH_NEW:
            ## Fully new entity for new simulation
            burst_config = burst_configuration.clone()
            if burst_config.name is None:
                new_id = dao.get_max_burst_id() + 1
                burst_config.name = 'simulation_' + str(new_id)
        else:
            ## Branch or Continue simulation
            burst_config = burst_configuration
            simulation_state = dao.get_generic_entity(SIMULATION_DATATYPE_MODULE + "." + SIMULATION_DATATYPE_CLASS,
                                                      burst_config.id, "fk_parent_burst")
            if simulation_state is None or len(simulation_state) < 1:
                exc = BurstServiceException("Simulation State not found for %s, "
                                            "thus we are unable to branch from it!" % burst_config.name)
                self.logger.error(exc)
                raise exc

            simulation_state = simulation_state[0]
            burst_config.update_simulation_parameter("simulation_state", simulation_state.gid)
            burst_config = burst_configuration.clone()

            count = dao.count_bursts_with_name(burst_config.name, burst_config.fk_project)
            burst_config.name = burst_config.name + "_" + launch_mode + str(count)

        ## 2. Create Operations and do the actual launch  
        if launch_mode in [LAUNCH_NEW, LAUNCH_BRANCH]:
            ## New Burst entry in the history
            burst_id = self._store_burst_config(burst_config)
            thread = threading.Thread(target=self._async_launch_and_prepare,
                                      kwargs={'burst_config': burst_config,
                                              'simulator_index': simulator_index,
                                              'simulator_id': simulator_id,
                                              'user_id': user_id})
            thread.start()
            return burst_id, burst_config.name
        else:
            ## Continue simulation
            ## TODO
            return burst_config.id, burst_config.name


    @transactional
    def _prepare_operations(self, burst_config, simulator_index, simulator_id, user_id):
        """
        Prepare all required operations for burst launch.
        """
        project_id = burst_config.fk_project
        burst_id = burst_config.id
        workflow_step_list = []
        starting_index = simulator_index + 1

        sim_algo = FlowService().get_algorithm_by_identifier(simulator_id)
        metadata = {DataTypeMetaData.KEY_BURST: burst_id}
        launch_data = burst_config.get_all_simulator_values()[0]
        operations, group = self.operation_service.prepare_operations(user_id, project_id, sim_algo,
                                                                      sim_algo.algorithm_category, metadata,
                                                                      **launch_data)
        group_launched = group is not None
        if group_launched:
            starting_index += 1

        for tab in burst_config.tabs:
            for portlet_cfg in tab.portlets:
                ### For each portlet configuration stored, update the step index ###
                ### and also change the dynamic parameters step indexes to point ###
                ### to the simulator outputs.                                     ##
                if portlet_cfg is not None:
                    analyzers = portlet_cfg.analyzers
                    visualizer = portlet_cfg.visualizer
                    for entry in analyzers:
                        entry.step_index = starting_index
                        self.workflow_service.set_dynamic_step_references(entry, simulator_index)
                        workflow_step_list.append(entry)
                        starting_index += 1
                    ### Change the dynamic parameters to point to the last adapter from this portlet execution.
                    visualizer.step_visible = False
                    if len(workflow_step_list) > 0 and isinstance(workflow_step_list[-1], model.WorkflowStep):
                        self.workflow_service.set_dynamic_step_references(visualizer, workflow_step_list[-1].step_index)
                    else:
                        self.workflow_service.set_dynamic_step_references(visualizer, simulator_index)
                    ### Only for a single operation have the step of visualization, otherwise is useless.
                    if not group_launched:
                        workflow_step_list.append(visualizer)

        if group_launched:
            ###  For a group of operations, make sure the metric for PSE view 
            ### is also computed, immediately after the simulation.
            metric_algo = FlowService().get_algorithm_by_module_and_class(MEASURE_METRICS_MODULE, MEASURE_METRICS_CLASS)
            metric_interface = FlowService().prepare_adapter(project_id, metric_algo)
            dynamics = {}
            for entry in metric_interface:
                # We have a select that should be the dataType and a select multiple with the 
                # required metric algorithms to be evaluated. Only dynamic parameter should be
                # the select type.
                if entry[KEY_TYPE] == TYPE_SELECT:
                    dynamics[entry[KEY_NAME]] = {WorkflowStepConfiguration.DATATYPE_INDEX_KEY: 0,
                                                 WorkflowStepConfiguration.STEP_INDEX_KEY: simulator_index}
            metric_step = model.WorkflowStep(algorithm_id=metric_algo.id, step_index=simulator_index + 1,
                                             static_param={}, dynamic_param=dynamics)
            metric_step.step_visible = False
            workflow_step_list.insert(0, metric_step)

        workflows = self.workflow_service.create_and_store_workflow(project_id, burst_id, simulator_index,
                                                                    simulator_id, operations)
        self.operation_service.prepare_operations_for_workflowsteps(workflow_step_list, workflows, user_id,
                                                                    burst_id, project_id, group, operations)
        operation_ids = [operation.id for operation in operations]
        return operation_ids


    def _async_launch_and_prepare(self, burst_config, simulator_index, simulator_id, user_id):
        """
        Prepare operations asynchronously.
        """
        try:
            operation_ids = self._prepare_operations(burst_config, simulator_index, simulator_id, user_id)
            self.logger.debug("Starting a total of %s workflows" % (len(operation_ids, )))
            wf_errs = 0
            for operation_id in operation_ids:
                try:
                    OperationService().launch_operation(operation_id, True)
                except Exception as excep:
                    self.logger.error(excep)
                    wf_errs += 1
                    self.workflow_service.mark_burst_finished(burst_config, error_message=str(excep))

            self.logger.debug("Finished launching workflows. " + str(len(operation_ids) - wf_errs) +
                              " were launched successfully, " + str(wf_errs) + " had error on pre-launch steps")
        except Exception as excep:
            self.logger.error(excep)
            self.workflow_service.mark_burst_finished(burst_config, error_message=str(excep))


    @staticmethod
    def launch_visualization(visualization, frame_width=None, frame_height=None, is_preview=True):
        """
        :param visualization: a visualization workflow step
        """
        dynamic_params = visualization.dynamic_param
        static_params = visualization.static_param
        parameters_dict = static_params
        current_project_id = 0
        # Current operation id needed for export mechanism. So far just use ##
        # the operation of the workflow_step from which the inputs are taken    ####
        for param in dynamic_params:
            step_index = dynamic_params[param][WorkflowStepConfiguration.STEP_INDEX_KEY]
            datatype_index = dynamic_params[param][WorkflowStepConfiguration.DATATYPE_INDEX_KEY]
            referred_workflow_step = dao.get_workflow_step_by_step_index(visualization.fk_workflow, step_index)
            referred_operation_id = referred_workflow_step.fk_operation
            referred_operation = dao.get_operation_by_id(referred_operation_id)
            current_project_id = referred_operation.fk_launched_in
            if type(datatype_index) is IntType:
                # Entry is the output of a previous step ##
                datatypes = dao.get_results_for_operation(referred_operation_id)
                parameters_dict[param] = datatypes[datatype_index].gid
            else:
                # Entry is the input of a previous step ###
                parameters_dict[param] = json.loads(referred_operation.parameters)[datatype_index]
        algorithm = dao.get_algorithm_by_id(visualization.fk_algorithm)
        adapter_instance = ABCAdapter.build_adapter(algorithm)
        adapter_instance.current_project_id = current_project_id
        prepared_inputs = adapter_instance.prepare_ui_inputs(parameters_dict)
        if frame_width is not None:
            prepared_inputs[ABCDisplayer.PARAM_FIGURE_SIZE] = (frame_width, frame_height)

        if is_preview:
            result = adapter_instance.generate_preview(**prepared_inputs)
        else:
            result = adapter_instance.launch(**prepared_inputs)
        return result, parameters_dict


    def update_history_status(self, id_list):
        """
        For each burst_id received in the id_list read new status from DB and return a list [id, new_status] pair.
        """
        result = []
        for b_id in id_list:
            burst = dao.get_burst_by_id(b_id)
            burst.prepare_after_load()
            if burst is not None:
                if burst.status == burst.BURST_RUNNING:
                    running_time = datetime.now() - burst.start_time
                else:
                    running_time = burst.finish_time - burst.start_time
                running_time = format_timedelta(running_time, most_significant2=False)

                if burst.status == burst.BURST_ERROR:
                    msg = 'Check Operations page for error Message'
                else:
                    msg = ''
                result.append([burst.id, burst.status, burst.is_group, msg, running_time])
            else:
                self.logger.debug("Could not find burst with id=" + str(b_id) + ". Might have been deleted by user!!")
        return result


    def stop_burst(self, burst_entity):
        """
        Stop all the entities for the current burst and set the burst status to canceled.
        """
        burst_wfs = dao.get_workflows_for_burst(burst_entity.id)
        any_stopped = False
        for workflow in burst_wfs:
            wf_steps = dao.get_workflow_steps(workflow.id)
            for step in wf_steps:
                if step.fk_operation is not None:
                    self.logger.debug("We will stop operation: %d" % step.fk_operation)
                    any_stopped = self.operation_service.stop_operation(step.fk_operation) or any_stopped

        if any_stopped and burst_entity.status != burst_entity.BURST_CANCELED:
            self.workflow_service.mark_burst_finished(burst_entity, model.BurstConfiguration.BURST_CANCELED)
            return True
        return False


    @transactional
    def cancel_or_remove_burst(self, burst_id):
        """
        Cancel (if burst is still running) or Remove the burst given by burst_id.
        :returns True when Remove operation was done and False when Cancel
        """
        burst_entity = dao.get_burst_by_id(burst_id)
        if burst_entity.status == burst_entity.BURST_RUNNING:
            self.stop_burst(burst_entity)
            return False

        service = ProjectService()
        ## Remove each DataType in current burst.
        ## We can not leave all on cascade, because it won't work on SQLite for mapped dataTypes.
        datatypes = dao.get_all_datatypes_in_burst(burst_id)
        ## Get operations linked to current burst before removing the burst or else 
        ##    the burst won't be there to identify operations any more.
        remaining_ops = dao.get_operations_in_burst(burst_id)

        # Remove burst first to delete work-flow steps which still hold foreign keys to operations.
        correct = dao.remove_entity(burst_entity.__class__, burst_id)
        if not correct:
            raise RemoveDataTypeException("Could not remove Burst entity!")

        for datatype in datatypes:
            service.remove_datatype(burst_entity.fk_project, datatype.gid, False)

        ## Remove all Operations remained.
        correct = True
        remaining_op_groups = set()
        project = dao.get_project_by_id(burst_entity.fk_project)

        for oper in remaining_ops:
            is_remaining = dao.get_generic_entity(oper.__class__, oper.id)
            if len(is_remaining) == 0:
                ### Operation removed cascaded.
                continue
            if oper.fk_operation_group is not None and oper.fk_operation_group not in remaining_op_groups:
                is_remaining = dao.get_generic_entity(model.OperationGroup, oper.fk_operation_group)
                if len(is_remaining) > 0:
                    remaining_op_groups.add(oper.fk_operation_group)
                    correct = correct and dao.remove_entity(model.OperationGroup, oper.fk_operation_group)
            correct = correct and dao.remove_entity(oper.__class__, oper.id)
            service.structure_helper.remove_operation_data(project.name, oper.id)

        if not correct:
            raise RemoveDataTypeException("Could not remove Burst because a linked operation could not be dropped!!")
        return True


    @staticmethod
    def get_portlet_status(portlet_cfg):
        """ 
        Get the status of a portlet configuration. 
        """
        if portlet_cfg.analyzers:
            for analyze_step in portlet_cfg.analyzers:
                operation = dao.try_get_operation_by_id(analyze_step.fk_operation)
                if operation is None:
                    return model.STATUS_ERROR, "Operation has been removed"
                if operation.status != model.STATUS_FINISHED:
                    return operation.status, operation.additional_info or ''
        else:
            ## Simulator is first step so now decide if we are waiting for input or output ##
            visualizer = portlet_cfg.visualizer
            wait_on_outputs = False
            for entry in visualizer.dynamic_param:
                if type(visualizer.dynamic_param[entry][WorkflowStepConfiguration.DATATYPE_INDEX_KEY]) == IntType:
                    wait_on_outputs = True
                    break
            if wait_on_outputs:
                simulator_step = dao.get_workflow_step_by_step_index(visualizer.fk_workflow, 0)
                operation = dao.try_get_operation_by_id(simulator_step.fk_operation)
                if operation is None:
                    error_msg = ("At least one simulation result was not found, it might have been removed. <br\>"
                                 "You can copy and relaunch current simulation, if you are interested in having "
                                 "your results re-computed.")
                    return model.STATUS_ERROR, error_msg
                else:
                    return operation.status, operation.additional_info or ''
        return model.STATUS_FINISHED, ''
