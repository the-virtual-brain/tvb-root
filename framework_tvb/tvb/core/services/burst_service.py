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
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.model.model_operation import OperationGroup, STATUS_ERROR, STATUS_FINISHED
from tvb.core.entities.transient.burst_configuration_entities import PortletConfiguration, WorkflowStepConfiguration
from tvb.core.entities.storage import dao, transactional
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.core.services.operation_service import OperationService
from tvb.core.services.workflow_service import WorkflowService
from tvb.core.services.project_service import ProjectService
from tvb.core.services.exceptions import RemoveDataTypeException, InvalidPortletConfiguration
from tvb.core.portlets.portlet_configurer import PortletConfigurer


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

        # for adapter_conf in portlet_interface:
        #     interface = adapter_conf.interface
        #     itree_mngr = InputTreeManager()
        #     interface = itree_mngr.fill_input_tree_with_options(interface, project_id,
        #                                                         adapter_conf.stored_adapter.fk_category)
        #     adapter_conf.interface = itree_mngr.prepare_param_names(interface)

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
            if type(datatype_index) is int:
                # Entry is the output of a previous step ##
                datatypes = dao.get_results_for_operation(referred_operation_id)
                parameters_dict[param] = datatypes[datatype_index].gid
            else:
                # Entry is the input of a previous step ###
                parameters_dict[param] = json.loads(referred_operation.parameters)[datatype_index]
        algorithm = dao.get_algorithm_by_id(visualization.fk_algorithm)
        adapter_instance = ABCAdapter.build_adapter(algorithm)
        adapter_instance.current_project_id = current_project_id
        prepared_inputs = {}  # adapter_instance.prepare_ui_inputs(parameters_dict)
        if frame_width is not None:
            prepared_inputs[ABCDisplayer.PARAM_FIGURE_SIZE] = (frame_width, frame_height)

        if is_preview:
            result = adapter_instance.generate_preview(**prepared_inputs)
        else:
            result = adapter_instance.launch(**prepared_inputs)
        return result, parameters_dict

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
                is_remaining = dao.get_generic_entity(OperationGroup, oper.fk_operation_group)
                if len(is_remaining) > 0:
                    remaining_op_groups.add(oper.fk_operation_group)
                    correct = correct and dao.remove_entity(OperationGroup, oper.fk_operation_group)
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
                    return STATUS_ERROR, "Operation has been removed"
                if operation.status != STATUS_FINISHED:
                    return operation.status, operation.additional_info or ''
        else:
            ## Simulator is first step so now decide if we are waiting for input or output ##
            visualizer = portlet_cfg.visualizer
            wait_on_outputs = False
            for entry in visualizer.dynamic_param:
                if type(visualizer.dynamic_param[entry][WorkflowStepConfiguration.DATATYPE_INDEX_KEY]) == int:
                    wait_on_outputs = True
                    break
            if wait_on_outputs:
                simulator_step = dao.get_workflow_step_by_step_index(visualizer.fk_workflow, 0)
                operation = dao.try_get_operation_by_id(simulator_step.fk_operation)
                if operation is None:
                    error_msg = ("At least one simulation result was not found, it might have been removed. <br\>"
                                 "You can copy and relaunch current simulation, if you are interested in having "
                                 "your results re-computed.")
                    return STATUS_ERROR, error_msg
                else:
                    return operation.status, operation.additional_info or ''
        return STATUS_FINISHED, ''
