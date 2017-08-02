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
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""

import json
from datetime import datetime
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.storage import dao
from tvb.core.entities import model
from tvb.core.services.exceptions import WorkflowInterStepsException
from tvb.core.entities.transient.burst_configuration_entities import WorkflowStepConfiguration
from types import IntType


DYNAMIC_PARAMS_KEY = "dynamic_params"
STATIC_PARAMS_KEY = "static_params"
EMPTY_OPTION = "empty"


class WorkflowService:
    """
    service layer for work-flow entity.
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__module__)
        self.file_helper = FilesHelper()


    def persist_operation_state(self, operation, operation_status, message=None):
        """
        Update Operation instance state. Store it in DB and on HDD/
        :param operation: Operation instance
        :param operation_status: new status
        :param message: message in case of error
        :return: operation instance changed
        """
        operation.mark_complete(operation_status, unicode(message))
        dao.store_entity(operation)
        operation = dao.get_operation_by_id(operation.id)
        self.file_helper.write_operation_metadata(operation)
        return operation


    @staticmethod
    def store_workflow_step(workflow_step):
        """
        Store a workflow step entity.
        """
        dao.store_entity(workflow_step)
    
    
    @staticmethod
    def create_and_store_workflow(project_id, burst_id, simulator_index, simulator_id, operations):
        """
        Create and store the workflow given the project, user and burst in which the workflow is created.
        :param simulator_index: the index of the simulator in the workflow
        :param simulator_id: the id of the simulator adapter
        :param operations: a list with the operations created for the simulator steps
        """
        workflows = []
        for operation in operations:
            new_workflow = model.Workflow(project_id, burst_id)
            new_workflow = dao.store_entity(new_workflow)
            workflows.append(new_workflow)
            simulation_step = model.WorkflowStep(algorithm_id=simulator_id, workflow_id=new_workflow.id,
                                                 step_index=simulator_index, static_param=operation.parameters)
            simulation_step.fk_operation = operation.id
            dao.store_entity(simulation_step)
        return workflows
        

    @staticmethod
    def set_dynamic_step_references(workflow_step, step_reference):
        """
        :param workflow_step: a valid instance of a workflow_step
        :param step_reference: the step to which every dataType reference index should be set
        
        For each dynamic parameter of the given workflow_step set the 'step_index' at step_reference. 
        """
        dynamic_params = workflow_step.dynamic_param
        for entry in dynamic_params:
            dynamic_params[entry][WorkflowStepConfiguration.STEP_INDEX_KEY] = step_reference
        workflow_step.dynamic_param = dynamic_params


    def prepare_next_step(self, last_executed_op_id):
        """
        If the operation with id 'last_executed_op_id' resulted after
        the execution of a workflow step then this method will launch
        the operation corresponding to the next step from the workflow.
        """
        try:
            current_step, next_workflow_step = self._get_data(last_executed_op_id)
            if next_workflow_step is not None:
                operation = dao.get_operation_by_id(next_workflow_step.fk_operation)
                dynamic_param_names = next_workflow_step.dynamic_workflow_param_names
                if len(dynamic_param_names) > 0:
                    op_params = json.loads(operation.parameters)
                    for param_name in dynamic_param_names:
                        dynamic_param = op_params[param_name]
                        former_step = dao.get_workflow_step_by_step_index(next_workflow_step.fk_workflow,
                                                                          dynamic_param[
                                                                              WorkflowStepConfiguration.STEP_INDEX_KEY])
                        if type(dynamic_param[WorkflowStepConfiguration.DATATYPE_INDEX_KEY]) is IntType:
                            datatypes = dao.get_results_for_operation(former_step.fk_operation)
                            op_params[param_name] = datatypes[
                                dynamic_param[WorkflowStepConfiguration.DATATYPE_INDEX_KEY]].gid
                        else:
                            previous_operation = dao.get_operation_by_id(former_step.fk_operation)
                            op_params[param_name] = json.loads(previous_operation.parameters)[
                                dynamic_param[WorkflowStepConfiguration.DATATYPE_INDEX_KEY]]
                    operation.parameters = json.dumps(op_params)
                    operation = dao.store_entity(operation)
                return operation.id
            elif current_step is not None:
                current_workflow = dao.get_workflow_by_id(current_step.fk_workflow)
                current_workflow.status = current_workflow.STATUS_FINISHED
                dao.store_entity(current_workflow)
                burst_entity = dao.get_burst_by_id(current_workflow.fk_burst)
                parallel_workflows = dao.get_workflows_for_burst(burst_entity.id)
                all_finished = True
                for workflow in parallel_workflows:
                    if workflow.status == workflow.STATUS_STARTED:
                        all_finished = False
                if all_finished:
                    self.mark_burst_finished(burst_entity)
            return None
        except Exception as excep:
            self.logger.error(excep)
            self.logger.exception(excep)
            raise WorkflowInterStepsException(excep)


    def update_executed_workflow_state(self, operation):
        """
        Used for updating the state of an executed workflow.
        Only if the operation with the specified id has resulted after the execution
        of an ExecutedWorkflowStep than the state of the ExecutedWorkflow
        to which belongs the step will be updated.
        """
        executed_step, _ = self._get_data(operation.id)
        if executed_step is not None:
            if operation.status == model.STATUS_ERROR:
                all_executed_steps = dao.get_workflow_steps(executed_step.fk_workflow)
                for step in all_executed_steps:
                    if step.step_index > executed_step.step_index:
                        self.logger.debug("Marking unreached operation %s with error." % step.fk_operation)
                        unreached_operation = dao.get_operation_by_id(step.fk_operation)
                        self.persist_operation_state(unreached_operation, model.STATUS_ERROR,
                                                     "Blocked by failure in step %s with message: \n\n%s." % (
                                                         executed_step.step_index, operation.additional_info))

            workflow = dao.get_workflow_by_id(executed_step.fk_workflow)
            burst = dao.get_burst_by_id(workflow.fk_burst)
            self.mark_burst_finished(burst, error_message=operation.additional_info)
            dao.store_entity(burst)


    @staticmethod
    def _get_data(operation_id):
        """
        For a given operation id, return the corresponding WorkflowStep and the NextWorkflowStep to be executed.
        """
        executed_step = dao.get_workflow_step_for_operation(operation_id)
        if executed_step is not None:
            next_workflow_step = dao.get_workflow_step_by_step_index(executed_step.fk_workflow,
                                                                     executed_step.step_index + 1)
            return executed_step, next_workflow_step
        else:
            return None, None
        
        
    def mark_burst_finished(self, burst_entity, burst_status=None, error_message=None):
        """
        Mark Burst status field.
        Also compute 'weight' for current burst: no of operations inside, estimate time on disk...
        
        :param burst_entity: BurstConfiguration to be updated, at finish time.
        :param burst_status: BurstConfiguration status. By default BURST_FINISHED
        :param error_message: If given, set the status to error and perpetuate the message.
        """
        if burst_status is None:
            burst_status = model.BurstConfiguration.BURST_FINISHED
        if error_message is not None:
            burst_status = model.BurstConfiguration.BURST_ERROR

        try:
            ### If there are any DataType Groups in current Burst, update their counter.
            burst_dt_groups = dao.get_generic_entity(model.DataTypeGroup, burst_entity.id, "fk_parent_burst")
            for dt_group in burst_dt_groups:
                dt_group.count_results = dao.count_datatypes_in_group(dt_group.id)
                dt_group.disk_size, dt_group.subject = dao.get_summary_for_group(dt_group.id)
                dao.store_entity(dt_group)

            ### Update actual Burst entity fields
            burst_entity.datatypes_number = dao.count_datatypes_in_burst(burst_entity.id)
            burst_entity.workflows_number = dao.get_workflows_for_burst(burst_entity.id, is_count=True)

            burst_entity.status = burst_status
            burst_entity.error_message = error_message
            burst_entity.finish_time = datetime.now()
            dao.store_entity(burst_entity)
        except Exception:
            self.logger.exception("Could not correctly update Burst status and meta-data!")
            burst_entity.status = burst_status
            burst_entity.error_message = "Error when updating Burst Status"
            burst_entity.finish_time = datetime.now()
            dao.store_entity(burst_entity)
