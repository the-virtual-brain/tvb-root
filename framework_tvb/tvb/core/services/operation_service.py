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
Module in charge with Launching an operation (creating the Operation entity as well, based on gathered parameters).

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Yann Gordon <yann@tvb.invalid>
"""

import os
import json
import zipfile
import sys
import six
from copy import copy
from cgi import FieldStorage
from datetime import datetime
from tvb.basic.traits.exceptions import TVBException
from tvb.basic.traits.types_basic import MapAsJson, Range
from tvb.basic.profile import TvbProfile
from tvb.basic.logger.builder import get_logger
from tvb.core import utils
from tvb.core.adapters import constants
from tvb.core.adapters.abcadapter import ABCAdapter, ABCSynchronous
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities import model
from tvb.core.entities.storage import dao
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.services.workflow_service import WorkflowService
from tvb.core.services.backend_client import BACKEND_CLIENT

try:
    from cherrypy._cpreqbody import Part
    # cover cases when the web interface is not available.
except Exception:
    Part = FieldStorage

TEMPORARY_PREFIX = ".tmp"

RANGE_PARAMETER_1 = model.RANGE_PARAMETER_1
RANGE_PARAMETER_2 = model.RANGE_PARAMETER_2

UIKEY_SUBJECT = "RESERVEDsubject"
UIKEY_USERGROUP = "RESERVEDusergroup"



class OperationService:
    """
    Class responsible for preparing an operation launch. 
    It will prepare parameters, and decide if the operation is to be executed
    immediately, or to be sent on the cluster.
    """
    ATT_UID = "uid"


    def __init__(self):
        self.logger = get_logger(self.__class__.__module__)
        self.workflow_service = WorkflowService()
        self.file_helper = FilesHelper()


    ##########################################################################################
    ######## Methods related to launching operations start here ##############################
    ##########################################################################################

    def initiate_operation(self, current_user, project_id, adapter_instance,
                           temporary_storage, visible=True, **kwargs):
        """
        Gets the parameters of the computation from the previous inputs form,
        and launches a computation (on the cluster or locally).
        
        Invoke custom method on an Adapter Instance. Make sure when the  
        operation has finished that the correct results are stored into DB. 
        """
        if not isinstance(adapter_instance, ABCAdapter):
            self.logger.warning("Inconsistent Adapter Class:" + str(adapter_instance.__class__))
            raise LaunchException("Developer Exception!!")

        # Prepare Files parameters
        files = {}
        kw2 = copy(kwargs)
        for i, j in six.iteritems(kwargs):
            if isinstance(j, FieldStorage) or isinstance(j, Part):
                files[i] = j
                del kw2[i]

        temp_files = {}
        try:
            for i, j in six.iteritems(files):
                if j.file is None:
                    kw2[i] = None
                    continue
                uq_name = utils.date2string(datetime.now(), True) + '_' + str(i)
                # We have to add original file name to end, in case file processing
                # involves file extension reading
                file_name = TEMPORARY_PREFIX + uq_name + '_' + j.filename
                file_name = os.path.join(temporary_storage, file_name)
                kw2[i] = file_name
                temp_files[i] = file_name
                with open(file_name, 'wb') as file_obj:
                    file_obj.write(j.file.read())
                self.logger.debug("Will store file:" + file_name)
            kwargs = kw2
        except Exception as excep:
            self._handle_exception(excep, temp_files, "Could not launch operation: invalid input files!")

        ### Store Operation entity. 
        algo = adapter_instance.stored_adapter
        algo_category = dao.get_category_by_id(algo.fk_category)

        operations = self.prepare_operations(current_user.id, project_id, algo, algo_category,
                                             {}, visible, **kwargs)[0]

        if isinstance(adapter_instance, ABCSynchronous):
            if len(operations) > 1:
                raise LaunchException("Synchronous operations are not supporting ranges!")
            if len(operations) < 1:
                self.logger.warning("No operation was defined")
                raise LaunchException("Invalid empty Operation!!!")
            return self.initiate_prelaunch(operations[0], adapter_instance, temp_files, **kwargs)
        else:
            return self._send_to_cluster(operations, adapter_instance, current_user.username)


    @staticmethod
    def _prepare_metadata(initial_metadata, algo_category, operation_group, submit_data):
        """
        Gather metadata from submitted fields and current to be execute algorithm.
        Will populate STATE, GROUP in metadata
        """
        metadata = copy(initial_metadata)

        user_group = None
        if DataTypeMetaData.KEY_OPERATION_TAG in submit_data:
            user_group = submit_data[DataTypeMetaData.KEY_OPERATION_TAG]

        if operation_group is not None:
            metadata[DataTypeMetaData.KEY_OPERATION_TAG] = operation_group.name

        if DataTypeMetaData.KEY_TAG_1 in submit_data:
            metadata[DataTypeMetaData.KEY_TAG_1] = submit_data[DataTypeMetaData.KEY_TAG_1]

        metadata[DataTypeMetaData.KEY_STATE] = algo_category.defaultdatastate

        return metadata, user_group


    @staticmethod
    def _read_set(values):
        """ Parse a committed UI possible list of values, into a set converted into string."""
        if isinstance(values, list):
            set_values = []
            values_str = ""
            for val in values:
                if val not in set_values:
                    set_values.append(val)
                    values_str = values_str + " " + str(val)
            values = values_str
        return str(values).strip()


    def group_operation_launch(self, user_id, project_id, algorithm_id, category_id, existing_dt_group=None, **kwargs):
        """
        Create and prepare the launch of a group of operations.
        """
        category = dao.get_category_by_id(category_id)
        algorithm = dao.get_algorithm_by_id(algorithm_id)
        ops, _ = self.prepare_operations(user_id, project_id, algorithm, category, {},
                                         existing_dt_group=existing_dt_group, **kwargs)
        for operation in ops:
            self.launch_operation(operation.id, True)


    def prepare_operations(self, user_id, project_id, algorithm, category, metadata,
                           visible=True, existing_dt_group=None, **kwargs):
        """
        Do all the necessary preparations for storing an operation. If it's the case of a 
        range of values create an operation group and multiple operations for each possible
        instance from the range.
        :param metadata: Initial MetaData with potential Burst identification inside.
        """
        operations = []

        available_args, group = self._prepare_group(project_id, existing_dt_group, kwargs)
        if len(available_args) > TvbProfile.current.MAX_RANGE_NUMBER:
            raise LaunchException("Too big range specified. You should limit the"
                                  " resulting operations to %d" % TvbProfile.current.MAX_RANGE_NUMBER)
        else:
            self.logger.debug("Launching a range with %d operations..." % len(available_args))
        group_id = None
        if group is not None:
            group_id = group.id
        metadata, user_group = self._prepare_metadata(metadata, category, group, kwargs)

        self.logger.debug("Saving Operation(userId=" + str(user_id) + ",projectId=" + str(project_id) + "," +
                          str(metadata) + ",algorithmId=" + str(algorithm.id) + ", ops_group= " + str(group_id) + ")")

        visible_operation = visible and category.display is False
        meta_str = json.dumps(metadata)
        for (one_set_of_args, range_vals) in available_args:
            range_values = json.dumps(range_vals) if range_vals else None
            operation = model.Operation(user_id, project_id, algorithm.id,
                                        json.dumps(one_set_of_args, cls=MapAsJson.MapAsJsonEncoder), meta_str,
                                        op_group_id=group_id, user_group=user_group, range_values=range_values)
            operation.visible = visible_operation
            operations.append(operation)
        operations = dao.store_entities(operations)

        if group is not None:
            burst_id = None
            if DataTypeMetaData.KEY_BURST in metadata:
                burst_id = metadata[DataTypeMetaData.KEY_BURST]
            if existing_dt_group is None:
                datatype_group = model.DataTypeGroup(group, operation_id=operations[0].id, fk_parent_burst=burst_id,
                                                     state=metadata[DataTypeMetaData.KEY_STATE])
                dao.store_entity(datatype_group)
            else:
                # Reset count
                existing_dt_group.count_results = None
                dao.store_entity(existing_dt_group)

        return operations, group


    def prepare_operations_for_workflowsteps(self, workflow_step_list, workflows, user_id, burst_id,
                                             project_id, group, sim_operations):
        """
        Create and store Operation entities from a list of Workflow Steps.
        Will be generated workflows x workflow_step_list Operations.
        For every step in workflow_step_list one OperationGroup and one DataTypeGroup will be created 
        (in case of PSE).
        """

        for step in workflow_step_list:
            operation_group = None
            if (group is not None) and not isinstance(step, model.WorkflowStepView):
                operation_group = model.OperationGroup(project_id=project_id, ranges=group.range_references)
                operation_group = dao.store_entity(operation_group)

            operation = None
            metadata = {DataTypeMetaData.KEY_BURST: burst_id}
            algo_category = dao.get_algorithm_by_id(step.fk_algorithm)
            if algo_category is not None:
                algo_category = algo_category.algorithm_category

            for wf_idx, workflow in enumerate(workflows):
                cloned_w_step = step.clone()
                cloned_w_step.fk_workflow = workflow.id
                dynamic_params = cloned_w_step.dynamic_param
                op_params = cloned_w_step.static_param
                op_params.update(dynamic_params)
                range_values = None
                group_id = None
                if operation_group is not None:
                    group_id = operation_group.id
                    range_values = sim_operations[wf_idx].range_values

                if not isinstance(step, model.WorkflowStepView):
                    ## For visualization steps, do not create operations, as those are not really needed.
                    metadata, user_group = self._prepare_metadata(metadata, algo_category, operation_group, op_params)
                    operation = model.Operation(user_id, project_id, step.fk_algorithm,
                                                json.dumps(op_params, cls=MapAsJson.MapAsJsonEncoder),
                                                meta=json.dumps(metadata),
                                                op_group_id=group_id, range_values=range_values, user_group=user_group)
                    operation.visible = step.step_visible
                    operation = dao.store_entity(operation)
                    cloned_w_step.fk_operation = operation.id

                dao.store_entity(cloned_w_step)

            if operation_group is not None and operation is not None:
                datatype_group = model.DataTypeGroup(operation_group, operation_id=operation.id,
                                                     fk_parent_burst=burst_id,
                                                     state=metadata[DataTypeMetaData.KEY_STATE])
                dao.store_entity(datatype_group)


    def initiate_prelaunch(self, operation, adapter_instance, temp_files, **kwargs):
        """
        Public method.
        This should be the common point in calling an adapter- method.
        """
        result_msg = ""
        try:
            unique_id = None
            if self.ATT_UID in kwargs:
                unique_id = kwargs[self.ATT_UID]
            filtered_kwargs = adapter_instance.prepare_ui_inputs(kwargs)
            self.logger.debug("Launching operation " + str(operation.id) + " with " + str(filtered_kwargs))
            operation = dao.get_operation_by_id(operation.id)   # Load Lazy fields

            params = dict()
            for k, value_ in filtered_kwargs.items():
                params[str(k)] = value_

            disk_space_per_user = TvbProfile.current.MAX_DISK_SPACE
            pending_op_disk_space = dao.compute_disk_size_for_started_ops(operation.fk_launched_by)
            user_disk_space = dao.compute_user_generated_disk_size(operation.fk_launched_by)    # From kB to Bytes
            available_space = disk_space_per_user - pending_op_disk_space - user_disk_space

            result_msg, nr_datatypes = adapter_instance._prelaunch(operation, unique_id, available_space, **params)
            operation = dao.get_operation_by_id(operation.id)
            ## Update DB stored kwargs for search purposes, to contain only valuable params (no unselected options)
            operation.parameters = json.dumps(kwargs)
            operation.mark_complete(model.STATUS_FINISHED)
            if nr_datatypes > 0:
                #### Write operation meta-XML only if some result are returned
                self.file_helper.write_operation_metadata(operation)
            dao.store_entity(operation)
            self._remove_files(temp_files)

        except zipfile.BadZipfile as excep:
            msg = "The uploaded file is not a valid ZIP!"
            self._handle_exception(excep, temp_files, msg, operation)
        except TVBException as excep:
            self._handle_exception(excep, temp_files, excep.message, operation)
        except MemoryError:
            msg = ("Could not execute operation because there is not enough free memory." +
                   " Please adjust operation parameters and re-launch it.")
            self._handle_exception(Exception(msg), temp_files, msg, operation)
        except Exception as excep1:
            msg = "Could not launch Operation with the given input data!"
            self._handle_exception(excep1, temp_files, msg, operation)

        ### Try to find next workflow Step. It might throw WorkflowException
        next_op_id = self.workflow_service.prepare_next_step(operation.id)
        self.launch_operation(next_op_id)
        return result_msg


    def _send_to_cluster(self, operations, adapter_instance, current_username="unknown"):
        """ Initiate operation on cluster"""
        for operation in operations:
            try:
                BACKEND_CLIENT.execute(str(operation.id), current_username, adapter_instance)
            except Exception as excep:
                self._handle_exception(excep, {}, "Could not start operation!", operation)

        return operations


    def launch_operation(self, operation_id, send_to_cluster=False, adapter_instance=None):
        """
        Method exposed for Burst-Workflow related calls.
        It is used for cascading operation in the same workflow.
        """
        if operation_id is not None:
            operation = dao.get_operation_by_id(operation_id)
            if adapter_instance is None:
                algorithm = operation.algorithm
                adapter_instance = ABCAdapter.build_adapter(algorithm)
            parsed_params = utils.parse_json_parameters(operation.parameters)

            if send_to_cluster:
                self._send_to_cluster([operation], adapter_instance, operation.user.username)
            else:
                self.initiate_prelaunch(operation, adapter_instance, {}, **parsed_params)


    def _handle_exception(self, exception, temp_files, message, operation=None):
        """
        Common way to treat exceptions:
            - remove temporary files, if any
            - set status ERROR on current operation (if any)
            - log exception
        """
        self.logger.exception(message)
        if operation is not None:
            self.workflow_service.persist_operation_state(operation, model.STATUS_ERROR, unicode(exception))
            self.workflow_service.update_executed_workflow_state(operation)
        self._remove_files(temp_files)
        exception.message = message
        raise exception, None, sys.exc_info()[2]  # when rethrowing in python this is required to preserve the stack trace


    def _remove_files(self, file_dictionary):
        """
        Remove any files that exist in the file_dictionary. 
        Currently used to delete temporary files created during an operation.
        """
        for pth in file_dictionary.itervalues():
            pth = str(pth)
            try:
                if os.path.exists(pth) and os.path.isfile(pth):
                    os.remove(pth)
                    self.logger.debug("We no longer need file:" + pth + " => deleted")
                else:
                    self.logger.warning("Trying to remove not existent file:" + pth)
            except OSError:
                self.logger.exception("Could not cleanup file!")


    @staticmethod
    def _range_name(range_no):
        return model.PARAM_RANGE_PREFIX + str(range_no)


    def _prepare_group(self, project_id, existing_dt_group, kwargs):
        """
        Create and store OperationGroup entity, or return None
        """
        # Standard ranges as accepted from UI
        range1_values = self.get_range_values(kwargs, self._range_name(1))
        range2_values = self.get_range_values(kwargs, self._range_name(2))
        available_args = self.__expand_arguments([(kwargs, None)], range1_values, self._range_name(1))
        available_args = self.__expand_arguments(available_args, range2_values, self._range_name(2))
        is_group = False
        ranges = []
        if self._range_name(1) in kwargs and range1_values is not None:
            is_group = True
            ranges.append(json.dumps((kwargs[self._range_name(1)], range1_values)))
        if self._range_name(2) in kwargs and range2_values is not None:
            is_group = True
            ranges.append(json.dumps((kwargs[self._range_name(2)], range2_values)))
        # Now for additional ranges which might be the case for the 'model exploration'
        last_range_idx = 3
        ranger_name = self._range_name(last_range_idx)
        while ranger_name in kwargs:
            values_for_range = self.get_range_values(kwargs, ranger_name)
            available_args = self.__expand_arguments(available_args, values_for_range, ranger_name)
            last_range_idx += 1
            ranger_name = self._range_name(last_range_idx)
        if last_range_idx > 3:
            ranges = []  # Since we only have 3 fields in db for this just hide it
        if not is_group:
            group = None
        elif existing_dt_group is None:
            group = model.OperationGroup(project_id=project_id, ranges=ranges)
            group = dao.store_entity(group)
        else:
            group = existing_dt_group.parent_operation_group

        return available_args, group


    def get_range_values(self, kwargs, ranger_name):
        """
        For the ranger given by ranger_name look in kwargs and return
        the array with all the possible values.
        """
        if ranger_name not in kwargs:
            return None
        if str(kwargs[ranger_name]) not in kwargs:
            return None

        range_values = []
        try:
            range_data = json.loads(str(kwargs[str(kwargs[ranger_name])]))
        except Exception:
            try:
                range_data = [x.strip() for x in str(kwargs[str(kwargs[ranger_name])]).split(',') if len(x.strip()) > 0]
                return range_data
            except Exception:
                self.logger.exception("Could not launch operation !")
                raise LaunchException("Could not launch with no data from:" + str(ranger_name))
        if type(range_data) in (list, tuple):
            return range_data

        if (constants.ATT_MINVALUE in range_data) and (constants.ATT_MAXVALUE in range_data):
            lo_val = float(range_data[constants.ATT_MINVALUE])
            hi_val = float(range_data[constants.ATT_MAXVALUE])
            step = float(range_data[constants.ATT_STEP])
            range_values = list(Range(lo=lo_val, hi=hi_val, step=step, mode=Range.MODE_INCLUDE_BOTH))

        else:
            for possible_value in range_data:
                if range_data[possible_value]:
                    range_values.append(possible_value)
        return range_values


    @staticmethod
    def __expand_arguments(arguments_list, range_values, range_title):
        """
        Parse the arguments submitted from UI (flatten form) 
        If any ranger is found, return a list of arguments for all possible operations.
        """
        if range_values is None:
            return arguments_list
        result = []
        for value in range_values:
            for args, range_ in arguments_list:
                kw_new = copy(args)
                range_new = copy(range_)
                kw_new[kw_new[range_title]] = value
                if range_new is None:
                    range_new = {}
                range_new[kw_new[range_title]] = value
                del kw_new[range_title]
                result.append((kw_new, range_new))
        return result


    ##########################################################################################
    ######## Methods related to stopping and restarting operations start here ################
    ##########################################################################################

    def stop_operation(self, operation_id):
        """
        Stop the operation given by the operation id.
        """
        return BACKEND_CLIENT.stop_operation(int(operation_id))



