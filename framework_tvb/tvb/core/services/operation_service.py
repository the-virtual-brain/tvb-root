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
Module in charge with Launching an operation (creating the Operation entity as well, based on gathered parameters).

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Yann Gordon <yann@tvb.invalid>
"""

import json
import os
import shutil
import sys
import uuid
import zipfile
from copy import copy
from inspect import isclass

from tvb.basic.exceptions import TVBException
from tvb.basic.logger.builder import get_logger
from tvb.basic.neotraits.api import Range
from tvb.basic.profile import TvbProfile
from tvb.config import MEASURE_METRICS_MODULE, MEASURE_METRICS_CLASS, MEASURE_METRICS_MODEL_CLASS, ALGORITHMS
from tvb.core.adapters import constants
from tvb.core.adapters.abcadapter import ABCAdapter, AdapterLaunchModeEnum
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.generic_attributes import GenericAttributes
from tvb.core.entities.load import get_class_by_name
from tvb.core.entities.model.model_burst import PARAM_RANGE_PREFIX, RANGE_PARAMETER_1, RANGE_PARAMETER_2, \
    BurstConfiguration
from tvb.core.entities.model.model_datatype import DataTypeGroup
from tvb.core.entities.model.model_operation import STATUS_FINISHED, STATUS_ERROR, OperationGroup, Operation
from tvb.core.entities.storage import dao, transactional
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.neocom import h5
from tvb.core.neotraits.h5 import ViewModelH5
from tvb.core.services.backend_client_factory import BackendClientFactory
from tvb.core.services.burst_service import BurstService
from tvb.core.services.exceptions import OperationException
from tvb.core.services.project_service import ProjectService
from tvb.datatypes.time_series import TimeSeries

RANGE_PARAMETER_1 = RANGE_PARAMETER_1
RANGE_PARAMETER_2 = RANGE_PARAMETER_2


class OperationService:
    """
    Class responsible for preparing an operation launch. 
    It will prepare parameters, and decide if the operation is to be executed
    immediately, or to be sent on the cluster.
    """
    ATT_UID = "uid"

    def __init__(self):
        self.logger = get_logger(self.__class__.__module__)
        self.file_helper = FilesHelper()

    ##########################################################################################
    ######## Methods related to launching operations start here ##############################
    ##########################################################################################

    def initiate_operation(self, current_user, project, adapter_instance, visible=True, model_view=None, **kwargs):
        """
        Gets the parameters of the computation from the previous inputs form,
        and launches a computation (on the cluster or locally).
        
        Invoke custom method on an Adapter Instance. Make sure when the
        operation has finished that the correct results are stored into DB.
        """
        if not isinstance(adapter_instance, ABCAdapter):
            self.logger.warning("Inconsistent Adapter Class:" + str(adapter_instance.__class__))
            raise LaunchException("Developer Exception!!")

        # Store Operation entity.
        algo = adapter_instance.stored_adapter
        algo_category = dao.get_category_by_id(algo.fk_category)

        operations = self.prepare_operations(current_user.id, project, algo, algo_category,
                                             visible, view_model=model_view, **kwargs)[0]

        if adapter_instance.launch_mode == AdapterLaunchModeEnum.SYNC_SAME_MEM:
            if len(operations) > 1:
                raise LaunchException("Synchronous operations are not supporting ranges!")
            if len(operations) < 1:
                self.logger.warning("No operation was defined")
                raise LaunchException("Invalid empty Operation!!!")
            return self.initiate_prelaunch(operations[0], adapter_instance, **kwargs)
        else:
            return self._send_to_cluster(operations, adapter_instance, current_user.username)

    @staticmethod
    def _prepare_metadata(algo_category, submit_data, operation_group=None, burst=None,
                          current_ga=GenericAttributes()):
        """
        Gather generic_metadata from submitted fields and current to be execute algorithm.
        Will populate STATE, GROUP, etc in generic_metadata
        """
        generic_metadata = GenericAttributes()
        generic_metadata.state = algo_category.defaultdatastate
        generic_metadata.parent_burst = burst
        if DataTypeMetaData.KEY_OPERATION_TAG in submit_data:
            generic_metadata.operation_tag = submit_data[DataTypeMetaData.KEY_OPERATION_TAG]
        if DataTypeMetaData.KEY_TAG_1 in submit_data:
            generic_metadata.user_tag_1 = submit_data[DataTypeMetaData.KEY_TAG_1]
        generic_metadata.fill_from(current_ga)
        return generic_metadata

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

    def group_operation_launch(self, user_id, project, algorithm_id, category_id, existing_dt_group=None, **kwargs):
        """
        Create and prepare the launch of a group of operations.
        """
        category = dao.get_category_by_id(category_id)
        algorithm = dao.get_algorithm_by_id(algorithm_id)
        ops, _ = self.prepare_operations(user_id, project, algorithm, category,
                                         existing_dt_group=existing_dt_group, **kwargs)
        for operation in ops:
            self.launch_operation(operation.id, True)

    def _prepare_metric_operation(self, sim_operation):
        # type: (Operation) -> Operation
        metric_algo = dao.get_algorithm_by_module(MEASURE_METRICS_MODULE, MEASURE_METRICS_CLASS)
        datatype_index = h5.REGISTRY.get_index_for_datatype(TimeSeries)
        time_series_index = dao.get_generic_entity(datatype_index, sim_operation.id, 'fk_from_operation')[0]
        ga = self._prepare_metadata(metric_algo.algorithm_category, {}, None, time_series_index.fk_parent_burst)
        ga.visible = False

        view_model = get_class_by_name("{}.{}".format(MEASURE_METRICS_MODULE, MEASURE_METRICS_MODEL_CLASS))()
        view_model.time_series = time_series_index.gid
        view_model.algorithms = tuple(ALGORITHMS.keys())
        view_model.generic_attributes = ga

        parent_burst = dao.get_generic_entity(BurstConfiguration, time_series_index.fk_parent_burst, 'gid')[0]
        metric_op_group = dao.get_operationgroup_by_id(parent_burst.fk_metric_operation_group)
        metric_operation_group_id = parent_burst.fk_metric_operation_group
        range_values = sim_operation.range_values
        view_model.operation_group_gid = uuid.UUID(metric_op_group.gid)
        view_model.ranges = json.dumps(parent_burst.ranges)
        view_model.range_values = range_values
        view_model.is_metric_operation = True
        metric_operation = Operation(view_model.gid.hex, sim_operation.fk_launched_by, sim_operation.fk_launched_in,
                                     metric_algo.id, user_group=ga.operation_tag, op_group_id=metric_operation_group_id,
                                     range_values=range_values)
        metric_operation.visible = False
        metric_operation = dao.store_entity(metric_operation)

        metrics_datatype_group = dao.get_generic_entity(DataTypeGroup, metric_operation_group_id,
                                                        'fk_operation_group')[0]
        if metrics_datatype_group.fk_from_operation is None:
            metrics_datatype_group.fk_from_operation = metric_operation.id
            dao.store_entity(metrics_datatype_group)

        self.store_view_model(metric_operation, sim_operation.project, view_model)
        return metric_operation

    @transactional
    def prepare_operation(self, user_id, project_id, algorithm, view_model_gid,
                          op_group=None, ranges=None, visible=True):

        op_group_id = None
        if op_group:
            op_group_id = op_group.id
        if isinstance(view_model_gid, uuid.UUID):
            view_model_gid = view_model_gid.hex

        operation = Operation(view_model_gid, user_id, project_id, algorithm.id,
                              op_group_id=op_group_id, range_values=ranges)
        self.logger.debug("Saving Operation(userId=" + str(user_id) + ",projectId=" + str(project_id) +
                          ",algorithmId=" + str(algorithm.id) + ", ops_group= " + str(op_group_id) + ")")

        operation.visible = visible
        operation = dao.store_entity(operation)
        return operation

    def prepare_operations(self, user_id, project, algorithm, category,
                           visible=True, existing_dt_group=None, view_model=None, **kwargs):
        """
        Do all the necessary preparations for storing an operation. If it's the case of a
        range of values create an operation group and multiple operations for each possible
        instance from the range.
        """
        operations = []

        available_args, group = self._prepare_group(project.id, existing_dt_group, kwargs)
        if len(available_args) > TvbProfile.current.MAX_RANGE_NUMBER:
            raise LaunchException("Too big range specified. You should limit the"
                                  " resulting operations to %d" % TvbProfile.current.MAX_RANGE_NUMBER)
        else:
            self.logger.debug("Launching a range with %d operations..." % len(available_args))
        group_id = None
        if group is not None:
            group_id = group.id
        ga = self._prepare_metadata(category, kwargs, group, current_ga=view_model.generic_attributes)
        ga.visible = visible
        view_model.generic_attributes = ga

        self.logger.debug("Saving Operation(userId=" + str(user_id) + ",projectId=" + str(project.id) +
                          ",algorithmId=" + str(algorithm.id) + ", ops_group= " + str(group_id) + ")")

        for (one_set_of_args, range_vals) in available_args:
            range_values = json.dumps(range_vals) if range_vals else None
            operation = Operation(view_model.gid.hex, user_id, project.id, algorithm.id,
                                  op_group_id=group_id, user_group=ga.operation_tag, range_values=range_values)
            operation.visible = visible
            operations.append(operation)
        operations = dao.store_entities(operations)

        if group is not None:
            if existing_dt_group is None:
                datatype_group = DataTypeGroup(group, operation_id=operations[0].id, state=category.defaultdatastate)
                dao.store_entity(datatype_group)
            else:
                # Reset count
                existing_dt_group.count_results = None
                dao.store_entity(existing_dt_group)

        for operation in operations:
            self.store_view_model(operation, project, view_model)

        return operations, group

    def store_view_model(self, operation, project, view_model):
        storage_path = FilesHelper().get_project_folder(project, str(operation.id))
        h5.store_view_model(view_model, storage_path)
        view_model_size_on_disk = FilesHelper.compute_recursive_h5_disk_usage(storage_path)
        operation.view_model_disk_size = view_model_size_on_disk
        dao.store_entity(operation)

    def initiate_prelaunch(self, operation, adapter_instance, **kwargs):
        """
        Public method.
        This should be the common point in calling an adapter- method.
        """
        result_msg = ""
        temp_files = []
        try:
            unique_id = None
            if self.ATT_UID in kwargs:
                unique_id = kwargs[self.ATT_UID]

            operation = dao.get_operation_by_id(operation.id)  # Load Lazy fields

            disk_space_per_user = TvbProfile.current.MAX_DISK_SPACE
            pending_op_disk_space = dao.compute_disk_size_for_started_ops(operation.fk_launched_by)
            user_disk_space = dao.compute_user_generated_disk_size(operation.fk_launched_by)  # From kB to Bytes
            available_space = disk_space_per_user - pending_op_disk_space - user_disk_space

            view_model = adapter_instance.load_view_model(operation)
            try:
                form = adapter_instance.get_form()
                form = form() if isclass(form) else form
                fields = form.get_upload_field_names()
                project = dao.get_project_by_id(operation.fk_launched_in)
                tmp_folder = self.file_helper.get_project_folder(project, self.file_helper.TEMP_FOLDER)
                for upload_field in fields:
                    if hasattr(view_model, upload_field):
                        file = getattr(view_model, upload_field)
                        if file.startswith(tmp_folder) or file.startswith(TvbProfile.current.TVB_TEMP_FOLDER):
                            temp_files.append(file)
            except AttributeError:
                # Skip if we don't have upload fields on current form
                pass
            result_msg, nr_datatypes = adapter_instance._prelaunch(operation, view_model, unique_id, available_space)
            operation = dao.get_operation_by_id(operation.id)
            # Update DB stored kwargs for search purposes, to contain only valuable params (no unselected options)
            operation.mark_complete(STATUS_FINISHED)
            dao.store_entity(operation)

            self._update_vm_generic_operation_tag(view_model, operation)
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

        if operation.fk_operation_group and 'SimulatorAdapter' in operation.algorithm.classname:
            next_op = self._prepare_metric_operation(operation)
            self.launch_operation(next_op.id)
        return result_msg

    def _send_to_cluster(self, operations, adapter_instance, current_username="unknown"):
        """ Initiate operation on cluster"""
        for operation in operations:
            try:
                BackendClientFactory.execute(str(operation.id), current_username, adapter_instance)
            except TVBException as ex:
                self._handle_exception(ex, {}, ex.message, operation)
            except Exception as excep:
                self._handle_exception(excep, {}, "Could not start operation!", operation)

        return operations

    @staticmethod
    def _update_vm_generic_operation_tag(view_model, operation):
        project = dao.get_project_by_id(operation.fk_launched_in)
        storage_path = FilesHelper().get_project_folder(project, str(operation.id))
        h5_path = h5.path_for(storage_path, ViewModelH5, view_model.gid, type(view_model).__name__)
        with ViewModelH5(h5_path, view_model) as vm_h5:
            vm_h5.operation_tag.store(operation.user_group)

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

            if send_to_cluster:
                self._send_to_cluster([operation], adapter_instance, operation.user.username)
            else:
                self.initiate_prelaunch(operation, adapter_instance)

    def _handle_exception(self, exception, temp_files, message, operation=None):
        """
        Common way to treat exceptions:
            - remove temporary files, if any
            - set status ERROR on current operation (if any)
            - log exception
        """
        self.logger.exception(message)
        if operation is not None:
            BurstService().persist_operation_state(operation, STATUS_ERROR, str(exception))
        self._remove_files(temp_files)
        exception.message = message
        raise exception.with_traceback(
            sys.exc_info()[2])  # when rethrowing in python this is required to preserve the stack trace

    def _remove_files(self, file_list):
        """
        Remove any files that exist in the file_dictionary. 
        Currently used to delete temporary files created during an operation.
        """
        for pth in file_list:
            if pth is not None:
                pth = str(pth)
                try:
                    if os.path.exists(pth) and os.path.isfile(pth):
                        os.remove(pth)
                        if len(os.listdir(os.path.dirname(pth))) == 0:
                            shutil.rmtree(os.path.dirname(pth))
                        self.logger.debug("We no longer need file:" + pth + " => deleted")
                    else:
                        self.logger.warning("Trying to remove not existent file:" + pth)
                except OSError:
                    self.logger.exception("Could not cleanup file!")

    @staticmethod
    def _range_name(range_no):
        return PARAM_RANGE_PREFIX + str(range_no)

    def _prepare_group(self, project_id, existing_dt_group, kwargs):
        """
        Create and store OperationGroup entity, or return None
        """
        # Standard ranges as accepted from UI
        range1_values = self._get_range_values(kwargs, self._range_name(1))
        range2_values = self._get_range_values(kwargs, self._range_name(2))
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
            values_for_range = self._get_range_values(kwargs, ranger_name)
            available_args = self.__expand_arguments(available_args, values_for_range, ranger_name)
            last_range_idx += 1
            ranger_name = self._range_name(last_range_idx)
        if last_range_idx > 3:
            ranges = []  # Since we only have 3 fields in db for this just hide it
        if not is_group:
            group = None
        elif existing_dt_group is None:
            group = OperationGroup(project_id=project_id, ranges=ranges)
            group = dao.store_entity(group)
        else:
            group = existing_dt_group.parent_operation_group

        return available_args, group

    def _get_range_values(self, kwargs, ranger_name):
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
            range_values = list(Range(lo=lo_val, hi=hi_val, step=step).to_array())  # , mode=Range.MODE_INCLUDE_BOTH))

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

    def fire_operation(self, adapter_instance, current_user, project_id, visible=True, view_model=None, **data):
        """
        Launch an operation, specified by AdapterInstance, for CurrentUser,
        Current Project and a given set of UI Input Data.
        """
        operation_name = str(adapter_instance.__class__.__name__)
        try:
            self.logger.info("Starting operation " + operation_name)
            project = dao.get_project_by_id(project_id)

            result = self.initiate_operation(current_user, project, adapter_instance, visible,
                                             model_view=view_model, **data)
            self.logger.info("Finished operation launch:" + operation_name)
            return result

        except TVBException as excep:
            self.logger.exception("Could not launch operation " + operation_name +
                                  " with the given set of input data, because: " + excep.message)
            raise OperationException(excep.message, excep)
        except Exception as excep:
            self.logger.exception("Could not launch operation " + operation_name + " with the given set of input data!")
            raise OperationException(str(excep))

    @staticmethod
    def load_operation(operation_id):
        """ Retrieve previously stored Operation from DB, and load operation.burst attribute"""
        operation = dao.get_operation_by_id(operation_id)
        operation.burst = dao.get_burst_for_operation_id(operation_id)
        return operation

    @staticmethod
    def stop_operation(operation_id, is_group=False, remove_after_stop=False):
        # type: (int, bool, bool) -> bool
        """
        Stop (also named Cancel) the operation given by operation_id,
        and potentially also remove it after (with all linked data).
        In case the Operation has a linked Burst, remove that too.
        :param operation_id: ID for Operation (or OperationGroup) to be canceled/removed
        :param is_group: When true stop all the operations from that group.
        :param remove_after_stop: if True, also remove the operation(s) after stopping
        :returns True if the stop step was successfully
        """
        result = False
        if is_group:
            op_group = ProjectService.get_operation_group_by_id(operation_id)
            operations_in_group = ProjectService.get_operations_in_group(op_group)
            for operation in operations_in_group:
                result = OperationService.stop_operation(operation.id, False, remove_after_stop) or result
        else:
            result = BackendClientFactory.stop_operation(operation_id)
            if remove_after_stop:
                burst_config = dao.get_burst_for_direct_operation_id(operation_id)
                ProjectService().remove_operation(operation_id)
                if burst_config is not None:
                    result = dao.remove_entity(BurstConfiguration, burst_config.id) or result
        return result
