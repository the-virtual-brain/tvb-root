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
Module in charge with Launching an operation (creating the Operation entity as well, based on gathered parameters).

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Yann Gordon <yann@tvb.invalid>
"""

import json
import os
import sys
import uuid
import zipfile
from inspect import isclass

from tvb.basic.exceptions import TVBException
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.config import MEASURE_METRICS_MODULE, MEASURE_METRICS_CLASS, MEASURE_METRICS_MODEL_CLASS, ALGORITHMS
from tvb.core.adapters.abcadapter import ABCAdapter, AdapterLaunchModeEnum
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.generic_attributes import GenericAttributes
from tvb.core.entities.load import get_class_by_name
from tvb.core.entities.model.model_burst import PARAM_RANGE_PREFIX, RANGE_PARAMETER_1, RANGE_PARAMETER_2, \
    BurstConfiguration
from tvb.core.entities.model.model_datatype import DataTypeGroup
from tvb.core.entities.model.model_operation import STATUS_FINISHED, STATUS_ERROR, Operation
from tvb.core.entities.storage import dao, transactional
from tvb.core.neocom import h5
from tvb.core.neotraits.h5 import ViewModelH5
from tvb.core.services.backend_client_factory import BackendClientFactory
from tvb.core.services.burst_service import BurstService
from tvb.core.services.exceptions import OperationException
from tvb.core.services.project_service import ProjectService
from tvb.datatypes.time_series import TimeSeries
from tvb.storage.storage_interface import StorageInterface

RANGE_PARAMETER_1 = RANGE_PARAMETER_1
RANGE_PARAMETER_2 = RANGE_PARAMETER_2

GROUP_BURST_PENDING = {}


class OperationService:
    """
    Class responsible for preparing an operation launch.
    It will prepare parameters, and decide if the operation is to be executed
    immediately, or to be sent on the cluster.
    """
    ATT_UID = "uid"

    def __init__(self):
        self.logger = get_logger(self.__class__.__module__)
        self.storage_interface = StorageInterface()

    ##########################################################################################
    ######## Methods related to launching operations start here ##############################
    ##########################################################################################

    def fits_max_operation_size(self, adapter_instance, view_model, project_id, range_length=1):
        project = dao.get_project_by_id(project_id)
        if project.max_operation_size is None:
            return True

        adapter_instance.configure(view_model)
        adapter_required_memory = adapter_instance.get_required_disk_size(view_model)
        return adapter_required_memory * range_length < project.max_operation_size

    def initiate_operation(self, current_user, project, adapter_instance, visible=True, model_view=None):
        """
        Gets the parameters of the computation from the previous inputs form,
        and launches a computation (on the cluster or locally).

        Invoke custom method on an Adapter Instance. Make sure when the
        operation has finished that the correct results are stored into DB.
        """
        if not isinstance(adapter_instance, ABCAdapter):
            self.logger.warning("Inconsistent Adapter Class:" + str(adapter_instance.__class__))
            raise LaunchException("Developer Exception!!")

        algo = adapter_instance.stored_adapter
        operation = self.prepare_operation(current_user.id, project, algo, visible, model_view)
        if adapter_instance.launch_mode == AdapterLaunchModeEnum.SYNC_SAME_MEM:
            return self.initiate_prelaunch(operation, adapter_instance)
        else:
            return self._send_to_cluster(operation, adapter_instance, current_user.username)

    @staticmethod
    def prepare_metadata(algo_category, burst=None, current_ga=GenericAttributes()):
        """
        Gather generic_metadata from submitted fields and current to be execute algorithm.
        Will populate STATE, GROUP, etc in generic_metadata
        """
        generic_metadata = GenericAttributes()
        generic_metadata.state = algo_category.defaultdatastate
        generic_metadata.parent_burst = burst
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

    def group_operation_launch(self, user_id, project, algorithm_id, category_id):
        """
        Create and prepare the launch of a group of operations.
        """
        algorithm = dao.get_algorithm_by_id(algorithm_id)
        ops, _ = self.prepare_operation(user_id, project, algorithm)
        for operation in ops:
            self.launch_operation(operation.id, True)

    def _prepare_metric_operation(self, sim_operation):
        # type: (Operation) -> Operation
        metric_algo = dao.get_algorithm_by_module(MEASURE_METRICS_MODULE, MEASURE_METRICS_CLASS)
        datatype_index = h5.REGISTRY.get_index_for_datatype(TimeSeries)
        time_series_index = dao.get_generic_entity(datatype_index, sim_operation.id, 'fk_from_operation')[0]
        ga = self.prepare_metadata(metric_algo.algorithm_category, time_series_index.fk_parent_burst)
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
    def prepare_operation(self, user_id, project, algorithm, visible=True, view_model=None, ranges=None,
                          burst_gid=None, op_group_id=None):
        """
        Do all the necessary preparations for storing an operation. If it's the case of a
        range of values create an operation group and multiple operations for each possible
        instance from the range.
        """
        algo_category = dao.get_category_by_id(algorithm.fk_category)
        ga = self.prepare_metadata(algo_category, current_ga=view_model.generic_attributes, burst=burst_gid)
        ga.visible = visible
        view_model.generic_attributes = ga

        self.logger.debug("Saving Operation(userId=" + str(user_id) + ",projectId=" + str(project.id) +
                          ",algorithmId=" + str(algorithm.id) + ")")

        operation = Operation(view_model.gid.hex, user_id, project.id, algorithm.id, user_group=ga.operation_tag,
                              op_group_id=op_group_id, range_values=ranges)
        operation = dao.store_entity(operation)

        self.store_view_model(operation, project, view_model)

        return operation

    @staticmethod
    def store_view_model(operation, project, view_model):
        storage_path = StorageInterface().get_project_folder(project.name, str(operation.id))
        h5.store_view_model(view_model, storage_path)
        view_model_size_on_disk = StorageInterface.compute_recursive_h5_disk_usage(storage_path)
        operation.view_model_disk_size = view_model_size_on_disk
        dao.store_entity(operation)

    def initiate_prelaunch(self, operation, adapter_instance):
        """
        Public method.
        This should be the common point in calling an adapter- method.
        """
        result_msg = ""
        nr_datatypes = 0
        temp_files = []
        try:
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
                tmp_folder = self.storage_interface.get_temp_folder(project.name)
                for upload_field in fields:
                    if hasattr(view_model, upload_field):
                        file = getattr(view_model, upload_field)
                        if file.startswith(tmp_folder) or file.startswith(TvbProfile.current.TVB_TEMP_FOLDER):
                            temp_files.append(file)
            except AttributeError:
                # Skip if we don't have upload fields on current form
                pass
            result_msg, nr_datatypes = adapter_instance._prelaunch(operation, view_model, available_space)
            operation = dao.get_operation_by_id(operation.id)
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

        if operation.fk_operation_group and 'SimulatorAdapter' in operation.algorithm.classname and nr_datatypes == 1:
            next_op = self._prepare_metric_operation(operation)
            self.launch_operation(next_op.id)
        return result_msg

    def _send_to_cluster(self, operation, adapter_instance, current_username="unknown"):
        """ Initiate operation on cluster"""
        try:
            BackendClientFactory.execute(str(operation.id), current_username, adapter_instance)
        except TVBException as ex:
            self._handle_exception(ex, {}, ex.message, operation)
        except Exception as excep:
            self._handle_exception(excep, {}, "Could not start operation!", operation)

        return operation

    @staticmethod
    def _update_vm_generic_operation_tag(view_model, operation):
        project = dao.get_project_by_id(operation.fk_launched_in)
        h5_path = h5.path_for(operation.id, ViewModelH5, view_model.gid, project.name, type(view_model).__name__)

        if not os.path.exists(h5_path):
            return

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
                self._send_to_cluster(operation, adapter_instance, operation.user.username)
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
                            self.storage_interface.remove_folder(os.path.dirname(pth))
                        self.logger.debug("We no longer need file:" + pth + " => deleted")
                    else:
                        self.logger.warning("Trying to remove not existent file:" + pth)
                except OSError:
                    self.logger.exception("Could not cleanup file!")

    @staticmethod
    def _range_name(range_no):
        return PARAM_RANGE_PREFIX + str(range_no)

    def fire_operation(self, adapter_instance, current_user, project_id, visible=True, view_model=None):
        """
        Launch an operation, specified by AdapterInstance, for current_user and project with project_id.
        """
        operation_name = str(adapter_instance.__class__.__name__)
        try:
            self.logger.info("Starting operation " + operation_name)
            project = dao.get_project_by_id(project_id)

            result = self.initiate_operation(current_user, project, adapter_instance, visible,
                                             model_view=view_model)
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
        elif dao.try_get_operation_by_id(operation_id) is not None:
            result = BackendClientFactory.stop_operation(operation_id)
            if remove_after_stop:
                burst_config = dao.get_burst_for_direct_operation_id(operation_id)
                ProjectService().remove_operation(operation_id)
                if burst_config is not None:
                    result = dao.remove_entity(BurstConfiguration, burst_config.id) or result

        return result
