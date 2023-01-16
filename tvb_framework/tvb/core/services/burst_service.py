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

import json
import os
from datetime import datetime

from tvb.basic.logger.builder import get_logger
from tvb.config import MEASURE_METRICS_MODULE, MEASURE_METRICS_CLASS
from tvb.core.entities.file.simulator.burst_configuration_h5 import BurstConfigurationH5
from tvb.core.entities.file.simulator.datatype_measure_h5 import DatatypeMeasureH5
from tvb.core.entities.file.simulator.view_model import SimulatorAdapterModel
from tvb.core.entities.generic_attributes import GenericAttributes
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.core.entities.model.model_datatype import DataTypeGroup
from tvb.core.entities.model.model_operation import Operation, STATUS_FINISHED, STATUS_PENDING, STATUS_CANCELED
from tvb.core.entities.model.model_operation import OperationGroup, STATUS_ERROR, STATUS_STARTED, has_finished
from tvb.core.entities.storage import dao
from tvb.core.entities.transient.range_parameter import RangeParameter
from tvb.core.neocom import h5
from tvb.core.neocom.h5 import DirLoader
from tvb.core.services.import_service import ImportService
from tvb.core.utils import format_bytes_human, format_timedelta
from tvb.storage.storage_interface import StorageInterface

MAX_BURSTS_DISPLAYED = 50
STATUS_FOR_OPERATION = {
    STATUS_PENDING: BurstConfiguration.BURST_RUNNING,
    STATUS_STARTED: BurstConfiguration.BURST_RUNNING,
    STATUS_CANCELED: BurstConfiguration.BURST_CANCELED,
    STATUS_ERROR: BurstConfiguration.BURST_ERROR,
    STATUS_FINISHED: BurstConfiguration.BURST_FINISHED
}


class BurstService(object):
    LAUNCH_NEW = 'new'
    LAUNCH_BRANCH = 'branch'

    def __init__(self):
        self.logger = get_logger(self.__class__.__module__)
        self.storage_interface = StorageInterface()

    def mark_burst_finished(self, burst_entity, burst_status=None, error_message=None, store_h5_file=True):
        """
        Mark Burst status field.
        Also compute 'weight' for current burst: no of operations inside, estimate time on disk...

        :param burst_entity: BurstConfiguration to be updated, at finish time.
        :param burst_status: BurstConfiguration status. By default BURST_FINISHED
        :param error_message: If given, set the status to error and perpetuate the message.
        """
        if burst_status is None:
            burst_status = BurstConfiguration.BURST_FINISHED
        if error_message is not None:
            burst_status = BurstConfiguration.BURST_ERROR

        try:
            # If there are any DataType Groups in current Burst, update their counter.
            burst_dt_groups = dao.get_generic_entity(DataTypeGroup, burst_entity.gid, "fk_parent_burst")
            for dt_group in burst_dt_groups:
                dt_group.count_results = dao.count_datatypes_in_group(dt_group.id)
                dt_group.disk_size, dt_group.subject = dao.get_summary_for_group(dt_group.id)
                dao.store_entity(dt_group)

            # Update actual Burst entity fields
            burst_entity.datatypes_number = dao.count_datatypes_in_burst(burst_entity.gid)

            burst_entity.status = burst_status
            burst_entity.error_message = error_message
            burst_entity.finish_time = datetime.now()
            dao.store_entity(burst_entity)
            if store_h5_file:
                self.store_burst_configuration(burst_entity)
        except Exception:
            self.logger.exception("Could not correctly update Burst status and meta-data!")
            burst_entity.status = burst_status
            burst_entity.error_message = "Error when updating Burst Status"
            burst_entity.finish_time = datetime.now()
            dao.store_entity(burst_entity)
            if store_h5_file:
                self.store_burst_configuration(burst_entity)

    def persist_operation_state(self, operation, operation_status, message=None):
        """
        Update Operation instance state. Store it in DB and on HDD/
        :param operation: Operation instance
        :param operation_status: new status
        :param message: message in case of error
        :return: operation instance changed
        """
        operation.mark_complete(operation_status, message)
        operation.queue_full = False
        operation = dao.store_entity(operation)
        # update burst also
        burst_config = self.get_burst_for_operation_id(operation.id)
        if burst_config is not None:
            burst_status = STATUS_FOR_OPERATION.get(operation_status)
            self.mark_burst_finished(burst_config, burst_status, message)
        return operation

    @staticmethod
    def get_burst_for_operation_id(operation_id, is_group=False):
        return dao.get_burst_for_operation_id(operation_id, is_group)

    def rename_burst(self, burst_id, new_name):
        """
        Rename the burst given by burst_id, setting it's new name to
        burst_name.
        """
        burst = dao.get_burst_by_id(burst_id)
        burst.name = new_name
        dao.store_entity(burst)
        self.store_burst_configuration(burst)

    @staticmethod
    def get_available_bursts(project_id):
        """
        Return all the burst for the current project.
        """
        bursts = dao.get_bursts_for_project(project_id, page_size=MAX_BURSTS_DISPLAYED) or []
        return bursts

    @staticmethod
    def populate_burst_disk_usage(bursts):
        """
        Adds a disk_usage field to each burst object.
        The disk usage is computed as the sum of the datatypes generated by a burst
        """
        sizes = dao.compute_bursts_disk_size([b.gid for b in bursts])
        for b in bursts:
            b.disk_size = format_bytes_human(sizes[b.gid])

    def update_history_status(self, id_list):
        """
        For each burst_id received in the id_list read new status from DB and return a list
        [id, new_status, is_group, message, running_time] tuple.
        """
        result = []
        for b_id in id_list:
            burst = dao.get_burst_by_id(b_id)
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

    @staticmethod
    def update_simulation_fields(burst, op_simulation_id, simulation_gid):
        burst.fk_simulation = op_simulation_id
        burst.simulator_gid = simulation_gid.hex
        burst = dao.store_entity(burst)
        return burst

    @staticmethod
    def load_burst_configuration(burst_config_id):
        # type: (int) -> BurstConfiguration
        burst_config = dao.get_burst_by_id(burst_config_id)
        return burst_config

    @staticmethod
    def remove_burst_configuration(burst_config_id):
        # type: (int) -> None
        dao.remove_entity(BurstConfiguration, burst_config_id)

    @staticmethod
    def prepare_burst_for_pse(burst_config):
        # type: (BurstConfiguration) -> (BurstConfiguration)
        operation_group = OperationGroup(burst_config.fk_project, ranges=burst_config.ranges)
        operation_group = dao.store_entity(operation_group)

        metric_operation_group = OperationGroup(burst_config.fk_project, ranges=burst_config.ranges)
        metric_operation_group = dao.store_entity(metric_operation_group)

        burst_config.operation_group = operation_group
        burst_config.fk_operation_group = operation_group.id
        burst_config.metric_operation_group = metric_operation_group
        burst_config.fk_metric_operation_group = metric_operation_group.id
        return dao.store_entity(burst_config)

    @staticmethod
    def store_burst_configuration(burst_config):
        project = dao.get_project_by_id(burst_config.fk_project)
        bc_path = h5.path_for(burst_config.fk_simulation, BurstConfigurationH5, burst_config.gid, project.name)
        with BurstConfigurationH5(bc_path) as bc_h5:
            bc_h5.store(burst_config)

    @staticmethod
    def load_burst_configuration_from_folder(simulator_folder, project):
        bc_h5_filename = DirLoader(simulator_folder, None).find_file_for_has_traits_type(BurstConfiguration)
        burst_config = BurstConfiguration(project.id)
        with BurstConfigurationH5(os.path.join(simulator_folder, bc_h5_filename)) as bc_h5:
            bc_h5.load_into(burst_config)
        return burst_config

    @staticmethod
    def prepare_simulation_name(burst, project_id):
        simulation_number = dao.get_number_of_bursts(project_id) + 1

        if burst.name is None:
            simulation_name = 'simulation_' + str(simulation_number)
        else:
            simulation_name = burst.name

        return simulation_name, simulation_number

    def prepare_indexes_for_simulation_results(self, operation, result_filenames, burst):
        indexes = list()
        self.logger.debug("Preparing indexes for simulation results in operation {}...".format(operation.id))
        for filename in result_filenames:
            try:
                self.logger.debug("Preparing index for filename: {}".format(filename))
                index = h5.index_for_h5_file(filename)()
                h5_class = h5.REGISTRY.get_h5file_for_index(type(index))

                with h5_class(filename) as index_h5:
                    index.fill_from_h5(index_h5)
                    index.fill_from_generic_attributes(index_h5.load_generic_attributes())

                index.fk_parent_burst = burst.gid
                index.fk_from_operation = operation.id
                if operation.fk_operation_group:
                    datatype_group = dao.get_datatypegroup_by_op_group_id(operation.fk_operation_group)
                    self.logger.debug(
                        "Found DatatypeGroup with id {} for operation {}".format(datatype_group.id, operation.id))
                    index.fk_datatype_group = datatype_group.id

                    # Update the operation group name
                    operation_group = dao.get_operationgroup_by_id(operation.fk_operation_group)
                    operation_group.fill_operationgroup_name("TimeSeriesRegionIndex")
                    dao.store_entity(operation_group)
                self.logger.debug(
                    "Prepared index {} for file {} in operation {}".format(index.summary_info, filename, operation.id))
                indexes.append(index)
            except Exception as e:
                self.logger.debug("Skip preparing index {} because there was an error.".format(filename))
                self.logger.error(e)
        self.logger.debug("Prepared {} indexes for results in operation {}...".format(len(indexes), operation.id))
        return indexes

    def prepare_index_for_metric_result(self, operation, result_filename, burst):
        self.logger.debug("Preparing index for metric result in operation {}...".format(operation.id))
        index = h5.index_for_h5_file(result_filename)()
        with DatatypeMeasureH5(result_filename) as dti_h5:
            index.gid = dti_h5.gid.load().hex
            index.metrics = json.dumps(dti_h5.metrics.load())
            index.fk_source_gid = dti_h5.analyzed_datatype.load().hex
        index.fk_from_operation = operation.id
        index.fk_parent_burst = burst.gid
        datatype_group = dao.get_datatypegroup_by_op_group_id(operation.fk_operation_group)
        self.logger.debug("Found DatatypeGroup with id {} for operation {}".format(datatype_group.id, operation.id))
        index.fk_datatype_group = datatype_group.id
        self.logger.debug("Prepared index {} for results in operation {}...".format(index.summary_info, operation.id))
        return index

    def _update_pse_burst_status(self, burst_config):
        operations_in_group = dao.get_operations_in_group(burst_config.fk_operation_group)
        if burst_config.fk_metric_operation_group:
            operations_in_group.extend(dao.get_operations_in_group(burst_config.fk_metric_operation_group))
        operation_statuses = list()
        for operation in operations_in_group:
            if not has_finished(operation.status):
                self.logger.debug(
                    'Operation {} in group {} is not finished, burst status will not be updated'.format(
                        operation.id, operation.fk_operation_group))
                return
            operation_statuses.append(operation.status)
        self.logger.debug(
            'All operations in burst {} have finished. Will update burst status'.format(burst_config.id))
        if STATUS_ERROR in operation_statuses:
            self.mark_burst_finished(burst_config, BurstConfiguration.BURST_ERROR,
                                     'Some operations in PSE have finished with errors')
        elif STATUS_CANCELED in operation_statuses:
            self.mark_burst_finished(burst_config, BurstConfiguration.BURST_CANCELED)
        else:
            self.mark_burst_finished(burst_config)

    def update_burst_status(self, burst_config):
        if burst_config.fk_operation_group:
            self._update_pse_burst_status(burst_config)
        else:
            operation = dao.get_operation_by_id(burst_config.fk_simulation)
            message = operation.additional_info
            if len(message) == 0:
                message = None
            self.mark_burst_finished(burst_config, STATUS_FOR_OPERATION[operation.status], message)

    @staticmethod
    def prepare_metrics_operation(operation):
        # TODO reuse from OperationService and do not duplicate logic here
        parent_burst = dao.get_generic_entity(BurstConfiguration, operation.fk_operation_group, 'fk_operation_group')[0]
        metric_operation_group_id = parent_burst.fk_metric_operation_group
        range_values = operation.range_values
        metric_algo = dao.get_algorithm_by_module(MEASURE_METRICS_MODULE, MEASURE_METRICS_CLASS)

        metric_operation = Operation(None, operation.fk_launched_by, operation.fk_launched_in, metric_algo.id,
                                     status=STATUS_FINISHED, op_group_id=metric_operation_group_id,
                                     range_values=range_values)
        metric_operation.visible = False
        metric_operation = dao.store_entity(metric_operation)
        op_dir = StorageInterface().get_project_folder(operation.project.name, str(metric_operation.id))
        return op_dir, metric_operation

    @staticmethod
    def get_range_param_by_name(param_name, all_range_parameters):
        for range_param in all_range_parameters:
            if param_name == range_param.name:
                return range_param

        return None

    @staticmethod
    def handle_range_params_at_loading(burst_config, all_range_parameters):
        param1, param2 = None, None
        if burst_config.range1:
            param1 = RangeParameter.from_json(burst_config.range1)
            param1.fill_from_default(BurstService.get_range_param_by_name(param1.name, all_range_parameters))
            if burst_config.range2 is not None:
                param2 = RangeParameter.from_json(burst_config.range2)
                param2.fill_from_default(BurstService.get_range_param_by_name(param2.name, all_range_parameters))

        return param1, param2

    def prepare_data_for_burst_copy(self, burst_config_id, burst_name_format, project):
        burst_config = self.load_burst_configuration(burst_config_id)
        burst_config_copy = burst_config.clone()
        count = dao.count_bursts_with_name(burst_config.name, burst_config.fk_project)
        burst_config_copy.name = burst_name_format.format(burst_config.name, count + 1)

        storage_path = self.storage_interface.get_project_folder(project.name, str(burst_config.fk_simulation))
        simulator = h5.load_view_model(burst_config.simulator_gid, storage_path)
        simulator.generic_attributes = GenericAttributes()
        return simulator, burst_config_copy

    @staticmethod
    def store_burst(burst_config):
        return dao.store_entity(burst_config)

    def load_simulation_from_zip(self, zip_file, project):
        import_service = ImportService()
        simulator_folder = import_service.import_simulator_configuration_zip(zip_file)

        simulator_h5_filename = DirLoader(simulator_folder, None).find_file_for_has_traits_type(SimulatorAdapterModel)
        simulator_h5_filepath = os.path.join(simulator_folder, simulator_h5_filename)
        simulator = h5.load_view_model_from_file(simulator_h5_filepath)

        burst_config = self.load_burst_configuration_from_folder(simulator_folder, project)
        burst_config_copy = burst_config.clone()
        simulator.generic_attributes.parent_burst = burst_config_copy.gid

        return simulator, burst_config_copy, simulator_folder
