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
import copy
import json
import uuid
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.file.simulator import h5_factory
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.sensors import SensorsEEG, SensorsInternal, SensorsMEG
from tvb.datatypes.surfaces import CorticalSurface
from tvb.simulator.monitors import EEG, Projection, MEG, iEEG
from tvb.simulator.simulator import Simulator
from tvb.adapters.datatypes.h5.region_mapping_h5 import RegionMappingH5
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.file.simulator.cortex_h5 import CortexH5
from tvb.core.entities.file.simulator.simulator_h5 import SimulatorH5
from tvb.core.entities.model.model_datatype import DataTypeGroup
from tvb.core.entities.model.model_operation import Operation
from tvb.core.entities.model.simulator.simulator import SimulatorIndex
from tvb.core.entities.storage import dao, transactional
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.services.burst_service2 import BurstService2
from tvb.core.services.operation_service import OperationService
from tvb.core.neocom import h5


class SimulatorService(object):
    MAX_BURSTS_DISPLAYED = 50
    LAUNCH_NEW = 'new'
    LAUNCH_BRANCH = 'branch'

    def __init__(self):
        self.logger = get_logger(self.__class__.__module__)
        self.operation_service = OperationService()
        self.files_helper = FilesHelper()

    @staticmethod
    def serialize_simulator(simulator, simulator_gid, simulation_state_gid, storage_path):
        simulator_path = h5.path_for(storage_path, SimulatorH5, simulator_gid)

        with SimulatorH5(simulator_path) as simulator_h5:
            simulator_h5.gid.store(uuid.UUID(simulator_gid))
            simulator_h5.store(simulator)
            simulator_h5.connectivity.store(simulator.connectivity.gid)
            if simulator.stimulus:
                simulator_h5.stimulus.store(uuid.UUID(simulator.stimulus.gid))
            if simulation_state_gid:
                simulator_h5.simulation_state.store(uuid.UUID(simulation_state_gid))

        return simulator_gid

    @staticmethod
    def deserialize_simulator(simulator_gid, storage_path):
        simulator_in_path = h5.path_for(storage_path, SimulatorH5, simulator_gid)
        simulator_in = Simulator()

        with SimulatorH5(simulator_in_path) as simulator_in_h5:
            simulator_in_h5.load_into(simulator_in)
            connectivity_gid = simulator_in_h5.connectivity.load()
            stimulus_gid = simulator_in_h5.stimulus.load()
            simulation_state_gid = simulator_in_h5.simulation_state.load()

        if isinstance(simulator_in.monitors[0], Projection):
            with SimulatorH5(simulator_in_path) as simulator_in_h5:
                monitor_h5_path = simulator_in_h5.get_reference_path(simulator_in.monitors[0].gid)

            monitor_h5_class = h5_factory.monitor_h5_factory(type(simulator_in.monitors[0]))

            with monitor_h5_class(monitor_h5_path) as monitor_h5:
                sensors = monitor_h5.sensors.load()
                region_mapping = monitor_h5.region_mapping.load()

            sensors_index = ABCAdapter.load_entity_by_gid(sensors.hex)
            sensors = h5.load_from_index(sensors_index)

            if isinstance(simulator_in.monitors[0], EEG):
                sensors = SensorsEEG.build_sensors_subclass(sensors)
            elif isinstance(simulator_in.monitors[0], MEG):
                sensors = SensorsMEG.build_sensors_subclass(sensors)
            elif isinstance(simulator_in.monitors[0], iEEG):
                sensors = SensorsInternal.build_sensors_subclass(sensors)

            simulator_in.monitors[0].sensors = sensors
            region_mapping_index = ABCAdapter.load_entity_by_gid(region_mapping.hex)
            region_mapping = h5.load_from_index(region_mapping_index)
            simulator_in.monitors[0].region_mapping = region_mapping

        conn_index = dao.get_datatype_by_gid(connectivity_gid.hex)
        conn = h5.load_from_index(conn_index)

        simulator_in.connectivity = conn

        if simulator_in.surface:
            cortex_path = h5.path_for(storage_path, CortexH5, simulator_in.surface.gid)
            with CortexH5(cortex_path) as cortex_h5:
                local_conn_gid = cortex_h5.local_connectivity.load()
                region_mapping_gid = cortex_h5.region_mapping_data.load()

            region_mapping_index = dao.get_datatype_by_gid(region_mapping_gid.hex)
            region_mapping_path = h5.path_for_stored_index(region_mapping_index)
            region_mapping = RegionMapping()
            with RegionMappingH5(region_mapping_path) as region_mapping_h5:
                region_mapping_h5.load_into(region_mapping)
                region_mapping.gid = region_mapping_h5.gid.load()
                surf_gid = region_mapping_h5.surface.load()

            surf_index = dao.get_datatype_by_gid(surf_gid.hex)
            surf_h5 = h5.h5_file_for_index(surf_index)
            surf = CorticalSurface()
            surf_h5.load_into(surf)
            surf_h5.close()
            region_mapping.surface = surf
            simulator_in.surface.region_mapping_data = region_mapping

            if local_conn_gid:
                local_conn_index = dao.get_datatype_by_gid(local_conn_gid.hex)
                local_conn = h5.load_from_index(local_conn_index)
                simulator_in.surface.local_connectivity = local_conn

        if stimulus_gid:
            stimulus_index = dao.get_datatype_by_gid(stimulus_gid.hex)
            stimulus = h5.load_from_index(stimulus_index)
            simulator_in.stimulus = stimulus

        return simulator_in, simulation_state_gid

    @transactional
    def _prepare_operation(self, project_id, user_id, simulator_id, simulator_index, algo_category, op_group, metadata,
                           ranges=None):
        operation_parameters = json.dumps({'simulator_gid': simulator_index.gid})
        metadata, user_group = self.operation_service._prepare_metadata(metadata, algo_category, op_group, {})
        meta_str = json.dumps(metadata)

        op_group_id = None
        if op_group:
            op_group_id = op_group.id

        operation = Operation(user_id, project_id, simulator_id, operation_parameters, op_group_id=op_group_id,
                              meta=meta_str, range_values=ranges)

        self.logger.debug("Saving Operation(userId=" + str(user_id) + ",projectId=" + str(project_id) + "," +
                          str(metadata) + ",algorithmId=" + str(simulator_id) + ", ops_group= " + str(
            op_group_id) + ")")

        # visible_operation = visible and category.display is False
        operation = dao.store_entity(operation)
        # operation.visible = visible_operation

        # TODO: prepare portlets/handle operation groups/no workflows

        return operation

    @staticmethod
    def _set_simulator_range_parameter(simulator, range_parameter_name, range_parameter_value):
        range_param_name_list = range_parameter_name.split('.')
        current_attr = simulator
        for param_name in range_param_name_list[:len(range_param_name_list) - 1]:
            current_attr = getattr(current_attr, param_name)
        setattr(current_attr, range_param_name_list[-1], range_parameter_value)

    def async_launch_and_prepare_simulation(self, burst_config, user, project, simulator_algo,
                                            session_stored_simulator, simulation_state_gid):
        try:
            simulator_index = SimulatorIndex()
            metadata = {}
            if burst_config:
                simulator_index.fk_parent_burst = burst_config.id
                metadata.update({DataTypeMetaData.KEY_BURST: burst_config.id})
            dao.store_entity(simulator_index)
            simulator_id = simulator_algo.id
            algo_category = simulator_algo.algorithm_category
            operation = self._prepare_operation(project.id, user.id, simulator_id, simulator_index, algo_category, None,
                                                metadata)

            simulator_index.fk_from_operation = operation.id
            dao.store_entity(simulator_index)

            storage_path = self.files_helper.get_project_folder(project, str(operation.id))
            self.serialize_simulator(session_stored_simulator, simulator_index.gid, simulation_state_gid, storage_path)

            wf_errs = 0
            try:
                OperationService().launch_operation(operation.id, True)
                return operation
            except Exception as excep:
                self.logger.error(excep)
                wf_errs += 1
                if burst_config:
                    BurstService2().mark_burst_finished(burst_config, error_message=str(excep))

            self.logger.debug("Finished launching workflow. The operation was launched successfully, " +
                              str(wf_errs) + " had error on pre-launch steps")

        except Exception as excep:
            self.logger.error(excep)
            if burst_config:
                BurstService2().mark_burst_finished(burst_config, error_message=str(excep))

    def async_launch_and_prepare_pse(self, burst_config, user, project, simulator_algo, range_param1, range_param2,
                                     session_stored_simulator):
        try:
            simulator_id = simulator_algo.id
            algo_category = simulator_algo.algorithm_category
            operation_group = burst_config.operation_group
            metric_operation_group = burst_config.metric_operation_group
            operations = []
            range_param2_values = []
            if range_param2:
                range_param2_values = range_param2.get_range_values()
            for param1_value in range_param1.get_range_values():
                for param2_value in range_param2_values:
                    simulator = copy.deepcopy(session_stored_simulator)
                    self._set_simulator_range_parameter(simulator, range_param1.name, param1_value)
                    self._set_simulator_range_parameter(simulator, range_param2.name, param2_value)

                    simulator_index = SimulatorIndex()
                    simulator_index.fk_parent_burst = burst_config.id
                    simulator_index = dao.store_entity(simulator_index)
                    ranges = json.dumps({range_param1.name: param1_value[0], range_param2.name: param2_value[0]})

                    operation = self._prepare_operation(project.id, user.id, simulator_id, simulator_index,
                                                        algo_category, operation_group,
                                                        {DataTypeMetaData.KEY_BURST: burst_config.id}, ranges)

                    simulator_index.fk_from_operation = operation.id
                    dao.store_entity(simulator_index)

                    storage_path = self.files_helper.get_project_folder(project, str(operation.id))
                    self.serialize_simulator(simulator, simulator_index.gid, None, storage_path)
                    operations.append(operation)

            first_operation = operations[0]
            datatype_group = DataTypeGroup(operation_group, operation_id=first_operation.id,
                                           fk_parent_burst=burst_config.id,
                                           state=json.loads(first_operation.meta_data)[DataTypeMetaData.KEY_STATE])
            dao.store_entity(datatype_group)

            metrics_datatype_group = DataTypeGroup(metric_operation_group, fk_parent_burst=burst_config.id)
            dao.store_entity(metrics_datatype_group)

            wf_errs = 0
            for operation in operations:
                try:
                    OperationService().launch_operation(operation.id, True)
                except Exception as excep:
                    self.logger.error(excep)
                    wf_errs += 1
                    BurstService2().mark_burst_finished(burst_config, error_message=str(excep))

            self.logger.debug("Finished launching workflows. " + str(len(operations) - wf_errs) +
                              " were launched successfully, " + str(wf_errs) + " had error on pre-launch steps")

        except Exception as excep:
            self.logger.error(excep)
            BurstService2().mark_burst_finished(burst_config, error_message=str(excep))
