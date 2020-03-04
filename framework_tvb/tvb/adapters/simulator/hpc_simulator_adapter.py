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

import os
import uuid
import typing
from tvb.adapters.analyzers.metrics_group_timeseries import TimeseriesMetricsAdapterModel, choices, \
    TimeseriesMetricsAdapter
from tvb.adapters.datatypes.db.mapped_value import DatatypeMeasureIndex
from tvb.adapters.datatypes.db.simulation_history import SimulationHistoryIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.adapters.simulator.simulator_adapter import SimulatorAdapter, SimulatorAdapterModel
from tvb.basic.neotraits.api import HasTraits
from tvb.core.neocom import h5


class HPCSimulatorAdapter(SimulatorAdapter):
    OUTPUT_FOLDER = 'output'

    def __init__(self, storage_path, is_group_launch):
        super(HPCSimulatorAdapter, self).__init__()
        self.storage_path = storage_path
        self.is_group_launch = is_group_launch

    def get_output(self):
        return [TimeSeriesIndex, SimulationHistoryIndex, DatatypeMeasureIndex]

    def load_traited_by_gid(self, data_gid, dt_class=None):
        # type: (uuid.UUID, typing.Type[HasTraits]) -> HasTraits
        """
        Load a generic HasTraits instance, specified by GID.
        """
        return h5.load_from_dir(self.storage_path, data_gid, dt_class=dt_class)

    def _try_load_region_mapping(self):
        return None, None

    def _is_group_launch(self):
        """
        Return true if this adapter is launched from a group of operations
        """
        return self.is_group_launch

    def _get_output_path(self):
        output_path = os.path.join(self.storage_path, self.OUTPUT_FOLDER)
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        return output_path

    def _extract_operation_data(self, operation=None):
        """
        Do nothing for HPC run.
        :param operation: None
        """

    def _update_operation_entity(self, operation, required_disk_space):
        """
        """

    def _capture_operation_results(self, result):
        """
        """
        return "", 2

    def _ensure_enough_resources(self, available_disk_space, view_model):
        return 0

    def launch(self, view_model):
    # type: (SimulatorAdapterModel) -> [TimeSeriesIndex, SimulationHistoryIndex]
        simulation_results = super(HPCSimulatorAdapter, self).launch(view_model)

        if not self.is_group_launch:
            return simulation_results

        for dt in simulation_results:
            if issubclass(type(dt), TimeSeriesIndex):
                metric_result = self._compute_metrics_for_pse_launch(dt)
                simulation_results.append(metric_result)

        return simulation_results

    def _compute_metrics_for_pse_launch(self, time_series_index):
        # type: (TimeSeriesIndex) -> [DatatypeMeasureIndex]
        metric_vm = TimeseriesMetricsAdapterModel()
        metric_vm.time_series = time_series_index.gid
        metric_vm.algorithms = tuple(choices.values())
        metric_adapter = HPCTimeseriesMetricsAdapter(self.storage_path, time_series_index)
        metric_result = metric_adapter.launch(metric_vm)
        return metric_result


class HPCTimeseriesMetricsAdapter(TimeseriesMetricsAdapter):
    OUTPUT_FOLDER = 'output'

    def __init__(self, storage_path, input_time_series_index):
        super(HPCTimeseriesMetricsAdapter,self).__init__()
        self.storage_path = storage_path
        self.input_time_series_index = input_time_series_index

    def configure(self, view_model):
        # type: (TimeseriesMetricsAdapterModel) -> None
        """
        Store the input shape to be later used to estimate memory usage.
        """
        self.input_shape = (self.input_time_series_index.data_length_1d,
                            self.input_time_series_index.data_length_2d,
                            self.input_time_series_index.data_length_3d,
                            self.input_time_series_index.data_length_4d)

    def load_traited_by_gid(self, data_gid, dt_class=None):
        # type: (uuid.UUID, typing.Type[HasTraits]) -> HasTraits
        """
        Load a generic HasTraits instance, specified by GID.
        """
        return h5.load_from_dir(os.path.join(self.storage_path, 'output'), data_gid, dt_class=dt_class)

    def _get_output_path(self):
        output_path = os.path.join(self.storage_path, self.OUTPUT_FOLDER)
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        return output_path