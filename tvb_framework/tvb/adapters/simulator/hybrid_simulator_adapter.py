# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
Adapter running one Hybrid simulation out of the configuration gathered by the Hybrid Simulator cockpit.

The configuration arrives as a HybridSimulatorAdapterModel: a Connectivity, the Subnetworks with their
dynamics, the shared integration step size, the Monitors and the simulation length. Everything the
scientific side needs is built by HybridSimulatorService, so this adapter only translates, runs and
persists.

.. moduleauthor:: TVB Team
"""

import json

from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.adapters.forms.hybrid_simulator_fragments import HybridConnectivityFragment
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.file.simulator.view_model import HybridSimulatorAdapterModel
from tvb.core.neocom import h5
from tvb.core.services.hybrid_simulator_service import HybridSimulatorService, HybridSubnetworkException


class HybridSimulatorAdapter(ABCAdapter):
    """
    Interface between the Hybrid Simulator and the Framework.
    """
    _ui_name = "Hybrid Simulation Core"

    def __init__(self):
        super(HybridSimulatorAdapter, self).__init__()
        self.hybrid_simulator_service = HybridSimulatorService()
        # the library Simulator, built in configure()
        self.algorithm = None
        self.connectivity = None
        self.layout = None

    def get_form_class(self):
        # the cockpit's own Connectivity step already is this form: the same field, the same filters
        return HybridConnectivityFragment

    def get_output(self):
        """
        :returns: list of classes for possible results of the Hybrid Simulator.
        """
        # No SimulationHistoryIndex: branching is not offered, and SimulationHistory.populate_from reads
        # a classic Simulator, which this one is not.
        return [TimeSeriesIndex]

    def configure(self, view_model):
        # type: (HybridSimulatorAdapterModel) -> None
        """
        Turn the stored configuration into a configured ``tvb.simulator.hybrid.Simulator``.
        """
        self.log.debug("%s: Configuring hybrid simulator adapter..." % str(self))

        self.connectivity = h5.load_from_gid(view_model.connectivity)
        self.layout = self.hybrid_simulator_service.output_layout(view_model.subnetworks)

        try:
            # Checked on the stored Monitors, before anything is loaded for them: what is refused here
            # depends only on the Monitor's class and on the layout, and a Monitor that cannot run is
            # better named than followed into loading datatypes it will never use.
            self.hybrid_simulator_service.validate_monitors_for_layout(view_model.monitors, self.layout)
            self.hybrid_simulator_service.validate_monitors(view_model.monitors,
                                                            view_model.simulation_length, view_model.dt)

            # the Monitors are stored as view models holding GIDs; this is what loads the Sensors, the
            # Projection matrix and the Region mapping a projection Monitor points at
            monitors = [self.view_model_to_has_traits(monitor) for monitor in view_model.monitors]

            network_set = self.hybrid_simulator_service.build_network_set(
                self.connectivity, view_model.subnetworks, view_model.dt)
            self.algorithm = self.hybrid_simulator_service.build_hybrid_simulator(
                network_set, monitors, view_model.simulation_length,
                # the Monitors that need a classic Simulator are only meaningful over region ordered
                # output, and validate_monitors_for_layout has just refused any other case
                connectivity=self.connectivity if self.layout['is_merged'] else None,
                layout=self.layout)
        except HybridSubnetworkException as excep:
            raise LaunchException(str(excep), excep)

    def get_required_memory_size(self, view_model):
        # type: (HybridSimulatorAdapterModel) -> int
        """
        Return the required memory to run this algorithm, in bytes.

        Estimated rather than asked for: the hybrid Simulator has no ``memory_requirement``. It is also
        genuinely larger than the classic one's, because ``Simulator.run`` returns whole arrays instead
        of yielding them, so every Monitor's output is held in memory until the run is over.
        """
        return self._recorded_bytes(view_model)

    def get_required_disk_size(self, view_model):
        # type: (HybridSimulatorAdapterModel) -> int
        """
        Return the required disk size this algorithm estimates it will take, in kB.
        """
        return self._recorded_bytes(view_model) / 2 ** 10

    def _recorded_bytes(self, view_model):
        """
        What every Monitor together will record: samples x variables x nodes, as float64.
        """
        layout = self.hybrid_simulator_service.output_layout(view_model.subnetworks)
        total = 0
        for monitor in view_model.monitors or []:
            period = max(float(monitor.period), view_model.dt)
            samples = max(int(view_model.simulation_length / period), 1)
            total += samples * layout['variables'] * layout['nodes'] * 8
        return int(total)

    def get_execution_time_approximation(self, view_model):
        # type: (HybridSimulatorAdapterModel) -> int
        """
        Approximate, in seconds, how long this simulation takes, so that a cluster node does not kill it
        before it is finished. The same brute approximation the classic adapter makes, over the node and
        variable counts this configuration actually holds.
        """
        magic_number = 6.57e-06  # seconds
        layout = self.hybrid_simulator_service.output_layout(view_model.subnetworks)
        approx_modes = 15
        estimation = (magic_number * max(layout['nodes'], 1) * max(layout['variables'], 1) *
                      approx_modes * view_model.simulation_length / max(view_model.dt, 1e-6))
        return max(int(estimation), 1)

    def launch(self, view_model):
        # type: (HybridSimulatorAdapterModel) -> [TimeSeriesIndex]
        """
        Run the configured Hybrid Simulator and store one TimeSeries per Monitor.
        """
        result_h5 = dict()
        result_indexes = dict()

        for monitor in self.algorithm.monitors:
            m_name = type(monitor).__name__
            ts = self._prepare_time_series(monitor)

            ts_index_class = h5.REGISTRY.get_index_for_datatype(type(ts))
            ts_index = ts_index_class()
            ts_index.fill_from_has_traits(ts)
            ts_index.data_ndim = 4
            ts_index.state = 'INTERMEDIATE'
            ts_index.labels_dimensions = json.dumps(ts.labels_dimensions)

            ts_h5_class = h5.REGISTRY.get_h5file_for_datatype(type(ts))
            ts_h5_path = h5.path_by_dir(self._get_output_path(), ts_h5_class, ts.gid)
            self.log.info("Generating Timeseries at: {}".format(ts_h5_path))
            ts_h5 = ts_h5_class(ts_h5_path)
            ts_h5.store(ts, scalars_only=True, store_references=False)
            ts_h5.sample_rate.store(ts.sample_rate)
            ts_h5.nr_dimensions.store(ts_index.data_ndim)
            ts_h5.store_generic_attributes(self.generic_attributes)
            ts_h5.store_references(ts)

            result_indexes[m_name] = ts_index
            result_h5[m_name] = ts_h5

        self.log.debug("Starting hybrid simulation...")
        # Unlike the classic Simulator, this one does not yield as it goes: it returns one (times, data)
        # pair per Monitor once the whole run is over.
        results = self.algorithm.run()

        for monitor, (times, data) in zip(self.algorithm.monitors, results):
            m_name = type(monitor).__name__
            if len(times) == 0:
                self.log.warning("The %s monitor recorded nothing over this simulation", m_name)
                continue
            result_h5[m_name].write_time_slice(times)
            result_h5[m_name].write_data_slice(data)

        self.log.debug("Completed hybrid simulation, storing the results")
        for monitor in self.algorithm.monitors:
            m_name = type(monitor).__name__
            result_indexes[m_name].fill_shape(result_h5[m_name].read_data_shape())
            result_h5[m_name].close()

        self.log.debug("%s: Adapter simulation finished!!" % str(self))
        return list(result_indexes.values())

    def _prepare_time_series(self, monitor):
        """
        The TimeSeries one Monitor will fill.

        Every Monitor that changes the node axis says for itself what it produces - EEG a TimeSeriesEEG
        over its Sensors, Global average a plain TimeSeries - so the Connectivity is simply withheld
        when the output is not in region order, and a Monitor that would have made a TimeSeriesRegion
        makes a plain TimeSeries instead. Claiming a region ordering the data does not have is the one
        thing worth refusing to write.
        """
        connectivity = self.connectivity if self.layout['is_merged'] else None
        ts = monitor.create_time_series(connectivity)
        ts.start_time = 0.0

        # The Subnetworks agree on the *number* of variables they watch, never on their names: JansenRit
        # watches y0, y1 where Generic2dOscillator watches V, W. Labelling by position claims nothing
        # that is untrue of any of them, and their own names stay on the stored configuration.
        state_variable_dimension_name = ts.labels_ordering[1]
        ts.labels_dimensions[state_variable_dimension_name] = [
            'Variable {}'.format(index + 1) for index in range(self.layout['variables'])]
        return ts

    def _get_output_path(self):
        return self.get_storage_path()
