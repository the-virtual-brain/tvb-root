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
Adapter that uses the traits module to generate interfaces to the Simulator.
Few supplementary steps are done here:

   * from submitted Monitor/Model... names, build transient entities
   * after UI parameters submit, compose transient Cortex entity to be passed to the Simulator.

.. moduleauthor:: Paula Popa <paula.popa@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""
import json
from tvb.adapters.simulator.model_forms import get_model_to_form_dict
from tvb.adapters.simulator.monitor_forms import get_monitor_to_form_dict
from tvb.adapters.simulator.simulator_fragments import *
from tvb.adapters.simulator.coupling_forms import get_ui_name_to_coupling_dict
from tvb.adapters.datatypes.db.simulation_history import SimulationHistoryIndex
from tvb.adapters.datatypes.db.region_mapping import RegionMappingIndex, RegionVolumeMappingIndex
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.basic.neotraits.api import Attr
from tvb.core.entities.file.simulator.simulation_history_h5 import SimulationHistory
from tvb.core.entities.file.simulator.view_model import SimulatorAdapterModel
from tvb.core.entities.storage import dao
from tvb.core.adapters.abcadapter import ABCAsynchronous, ABCAdapterForm
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.neotraits.forms import DataTypeSelectField, FloatField, SelectField
from tvb.core.neocom import h5
from tvb.core.services.simulator_serializer import SimulatorSerializer
from tvb.datatypes.cortex import Cortex
from tvb.datatypes.patterns import StimuliSurface
from tvb.datatypes.surfaces import CorticalSurface
from tvb.simulator.coupling import Coupling
from tvb.simulator.simulator import Simulator


class SimulatorAdapterForm(ABCAdapterForm):

    def __init__(self, prefix='', project_id=None):
        super(SimulatorAdapterForm, self).__init__(prefix, project_id)
        self.coupling_choices = get_ui_name_to_coupling_dict()
        default_coupling = list(self.coupling_choices.values())[0]

        self.connectivity = DataTypeSelectField(self.get_required_datatype(), self, name=self.get_input_name(),
                                                required=True, label="Connectivity",
                                                doc=Simulator.connectivity.doc,
                                                conditions=self.get_filters())
        self.coupling = SelectField(
            Attr(Coupling, default=default_coupling, label="Coupling", doc=Simulator.coupling.doc), self,
            name='coupling', choices=self.coupling_choices)
        self.conduction_speed = FloatField(Simulator.conduction_speed, self)
        self.ordered_fields = (self.connectivity, self.conduction_speed, self.coupling)
        self.range_params = [Simulator.connectivity, Simulator.conduction_speed]

    def fill_from_trait(self, trait):
        # type: (Simulator) -> None
        if hasattr(trait, 'connectivity'):
            self.connectivity.data = trait.connectivity.hex
        self.coupling.data = trait.coupling.__class__
        self.conduction_speed.data = trait.conduction_speed

    @staticmethod
    def get_view_model():
        return SimulatorAdapterModel

    @staticmethod
    def get_input_name():
        return 'connectivity'

    @staticmethod
    def get_filters():
        return None

    @staticmethod
    def get_required_datatype():
        return ConnectivityIndex

    def get_traited_datatype(self):
        return Simulator()

    def __str__(self):
        pass


class SimulatorAdapter(ABCAsynchronous):
    """
    Interface between the Simulator and the Framework.
    """
    _ui_name = "Simulation Core"

    algorithm = None
    branch_simulation_state_gid = None

    # This is a list with the monitors that actually return multi dimensions for the state variable dimension.
    # We exclude from this for example EEG, MEG or Bold which return 
    HAVE_STATE_VARIABLES = ["GlobalAverage", "SpatialAverage", "Raw", "SubSample", "TemporalAverage"]

    def __init__(self):
        super(SimulatorAdapter, self).__init__()
        self.log.debug("%s: Initialized..." % str(self))

    def get_form_class(self):
        return SimulatorAdapterForm

    def get_adapter_fragments(self, view_model):
        # type (SimulatorAdapterModel) -> dict
        forms = {None: [SimulatorSurfaceFragment, SimulatorRMFragment, SimulatorStimulusFragment,
                        SimulatorModelFragment, SimulatorIntegratorFragment, SimulatorMonitorFragment,
                        SimulatorFinalFragment]}

        current_model_class = type(view_model.model)
        all_model_forms = get_model_to_form_dict()
        forms["model"] = [all_model_forms.get(current_model_class)]

        all_monitor_forms = get_monitor_to_form_dict()
        selected_monitor_forms = []
        for monitor in view_model.monitors:
            current_monitor_class = type(monitor)
            selected_monitor_forms.append(all_monitor_forms.get(current_monitor_class))

        forms["monitors"] = selected_monitor_forms
        # Not sure if where we should in fact include the entire tree, or it will become too tedious.
        # For now I think it is ok if we rename this section "Summary" and filter what is shown
        return forms

    def load_view_model(self, operation):
        storage_path = self.file_handler.get_project_folder(operation.project, str(operation.id))
        input_gid = json.loads(operation.parameters)['gid']
        return SimulatorSerializer().deserialize_simulator(input_gid, storage_path)

    def get_output(self):
        """
        :returns: list of classes for possible results of the Simulator.
        """
        return [TimeSeriesIndex, SimulationHistoryIndex]

    def _prepare_simulator_from_view_model(self, view_model):
        simulator = Simulator()
        simulator.gid = view_model.gid

        conn = self.load_traited_by_gid(view_model.connectivity)
        simulator.connectivity = conn

        simulator.conduction_speed = view_model.conduction_speed
        simulator.coupling = view_model.coupling

        rm_surface = None

        if view_model.surface:
            simulator.surface = Cortex()
            rm_index = self.load_entity_by_gid(view_model.surface.region_mapping_data.hex)
            rm = h5.load_from_index(rm_index)

            rm_surface_index = self.load_entity_by_gid(rm_index.fk_surface_gid)
            rm_surface = h5.load_from_index(rm_surface_index, CorticalSurface)
            rm.surface = rm_surface
            rm.connectivity = conn

            simulator.surface.region_mapping_data = rm
            if simulator.surface.local_connectivity:
                lc = self.load_traited_by_gid(view_model.surface.local_connectivity)
                assert lc.surface.gid == rm_index.fk_surface_gid
                lc.surface = rm_surface
                simulator.surface.local_connectivity = lc

        if view_model.stimulus:
            stimulus_index = self.load_entity_by_gid(view_model.stimulus.hex)
            stimulus = h5.load_from_index(stimulus_index)
            simulator.stimulus = stimulus

            if isinstance(stimulus, StimuliSurface):
                simulator.stimulus.surface = rm_surface
            else:
                simulator.stimulus.connectivity = simulator.connectivity

        simulator.model = view_model.model
        simulator.integrator = view_model.integrator
        simulator.initial_conditions = view_model.initial_conditions
        simulator.monitors = view_model.monitors
        simulator.simulation_length = view_model.simulation_length

        # TODO: why not load history here?
        # if view_model.history:
        #     history_index = dao.get_datatype_by_gid(view_model.history.hex)
        #     history = h5.load_from_index(history_index)
        #     assert isinstance(history, SimulationHistory)
        #     history.fill_into(self.algorithm)
        return simulator

    def configure(self, view_model):
        # type: (SimulatorAdapterModel) -> None
        """
        Make preparations for the adapter launch.
        """
        self.log.debug("%s: Configuring simulator adapter..." % str(self))
        self.algorithm = self._prepare_simulator_from_view_model(view_model)
        self.branch_simulation_state_gid = view_model.history_gid

        # for monitor in self.algorithm.monitors:
        #     if issubclass(monitor, Projection):
        #         # TODO: add a service that loads a RM with Surface and Connectivity
        #         pass

        try:
            self.algorithm.preconfigure()
        except ValueError as err:
            raise LaunchException("Failed to configure simulator due to invalid Input Values. It could be because "
                                  "of an incompatibility between different version of TVB code.", err)

    def get_required_memory_size(self, view_model):
        # type: (SimulatorAdapterModel) -> int
        """
        Return the required memory to run this algorithm.
        """
        return self.algorithm.memory_requirement()

    def get_required_disk_size(self, view_model):
        # type: (SimulatorAdapterModel) -> int
        """
        Return the required disk size this algorithm estimates it will take. (in kB)
        """
        return self.algorithm.storage_requirement() / 2 ** 10

    def get_execution_time_approximation(self, view_model):
        # type: (SimulatorAdapterModel) -> int
        """
        Method should approximate based on input arguments, the time it will take for the operation 
        to finish (in seconds).
        """
        # This is just a brute approx so cluster nodes won't kill operation before
        # it's finished. This should be done with a higher grade of sensitivity
        # Magic number connecting simulation length to simulation computation time
        # This number should as big as possible, as long as it is still realistic, to
        magic_number = 6.57e-06  # seconds
        approx_number_of_nodes = 500
        approx_nvar = 15
        approx_modes = 15

        approx_integrator_dt = self.algorithm.integrator.dt
        if approx_integrator_dt == 0.0:
            approx_integrator_dt = 1.0

        if self.algorithm.is_surface_simulation:
            approx_number_of_nodes *= approx_number_of_nodes

        estimation = (magic_number * approx_number_of_nodes * approx_nvar *
                      approx_modes * self.algorithm.simulation_length / approx_integrator_dt)
        return max(int(estimation), 1)

    def _try_find_mapping(self, mapping_class, connectivity_gid):
        """
        Try to find a DataType instance of class "mapping_class", linked to the given Connectivity.
        Entities in the current project will have priority.

        :param mapping_class: DT class, with field "_connectivity" on it
        :param connectivity_gid: GUID
        :return: None or instance of "mapping_class"
        """

        dts_list = dao.get_generic_entity(mapping_class, connectivity_gid, 'fk_connectivity_gid')
        if len(dts_list) < 1:
            return None

        for dt in dts_list:
            dt_operation = dao.get_operation_by_id(dt.fk_from_operation)
            if dt_operation.fk_launched_in == self.current_project_id:
                return dt
        return dts_list[0]

    def _try_load_region_mapping(self):
        region_map = None
        region_volume_map = None

        region_map_index = self._try_find_mapping(RegionMappingIndex, self.algorithm.connectivity.gid.hex)
        region_volume_map_index = self._try_find_mapping(RegionVolumeMappingIndex, self.algorithm.connectivity.gid.hex)

        if region_map_index:
            region_map = h5.load_from_index(region_map_index)

        if region_volume_map_index:
            region_volume_map = h5.load_from_index(region_volume_map_index)

        return region_map, region_volume_map

    def launch(self, view_model):
        # type: (SimulatorAdapterModel) -> [TimeSeriesIndex, SimulationHistoryIndex]
        """
        Called from the GUI to launch a simulation.
          *: string class name of chosen model, etc...
          *_parameters: dictionary of parameters for chosen model, etc...
          connectivity: tvb.datatypes.connectivity.Connectivity object.
          surface: tvb.datatypes.surfaces.CorticalSurface: or None.
          stimulus: tvb.datatypes.patters.* object
        """
        result_h5 = dict()
        result_indexes = dict()
        start_time = self.algorithm.current_step * self.algorithm.integrator.dt

        self.algorithm.configure(full_configure=False)
        if self.branch_simulation_state_gid is not None:
            history_index = dao.get_datatype_by_gid(self.branch_simulation_state_gid.hex)
            history = h5.load_from_index(history_index)
            assert isinstance(history, SimulationHistory)
            history.fill_into(self.algorithm)

        region_map, region_volume_map = self._try_load_region_mapping()

        for monitor in self.algorithm.monitors:
            m_name = type(monitor).__name__
            ts = monitor.create_time_series(self.algorithm.connectivity, self.algorithm.surface, region_map,
                                            region_volume_map)
            self.log.debug("Monitor created the TS")
            ts.start_time = start_time

            ts_index_class = h5.REGISTRY.get_index_for_datatype(type(ts))
            ts_index = ts_index_class()
            ts_index.fill_from_has_traits(ts)
            ts_index.data_ndim = 4
            ts_index.state = 'INTERMEDIATE'

            state_variable_dimension_name = ts.labels_ordering[1]
            if m_name in self.HAVE_STATE_VARIABLES:
                selected_vois = [self.algorithm.model.variables_of_interest[idx] for idx in monitor.voi]
                ts.labels_dimensions[state_variable_dimension_name] = selected_vois
                ts_index.labels_dimensions = json.dumps(ts.labels_dimensions)

            ts_h5_class = h5.REGISTRY.get_h5file_for_datatype(type(ts))
            ts_h5_path = h5.path_for(self.storage_path, ts_h5_class, ts.gid)
            ts_h5 = ts_h5_class(ts_h5_path)
            ts_h5.store(ts, scalars_only=True, store_references=False)
            ts_h5.sample_rate.store(ts.sample_rate)
            ts_h5.nr_dimensions.store(ts_index.data_ndim)

            ts_h5.store_references(ts)

            result_indexes[m_name] = ts_index
            result_h5[m_name] = ts_h5

        # Run simulation
        self.log.debug("Starting simulation...")
        for result in self.algorithm(simulation_length=self.algorithm.simulation_length):
            for j, monitor in enumerate(self.algorithm.monitors):
                if result[j] is not None:
                    m_name = type(monitor).__name__
                    ts_h5 = result_h5[m_name]
                    ts_h5.write_time_slice([result[j][0]])
                    ts_h5.write_data_slice([result[j][1]])

        self.log.debug("Completed simulation, starting to store simulation state ")
        # Now store simulator history, at the simulation end
        results = []
        if not self._is_group_launch():
            simulation_history = SimulationHistory()
            simulation_history.populate_from(self.algorithm)
            history_index = h5.store_complete(simulation_history, self.storage_path)
            results.append(history_index)

        self.log.debug("Simulation state persisted, returning results ")
        for monitor in self.algorithm.monitors:
            m_name = type(monitor).__name__
            ts_shape = result_h5[m_name].read_data_shape()
            result_indexes[m_name].fill_shape(ts_shape)
            result_h5[m_name].close()
        self.log.debug("%s: Adapter simulation finished!!" % str(self))
        results.extend(result_indexes.values())
        return results
