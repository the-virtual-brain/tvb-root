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
Adapter that uses the traits module to generate interfaces to the Simulator.
Few supplementary steps are done here:

   * from submitted Monitor/Model... names, build transient entities
   * after UI parameters submit, compose transient Cortex entity to be passed to the Simulator.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""
import json
import os
import uuid

import numpy
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.simulator import Simulator
from tvb.adapters.simulator.coupling_forms import get_ui_name_to_coupling_dict
from tvb.core.entities.file.datatypes.connectivity_h5 import ConnectivityH5
from tvb.core.entities.model.datatypes.connectivity import ConnectivityIndex
from tvb.core.entities.model.datatypes.time_series import TimeSeriesIndex
from tvb.core.entities.storage import dao
from tvb.core.adapters.abcadapter import ABCAsynchronous, ABCAdapterForm
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.neotraits._forms import DataTypeSelectField, SimpleSelectField, FloatField, jinja_env
from tvb.core.services.simulator_service import SimulatorService
from tvb.interfaces.neocom._h5loader import DirLoader
from tvb.interfaces.neocom.config import registry


class SimulatorAdapterForm(ABCAdapterForm):

    def __init__(self, prefix='', project_id=None):
        super(SimulatorAdapterForm, self).__init__(prefix, project_id)
        self.connectivity = DataTypeSelectField(self.get_required_datatype(), self, name=self.get_input_name(),
                                                required=True, label="Connectivity",
                                                doc=Simulator.connectivity.doc,
                                                conditions=self.get_filters())
        self.coupling_choices = get_ui_name_to_coupling_dict()
        self.coupling = SimpleSelectField(choices=self.coupling_choices, form=self, name='coupling', required=True,
                                          label="Coupling", doc=Simulator.coupling.doc)
        self.coupling.template = 'select_field.jinja2'
        self.conduction_speed = FloatField(Simulator.conduction_speed, self)
        self.ordered_fields = (self.connectivity, self.conduction_speed, self.coupling)

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
        #TODO: get rid of this
        return jinja_env.get_template('wizzard_form.jinja2').render(form=self, action="/burst/set_connectivity",
                                                            is_first_fragment=True, is_last_fragment=False)



class SimulatorAdapter(ABCAsynchronous):
    """
    Interface between the Simulator and the Framework.
    """
    _ui_name = "Simulation Core"

    algorithm = None

    # This is a list with the monitors that actually return multi dimensions for the state variable dimension.
    # We exclude from this for example EEG, MEG or Bold which return 
    HAVE_STATE_VARIABLES = ["GlobalAverage", "SpatialAverage", "Raw", "SubSample", "TemporalAverage"]

    form = None

    def get_form(self):
        if not self.form:
            return SimulatorAdapterForm
        return self.form

    def set_form(self, form):
        self.form = form

    def __init__(self):
        super(SimulatorAdapter, self).__init__()
        self.log.debug("%s: Initialized..." % str(self))

    def get_input_tree2(self):
        return None
        # sim = Simulator()
        # sim.trait.bound = self.INTERFACE_ATTRIBUTES_ONLY
        # result = sim.interface_experimental
        # return result

    def get_input_tree(self):
        """
        Return a list of lists describing the interface to the simulator. This
        is used by the GUI to generate the menus and fields necessary for
        defining a simulation.
        """
        # sim = Simulator()
        # sim.trait.bound = self.INTERFACE_ATTRIBUTES_ONLY
        # result = sim.interface[self.INTERFACE_ATTRIBUTES]
        # #We should add as hidden the Simulator State attribute.
        # result.append({self.KEY_NAME: 'simulation_state',
        #                self.KEY_TYPE: 'tvb.datatypes.simulation_state.SimulationState',
        #                self.KEY_LABEL: "Continuation of", self.KEY_REQUIRED: False, self.KEY_UI_HIDE: True})
        return None

    def get_output(self):
        """
        :returns: list of classes for possible results of the Simulator.
        """
        return [TimeSeriesIndex]

    def configure(self, simulator_gid):
        """
        Make preparations for the adapter launch.
        """
        self.log.debug("%s: Instantiating requested simulator..." % str(self))

        simulator_service = SimulatorService()
        self.algorithm, connectivity_gid = simulator_service.deserialize_simulator(simulator_gid, self.storage_path)

        connectivity_index = dao.get_datatype_by_gid(connectivity_gid.hex)
        dir_loader = DirLoader(os.path.join(os.path.dirname(self.storage_path),
                                            str(connectivity_index.fk_from_operation)))
        connectivity_path = dir_loader.path_for(ConnectivityH5, connectivity_gid)
        connectivity = Connectivity()
        with ConnectivityH5(connectivity_path) as connectivity_h5:
            connectivity_h5.load_into(connectivity)

        self.algorithm.connectivity = connectivity
        self.simulation_length = self.algorithm.simulation_length
        print('Storage path is: %s' % self.storage_path)
        self.log.debug("%s: Initializing storage..." % str(self))
        try:
            self.algorithm.preconfigure()
        except ValueError as err:
            raise LaunchException("Failed to configure simulator due to invalid Input Values. It could be because "
                                  "of an incompatibility between different version of TVB code.", err)

    def get_required_memory_size(self, **kwargs):
        """
        Return the required memory to run this algorithm.
        """
        return self.algorithm.memory_requirement()

    def get_required_disk_size(self, **kwargs):
        """
        Return the required disk size this algorithm estimates it will take. (in kB)
        """
        return self.algorithm.storage_requirement(self.simulation_length) / 2 ** 10

    def get_execution_time_approximation(self, **kwargs):
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

        simulation_length = int(float(kwargs['simulation_length']))
        approx_integrator_dt = float(kwargs['integrator_parameters']['dt'])

        if approx_integrator_dt == 0.0:
            approx_integrator_dt = 1.0

        if 'surface' in kwargs and kwargs['surface'] is not None and kwargs['surface'] != '':
            approx_number_of_nodes *= approx_number_of_nodes

        estimation = magic_number * approx_number_of_nodes * approx_nvar * approx_modes * simulation_length \
                     / approx_integrator_dt

        return max(int(estimation), 1)

    def _try_find_mapping(self, mapping_class, connectivity_gid):
        """
        Try to find a DataType instance of class "mapping_class", linked to the given Connectivity.
        Entities in the current project will have priority.

        :param mapping_class: DT class, with field "_connectivity" on it
        :param connectivity_gid: GUID
        :return: None or instance of "mapping_class"
        """

        dts_list = dao.get_generic_entity(mapping_class, connectivity_gid, '_connectivity')
        if len(dts_list) < 1:
            return None

        for dt in dts_list:
            dt_operation = dao.get_operation_by_id(dt.fk_from_operation)
            if dt_operation.fk_launched_in == self.current_project_id:
                return dt
        return dts_list[0]

    def launch(self, simulator_gid):
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
        # TODO: handle SimulationState
        # if simulation_state is not None:
        #     simulation_state.fill_into(self.algorithm)

        # region_map = self._try_find_mapping(region_mapping.RegionMapping, connectivity.gid)
        # region_volume_map = self._try_find_mapping(region_mapping.RegionVolumeMapping, connectivity.gid)

        dir_loader = DirLoader(self.storage_path)

        for monitor in self.algorithm.monitors:
            m_name = monitor.__class__.__name__
            ts = monitor.create_time_series(self.algorithm.connectivity)
            self.log.debug("Monitor created the TS")
            ts.start_time = start_time

            ts_index = registry.get_index_for_datatype(type(ts))()
            ts_index.time_series_type = type(ts).__name__
            ts_index.sample_period_unit = ts.sample_period_unit
            ts_index.sample_period = ts.sample_period
            ts_index.labels_ordering = json.dumps(ts.labels_ordering)
            ts_index.data_ndim = 4
            ts_index.connectivity_id = 1
            ts_index.state = 'INTERMEDIATE'

            # state_variable_dimension_name = ts.labels_ordering[1]
            # if ts_index.user_tag_1:
            #     ts_index.labels_dimensions[state_variable_dimension_name] = ts.user_tag_1.split(';')
            # elif m_name in self.HAVE_STATE_VARIABLES:
            #     selected_vois = [self.algorithm.model.variables_of_interest[idx] for idx in monitor.voi]
            #     ts.labels_dimensions[state_variable_dimension_name] = selected_vois

            result_indexes[m_name] = ts_index

            ts_h5_class = registry.get_h5file_for_datatype(type(ts))
            ts_h5_path = dir_loader.path_for(ts_h5_class, ts_index.gid)
            ts_h5 = ts_h5_class(ts_h5_path)
            ts_h5.store(ts, scalars_only=True, store_references=False)
            ts_h5.gid.store(uuid.UUID(ts_index.gid))
            result_h5[m_name] = ts_h5

        #### Create Simulator State entity and persist it in DB. H5 file will be empty now.
        # if not self._is_group_launch():
        #     simulation_state = SimulationState(storage_path=self.storage_path)
        #     self._capture_operation_results([simulation_state])

        ### Run simulation
        self.log.debug("Starting simulation...")
        for result in self.algorithm(simulation_length=self.simulation_length):
            for j, monitor in enumerate(self.algorithm.monitors):
                if result[j] is not None:
                    m_name = monitor.__class__.__name__
                    ts_h5 = result_h5[m_name]
                    ts_h5.write_time_slice([result[j][0]])
                    ts_h5.write_data_slice([result[j][1]])

        self.log.debug("Completed simulation, starting to store simulation state ")
        ### Populate H5 file for simulator state. This step could also be done while running sim, in background.
        # if not self._is_group_launch():
        #     simulation_state.populate_from(self.algorithm)
        #     self._capture_operation_results([simulation_state])

        self.log.debug("Simulation state persisted, returning results ")
        for result in result_h5.values():
            result.close()
        # self.log.info("%s: Adapter simulation finished!!" % str(self))
        return result_indexes.values()

    def _validate_model_parameters(self, model_instance, connectivity, surface):
        """
        Checks if the size of the model parameters is set correctly.
        """
        ui_configurable_params = model_instance.ui_configurable_parameters
        for param in ui_configurable_params:
            param_value = eval('model_instance.' + param)
            if isinstance(param_value, numpy.ndarray):
                if len(param_value) == 1 or connectivity is None:
                    continue
                if surface is not None:
                    if (len(param_value) != surface.number_of_vertices
                            and len(param_value) != connectivity.number_of_regions):
                        msg = str(surface.number_of_vertices) + ' or ' + str(connectivity.number_of_regions)
                        msg = self._get_exception_message(param, msg, len(param_value))
                        self.log.error(msg)
                        raise LaunchException(msg)
                elif len(param_value) != connectivity.number_of_regions:
                    msg = self._get_exception_message(param, connectivity.number_of_regions, len(param_value))
                    self.log.error(msg)
                    raise LaunchException(msg)


    @staticmethod
    def _get_exception_message(param_name, expected_size, actual_size):
        """
        Creates the message that will be displayed to the user when the size of a model parameter is incorrect.
        """
        msg = "The length of the parameter '" + param_name + "' is not correct."
        msg += " It is expected to be an array of length " + str(expected_size) + "."
        msg += " It is an array of length " + str(actual_size) + "."
        return msg

    @staticmethod
    def _is_surface_simulation(surface, surface_parameters):
        """
        Is this a surface simulation?
        """
        return surface is not None and surface_parameters is not None

