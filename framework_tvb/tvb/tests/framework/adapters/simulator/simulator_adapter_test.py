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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

from os import path
import pytest
from copy import copy
import tvb_data
from tvb.adapters.datatypes.db.time_series import TimeSeriesRegionIndex
from tvb.config.init.introspector_registry import IntrospectionRegistry
from tvb.core.entities.file.simulator.view_model import CortexViewModel, SimulatorAdapterModel
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.model.model_burst import RANGE_PARAMETER_1, RANGE_PARAMETER_2
from tvb.core.entities.storage import dao
from tvb.core.services.project_service import initialize_storage
from tvb.core.services.operation_service import OperationService
from tvb.datatypes.time_series import TimeSeriesRegion
from tvb.datatypes.surfaces import CORTICAL
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.core.base_testcase import TransactionalTestCase

# Default values for simulator's input. These values can be replace with adapter.get_flatten_interface...
SIMULATOR_PARAMETERS = {
    "model": "Generic2dOscillator",
    "model_parameters_option_Generic2dOscillator_state_variable_range_parameters_parameters_V": "[-2.0  4.0]",
    "model_parameters_option_Generic2dOscillator_state_variable_range_parameters_parameters_W": "[-6.0  6.0]",
    "model_parameters_option_Generic2dOscillator_tau": "[1.0]",
    "model_parameters_option_Generic2dOscillator_I": "[0.0]",
    "model_parameters_option_Generic2dOscillator_a": "[-2.0]",
    "model_parameters_option_Generic2dOscillator_b": "[-10.0]",
    "model_parameters_option_Generic2dOscillator_c": "[0.0]",
    "model_parameters_option_Generic2dOscillator_d": "[0.02]",
    "model_parameters_option_Generic2dOscillator_e": "[3.0]",
    "model_parameters_option_Generic2dOscillator_f": "[1.0]",
    "model_parameters_option_Generic2dOscillator_g": "[0.0]",
    "model_parameters_option_Generic2dOscillator_alpha": "[1.0]",
    "model_parameters_option_Generic2dOscillator_beta": "[1.0]",
    "model_parameters_option_Generic2dOscillator_gamma": "[1.0]",
    "model_parameters_option_Generic2dOscillator_noise": "Noise",
    "model_parameters_option_Generic2dOscillator_noise_parameters_option_Noise_ntau": "0.0",
    "model_parameters_option_Generic2dOscillator_noise_parameters_option_Noise_random_stream": "RandomStream",
    "model_parameters_option_Generic2dOscillator_noise_parameters_option_Noise_random_stream_parameters_option_RandomStream_init_seed": "88",
    "connectivity": "7eadbaeb-afdc-11e1-ab21-68a86d1bd4fa",
    "conduction_speed": 3.0,
    "surface": "",
    "stimulus": "",
    "currentAlgoId": "10",
    RANGE_PARAMETER_1: "0",
    RANGE_PARAMETER_2: "0",
    "integrator": "HeunDeterministic",
    "integrator_parameters_option_HeunDeterministic_dt": "0.01220703125",
    "monitors": "TemporalAverage",
    "monitors_parameters_option_TemporalAverage_period": "0.9765625",
    "coupling": "Linear",
    "coupling_parameters_option_Linear_a": "[0.00390625]",
    "coupling_parameters_option_Linear_b": "[0.0]",
    "simulation_length": "32"}


class TestSimulatorAdapter(TransactionalTestCase):
    """
    Basic testing that Simulator is still working from UI.
    """
    CONNECTIVITY_NODES = 74

    def transactional_setup_method(self):
        """
        Reset the database before each test.
        """
        initialize_storage()

        algorithm = dao.get_algorithm_by_module(IntrospectionRegistry.SIMULATOR_MODULE,
                                                IntrospectionRegistry.SIMULATOR_CLASS)
        self.simulator_adapter = ABCAdapter.build_adapter(algorithm)

    def test_happy_flow_launch(self, connectivity_index_factory, operation_factory):
        """
        Test that launching a simulation from UI works.
        """
        model = SimulatorAdapterModel()
        model.connectivity = connectivity_index_factory(self.CONNECTIVITY_NODES).gid
        model.simulation_length = 32

        self.operation = operation_factory()
        # TODO: should store model in H5 and keep GID as param on operation to fix this

        OperationService().initiate_prelaunch(self.operation, self.simulator_adapter)
        sim_result = dao.get_generic_entity(TimeSeriesRegionIndex, 'TimeSeriesRegion', 'time_series_type')[0]
        assert (sim_result.data_length_1d, sim_result.data_length_2d, sim_result.data_length_3d,
                sim_result.data_length_4d) == (32, 1, self.CONNECTIVITY_NODES, 1)

    def _estimate_hdd(self, model):
        """ Private method, to return HDD estimation for a given a model"""
        self.simulator_adapter.configure(model)
        return self.simulator_adapter.get_required_disk_size(model)

    def test_estimate_hdd(self, connectivity_index_factory):
        """
        Test that occupied HDD estimation for simulation results considers simulation length.
        """
        factor = 5

        model = SimulatorAdapterModel()
        model.connectivity = connectivity_index_factory(self.CONNECTIVITY_NODES).gid
        estimate1 = self._estimate_hdd(model)
        assert estimate1 > 1

        ## Change simulation length and monitor period, we expect a direct proportial increase in estimated HDD
        model.simulation_length = float(model.simulation_length) * factor
        period = float(model.monitors[0].period)
        model.monitors[0].period = period / factor
        estimate2 = self._estimate_hdd(model)
        assert estimate1 == estimate2 // factor // factor

        ## Change number of nodes in connectivity. Expect HDD estimation increase.
        model.connectivity = connectivity_index_factory(self.CONNECTIVITY_NODES * factor).gid
        estimate3 = self._estimate_hdd(model)
        assert estimate2 == estimate3 / factor

    def test_estimate_execution_time(self, connectivity_index_factory):
        """
        Test that get_execution_time_approximation considers the correct params
        """
        ## Compute reference estimation
        self.test_user = TestFactory.create_user("Simulator_Adapter_User")
        self.test_project = TestFactory.create_project(self.test_user, "Simulator_Adapter_Project")

        simulator_adapter_model = SimulatorAdapterModel()
        connectivity = connectivity_index_factory(76)
        simulator_adapter_model.connectivity = connectivity.gid

        self.simulator_adapter.configure(simulator_adapter_model)
        estimation1 = self.simulator_adapter.get_execution_time_approximation(simulator_adapter_model)

        # import surfaceData
        cortex_data = path.join(path.dirname(tvb_data.__file__), 'surfaceData', 'cortex_16384.zip')
        surface = TestFactory.import_surface_zip(self.test_user, self.test_project, cortex_data, CORTICAL)
        cortex_model = CortexViewModel()

        # import region mapping for cortex_model (surface)
        text_file = path.join(path.dirname(tvb_data.__file__), 'regionMapping', 'regionMapping_16k_76.txt')
        region_mapping = TestFactory.import_region_mapping(self.test_user, self.test_project, text_file, surface.gid, connectivity.gid)

        cortex_model.region_mapping_data = region_mapping.gid
        cortex_model.fk_surface_gid = surface.gid
        simulator_adapter_model.surface = cortex_model

        ## Estimation when the surface input parameter is set
        self.simulator_adapter.configure(simulator_adapter_model)
        estimation2 = self.simulator_adapter.get_execution_time_approximation(simulator_adapter_model)

        assert estimation1 == estimation2 // 500
        simulator_adapter_model.surface = None

        ## Modify integration step and simulation length:
        initial_simulation_length = simulator_adapter_model.simulation_length
        initial_integration_step = simulator_adapter_model.integrator.dt

        for factor in (2, 4, 10):
            simulator_adapter_model.simulation_length = initial_simulation_length * factor
            simulator_adapter_model.integrator.dt = initial_integration_step / factor
            self.simulator_adapter.configure(simulator_adapter_model)

            estimation3 = self.simulator_adapter.get_execution_time_approximation(simulator_adapter_model)

            assert estimation1 == estimation3 // factor // factor

        ## Check that no division by zero happens
        simulator_adapter_model.integrator.dt = 0
        estimation4 = self.simulator_adapter.get_execution_time_approximation(simulator_adapter_model)
        assert estimation4 > 0

        ## even with length zero, still a positive estimation should be returned
        simulator_adapter_model.simulation_length = 0
        estimation5 = self.simulator_adapter.get_execution_time_approximation(simulator_adapter_model)
        assert estimation5 > 0

    def test_noise_2d_bad_shape(self):
        """
        Test a simulation with noise. Pass a wrong shape and expect exception to be raised.
        """
        params = copy(SIMULATOR_PARAMETERS)
        params['integrator'] = 'HeunStochastic'
        noise_4d_config = [[1 for _ in range(self.CONNECTIVITY_NODES)] for _ in range(4)]
        params['integrator_parameters_option_HeunStochastic_dt'] = '0.01220703125'
        params['integrator_parameters_option_HeunStochastic_noise'] = 'Additive'
        params['integrator_parameters_option_HeunStochastic_noise_parameters_option_Additive_nsig'] = str(noise_4d_config)
        params['integrator_parameters_option_HeunStochastic_noise_parameters_option_Additive_ntau'] = '0.0'
        params['integrator_parameters_option_HeunStochastic_noise_parameters_option_Additive_random_stream'] = 'RandomStream'
        params['integrator_parameters_option_HeunStochastic_noise_parameters_option_Additive_random_stream_parameters_option_RandomStream_init_seed'] = '42'
        filtered_params = self.simulator_adapter.prepare_ui_inputs(params)
        self.simulator_adapter.configure(**filtered_params)
        if hasattr(self.simulator_adapter, 'algorithm'):
            assert (4, 74) == self.simulator_adapter.algorithm.integrator.noise.nsig.shape
        else:
            raise AssertionError("Simulator adapter was not initialized properly")
        with pytest.raises(Exception):
            OperationService().initiate_prelaunch(self.operation,self.simulator_adapter, {}, **params)

    def test_noise_2d_happy_flow(self):
        """
        Test a simulation with noise.
        """
        params = copy(SIMULATOR_PARAMETERS)
        params['integrator'] = 'HeunStochastic'
        noise_2d_config = [[1 for _ in range(self.CONNECTIVITY_NODES)] for _ in range(2)]
        params['integrator_parameters_option_HeunStochastic_dt'] = '0.01220703125'
        params['integrator_parameters_option_HeunStochastic_noise'] = 'Additive'
        params['integrator_parameters_option_HeunStochastic_noise_parameters_option_Additive_nsig'] = str(noise_2d_config)
        params['integrator_parameters_option_HeunStochastic_noise_parameters_option_Additive_ntau'] = '0.0'
        params['integrator_parameters_option_HeunStochastic_noise_parameters_option_Additive_random_stream'] = 'RandomStream'
        params['integrator_parameters_option_HeunStochastic_noise_parameters_option_Additive_random_stream_parameters_option_RandomStream_init_seed'] = '42'

        self._launch_and_check_noise(params, (2, 74))

        sim_result = dao.get_generic_entity(TimeSeriesRegion, 'TimeSeriesRegion', 'type')[0]
        assert sim_result.read_data_shape() == (32, 1, self.CONNECTIVITY_NODES, 1)

        params['integrator_parameters_option_HeunStochastic_noise_parameters_option_Additive_nsig'] = '[1]'
        self._launch_and_check_noise(params, (1,))

    def _launch_and_check_noise(self, params, expected_noise_shape):

        filtered_params = self.simulator_adapter.prepare_ui_inputs(params)
        self.simulator_adapter.configure(**filtered_params)

        if hasattr(self.simulator_adapter, 'algorithm'):
            assert expected_noise_shape == self.simulator_adapter.algorithm.integrator.noise.nsig.shape
        else:
            raise AssertionError("Simulator adapter was not initialized properly")

        OperationService().initiate_prelaunch(self.operation, self.simulator_adapter, **params)

    def test_simulation_with_stimulus(self, stimulus_factory):
        """
        Test a simulation with noise.
        """
        params = copy(SIMULATOR_PARAMETERS)
        params["stimulus"] = stimulus_factory.gid

        filtered_params = self.simulator_adapter.prepare_ui_inputs(params)
        self.simulator_adapter.configure(**filtered_params)
        OperationService().initiate_prelaunch(self.operation, self.simulator_adapter,  **params)
