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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import json
import pytest
from copy import copy
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.config import SIMULATOR_CLASS, SIMULATOR_MODULE
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities import model
from tvb.core.entities.storage import dao
from tvb.core.services.project_service import initialize_storage
from tvb.core.services.operation_service import OperationService
from tvb.datatypes.time_series import TimeSeriesRegion
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.datatypes.datatypes_factory import DatatypesFactory


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
    model.RANGE_PARAMETER_1: "0",
    model.RANGE_PARAMETER_2: "0",
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
        self.datatypes_factory = DatatypesFactory()
        self.test_user = self.datatypes_factory.get_user()
        self.test_project = self.datatypes_factory.get_project()
        self.connectivity = self.datatypes_factory.create_connectivity(self.CONNECTIVITY_NODES)[1]

        algorithm = dao.get_algorithm_by_module(SIMULATOR_MODULE, SIMULATOR_CLASS)
        self.simulator_adapter = ABCAdapter.build_adapter(algorithm)
        self.operation = TestFactory.create_operation(algorithm, self.test_user, self.test_project,
                                                      model.STATUS_STARTED, json.dumps(SIMULATOR_PARAMETERS))

        SIMULATOR_PARAMETERS['connectivity'] = self.connectivity.gid


    def test_happy_flow_launch(self):
        """
        Test that launching a simulation from UI works.
        """
        OperationService().initiate_prelaunch(self.operation, self.simulator_adapter, {}, **SIMULATOR_PARAMETERS)
        sim_result = dao.get_generic_entity(TimeSeriesRegion, 'TimeSeriesRegion', 'type')[0]
        assert sim_result.read_data_shape() == (32, 1, self.CONNECTIVITY_NODES, 1)


    def _estimate_hdd(self, new_parameters_dict):
        """ Private method, to return HDD estimation for a given set of input parameters"""
        filtered_params = self.simulator_adapter.prepare_ui_inputs(new_parameters_dict)
        self.simulator_adapter.configure(**filtered_params)
        return self.simulator_adapter.get_required_disk_size(**filtered_params)


    def test_estimate_hdd(self):
        """
        Test that occupied HDD estimation for simulation results considers simulation length.
        """
        factor = 5
        simulation_parameters = copy(SIMULATOR_PARAMETERS)
        ## Estimate HDD with default simulation parameters
        estimate1 = self._estimate_hdd(simulation_parameters)
        assert estimate1 > 1

        ## Change simulation length and monitor period, we expect a direct proportial increase in estimated HDD
        simulation_parameters['simulation_length'] = float(simulation_parameters['simulation_length']) * factor
        period = float(simulation_parameters['monitors_parameters_option_TemporalAverage_period'])
        simulation_parameters['monitors_parameters_option_TemporalAverage_period'] = period / factor
        estimate2 = self._estimate_hdd(simulation_parameters)
        assert estimate1 == estimate2 / factor / factor

        ## Change number of nodes in connectivity. Expect HDD estimation increase.
        large_conn_gid = self.datatypes_factory.create_connectivity(self.CONNECTIVITY_NODES * factor)[1].gid
        simulation_parameters['connectivity'] = large_conn_gid
        estimate3 = self._estimate_hdd(simulation_parameters)
        assert estimate2 == estimate3 / factor


    def test_estimate_execution_time(self):
        """
        Test that get_execution_time_approximation considers the correct params
        """
        ## Compute reference estimation
        params = self.simulator_adapter.prepare_ui_inputs(SIMULATOR_PARAMETERS)
        estimation1 = self.simulator_adapter.get_execution_time_approximation(**params)

        ## Estimation when the surface input parameter is set
        params['surface'] = "GID_surface"
        estimation2 = self.simulator_adapter.get_execution_time_approximation(**params)

        assert estimation1 == estimation2 / 500
        params['surface'] = ""

        ## Modify integration step and simulation length:
        initial_simulation_length = float(params['simulation_length'])
        initial_integration_step = float(params['integrator_parameters']['dt'])

        for factor in (2, 4, 10):
            params['simulation_length'] = initial_simulation_length * factor
            params['integrator_parameters']['dt'] = initial_integration_step / factor

            estimation3 = self.simulator_adapter.get_execution_time_approximation(**params)

            assert estimation1 == estimation3 / factor / factor

        ## Check that no division by zero happens
        params['integrator_parameters']['dt'] = 0
        estimation4 = self.simulator_adapter.get_execution_time_approximation(**params)
        assert estimation4 > 0

        ## even with length zero, still a positive estimation should be returned
        params['simulation_length'] = 0
        estimation5 = self.simulator_adapter.get_execution_time_approximation(**params)
        assert estimation5 > 0


    def test_noise_2d_bad_shape(self):
        """
        Test a simulation with noise. Pass a wrong shape and expect exception to be raised.
        """
        params = copy(SIMULATOR_PARAMETERS)
        params['integrator'] = u'HeunStochastic'
        noise_4d_config = [[1 for _ in range(self.CONNECTIVITY_NODES)] for _ in range(4)]
        params['integrator_parameters_option_HeunStochastic_dt'] = u'0.01220703125'
        params['integrator_parameters_option_HeunStochastic_noise'] = u'Additive'
        params['integrator_parameters_option_HeunStochastic_noise_parameters_option_Additive_nsig'] = str(noise_4d_config)
        params['integrator_parameters_option_HeunStochastic_noise_parameters_option_Additive_ntau'] = u'0.0'
        params['integrator_parameters_option_HeunStochastic_noise_parameters_option_Additive_random_stream'] = u'RandomStream'
        params['integrator_parameters_option_HeunStochastic_noise_parameters_option_Additive_random_stream_parameters_option_RandomStream_init_seed'] = u'42'
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
        params['integrator'] = u'HeunStochastic'
        noise_2d_config = [[1 for _ in range(self.CONNECTIVITY_NODES)] for _ in range(2)]
        params['integrator_parameters_option_HeunStochastic_dt'] = u'0.01220703125'
        params['integrator_parameters_option_HeunStochastic_noise'] = u'Additive'
        params['integrator_parameters_option_HeunStochastic_noise_parameters_option_Additive_nsig'] = str(noise_2d_config)
        params['integrator_parameters_option_HeunStochastic_noise_parameters_option_Additive_ntau'] = u'0.0'
        params['integrator_parameters_option_HeunStochastic_noise_parameters_option_Additive_random_stream'] = u'RandomStream'
        params['integrator_parameters_option_HeunStochastic_noise_parameters_option_Additive_random_stream_parameters_option_RandomStream_init_seed'] = u'42'

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

        OperationService().initiate_prelaunch(self.operation, self.simulator_adapter, {}, **params)


    def test_simulation_with_stimulus(self):
        """
        Test a simulation with noise.
        """
        params = copy(SIMULATOR_PARAMETERS)
        params["stimulus"] = self.datatypes_factory.create_stimulus(self.connectivity).gid

        filtered_params = self.simulator_adapter.prepare_ui_inputs(params)
        self.simulator_adapter.configure(**filtered_params)
        OperationService().initiate_prelaunch(self.operation, self.simulator_adapter, {}, **params)
