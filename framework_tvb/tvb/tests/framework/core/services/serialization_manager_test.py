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
# CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
import json
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.core.entities.model import BurstConfiguration
from tvb.core.services.burst_config_serialization import INTEGRATOR_PARAMETERS, MODEL_PARAMETERS, SerializationManager
from tvb.simulator.integrators import HeunStochastic
from tvb.simulator.models import Hopfield, Generic2dOscillator
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.datatypes.datatypes_factory import DatatypesFactory


class TestSerializationManager(TransactionalTestCase):
    CONF_HOPFIELD_HEUN_STOCH_RANGES = r"""
    {"": {"value": "0.1"},
    "model_parameters_option_Hopfield_noise_parameters_option_Noise_random_stream": {"value": "RandomStream"},
     "model_parameters_option_Hopfield_state_variable_range_parameters_theta": {"value": "[ 0.  1.]"},
     "integrator": {"value": "HeunStochastic"},
     "model_parameters_option_Hopfield_variables_of_interest": {"value": ["x"]},
     "surface": {"value": ""},
     "simulation_length": {"value": "100.0"},
     "monitors_parameters_option_TemporalAverage_period": {"value": "0.9765625"},
     "integrator_parameters_option_HeunStochastic_noise_parameters_option_Additive_nsig": {"value": "[0.00123]"},
     "monitors": {"value": ["TemporalAverage"]},
     "model_parameters_option_Hopfield_noise_parameters_option_Noise_random_stream_parameters_option_RandomStream_init_seed": {"value": "42"},
     "conduction_speed": {"value": "3.0"},
     "model_parameters_option_Hopfield_noise_parameters_option_Noise_ntau": {"value": "0.0"},
     "currentAlgoId": {"value": 64},
     "integrator_parameters_option_HeunStochastic_noise_parameters_option_Multiplicative_random_stream_parameters_option_RandomStream_init_seed": {"value": "42"},
     "integrator_parameters_option_HeunStochastic_noise_parameters_option_Additive_random_stream": {"value": "RandomStream"},
     "connectivity": {"value": "be827732-1655-11e4-ae16-c860002c3492"},
     "model_parameters_option_Hopfield_noise": {"value": "Noise"},
     "range_1": {"value": "model_parameters_option_Hopfield_taux"},
     "model_parameters_option_Hopfield_taux": {"value": "{\"minValue\":0.7,\"maxValue\":1,\"step\":0.1}"},
     "range_2": {"value": "0"},
     "coupling_parameters_option_Linear_b": {"value": "[0.0]"},
     "coupling_parameters_option_Linear_a": {"value": "[0.00390625]"},
     "coupling": {"value": "Linear"},
     "model_parameters_option_Hopfield_state_variable_range_parameters_x": {"value": "[-1.  2.]"},
     "stimulus": {"value": ""},
     "integrator_parameters_option_HeunStochastic_dt": {"value": "0.09765625"},
     "model_parameters_option_Hopfield_dynamic": {"value": "[0]"},
     "integrator_parameters_option_HeunStochastic_noise_parameters_option_Additive_ntau": {"value": "0.0"},
     "model_parameters_option_Hopfield_tauT": {"value": "[5.0]"},
     "integrator_parameters_option_HeunStochastic_noise": {"value": "Additive"},
     "model": {"value": "Hopfield"},
     "integrator_parameters_option_HeunStochastic_noise_parameters_option_Additive_random_stream_parameters_option_RandomStream_init_seed": {"value": "42"}
     }
    """

    def transactional_setup_method(self):
        _, self.connectivity = DatatypesFactory().create_connectivity()
        self.test_user = TestFactory.create_user(username="test_user")
        self.test_project = TestFactory.create_project(self.test_user, "Test")

        burst_conf = BurstConfiguration(self.test_project.id)
        burst_conf._simulator_configuration = self.CONF_HOPFIELD_HEUN_STOCH_RANGES
        burst_conf.prepare_after_load()
        burst_conf.simulator_configuration['connectivity'] = {'value': self.connectivity.gid}

        self.s_manager = SerializationManager(burst_conf)
        self.empty_manager = SerializationManager(BurstConfiguration(None))


    def test_has_model_pse_ranges(self):
        assert self.s_manager.has_model_pse_ranges()
        assert not self.empty_manager.has_model_pse_ranges()


    def test_get_params_dict(self):
        d = self.s_manager._get_params_dict()
        mp = d[MODEL_PARAMETERS]
        ip = d[INTEGRATOR_PARAMETERS]
        # test model param deserialization
        assert [5] == mp['tauT'].tolist()
        assert [{'step': 0.1, 'maxValue': 1, 'minValue': 0.7}] == mp['taux'].tolist()
        # test integrator param deserialization
        assert 0.09765625 == ip['dt']
        assert [ 0.00123] == ip['noise_parameters']['nsig'].tolist()


    def test_make_model_and_integrator(self):
        m ,i = self.s_manager.make_model_and_integrator()
        assert isinstance(m,Hopfield)
        assert isinstance(i, HeunStochastic)


    def test_group_parameter_values_by_name(self):
        gp = SerializationManager.group_parameter_values_by_name(
            [{"a": 2.0, 'b': 1.0},
             {"a": 3.0, 'b': 7.0}])
        assert {'a': [2.0, 3.0], 'b': [1.0, 7.0]} == gp


    def test_write_model_parameters_one_dynamic(self):
        m_name = Generic2dOscillator.__name__
        m_parms = {'I': 0.0, 'a': 1.75, 'alpha': 1.0, 'b': -10.0, 'beta': 1.0, 'c': 0.0,
               'd': 0.02, 'e': 3.0, 'f': 1.0, 'g': 0.0, 'gamma': 1.0, 'tau': 1.47}

        self.s_manager.write_model_parameters(m_name, [m_parms.copy() for _ in range(self.connectivity.number_of_regions)])

        sc = self.s_manager.conf.simulator_configuration
        # Default model in these tests is Hopfield. Test if the model was changed to Generic2dOscillator
        assert Generic2dOscillator.__name__ == sc['model']['value']

        # a modified parameter
        expected = [1.75]  # we expect same value arrays to contract to 1 element
        actual = json.loads(sc['model_parameters_option_Generic2dOscillator_a']['value'])
        assert expected == actual
        # one with the same value as the default
        expected = [-10.0]
        actual = json.loads(sc['model_parameters_option_Generic2dOscillator_b']['value'])
        assert expected == actual


    def test_write_model_parameters_two_dynamics(self):
        m_name = Generic2dOscillator.__name__
        m_parms_1 = {'I': 0.0, 'a': 1.75, 'alpha': 1.0, 'b': -10.0, 'beta': 1.0, 'c': 0.0,
               'd': 0.02, 'e': 3.0, 'f': 1.0, 'g': 0.0, 'gamma': 1.0, 'tau': 1.47}
        m_parms_2 = {'I': 0.0, 'a': 1.75, 'alpha': 1.0, 'b': -5.0, 'beta': 1.0, 'c': 0.0,
               'd': 0.02, 'e': 3.0, 'f': 1.0, 'g': 0.0, 'gamma': 1.0, 'tau': 1.47}
        # all nodes except the first have dynamic 1
        model_parameter_list = [m_parms_1.copy() for _ in range(self.connectivity.number_of_regions)]
        model_parameter_list[0] = m_parms_2

        self.s_manager.write_model_parameters(m_name, model_parameter_list)

        sc = self.s_manager.conf.simulator_configuration
        # Default model in these tests is Hopfield. Test if the model was changed to Generic2dOscillator
        assert Generic2dOscillator.__name__ == sc['model']['value']

        expected = [1.75]  # array contracted to one value
        actual = json.loads(sc['model_parameters_option_Generic2dOscillator_a']['value'])
        assert expected == actual

        # b is not the same across models. We will have a full array
        expected = [-10.0 for _ in range(self.connectivity.number_of_regions)]
        expected[0] = -5.0
        actual = json.loads(sc['model_parameters_option_Generic2dOscillator_b']['value'])
        assert expected == actual


    def test_write_noise_parameters(self):
        disp = [{"x":4,"theta":2} for _ in range(self.connectivity.number_of_regions)]
        self.s_manager.write_noise_parameters(disp)

        sc = self.s_manager.conf.simulator_configuration
        assert HeunStochastic.__name__ == sc['integrator']['value']
        nodes_nr = self.connectivity.number_of_regions
        expected = [[4] * nodes_nr , [2] * nodes_nr]
        actual = json.loads(sc['integrator_parameters_option_HeunStochastic_noise_parameters_option_Additive_nsig']['value'])
        assert expected == actual




