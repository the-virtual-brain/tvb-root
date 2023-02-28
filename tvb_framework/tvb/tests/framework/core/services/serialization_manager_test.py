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
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

from os import path
import tvb_data

from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.forms.model_forms import ModelsEnum
from tvb.core.services.burst_config_serialization import SerializationManager
from tvb.simulator.simulator import Simulator
from tvb.simulator.integrators import HeunStochastic
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.tests.framework.core.factory import TestFactory


class TestSerializationManager(TransactionalTestCase):

    def transactional_setup_method(self):
        self.test_user = TestFactory.create_user(username="test_user")
        self.test_project = TestFactory.create_project(self.test_user, "Test")
        zip_path = path.join(path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_66.zip')
        TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path, "John")
        self.connectivity = TestFactory.get_entity(self.test_project, ConnectivityIndex)

        sim_conf = Simulator(model=ModelsEnum.HOPFIELD.instance, integrator=HeunStochastic())

        self.s_manager = SerializationManager(sim_conf)
        self.empty_manager = SerializationManager(None)

    def test_group_parameter_values_by_name(self):
        gp = SerializationManager.group_parameter_values_by_name(
            [{"a": 2.0, 'b': 1.0},
             {"a": 3.0, 'b': 7.0}])
        assert {'a': [2.0, 3.0], 'b': [1.0, 7.0]} == gp

    def test_write_model_parameters_one_dynamic(self, connectivity_factory):
        connectivity = connectivity_factory()
        m_name = ModelsEnum.GENERIC_2D_OSCILLATOR.value.__name__
        m_parms = {'I': 0.0, 'a': 1.75, 'alpha': 1.0, 'b': -10.0, 'beta': 1.0, 'c': 0.0,
                   'd': 0.02, 'e': 3.0, 'f': 1.0, 'g': 0.0, 'gamma': 1.0, 'tau': 1.47}

        self.s_manager.write_model_parameters(m_name, [m_parms.copy() for _ in range(connectivity.number_of_regions)])

        sc = self.s_manager.conf
        # Default model in these tests is Hopfield. Test if the model was changed to Generic2dOscillator
        assert isinstance(sc.model, ModelsEnum.GENERIC_2D_OSCILLATOR.value)

        # a modified parameter
        expected = [1.75]  # we expect same value arrays to contract to 1 element
        actual = sc.model.a
        assert expected == actual
        # one with the same value as the default
        expected = [-10.0]
        actual = sc.model.b
        assert expected == actual

    def test_write_model_parameters_two_dynamics(self, connectivity_factory):
        connectivity = connectivity_factory()
        m_name = ModelsEnum.GENERIC_2D_OSCILLATOR.value.__name__
        m_parms_1 = {'I': 0.0, 'a': 1.75, 'alpha': 1.0, 'b': -10.0, 'beta': 1.0, 'c': 0.0,
                     'd': 0.02, 'e': 3.0, 'f': 1.0, 'g': 0.0, 'gamma': 1.0, 'tau': 1.47}
        m_parms_2 = {'I': 0.0, 'a': 1.75, 'alpha': 1.0, 'b': -5.0, 'beta': 1.0, 'c': 0.0,
                     'd': 0.02, 'e': 3.0, 'f': 1.0, 'g': 0.0, 'gamma': 1.0, 'tau': 1.47}
        # all nodes except the first have dynamic 1
        model_parameter_list = [m_parms_1.copy() for _ in range(connectivity.number_of_regions)]
        model_parameter_list[0] = m_parms_2

        self.s_manager.write_model_parameters(m_name, model_parameter_list)

        sc = self.s_manager.conf
        # Default model in these tests is Hopfield. Test if the model was changed to Generic2dOscillator
        assert isinstance(sc.model, ModelsEnum.GENERIC_2D_OSCILLATOR.value)

        expected = [1.75]  # array contracted to one value
        actual = sc.model.a
        assert expected == actual

        # b is not the same across models. We will have a full array
        expected = [-10.0 for _ in range(connectivity.number_of_regions)]
        expected[0] = -5.0
        actual = sc.model.b
        assert expected == actual.tolist()

    def test_write_noise_parameters(self, connectivity_factory):
        connectivity = connectivity_factory()
        disp = [{"x": 4, "theta": 2} for _ in range(connectivity.number_of_regions)]
        self.s_manager.write_noise_parameters(disp)

        assert isinstance(self.s_manager.conf.integrator, HeunStochastic)
        nodes_nr = connectivity.number_of_regions
        expected = [[4] * nodes_nr, [2] * nodes_nr]
        actual = self.s_manager.conf.integrator.noise.nsig
        assert expected == actual.tolist()
