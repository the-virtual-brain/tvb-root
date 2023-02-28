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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import json
import pytest

from tvb.adapters.forms.model_forms import ModelsEnum
from tvb.interfaces.web.controllers.simulator.simulator_controller import SimulatorController
from tvb.tests.framework.interfaces.web.controllers.base_controller_test import BaseTransactionalControllerTest
from tvb.core.entities.model.model_burst import Dynamic
from tvb.core.entities.storage import dao
from tvb.interfaces.web.controllers.burst.region_model_parameters_controller import RegionsModelParametersController
from tvb.simulator.integrators import HeunDeterministic


class TestRegionsModelParametersController(BaseTransactionalControllerTest):
    """ Unit tests for RegionsModelParametersController """

    def transactional_setup_method(self):
        """
        Sets up the environment for testing;
        creates a `RegionsModelParametersController` and a connectivity
        """
        self.init()
        self.region_m_p_c = RegionsModelParametersController()
        simulator_controller = SimulatorController()
        simulator_controller.index()
        self.simulator = simulator_controller.context.simulator
        self._setup_dynamic()

    def transactional_teardown_method(self):
        """ Cleans the testing environment """
        self.cleanup()

    def _setup_dynamic(self):
        dynamic_g = Dynamic("test_dyn", self.test_user.id, ModelsEnum.GENERIC_2D_OSCILLATOR.value.__name__,
                            '[["tau", 1.0], ["a", 5.0], ["b", -10.0], ["c", 10.0], ["I", 0.0], ["d", 0.02], '
                            '["e", 3.0], ["f", 1.0], ["g", 0.0], ["alpha", 1.0], ["beta", 5.0], ["gamma", 1.0]]',
                            HeunDeterministic.__name__, None)

        dynamic_k = Dynamic("test_dyn_kura", self.test_user.id, ModelsEnum.KURAMOTO.value.__name__,
                            '[["omega", 1.0]]', HeunDeterministic.__name__, None)

        self.dynamic_g = dao.store_entity(dynamic_g)
        self.dynamic_k = dao.store_entity(dynamic_k)

    def test_index(self, connectivity_index_factory):
        """
        Verifies that result dictionary has the expected keys / values after call to
        `edit_model_parameters()`
        """
        self.connectivity_index = connectivity_index_factory()
        self.simulator.connectivity = self.connectivity_index.gid
        result_dict = self.region_m_p_c.index()
        assert self.connectivity_index.gid == result_dict['connectivity_entity'].gid.hex
        assert result_dict['mainContent'] == 'burst/model_param_region'
        assert result_dict['submit_parameters_url'] == '/burst/modelparameters/regions/submit_model_parameters'
        assert 'dynamics' in result_dict
        assert 'dynamics_json' in result_dict
        assert 'pointsLabels' in result_dict
        assert 'positions' in result_dict

        json.loads(result_dict['dynamics_json'])

    def test_submit_model_parameters_happy(self, connectivity_index_factory):
        """
        Verifies call to `submit_model_parameters(...)` correctly redirects to '/burst/'
        """
        self.connectivity_index = connectivity_index_factory()
        self.simulator.connectivity = self.connectivity_index.gid
        self.region_m_p_c.index()

        dynamic_ids = json.dumps([self.dynamic_g.id for _ in range(self.connectivity_index.number_of_regions)])

        self._expect_redirect('/burst/', self.region_m_p_c.submit_model_parameters, dynamic_ids)

    def test_submit_model_parameters_inconsistent_models(self, connectivity_index_factory):
        self.connectivity_index = connectivity_index_factory()
        self.simulator.connectivity = self.connectivity_index.gid
        self.region_m_p_c.index()

        dynamic_ids = [self.dynamic_g.id for _ in range(self.connectivity_index.number_of_regions)]
        dynamic_ids[-1] = self.dynamic_k.id
        dynamic_ids = json.dumps(dynamic_ids)

        with pytest.raises(Exception):
            self.region_m_p_c.submit_model_parameters(dynamic_ids)
