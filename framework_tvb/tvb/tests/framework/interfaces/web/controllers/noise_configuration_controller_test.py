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
.. moduleauthor:: Andrei Mihai <mihai.andrei@codemart.ro>
"""

import json
import cherrypy
import pytest
from tvb.interfaces.web.controllers.simulator_controller import SimulatorController
from tvb.tests.framework.interfaces.web.controllers.base_controller_test import BaseTransactionalControllerTest
from tvb.tests.framework.adapters.simulator.simulator_adapter_test import SIMULATOR_PARAMETERS
from tvb.core.entities.model.model_burst import PARAM_INTEGRATOR, PARAM_MODEL
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.burst.noise_configuration_controller import NoiseConfigurationController
from tvb.interfaces.web.controllers.spatial.base_spatio_temporal_controller import INTEGRATOR_PARAMETERS
from tvb.simulator.integrators import EulerStochastic, HeunStochastic


class TestNoiseConfigurationController(BaseTransactionalControllerTest):

    @pytest.fixture()
    def transactional_setup_fixture(self, connectivity_factory):
        self.init()
        self.noise_controller = NoiseConfigurationController()
        SimulatorController().index()
        self.simulator = cherrypy.session[common.KEY_SIMULATOR_CONFIG]
        self.connectivity = connectivity_factory()
        self.simulator.connectivity = self.connectivity.gid
        self.simulator.integrator = HeunStochastic()

    def transactional_teardown_method(self):
        """ Cleans the testing environment """
        self.clean_database()
        self.cleanup()

    def test_submit_noise_configuration_happy(self, transactional_setup_fixture):
        """
        Submit noise configuration writes the noise array on the required key in the burst configuration
        """
        # a noise configuration in the format expected by submit. Assumes Generic2dOscillator model.
        nodes_range = list(range(self.connectivity.number_of_regions))
        noise_in = [{'V': 1.0, 'W': 2.0} for _ in nodes_range]
        noise_in = json.dumps(noise_in)

        self._expect_redirect('/burst/', self.noise_controller.submit, noise_in)

        expected_noise_arr = [[1.0 for _ in nodes_range], [2.0 for _ in nodes_range]]
        actual_noise_arr = self.simulator.integrator.noise.nsig
        assert (expected_noise_arr == actual_noise_arr).all()