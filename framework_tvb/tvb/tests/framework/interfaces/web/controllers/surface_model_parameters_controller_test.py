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
"""

import pytest
import cherrypy
from tvb.interfaces.web.controllers.simulator_controller import SimulatorController
from tvb.tests.framework.interfaces.web.controllers.base_controller_test import BaseTransactionalControllerTest
from tvb.interfaces.web.controllers.spatial.surface_model_parameters_controller import SurfaceModelParametersController
from tvb.tests.framework.adapters.simulator.simulator_adapter_test import SIMULATOR_PARAMETERS
import tvb.interfaces.web.controllers.common as common


class TestSurfaceModelParametersController(BaseTransactionalControllerTest):
    """ Unit tests for SurfaceModelParametersController """

    @pytest.fixture()
    def transactional_setup_fixture(self, connectivity_factory, surface_factory):
        self.init()
        self.surface_m_p_c = SurfaceModelParametersController()
        SimulatorController().index()
        stored_burst = cherrypy.session[common.KEY_BURST_CONFIG]
        _, self.connectivity = connectivity_factory
        _, self.surface = surface_factory
        new_params = {}
        for key, val in SIMULATOR_PARAMETERS.items():
            new_params[key] = {'value': val}
        new_params['connectivity'] = {'value': self.connectivity.gid}
        new_params['surface'] = {'value': self.surface.gid}
        stored_burst.simulator_configuration = new_params


    def transactional_teardown_method(self):
        """ Cleans the testing environment """
        self.clean_database()
        self.cleanup()


    def test_edit_model_parameters(self, transactional_setup_fixture):
        result_dict = self.surface_m_p_c.edit_model_parameters()
        expected_keys = ['urlNormals', 'urlNormalsPick', 'urlTriangles', 'urlTrianglesPick', 
                         'urlVertices', 'urlVerticesPick', 'mainContent', 'inputList',
                         'equationViewerUrl', 'equationsPrefixes', 'data', 'brainCenter',
                         'applied_equations']
        # map(lambda x: self.assertTrue(x in result_dict), expected_keys)
        assert all(x in result_dict for x in expected_keys)
        assert result_dict['equationViewerUrl'] == '/spatial/modelparameters/surface/get_equation_chart'
        assert result_dict['mainContent'] == 'spatial/model_param_surface_main'
