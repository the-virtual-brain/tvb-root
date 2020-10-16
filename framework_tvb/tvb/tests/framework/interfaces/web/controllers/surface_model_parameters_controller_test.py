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

import cherrypy
from tvb.core.entities.file.simulator.view_model import CortexViewModel
from tvb.interfaces.web.controllers.simulator_controller import SimulatorController
from tvb.tests.framework.interfaces.web.controllers.base_controller_test import BaseTransactionalControllerTest
from tvb.interfaces.web.controllers.spatial.surface_model_parameters_controller import SurfaceModelParametersController
import tvb.interfaces.web.controllers.common as common


class TestSurfaceModelParametersController(BaseTransactionalControllerTest):
    """ Unit tests for SurfaceModelParametersController """

    def transactional_teardown_method(self):
        self.cleanup()

    def test_edit_model_parameters(self, region_mapping_index_factory):
        self.init()
        surface_m_p_c = SurfaceModelParametersController()
        SimulatorController().index()
        simulator = cherrypy.session[common.KEY_SIMULATOR_CONFIG]
        region_mapping_index = region_mapping_index_factory()
        simulator.connectivity = region_mapping_index.fk_connectivity_gid
        simulator.surface = CortexViewModel()
        simulator.surface.surface_gid = region_mapping_index.fk_surface_gid
        simulator.surface.region_mapping_data = region_mapping_index.gid

        result_dict = surface_m_p_c.edit_model_parameters()
        expected_keys = ['urlNormals', 'urlNormalsPick', 'urlTriangles', 'urlTrianglesPick',
                         'urlVertices', 'urlVerticesPick', 'mainContent', 'parametersEquationPlotForm',
                         'baseUrl', 'equationsPrefixes', 'brainCenter', 'applied_equations']
        # map(lambda x: self.assertTrue(x in result_dict), expected_keys)
        assert all(x in result_dict for x in expected_keys)
        assert result_dict['baseUrl'] == '/spatial/modelparameters/surface'
        assert result_dict['mainContent'] == 'spatial/model_param_surface_main'
