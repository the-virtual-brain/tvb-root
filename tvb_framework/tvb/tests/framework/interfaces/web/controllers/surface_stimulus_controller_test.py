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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

from tvb.tests.framework.interfaces.web.controllers.base_controller_test import BaseTransactionalControllerTest
from tvb.interfaces.web.controllers.common import get_from_session
from tvb.interfaces.web.controllers.spatial.surface_stimulus_controller import SurfaceStimulusController
from tvb.interfaces.web.controllers.spatial.surface_stimulus_controller import KEY_SURFACE_STIMULUS


class TestSurfaceStimulusController(BaseTransactionalControllerTest):
    """ Unit tests for SurfaceStimulusController """
    
    def transactional_setup_method(self):
        self.init()
        self.surface_s_c = SurfaceStimulusController()

    def transactional_teardown_method(self):
        """ Cleans the testing environment """
        self.cleanup()

    def test_step_1(self):
        self.surface_s_c.step_1_submit(1, 1)
        result_dict = self.surface_s_c.step_1()
        expected_keys = ['temporalPlotInputList', 'spatialPlotInputList', 'baseUrl', 'definedFocalPoints',
                         'mainContent', 'surfaceStimulusSelectForm', 'surfaceStimulusCreateForm']
        assert all(x in result_dict for x in expected_keys)
        assert result_dict['mainContent'] == 'spatial/stimulus_surface_step1_main'
        assert result_dict['next_step_url'] == '/spatial/stimulus/surface/step_1_submit'

    def test_step_2(self, surface_index_factory):
        surface_index, _ = surface_index_factory()
        self.surface_s_c.step_1_submit(1, 1)
        context = get_from_session(KEY_SURFACE_STIMULUS)
        context.surface = surface_index.gid
        result_dict = self.surface_s_c.step_2()
        expected_keys = ['urlVerticesPick', 'urlVertices', 'urlTrianglesPick', 'urlTriangles',
                         'urlNormalsPick', 'urlNormals', 'baseUrl', 'surfaceGID', 'mainContent',
                         'loadExistentEntityUrl', 'surfaceStimulusSelectForm', 'definedFocalPoints']
        assert all(x in result_dict for x in expected_keys)
        assert result_dict['next_step_url'] == '/spatial/stimulus/surface/step_2_submit'
        assert result_dict['mainContent'] == 'spatial/stimulus_surface_step2_main'
        assert result_dict['loadExistentEntityUrl'] == '/spatial/stimulus/surface/load_surface_stimulus'
