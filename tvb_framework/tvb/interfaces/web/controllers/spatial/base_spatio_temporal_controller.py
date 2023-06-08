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
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""

import json
from tvb.adapters.forms.model_forms import get_model_to_form_dict
from tvb.adapters.visualizers.surface_view import SurfaceURLGenerator
from tvb.basic.logger.builder import get_logger
from tvb.core.neocom import h5
from tvb.core.services.operation_service import OperationService
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.autologging import traced
from tvb.interfaces.web.controllers.base_controller import BaseController
from tvb.interfaces.web.controllers.common import MissingDataException
from tvb.interfaces.web.controllers.decorators import settings, expose_page, using_template
from tvb.interfaces.web.structure import WebStructure

PARAM_SURFACE = "surface"


@traced
class SpatioTemporalController(BaseController):
    """
    Base class which contains methods related to spatio-temporal actions.
    """
    MSG_MISSING_SURFACE = "There is no surface in the current project. Please upload a CORTICAL one to continue!"

    def __init__(self):
        BaseController.__init__(self)
        self.operation_service = OperationService()
        self.logger = get_logger(__name__)
        editable_entities = [dict(link=self.build_path('/spatial/stimulus/region/step_1_submit/1/1'), title='Region Stimulus',
                                  subsection='regionstim', description='Create a new Stimulus on Region level'),
                             dict(link=self.build_path('/spatial/stimulus/surface/step_1_submit/1/1'), title='Surface Stimulus',
                                  subsection='surfacestim', description='Create a new Stimulus on Surface level')]
        self.submenu_list = editable_entities

    @expose_page
    @settings
    def index(self, **data):
        """
        Displays the main page for the spatio temporal section.
        """
        template_specification = {'title': "Spatio temporal", 'data': data, 'mainContent': 'header_menu'}
        return self.fill_default_attributes(template_specification)

    @staticmethod
    def display_surface(surface_gid, region_mapping_gid=None):
        """
        Generates the HTML for displaying the surface with the given ID.
        """
        surface_h5 = h5.h5_file_for_gid(surface_gid)
        if surface_h5 is None:
            raise MissingDataException(SpatioTemporalController.MSG_MISSING_SURFACE + "!!")
        common.add2session(PARAM_SURFACE, surface_gid)

        url_vertices_pick, url_normals_pick, url_triangles_pick = SurfaceURLGenerator.get_urls_for_pick_rendering(
            surface_h5)
        url_vertices, url_normals, _, url_triangles, _ = SurfaceURLGenerator.get_urls_for_rendering(surface_h5,
                                                                                                    region_mapping_gid)
        surface_h5.close()

        return {
            'urlVerticesPick': json.dumps(url_vertices_pick),
            'urlTrianglesPick': json.dumps(url_triangles_pick),
            'urlNormalsPick': json.dumps(url_normals_pick),
            'urlVertices': json.dumps(url_vertices),
            'urlTriangles': json.dumps(url_triangles),
            'urlNormals': json.dumps(url_normals),
            'brainCenter': json.dumps(surface_h5.center())
        }

    @staticmethod
    def get_series_json(data, label):
        """ For each data point entry, build the FLOT specific JSON. """
        return json.dumps([{'data': data, 'label': label}])

    @staticmethod
    def get_ui_message(list_of_equation_names):
        """
        The message returned by this method should be displayed if
        the equation with the given name couldn't be evaluated in all points.
        """
        if list_of_equation_names:
            return ("Could not evaluate the " + ", ".join(list_of_equation_names) + " equation(s) "
                                                                                    "in all the points. Some of the values were changed.")
        else:
            return ""

    def fill_default_attributes(self, template_dictionary, subsection='stimulus'):
        """
        Overwrite base controller to add required parameters for adapter templates.
        """
        template_dictionary[common.KEY_SECTION] = WebStructure.SECTION_STIMULUS
        template_dictionary[common.KEY_SUB_SECTION] = subsection
        template_dictionary[common.KEY_SUBMENU_LIST] = self.submenu_list
        template_dictionary[common.KEY_INCLUDE_RESOURCES] = 'spatial/included_resources'
        BaseController.fill_default_attributes(self, template_dictionary)
        return template_dictionary

    def get_x_axis_range(self, min_x_str, max_x_str):
        """
        Fill range for the X-axis displayed in 2D graph.
        """
        try:
            min_x = int(min_x_str)
        except ValueError:
            return 0, 100, "The min value for the x-axis should be an integer value."

        try:
            max_x = int(max_x_str)
        except ValueError:
            return 0, 100, "The max value for the x-axis should be an integer value."

        if min_x >= max_x:
            return 0, 100, "The min value for the x-axis should be smaller then the max value of the x-axis."

        return min_x, max_x, ''

    @using_template('spatial/spatial_fragment')
    def render_spatial_form(self, adapter_form):
        show_online_help = common.get_logged_user().is_online_help_active()
        rendering_dict = adapter_form.get_rendering_dict()
        rendering_dict.update({'showOnlineHelp': show_online_help})
        rendering_dict.update({'isCallout': False})
        return rendering_dict

    def _prepare_model_params_list(self, model):
        model_form = get_model_to_form_dict().get(type(model))
        model_params = model_form().get_params_configurable_in_phase_plane()
        if len(model_params) == 0:
            self.logger.warning("The list with configurable parameters for the current model is empty!")

        return model_params
