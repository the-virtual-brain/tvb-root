# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""

import json
from copy import deepcopy

import cherrypy

from tvb.basic.traits import traited_interface
from tvb.basic.logger.builder import get_logger
from tvb.basic.traits.parameters_factory import get_traited_instance_for_name
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.model import RANGE_PARAMETER_1, RANGE_PARAMETER_2, PARAM_MODEL, PARAM_INTEGRATOR
from tvb.core.entities.model import PARAM_CONNECTIVITY, PARAM_SURFACE
from tvb.core.services.flow_service import FlowService
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.base_controller import BaseController
from tvb.interfaces.web.controllers.decorators import settings, expose_page
from tvb.interfaces.web.controllers.flow_controller import SelectedAdapterContext
from tvb.simulator.models import Model
from tvb.simulator.integrators import Integrator
from tvb.config import SIMULATOR_CLASS, SIMULATOR_MODULE
from tvb.datatypes import noise_framework


MODEL_PARAMETERS = 'model_parameters'
INTEGRATOR_PARAMETERS = 'integrator_parameters'


class SpatioTemporalController(BaseController):
    """
    Base class which contains methods related to spatio-temporal actions.
    """

    def __init__(self):
        BaseController.__init__(self)
        self.flow_service = FlowService()
        self.logger = get_logger(__name__)
        editable_entities = [dict(link='/spatial/stimulus/region/step_1_submit/1/1', title='Region Stimulus',
                                  subsection='regionstim', description='Create a new Stimulus on Region level'),
                             dict(link='/spatial/stimulus/surface/step_1_submit/1/1', title='Surface Stimulus',
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


    def get_data_from_burst_configuration(self):
        """
        Returns the model, integrator, connectivity and surface instances from the burst configuration.
        """
        ### Read from session current burst-configuration
        burst_configuration = common.get_from_session(common.KEY_BURST_CONFIG)
        if burst_configuration is None:
            return None, None, None, None
        first_range = burst_configuration.get_simulation_parameter_value(RANGE_PARAMETER_1)
        second_range = burst_configuration.get_simulation_parameter_value(RANGE_PARAMETER_2)
        if ((first_range is not None and str(first_range).startswith(MODEL_PARAMETERS)) or
                (second_range is not None and str(second_range).startswith(MODEL_PARAMETERS))):
            common.set_error_message("When configuring model parameters you are not allowed to specify range values.")
            raise cherrypy.HTTPRedirect("/burst/")
        group = self.flow_service.get_algorithm_by_module_and_class(SIMULATOR_MODULE, SIMULATOR_CLASS)[1]
        simulator_adapter = self.flow_service.build_adapter_instance(group)
        try:
            params_dict = simulator_adapter.convert_ui_inputs(burst_configuration.get_all_simulator_values()[0], False)
        except Exception:
            self.logger.exception("Some of the provided parameters have an invalid value.")
            common.set_error_message("Some of the provided parameters have an invalid value.")
            raise cherrypy.HTTPRedirect("/burst/")
        ### Prepare Model instance
        model = burst_configuration.get_simulation_parameter_value(PARAM_MODEL)
        model_parameters = params_dict[MODEL_PARAMETERS]
        noise_framework.build_noise(model_parameters)
        try:
            model = get_traited_instance_for_name(model, Model, model_parameters)
        except Exception:
            self.logger.exception("Could not create the model instance with the given parameters. "
                                  "A new model instance will be created with the default values.")
            model = get_traited_instance_for_name(model, Model, {})
        ### Prepare Integrator instance
        integrator = burst_configuration.get_simulation_parameter_value(PARAM_INTEGRATOR)
        integrator_parameters = params_dict[INTEGRATOR_PARAMETERS]
        noise_framework.build_noise(integrator_parameters)
        try:
            integrator = get_traited_instance_for_name(integrator, Integrator, integrator_parameters)
        except Exception:
            self.logger.exception("Could not create the integrator instance with the given parameters. "
                                  "A new integrator instance will be created with the default values.")
            integrator = get_traited_instance_for_name(integrator, Integrator, {})
        ### Prepare Connectivity
        connectivity_gid = burst_configuration.get_simulation_parameter_value(PARAM_CONNECTIVITY)
        connectivity = ABCAdapter.load_entity_by_gid(connectivity_gid)
        ### Prepare Surface
        surface_gid = burst_configuration.get_simulation_parameter_value(PARAM_SURFACE)
        surface = None
        if surface_gid is not None and len(surface_gid):
            surface = ABCAdapter.load_entity_by_gid(surface_gid)
        return model, integrator, connectivity, surface


    @staticmethod
    def display_surface(surface_gid):
        """
        Generates the HTML for displaying the surface with the given ID.
        """
        surface = ABCAdapter.load_entity_by_gid(surface_gid)
        common.add2session(PARAM_SURFACE, surface_gid)
        url_vertices_pick, url_normals_pick, url_triangles_pick = surface.get_urls_for_pick_rendering()
        url_vertices, url_normals, _, url_triangles, alphas, alphas_indices = surface.get_urls_for_rendering(True, None)

        return {
            'urlVerticesPick': json.dumps(url_vertices_pick),
            'urlTrianglesPick': json.dumps(url_triangles_pick),
            'urlNormalsPick': json.dumps(url_normals_pick),
            'urlVertices': json.dumps(url_vertices),
            'urlTriangles': json.dumps(url_triangles),
            'urlNormals': json.dumps(url_normals),
            'alphas': json.dumps(alphas),
            'alphas_indices': json.dumps(alphas_indices),
            'brainCenter': json.dumps(surface.center())
        }


    @staticmethod
    def prepare_entity_interface(input_list):
        """
        Prepares the input tree obtained from a creator.
        """
        return {'inputList': input_list,
                common.KEY_PARAMETERS_CONFIG: False}


    def get_creator_and_interface(self, creator_module, creator_class, datatype_instance, lock_midpoint_for_eq=None):
        """
        Returns a Tuple: a creator instance and a dictionary for the creator interface.
        The interface is prepared for rendering, it is populated with existent data, in case of a
        parameter of type DataType. The name of the attributes are also prefixed to identify groups.
        """
        algo_group = self.flow_service.get_algorithm_by_module_and_class(creator_module, creator_class)[1]
        group, _ = self.flow_service.prepare_adapter(common.get_current_project().id, algo_group)

        #I didn't use the interface(from the above line) returned by the method 'prepare_adapter' from flow service
        # because the selects that display dataTypes will also have the 'All' entry.
        datatype_instance.trait.bound = traited_interface.INTERFACE_ATTRIBUTES_ONLY
        input_list = datatype_instance.interface[traited_interface.INTERFACE_ATTRIBUTES]
        if lock_midpoint_for_eq is not None:
            for idx in lock_midpoint_for_eq:
                input_list[idx] = self._lock_midpoints(input_list[idx])
        category = self.flow_service.get_visualisers_category()
        input_list = self.flow_service.prepare_parameters(input_list, common.get_current_project().id, category.id)
        input_list = ABCAdapter.prepare_param_names(input_list)

        return self.flow_service.build_adapter_instance(group), input_list


    @staticmethod
    def get_series_json(data, label):
        """ For each data point entry, build the FLOT specific JSON. """
        return '{"data": %s, "label": "%s"}' % (json.dumps(data), label)


    @staticmethod
    def build_final_json(list_of_series):
        """ Given a list with all the data points, build the final FLOT json. """
        return '[' + ','.join(list_of_series) + ']'


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


    def get_select_existent_entities(self, label, entity_type, entity_gid=None):
        """
        Returns the dictionary needed for drawing the select which display all
        the created entities of the specified type.
        """
        project_id = common.get_current_project().id
        category = self.flow_service.get_visualisers_category()

        interface = [{'name': 'existentEntitiesSelect', 'label': label, 'type': entity_type}]
        if entity_gid is not None:
            interface[0]['default'] = entity_gid
        interface = self.flow_service.prepare_parameters(interface, project_id, category.id)
        interface = ABCAdapter.prepare_param_names(interface)

        return interface


    @staticmethod
    def add_interface_to_session(left_input_tree, right_input_tree):
        """
        left_input_tree and right_input_tree are expected to be lists of dictionaries.

        Those 2 given lists will be concatenated and added to session.
        In order to work the filters, the interface should be added to session.
        """
        entire_tree = deepcopy(left_input_tree)
        entire_tree.extend(right_input_tree)
        SelectedAdapterContext().add_adapter_to_session(None, entire_tree)


    def fill_default_attributes(self, template_dictionary, subsection='stimulus'):
        """
        Overwrite base controller to add required parameters for adapter templates.
        """
        template_dictionary[common.KEY_SECTION] = 'stimulus'
        template_dictionary[common.KEY_SUB_SECTION] = subsection
        template_dictionary[common.KEY_SUBMENU_LIST] = self.submenu_list
        template_dictionary[common.KEY_INCLUDE_RESOURCES] = 'spatial/included_resources'
        BaseController.fill_default_attributes(self, template_dictionary)
        return template_dictionary


    def get_x_axis_range(self, min_x_str, max_x_str):
        """
        Fill range for the X-axis displayed in 2D graph.
        """
        min_x = 0
        max_x = 100
        error_msg = ''
        if self.is_int(min_x_str):
            min_x = int(min_x_str)

            if self.is_int(max_x_str):
                max_x = int(max_x_str)
            else:
                min_x = 0
                error_msg = "The max value for the x-axis should be an integer value."

            if min_x >= max_x:
                error_msg = "The min value for the x-axis should be smaller then the max value of the x-axis."
                min_x = 0
                max_x = 100
        else:
            error_msg = "The min value for the x-axis should be an integer value."

        return min_x, max_x, error_msg


    @staticmethod
    def _lock_midpoints(equations_dict):
        """
        Set mid-points for gaussian / double gausians as locked to 0.0 in case of spatial equations.
        """
        for equation in equations_dict[ABCAdapter.KEY_OPTIONS]:
            if equation[ABCAdapter.KEY_NAME] == 'Gaussian':
                for entry in equation[ABCAdapter.KEY_ATTRIBUTES][1][ABCAdapter.KEY_ATTRIBUTES]:
                    if entry[ABCAdapter.KEY_NAME] == 'midpoint':
                        entry['locked'] = True
            if equation[ABCAdapter.KEY_NAME] == 'DoubleGaussian':
                for entry in equation[ABCAdapter.KEY_ATTRIBUTES][1][ABCAdapter.KEY_ATTRIBUTES]:
                    if entry[ABCAdapter.KEY_NAME] == 'midpoint1':
                        entry['locked'] = True
        return equations_dict


    @staticmethod
    def is_int(str_value):
        """
        Checks if the given string may be converted to an int value.
        """
        try:
            int(str_value)
            return True
        except ValueError:
            return False
        
        
    def get_data_for_param_sliders(self, connectivity_node_index, context_model_parameters):
        """
        Method used only for handling the exception.
        """
        try:
            return context_model_parameters.get_data_for_param_sliders(connectivity_node_index)
        except ValueError:
            self.logger.exception("All the model parameters that are configurable should be valid arrays or numbers.")
            common.set_error_message("All the model parameters that are configurable should be valid arrays or numbers")
            raise cherrypy.HTTPRedirect("/burst/")
        
        
        
