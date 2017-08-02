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
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""

import cherrypy
import json
from copy import deepcopy

import tvb.basic.traits.traited_interface as interface
from tvb.basic.traits.parameters_factory import get_traited_instance_for_name, collapse_params
from tvb.core.adapters.abcadapter import KEY_EQUATION, KEY_FOCAL_POINTS
from tvb.core.adapters.input_tree import InputTreeManager
from tvb.core.entities.model import PARAMS_MODEL_PATTERN
from tvb.core.services.burst_config_serialization import SerializationManager
from tvb.datatypes import equations
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.base_controller import BaseController
from tvb.interfaces.web.controllers.decorators import expose_page, expose_fragment, handle_error, check_user
from tvb.interfaces.web.controllers.spatial.base_spatio_temporal_controller import SpatioTemporalController
from tvb.interfaces.web.entities.context_model_parameters import SurfaceContextModelParameters, EquationDisplayer


MODEL_PARAM = 'model_param'
MODEL_PARAM_EQUATION = 'model_param_equation'
PARAM_SUFFIX = '_parameters'
PARAMETERS = 'parameters'

### SESSION KEY for ContextModelParameter entity.
KEY_CONTEXT_MPS = "ContextForModelParametersOnSurface"


class SurfaceModelParametersController(SpatioTemporalController):
    """
    Control for defining parameters of a model in a visual manner.
    Here we focus on model-parameters spread over a brain surface.
    """

    def __init__(self):
        SpatioTemporalController.__init__(self)
        self.plotted_equations_prefixes = ['model_param', 'min_x', 'max_x']


    def get_data_from_burst_configuration(self):
        """
        Returns the model and surface instances from the burst configuration.
        """
        des = SerializationManager(common.get_from_session(common.KEY_BURST_CONFIG))
        ### Read from session current burst-configuration
        if des.conf is None:
            return None, None
        if des.has_model_pse_ranges():
            common.set_error_message("When configuring model parameters you are not allowed to specify range values.")
            raise cherrypy.HTTPRedirect("/burst/")

        try:
            model, integrator = des.make_model_and_integrator()
        except Exception:
            self.logger.exception("Some of the provided parameters have an invalid value.")
            common.set_error_message("Some of the provided parameters have an invalid value.")
            raise cherrypy.HTTPRedirect("/burst/")

        surface = des.get_surface()
        return model, surface


    @expose_page
    def edit_model_parameters(self):
        """
        Main method, to initialize Model-Parameter visual-set.
        """
        model, surface = self.get_data_from_burst_configuration()
        context_model_parameters = SurfaceContextModelParameters(surface, model)
        common.add2session(KEY_CONTEXT_MPS, context_model_parameters)

        template_specification = dict(title="Spatio temporal - Model parameters")
        template_specification.update(self.display_surface(surface.gid))
        model_params_data = self.get_surface_model_parameters_data()
        model_params_data = self._add_entra_equation_entries(model_params_data)
        template_specification.update(model_params_data)
        template_specification.update(
            submit_parameters_url='/spatial/modelparameters/surface/submit_model_parameters',
            mainContent='spatial/model_param_surface_main',
            equationViewerUrl='/spatial/modelparameters/surface/get_equation_chart',
            equationsPrefixes=json.dumps(self.plotted_equations_prefixes),
            submitSurfaceParametersBtn=True
        )
        return self.fill_default_attributes(template_specification)


    @expose_fragment('spatial/model_param_surface_left')
    def apply_equation(self, **kwargs):
        """
        Applies an equations for computing a model parameter.
        """
        submitted_data = collapse_params(kwargs, ['model_param'])
        model_param, equation = self._compute_equation(submitted_data)
        context_model_parameters = common.get_from_session(KEY_CONTEXT_MPS)
        context_model_parameters.apply_equation(model_param, equation)
        common.add2session(KEY_CONTEXT_MPS, context_model_parameters)
        template_specification = self.get_surface_model_parameters_data(model_param)
        template_specification = self._add_entra_equation_entries(template_specification,
                                                                  kwargs['min_x'], kwargs['max_x'])
        template_specification['equationViewerUrl'] = '/spatial/modelparameters/surface/get_equation_chart'
        template_specification['equationsPrefixes'] = json.dumps(self.plotted_equations_prefixes)
        return self.fill_default_attributes(template_specification)


    @expose_fragment('spatial/model_param_surface_focal_points')
    def apply_focal_point(self, model_param, triangle_index):
        """
        Adds the given focal point to the list of focal points specified for
        the equation used for computing the values for the specified model param.
        """
        template_specification = {}
        context_model_parameters = common.get_from_session(KEY_CONTEXT_MPS)
        if context_model_parameters.get_equation_for_parameter(model_param) is not None:
            context_model_parameters.apply_focal_point(model_param, triangle_index)
        else:
            template_specification['error_msg'] = "You have no equation applied for this parameter."
        template_specification['focal_points'] = context_model_parameters.get_focal_points_for_parameter(model_param)
        template_specification['focal_points_json'] = json.dumps(context_model_parameters.get_focal_points_for_parameter(model_param))
        return template_specification


    @expose_fragment('spatial/model_param_surface_focal_points')
    def remove_focal_point(self, model_param, vertex_index):
        """
        Removes the given focal point from the list of focal points specified for
        the equation used for computing the values for the specified model param.
        """
        context_model_parameters = common.get_from_session(KEY_CONTEXT_MPS)
        context_model_parameters.remove_focal_point(model_param, vertex_index)
        return {'focal_points': context_model_parameters.get_focal_points_for_parameter(model_param),
                'focal_points_json': json.dumps(context_model_parameters.get_focal_points_for_parameter(model_param))}


    @expose_fragment('spatial/model_param_surface_focal_points')
    def get_focal_points(self, model_param):
        """
        Returns the html which displays the list of focal points selected for the
        equation used for computing the values for the given model parameter.
        """
        context_model_parameters = common.get_from_session(KEY_CONTEXT_MPS)
        return {'focal_points': context_model_parameters.get_focal_points_for_parameter(model_param),
                'focal_points_json': json.dumps(context_model_parameters.get_focal_points_for_parameter(model_param))}


    @cherrypy.expose
    @handle_error(redirect=True)
    @check_user
    def submit_model_parameters(self, submit_action="cancel_action"):
        """
        Collects the model parameters values from all the models used for the surface vertices.
        @:param submit_action: a post parameter. It distinguishes if this request is a cancel or a submit
        """
        if submit_action == "submit_action":
            context_model_parameters = common.get_from_session(KEY_CONTEXT_MPS)
            burst_configuration = common.get_from_session(common.KEY_BURST_CONFIG)
            for original_param, modified_param in context_model_parameters.prepared_model_parameter_names.items():
                full_name = PARAMS_MODEL_PATTERN % (context_model_parameters.model_name, original_param)
                param_data = context_model_parameters.get_data_for_model_param(original_param, modified_param)
                if isinstance(param_data, dict):
                    equation = param_data[KEY_EQUATION]
                    focal_points = param_data[KEY_FOCAL_POINTS]
                    # if focal points or the equation are missing do not update this model parameter
                    if not focal_points or not equation:
                        continue
                    param_data[KEY_EQUATION] = equation.to_json(equation)
                    param_data[KEY_FOCAL_POINTS] = json.dumps(focal_points)
                    param_data = json.dumps(param_data)
                burst_configuration.update_simulation_parameter(full_name, param_data)
            ### Update in session BURST configuration for burst-page.
            common.add2session(common.KEY_BURST_CONFIG, burst_configuration.clone())

        ### Clean from session drawing context
        common.remove_from_session(KEY_CONTEXT_MPS)
        raise cherrypy.HTTPRedirect("/burst/")


    def _add_entra_equation_entries(self, input_list, min_x=0, max_x=100):
        """
        Add additional entries for the min and max of the plot.
        """
        plot_axis_parameters = []
        min_x = {'name': 'min_x', 'label': 'Min distance(mm)', 'type': 'str', "disabled": "False", "default": min_x,
                 "description": "The minimum value of the x-axis for spatial equation plot."}
        max_x = {'name': 'max_x', 'label': 'Max distance(mm)', 'type': 'str', "disabled": "False", "default": max_x,
                 "description": "The maximum value of the x-axis for spatial equation plot."}
        plot_axis_parameters.append(min_x)
        plot_axis_parameters.append(max_x)
        input_list['parametersEquationPlotDict'] = plot_axis_parameters
        return input_list


    def get_surface_model_parameters_data(self, default_selected_model_param=None):
        """
        Returns a dictionary which contains all the data needed for drawing the
        model parameters.
        """
        context_model_parameters = common.get_from_session(KEY_CONTEXT_MPS)
        if default_selected_model_param is None:
            default_selected_model_param = context_model_parameters.prepared_model_parameter_names.values()[0]

        equation_displayer = EquationDisplayer()
        equation_displayer.trait.bound = interface.INTERFACE_ATTRIBUTES_ONLY
        input_list = equation_displayer.interface[interface.INTERFACE_ATTRIBUTES]
        input_list[0] = self._lock_midpoints(input_list[0])

        options = []
        for original_param, modified_param in context_model_parameters.prepared_model_parameter_names.items():
            attributes = deepcopy(input_list)
            self._fill_default_values(attributes, modified_param)
            option = {'name': original_param, 'value': modified_param, 'attributes': attributes}
            options.append(option)

        input_list = [{'name': 'model_param', 'type': 'select', 'default': default_selected_model_param,
                       'label': 'Model param', 'required': True, 'options': options}]
        input_list = InputTreeManager.prepare_param_names(input_list)
        return {common.KEY_PARAMETERS_CONFIG: False, 'inputList': input_list,
                'applied_equations': context_model_parameters.get_configure_info()}


    @staticmethod
    def _fill_default_values(input_list, model_param):
        """
        If the user already applied an equation, for the given model parameter,
        than the form should be filled with the provided data for that equation.
        """
        #TODO: try to use fill_defaults from abcadapter

        context_model_parameters = common.get_from_session(KEY_CONTEXT_MPS)
        if model_param in context_model_parameters.applied_equations:
            model_param_data = context_model_parameters.applied_equations[model_param]
            if KEY_EQUATION in model_param_data:
                equation = model_param_data[KEY_EQUATION]
                equation_name = equation.__class__.__name__
                equation_params = equation.parameters
                for input_ in input_list:
                    if input_['name'] == 'model_param_equation':
                        input_['default'] = equation_name
                        for option in input_['options']:
                            if option['name'] == equation_name:
                                for attr in option['attributes']:
                                    if attr['name'] == 'parameters':
                                        for attribute in attr['attributes']:
                                            attribute['default'] = equation_params[attribute['name']]
                                        return

    @staticmethod
    def _compute_equation(parameters):
        """
        This method will return an equation and the model parameter on
        which should be applied the equation.
        The equation is constructed based on the parameters collected from the UI.
        """
        model_param = parameters[MODEL_PARAM]
        equation = parameters[MODEL_PARAM + PARAM_SUFFIX][MODEL_PARAM_EQUATION]
        equation_params = parameters[MODEL_PARAM + PARAM_SUFFIX][MODEL_PARAM_EQUATION + PARAM_SUFFIX][equation]
        equation_params = collapse_params(equation_params, [])
        if PARAMETERS + PARAM_SUFFIX in equation_params:
            equation_params = equation_params[PARAMETERS + PARAM_SUFFIX]
        else:
            equation_params = {}
        for param in equation_params:
            equation_params[param] = float(equation_params[param])
        selected_equation = get_traited_instance_for_name(equation, equations.Equation, {PARAMETERS: equation_params})
        return model_param, selected_equation


    def fill_default_attributes(self, template_dictionary):
        """
        Overwrite base controller to add required parameters for adapter templates.
        """
        template_dictionary[common.KEY_SECTION] = 'burst'
        template_dictionary[common.KEY_SUB_SECTION] = 'surfacemodel'
        template_dictionary[common.KEY_INCLUDE_RESOURCES] = 'spatial/included_resources'
        BaseController.fill_default_attributes(self, template_dictionary)
        return template_dictionary


    @expose_fragment('spatial/equation_displayer')
    def get_equation_chart(self, **form_data):
        """
        Returns the html which contains the plot with the equation selected by the user for a certain model param.
        """
        try:
            min_x, max_x, ui_message = self.get_x_axis_range(form_data['min_x'], form_data['max_x'])
            form_data = collapse_params(form_data, self.plotted_equations_prefixes)
            _, equation = self._compute_equation(form_data)
            series_data, display_ui_message = equation.get_series_data(min_range=min_x, max_range=max_x)
            all_series = self.get_series_json(series_data, "Spatial")

            ui_message = ''
            if display_ui_message:
                ui_message = self.get_ui_message(["spatial"])

            return {'allSeries': all_series, 'prefix': self.plotted_equations_prefixes[0], 'message': ui_message}
        except NameError as ex:
            self.logger.exception(ex)
            return {'allSeries': None, 'errorMsg': "Incorrect parameters for equation passed."}
        except SyntaxError as ex:
            self.logger.exception(ex)
            return {'allSeries': None, 'errorMsg': "Some of the parameters hold invalid characters."}
        except Exception as ex:
            self.logger.exception(ex)
            return {'allSeries': None, 'errorMsg': ex.message}
        