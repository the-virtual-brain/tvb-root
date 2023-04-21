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
import cherrypy
from tvb.adapters.forms.equation_forms import get_form_for_equation
from tvb.adapters.forms.equation_plot_forms import EquationPlotForm
from tvb.adapters.forms.surface_model_parameters_form import SurfaceModelParametersForm, KEY_CONTEXT_MPS
from tvb.core.entities import load
from tvb.core.services.burst_config_serialization import SerializationManager
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.autologging import traced
from tvb.interfaces.web.controllers.base_controller import BaseController
from tvb.interfaces.web.controllers.decorators import expose_page, expose_fragment, handle_error, check_user
from tvb.interfaces.web.controllers.simulator.simulator_controller import SimulatorWizzardURLs
from tvb.interfaces.web.controllers.spatial.base_spatio_temporal_controller import SpatioTemporalController
from tvb.interfaces.web.entities.context_model_parameters import SurfaceContextModelParameters
from tvb.interfaces.web.entities.context_simulator import SimulatorContext
from tvb.interfaces.web.structure import WebStructure


@traced
class SurfaceModelParametersController(SpatioTemporalController):
    """
    Control for defining parameters of a model in a visual manner.
    Here we focus on model-parameters spread over a brain surface.
    """
    MODEL_PARAM_FIELD = 'set_model_parameter'
    EQUATION_FIELD = 'set_equation'
    EQUATION_PARAMS_FIELD = 'set_equation_param'
    base_url = '/spatial/modelparameters/surface'

    def __init__(self):
        super(SurfaceModelParametersController, self).__init__()
        self.simulator_context = SimulatorContext()
        self.model_params_list = None

    def get_data_from_burst_configuration(self):
        """
        Returns the model and surface instances from the burst configuration.
        """
        des = SerializationManager(self.simulator_context.simulator)
        ### Read from session current burst-configuration
        if des.conf is None:
            return None, None
        # if des.has_model_pse_ranges():
        #     common.set_error_message("When configuring model parameters you are not allowed to specify range values.")
        #     raise cherrypy.HTTPRedirect("/burst/")

        try:
            model = des.conf.model
        except Exception:
            self.logger.exception("Some of the provided parameters have an invalid value.")
            common.set_error_message("Some of the provided parameters have an invalid value.")
            self.redirect("/burst/")

        cortex = des.conf.surface
        return model, cortex

    @staticmethod
    def _fill_form_from_context(config_form, context):
        if context.current_model_param in context.applied_equations:
            current_equation = context.get_equation_for_parameter(context.current_model_param)
            context.current_equation = current_equation
            config_form.equation.data = type(current_equation)
            config_form.equation.subform_field.form = get_form_for_equation(type(current_equation))()
            config_form.equation.subform_field.form.fill_from_trait(current_equation)
        else:
            context.current_equation = SurfaceModelParametersForm.default_equation.instance
            config_form.equation.data = type(context.current_equation)
            config_form.equation.subform_field.form.fill_from_trait(context.current_equation)

    def _prepare_reload(self, context):
        template_specification = {
            'baseUrl': self.base_url,
            'equationsPrefixes': self.plotted_equation_prefixes
        }
        template_specification.update({'applied_equations': context.get_configure_info()})

        config_form = SurfaceModelParametersForm(self.model_params_list)
        config_form.model_param.data = context.current_model_param
        self._fill_form_from_context(config_form, context)
        template_specification.update({'adapter_form': self.render_adapter_form(config_form)})

        parameters_equation_plot_form = EquationPlotForm()
        template_specification.update({'parametersEquationPlotForm': self.render_adapter_form(
            parameters_equation_plot_form)})
        return template_specification

    @expose_page
    def edit_model_parameters(self):
        """
        Main method, to initialize Model-Parameter visual-set.
        """
        model, cortex = self.get_data_from_burst_configuration()
        surface_gid = cortex.surface_gid
        surface_index = load.load_entity_by_gid(surface_gid)

        self.model_params_list = self._prepare_model_params_list(model)
        context_model_parameters = SurfaceContextModelParameters(surface_index, model,
                                                                 SurfaceModelParametersForm.default_equation,
                                                                 self.model_params_list[0].name)
        common.add2session(KEY_CONTEXT_MPS, context_model_parameters)

        template_specification = dict(title="Spatio temporal - Model parameters")
        template_specification.update(self.display_surface(surface_gid.hex, cortex.region_mapping_data))

        dummy_form_for_initialization = SurfaceModelParametersForm(self.model_params_list)
        self.plotted_equation_prefixes = {
            self.MODEL_PARAM_FIELD: dummy_form_for_initialization.model_param.name,
            self.EQUATION_FIELD: dummy_form_for_initialization.equation.name,
            self.EQUATION_PARAMS_FIELD: dummy_form_for_initialization.equation.subform_field.name[1:]
        }
        template_specification.update(self._prepare_reload(context_model_parameters))
        template_specification.update(
            submit_parameters_url=self.build_path('/spatial/modelparameters/surface/submit_model_parameters'),
            mainContent='spatial/model_param_surface_main',
            submitSurfaceParametersBtn=True
        )
        return self.fill_default_attributes(template_specification)

    @expose_fragment('spatial/model_param_surface_left')
    def set_model_parameter(self, model_parameter):
        context = common.get_from_session(KEY_CONTEXT_MPS)
        context.current_model_param = model_parameter

        template_specification = self._prepare_reload(context)
        return self.fill_default_attributes(template_specification)

    @cherrypy.expose
    def set_equation_param(self, **param):
        context = common.get_from_session(KEY_CONTEXT_MPS)
        eq_params_form_class = get_form_for_equation(type(context.current_equation))
        eq_params_form = eq_params_form_class()
        eq_params_form.fill_from_trait(context.current_equation)
        eq_params_form.fill_from_post(param)
        eq_params_form.fill_trait(context.current_equation)

    @expose_fragment('spatial/model_param_surface_left')
    def apply_equation(self, **kwargs):
        """
        Applies an equations for computing a model parameter.
        """
        context_model_parameters = common.get_from_session(KEY_CONTEXT_MPS)
        context_model_parameters.apply_equation(context_model_parameters.current_model_param,
                                                context_model_parameters.current_equation)
        template_specification = self._prepare_reload(context_model_parameters)
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
        template_specification['focal_points_json'] = json.dumps(
            context_model_parameters.get_focal_points_for_parameter(model_param))
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
            simulator = self.simulator_context.simulator

            for param in list(self.model_params_list):
                param_data = context_model_parameters.get_data_for_model_param(param.name)
                if param_data is None:
                    continue
                setattr(simulator.model, param.name, param_data)
            ### Update in session the last loaded URL for burst-page.
            self.simulator_context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_INTEGRATOR_URL)

        ### Clean from session drawing context
        common.remove_from_session(KEY_CONTEXT_MPS)
        self.redirect("/burst/")

    def fill_default_attributes(self, template_dictionary):
        """
        Overwrite base controller to add required parameters for adapter templates.
        """
        template_dictionary[common.KEY_SECTION] = WebStructure.SECTION_BURST
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
            plot_form = EquationPlotForm()
            if form_data:
                plot_form.fill_from_post(form_data)

            min_x, max_x, ui_message = self.get_x_axis_range(plot_form.min_x.value, plot_form.max_x.value)
            context_mps = common.get_from_session(KEY_CONTEXT_MPS)

            equation = context_mps.current_equation
            series_data, display_ui_message = equation.get_series_data(min_range=min_x, max_range=max_x)
            all_series = self.get_series_json(series_data, "Spatial")

            ui_message = ''
            if display_ui_message:
                ui_message = self.get_ui_message(["spatial"])
            return {'allSeries': all_series, 'prefix': 'spatial', 'message': ui_message}
        except NameError as ex:
            self.logger.exception(ex)
            return {'allSeries': None, 'errorMsg': "Incorrect parameters for equation passed."}
        except SyntaxError as ex:
            self.logger.exception(ex)
            return {'allSeries': None, 'errorMsg': "Some of the parameters hold invalid characters."}
        except Exception as ex:
            self.logger.exception(ex)
            return {'allSeries': None, 'errorMsg': ex}
