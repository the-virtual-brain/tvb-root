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
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""

import cherrypy
import json
from tvb.adapters.simulator.equation_forms import get_form_for_equation
from tvb.adapters.simulator.model_forms import get_model_to_form_dict
from tvb.adapters.simulator.subform_helper import SubformHelper
from tvb.adapters.simulator.subforms_mapping import get_ui_name_to_equation_dict, GAUSSIAN_EQUATION, SIGMOID_EQUATION
from tvb.basic.neotraits.api import Attr
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.entities.storage import dao
from tvb.core.neotraits.forms import Form, SimpleFloatField, FormField, SelectField
from tvb.core.neotraits.view_model import Str
from tvb.core.services.burst_config_serialization import SerializationManager
from tvb.datatypes.equations import Gaussian, SpatialApplicableEquation
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.autologging import traced
from tvb.interfaces.web.controllers.base_controller import BaseController
from tvb.interfaces.web.controllers.decorators import expose_page, expose_fragment, handle_error, check_user, \
    using_template
from tvb.interfaces.web.controllers.simulator_controller import SimulatorWizzardURLs
from tvb.interfaces.web.controllers.spatial.base_spatio_temporal_controller import SpatioTemporalController
from tvb.interfaces.web.entities.context_model_parameters import SurfaceContextModelParameters

### SESSION KEY for ContextModelParameter entity.
KEY_CONTEXT_MPS = "ContextForModelParametersOnSurface"


class SurfaceModelParametersForm(ABCAdapterForm):
    NAME_EQATION_PARAMS_DIV = 'equation_params'
    default_equation = Gaussian
    ui_name_to_equation_dict = get_ui_name_to_equation_dict()
    equation_choices = {GAUSSIAN_EQUATION: ui_name_to_equation_dict.get(GAUSSIAN_EQUATION),
                        SIGMOID_EQUATION: ui_name_to_equation_dict.get(SIGMOID_EQUATION)}

    def __init__(self, model_params, prefix=''):
        super(SurfaceModelParametersForm, self).__init__(prefix)

        self.model_param = SelectField(Str(label='Model parameter'), self, choices=model_params, name='model_param')
        self.equation = SelectField(Attr(SpatialApplicableEquation, label='Equation', default=self.default_equation),
                                    self, choices=self.equation_choices, name='equation',
                                    subform=get_form_for_equation(self.default_equation))

    @staticmethod
    def get_required_datatype():
        return None

    @staticmethod
    def get_input_name():
        return None

    @staticmethod
    def get_filters():
        return None

    def fill_from_trait(self, trait):
        self.equation.data = type(trait)
        self.equation.subform_field = FormField(get_form_for_equation(type(trait)), self,
                                                self.NAME_EQATION_PARAMS_DIV)
        self.equation.subform_field.form.fill_from_trait(trait)


class EquationPlotForm(Form):
    def __init__(self):
        super(EquationPlotForm, self).__init__()
        self.min_x = SimpleFloatField(self, name='min_x', label='Min distance(mm)',
                                      doc="The minimum value of the x-axis for spatial equation plot.", default=0)
        self.max_x = SimpleFloatField(self, name='max_x', label='Max distance(mm)',
                                      doc="The maximum value of the x-axis for spatial equation plot.", default=100)

    def fill_from_post(self, form_data):
        if self.min_x.name in form_data:
            self.min_x.fill_from_post(form_data)
        if self.max_x.name in form_data:
            self.max_x.fill_from_post(form_data)


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

    def get_data_from_burst_configuration(self):
        """
        Returns the model and surface instances from the burst configuration.
        """
        des = SerializationManager(common.get_from_session(common.KEY_SIMULATOR_CONFIG))
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
            raise cherrypy.HTTPRedirect("/burst/")

        cortex = des.conf.surface
        return model, cortex

    def _prepare_model_params_dict(self, model):
        model_form = get_model_to_form_dict().get(type(model))
        model_params = model_form.get_params_configurable_in_phase_plane()
        if len(model_params) == 0:
            self.logger.warning("The list with configurable parameters for the current model is empty!")
        model_params_dict = {}

        for param in model_params:
            model_params_dict.update({param: param})
        return model_params_dict

    def _fill_form_from_context(self, config_form, context):
        if context.current_model_param in context.applied_equations:
            current_equation = context.get_equation_for_parameter(context.current_model_param)
            context.current_equation = current_equation
            config_form.equation.data = type(current_equation)
            config_form.equation.subform_field.form = get_form_for_equation(type(current_equation))()
            config_form.equation.subform_field.form.fill_from_trait(current_equation)
        else:
            context.current_equation = SurfaceModelParametersForm.default_equation()
            config_form.equation.data = type(context.current_equation)
            config_form.equation.subform_field.form.fill_from_trait(context.current_equation)

    def _prepare_reload(self, context):
        template_specification = {
            'baseUrl': self.base_url,
            'equationsPrefixes': self.plotted_equation_prefixes
        }
        template_specification.update({'applied_equations': context.get_configure_info()})

        config_form = SurfaceModelParametersForm(self.model_params_dict)
        config_form.model_param.data = context.current_model_param
        self._fill_form_from_context(config_form, context)
        template_specification.update({'adapter_form': self.render_adapter_form(config_form)})

        parameters_equation_plot_form = EquationPlotForm()
        template_specification.update({'parametersEquationPlotForm': self.render_adapter_form(parameters_equation_plot_form)})
        return template_specification

    @expose_page
    def edit_model_parameters(self):
        """
        Main method, to initialize Model-Parameter visual-set.
        """
        model, cortex = self.get_data_from_burst_configuration()
        surface_gid = cortex.surface_gid
        surface_index = dao.get_datatype_by_gid(surface_gid.hex)

        self.model_params_dict = self._prepare_model_params_dict(model)
        context_model_parameters = SurfaceContextModelParameters(surface_index, model,
                                                                 SurfaceModelParametersForm.default_equation(),
                                                                 list(self.model_params_dict.values())[0])
        common.add2session(KEY_CONTEXT_MPS, context_model_parameters)

        template_specification = dict(title="Spatio temporal - Model parameters")
        template_specification.update(self.display_surface(surface_gid.hex, cortex.region_mapping_data))

        dummy_form_for_initialization = SurfaceModelParametersForm(self.model_params_dict)
        self.plotted_equation_prefixes = {
            self.MODEL_PARAM_FIELD: dummy_form_for_initialization.model_param.name,
            self.EQUATION_FIELD: dummy_form_for_initialization.equation.name,
            self.EQUATION_PARAMS_FIELD: dummy_form_for_initialization.equation.subform_field.name[1:]
        }
        template_specification.update(self._prepare_reload(context_model_parameters))
        template_specification.update(
            submit_parameters_url='/spatial/modelparameters/surface/submit_model_parameters',
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
    @using_template("form_fields/form_field")
    @handle_error(redirect=False)
    @check_user
    def refresh_subform(self, equation, mapping_key):
        eq_class = get_ui_name_to_equation_dict().get(equation)
        context = common.get_from_session(KEY_CONTEXT_MPS)
        context.current_equation = eq_class()

        eq_params_form = SubformHelper.get_subform_for_field_value(equation, mapping_key)
        return {'adapter_form': eq_params_form, 'equationsPrefixes': self.plotted_equation_prefixes}

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
            simulator = common.get_from_session(common.KEY_SIMULATOR_CONFIG)

            for param_name in self.model_params_dict.values():
                param_data = context_model_parameters.get_data_for_model_param(param_name)
                if param_data is None:
                    continue
                setattr(simulator.model, param_name, param_data)
            ### Update in session the last loaded URL for burst-page.
            common.add2session(common.KEY_LAST_LOADED_FORM_URL, SimulatorWizzardURLs.SET_INTEGRATOR_URL)

        ### Clean from session drawing context
        common.remove_from_session(KEY_CONTEXT_MPS)
        raise cherrypy.HTTPRedirect("/burst/")

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
