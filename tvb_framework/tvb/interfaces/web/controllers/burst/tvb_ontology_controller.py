# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""

import uuid
import cherrypy

from tvb.adapters.datatypes.db.graph import ConnectivityMeasureIndex
from tvb.adapters.forms.equation_forms import get_form_for_equation
from tvb.adapters.forms.equation_plot_forms import EquationPlotForm
from tvb.adapters.forms.ontology_form import KEY_ONTOLOGY, TVBOntologyForm
from tvb.adapters.visualizers.histogram import HistogramViewer
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5
from tvb.interfaces.web.controllers.autologging import traced
from tvb.interfaces.web.controllers.base_controller import BaseController
from tvb.interfaces.web.controllers.decorators import expose_page, handle_error, check_user, \
    expose_fragment
from tvb.interfaces.web.controllers.simulator.simulator_wizzard_urls import SimulatorWizzardURLs
from tvb.interfaces.web.controllers.spatial.base_spatio_temporal_controller import SpatioTemporalController
from tvb.interfaces.web.entities.context_ontology import TVBOntologyContext, KEY_RESULT
from tvb.interfaces.web.entities.context_simulator import SimulatorContext
from tvb.interfaces.web.controllers import common


@traced
class TVBOntologyController(SpatioTemporalController):
    """
    Controller class for TVB-O integration
    """

    MODEL_PARAM_FIELD = 'set_model_parameter'
    CONNECTIVITY_MEASURE_FIELD = 'set_connectivity_measure'
    EQUATION_FIELD = 'set_transfer_function'
    EQUATION_PARAMS_FIELD = 'set_transfer_function_param'

    base_url = '/burst/tvb-o/'
    title = 'TVB Ontology: Apply Transfer Function on Model Parameter'

    def __init__(self):
        super(TVBOntologyController, self).__init__()
        self.simulator_context = SimulatorContext()
        self.model_params_list = None
        self.plotted_tf_prefixes = None

    def no_connectivity_measure_page(self):
        params = ({
            'title': self.title,
            'mainContent': 'burst/transfer_function_apply_empty',
        })
        return self.fill_default_attributes(params)

    @staticmethod
    def _fill_form_from_context(config_form, context):
        if context.current_model_param in context.applied_transfer_functions:
            current_transfer_function = context.get_transfer_function_for_parameter(context.current_model_param)
            context.current_transfer_function = current_transfer_function
            config_form.transfer_function.data = type(current_transfer_function)
            config_form.transfer_function.subform_field.form = get_form_for_equation(type(current_transfer_function))()
            config_form.transfer_function.subform_field.form.fill_from_trait(current_transfer_function)
        else:
            context.current_transfer_function = TVBOntologyForm.default_transfer_function.instance
            config_form.transfer_function.data = type(context.current_transfer_function)
            config_form.transfer_function.subform_field.form.fill_from_trait(context.current_transfer_function)

    def _prepare_reload(self, ontology_context):
        template_specification = HistogramViewer().prepare_parameters(ontology_context.current_connectivity_measure)
        template_specification['baseUrl'] = self.base_url
        template_specification['transferFunctionsPrefixes'] = self.plotted_tf_prefixes
        template_specification['applied_transfer_functions'] = ontology_context.get_configure_info()

        ontology_form = TVBOntologyForm(self.model_params_list)
        ontology_form.model_param.data = ontology_context.current_model_param
        ontology_form.connectivity_measure.data = ontology_context.current_connectivity_measure
        self._fill_form_from_context(ontology_form, ontology_context)
        ontology_form = self.algorithm_service.prepare_adapter_form(form_instance=ontology_form)
        template_specification['applyTransferFunctionForm'] = self.render_adapter_form(ontology_form)

        transfer_function_plot_form = EquationPlotForm()
        template_specification['parametersTransferFunctionPlotForm'] = self.render_adapter_form(
            transfer_function_plot_form)

        return template_specification

    @expose_page
    def index(self):
        sim_config = self.simulator_context.simulator
        connectivity_measures = dao.get_generic_entity(ConnectivityMeasureIndex, sim_config.connectivity.hex,
                                                       "fk_connectivity_gid")

        if not connectivity_measures:
            return self.no_connectivity_measure_page()

        self.model_params_list = self._prepare_model_params_list(self.simulator_context.simulator.model)

        ontology_context = TVBOntologyContext(TVBOntologyForm.default_transfer_function, self.model_params_list[0].name,
                                              uuid.UUID(connectivity_measures[0].gid))
        common.add2session(KEY_ONTOLOGY, ontology_context)
        params = self._prepare_reload(ontology_context)

        tvb_ontology_form = TVBOntologyForm(self.model_params_list)
        self.plotted_tf_prefixes = {
            self.MODEL_PARAM_FIELD: tvb_ontology_form.model_param.name,
            self.CONNECTIVITY_MEASURE_FIELD: tvb_ontology_form.connectivity_measure.name,
            self.EQUATION_FIELD: tvb_ontology_form.transfer_function.name,
            self.EQUATION_PARAMS_FIELD: tvb_ontology_form.transfer_function.subform_field.name
        }

        params.update({'mainContent': 'burst/transfer_function_apply',
                       'submit_parameters_url': self.build_path('/burst/tvb-o/submit_model_parameter'),
                       'isSingleMode': True,
                       'title': self.title,
                       'transferFunctionsPrefixes': self.plotted_tf_prefixes})

        return self.fill_default_attributes(params)

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
            ontology_context = common.get_from_session(KEY_ONTOLOGY)

            transfer_function = ontology_context.current_transfer_function
            series_data, display_ui_message = transfer_function.get_series_data(min_range=min_x, max_range=max_x)
            all_series = self.get_series_json(series_data, "Temporal")

            ui_message = ''
            if display_ui_message:
                ui_message = self.get_ui_message(["temporal"])
            return {'allSeries': all_series, 'prefix': 'temporal', 'message': ui_message}
        except NameError as ex:
            self.logger.exception(ex)
            return {'allSeries': None, 'errorMsg': "Incorrect parameters for equation passed."}
        except SyntaxError as ex:
            self.logger.exception(ex)
            return {'allSeries': None, 'errorMsg': "Some of the parameters hold invalid characters."}
        except Exception as ex:
            self.logger.exception(ex)
            return {'allSeries': None, 'errorMsg': ex}

    @expose_fragment('burst/transfer_function_apply')
    def set_model_parameter(self, model_param):
        ontology_context = common.get_from_session(KEY_ONTOLOGY)
        ontology_context.current_model_param = model_param

        template_specification = self._prepare_reload(ontology_context)
        return self.fill_default_attributes(template_specification)

    @expose_fragment('burst/transfer_function_apply')
    def set_connectivity_measure(self, connectivity_measure):
        ontology_context = common.get_from_session(KEY_ONTOLOGY)
        ontology_context.current_connectivity_measure = uuid.UUID(connectivity_measure)

        template_specification = self._prepare_reload(ontology_context)
        return self.fill_default_attributes(template_specification)

    @cherrypy.expose
    def set_transfer_function_param(self, **param):
        ontology_context = common.get_from_session(KEY_ONTOLOGY)
        eq_params_form_class = get_form_for_equation(type(ontology_context.transfer_function))
        eq_params_form = eq_params_form_class()
        eq_params_form.fill_from_trait(ontology_context.transfer_function)
        eq_params_form.fill_from_post(param)
        eq_params_form.fill_trait(ontology_context.transfer_function)

    @expose_fragment('burst/transfer_function_apply')
    def apply_transfer_function(self):
        ontology_context = common.get_from_session(KEY_ONTOLOGY)
        simulator = self.simulator_context.simulator
        result = ontology_context.apply_transfer_function()

        template_specification = self._prepare_reload(ontology_context)
        connectivity = h5.load_from_gid(simulator.connectivity)
        new_cm_params = HistogramViewer().gather_params_dict(connectivity.region_labels.tolist(), result.tolist(), '')

        template_specification.update(new_cm_params)
        template_specification['title'] = self.title
        return self.fill_default_attributes(template_specification)

    @expose_fragment('burst/transfer_function_apply')
    def clear_histogram(self):
        ontology_context = common.get_from_session(KEY_ONTOLOGY)
        template_specification = self._prepare_reload(ontology_context)
        return self.fill_default_attributes(template_specification)

    @cherrypy.expose
    @handle_error(redirect=True)
    @check_user
    def submit_model_parameter(self, submit_action='cancel_action'):
        if submit_action != "cancel_action":
            ontology_model = common.get_from_session(KEY_ONTOLOGY)
            for model_param, applied_tf in ontology_model.applied_transfer_functions.items():
                setattr(self.simulator_context.simulator.model, model_param, applied_tf[KEY_RESULT])

            self.simulator_context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_INTEGRATOR_URL)

        common.remove_from_session(KEY_ONTOLOGY)
        self.redirect("/burst/")

    def fill_default_attributes(self, template_dictionary, subsection='tvb-o'):
        """
        Overwrite base controller to add required parameters for adapter templates.
        """
        template_dictionary[common.KEY_SECTION] = 'burst'
        template_dictionary[common.KEY_SUBMENU_LIST] = self.burst_submenu
        template_dictionary[common.KEY_SUB_SECTION] = subsection
        template_dictionary[common.KEY_INCLUDE_RESOURCES] = 'spatial/included_resources'
        BaseController.fill_default_attributes(self, template_dictionary)
        return template_dictionary
