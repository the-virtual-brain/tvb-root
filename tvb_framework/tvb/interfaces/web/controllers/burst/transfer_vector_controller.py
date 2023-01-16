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
.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""

import uuid
import cherrypy
from tvb.adapters.datatypes.db.graph import ConnectivityMeasureIndex
from tvb.adapters.forms.equation_forms import get_form_for_equation
from tvb.adapters.forms.equation_plot_forms import EquationPlotForm
from tvb.adapters.forms.transfer_vector_form import KEY_TRANSFER, TransferVectorForm
from tvb.adapters.visualizers.histogram import HistogramViewer
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5
from tvb.interfaces.web.controllers.autologging import traced
from tvb.interfaces.web.controllers.base_controller import BaseController
from tvb.interfaces.web.controllers.decorators import expose_page, handle_error, check_user, expose_fragment, jsonify
from tvb.interfaces.web.controllers.simulator.simulator_wizzard_urls import SimulatorWizzardURLs
from tvb.interfaces.web.controllers.spatial.base_spatio_temporal_controller import SpatioTemporalController
from tvb.interfaces.web.entities.context_transfer_vector import TransferVectorContext, KEY_RESULT
from tvb.interfaces.web.entities.context_simulator import SimulatorContext
from tvb.interfaces.web.controllers import common


@traced
class TransferVectorController(SpatioTemporalController):
    """
    Controller class for Applying a transfer function plus a ConnectivityMeasure DT
    and producing a Spatial Vector distribution on Model parameters
    """

    base_url = '/burst/transfer/'
    title = 'Apply Spatial Vector on Model Parameter(s)'

    def __init__(self):
        super(TransferVectorController, self).__init__()
        self.simulator_context = SimulatorContext()

    def no_connectivity_measure_page(self):
        params = ({
            'title': self.title,
            'mainContent': 'burst/transfer_function_apply_empty',
        })
        return self.fill_default_attributes(params)

    @expose_page
    def index(self):
        sim_config = self.simulator_context.simulator
        connectivity_measures = dao.get_generic_entity(ConnectivityMeasureIndex, sim_config.connectivity.hex,
                                                       "fk_connectivity_gid")

        if not connectivity_measures:
            return self.no_connectivity_measure_page()

        model_params_list = self._prepare_model_params_list(self.simulator_context.simulator.model)

        context = TransferVectorContext(TransferVectorForm.default_transfer_function.instance,
                                        model_params_list[0].name,
                                        uuid.UUID(connectivity_measures[-1].gid))
        common.add2session(KEY_TRANSFER, context)

        params = HistogramViewer().prepare_parameters(context.current_connectivity_measure)

        config_form = TransferVectorForm(model_params_list)
        config_form.model_param.data = context.current_model_param
        config_form.connectivity_measure.data = context.current_connectivity_measure
        config_form.transfer_function.data = type(context.current_transfer_function)
        config_form.transfer_function.subform_field.form.fill_from_trait(context.current_transfer_function)
        config_form = self.algorithm_service.prepare_adapter_form(form_instance=config_form)

        eq_plot_form = EquationPlotForm()
        params.update({'mainContent': 'burst/transfer_function_apply',
                       'submit_parameters_url': self.build_path('/burst/transfer/submit_model_parameter'),
                       'applyTransferFunctionForm': self.render_adapter_form(config_form),
                       'parametersTransferFunctionPlotForm': self.render_adapter_form(eq_plot_form),
                       'isSingleMode': False,
                       'title': self.title,
                       'baseUrl': self.base_url})

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
            ontology_context = common.get_from_session(KEY_TRANSFER)

            transfer_function = ontology_context.current_transfer_function
            series_data, display_ui_message = transfer_function.get_series_data(min_range=min_x, max_range=max_x)
            all_series = self.get_series_json(series_data, "vector")

            ui_message = ''
            if display_ui_message:
                ui_message = self.get_ui_message(["vector"])
            return {'allSeries': all_series, 'prefix': 'vector', 'message': ui_message}
        except NameError as ex:
            self.logger.exception(ex)
            return {'allSeries': None, 'errorMsg': "Incorrect parameters for equation passed."}
        except SyntaxError as ex:
            self.logger.exception(ex)
            return {'allSeries': None, 'errorMsg': "Some of the parameters hold invalid characters."}
        except Exception as ex:
            self.logger.exception(ex)
            return {'allSeries': None, 'errorMsg': ex}

    @cherrypy.expose
    @handle_error(redirect=False)
    def set_model_parameter(self, model_param):
        ontology_context = common.get_from_session(KEY_TRANSFER)
        ontology_context.current_model_param = model_param

    @cherrypy.expose
    @handle_error(redirect=False)
    @jsonify
    def set_connectivity_measure(self, connectivity_measure):
        ontology_context = common.get_from_session(KEY_TRANSFER)
        ontology_context.current_connectivity_measure = uuid.UUID(connectivity_measure)

        result_dict = HistogramViewer().prepare_parameters(ontology_context.current_connectivity_measure)
        return result_dict

    @cherrypy.expose
    @handle_error(redirect=False)
    def set_transfer_function_param(self, **param):
        ontology_context = common.get_from_session(KEY_TRANSFER)
        eq_params_form_class = get_form_for_equation(type(ontology_context.current_transfer_function))
        eq_params_form = eq_params_form_class()
        eq_params_form.fill_from_trait(ontology_context.current_transfer_function)
        eq_params_form.fill_from_post(param)
        eq_params_form.fill_trait(ontology_context.current_transfer_function)

    @cherrypy.expose
    @handle_error(redirect=False)
    @jsonify
    def apply_transfer_function(self):
        context = common.get_from_session(KEY_TRANSFER)
        simulator = self.simulator_context.simulator
        result = context.apply_transfer_function()

        connectivity = h5.load_from_gid(simulator.connectivity)
        result_dict = HistogramViewer().gather_params_dict(connectivity.region_labels.tolist(),
                                                           result.tolist(), self.title)
        result_dict['applied_transfer_functions'] = context.get_applied_transfer_functions()
        return result_dict

    @cherrypy.expose
    @handle_error(redirect=False)
    @jsonify
    def clear_histogram(self):
        ontology_context = common.get_from_session(KEY_TRANSFER)
        result_dict = HistogramViewer().prepare_parameters(ontology_context.current_connectivity_measure)
        return result_dict

    @cherrypy.expose
    @handle_error(redirect=True)
    @check_user
    def submit_model_parameter(self, submit_action='cancel_action'):
        if submit_action != "cancel_action":
            ontology_model = common.get_from_session(KEY_TRANSFER)
            for model_param, applied_tf in ontology_model.applied_transfer_functions.items():
                setattr(self.simulator_context.simulator.model, model_param, applied_tf[KEY_RESULT])

            self.simulator_context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_INTEGRATOR_URL)

        common.remove_from_session(KEY_TRANSFER)
        self.redirect("/burst/")

    def fill_default_attributes(self, template_dictionary, subsection='transfer'):
        """
        Overwrite base controller to add required parameters for adapter templates.
        """
        template_dictionary[common.KEY_SECTION] = 'burst'
        template_dictionary[common.KEY_SUBMENU_LIST] = self.burst_submenu
        template_dictionary[common.KEY_SUB_SECTION] = subsection
        template_dictionary[common.KEY_INCLUDE_RESOURCES] = 'spatial/included_resources'
        BaseController.fill_default_attributes(self, template_dictionary)
        return template_dictionary
