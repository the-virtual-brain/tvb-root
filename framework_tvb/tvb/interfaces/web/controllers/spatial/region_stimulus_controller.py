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

import json
import cherrypy
import numpy
from tvb.adapters.creators.stimulus_creator import *
from tvb.adapters.datatypes.h5.patterns_h5 import StimuliRegionH5
from tvb.adapters.simulator.equation_forms import get_form_for_equation
from tvb.adapters.simulator.subform_helper import SubformHelper
from tvb.adapters.simulator.subforms_mapping import get_ui_name_to_equation_dict
from tvb.adapters.visualizers.connectivity import ConnectivityViewer
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.load import try_get_last_datatype
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5
from tvb.datatypes.patterns import StimuliRegion
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.autologging import traced
from tvb.interfaces.web.controllers.common import MissingDataException
from tvb.interfaces.web.controllers.decorators import handle_error, expose_page, expose_fragment, using_template, \
    check_user
from tvb.interfaces.web.controllers.spatial.base_spatio_temporal_controller import SpatioTemporalController
from tvb.interfaces.web.controllers.spatial.surface_model_parameters_controller import EquationPlotForm

LOAD_EXISTING_URL = '/spatial/stimulus/region/load_region_stimulus'
RELOAD_DEFAULT_PAGE_URL = '/spatial/stimulus/region/reset_region_stimulus'

KEY_REGION_STIMULUS = "stim-region"
KEY_REGION_STIMULUS_NAME = "stim-region-name"


class TemporalPlotForm(EquationPlotForm):
    def __init__(self):
        super(TemporalPlotForm, self).__init__()
        self.min_x.label = 'Temporal Start Time(ms)'
        self.min_x.doc = "The minimum value of the x-axis for temporal equation plot. " \
                         "Not persisted, used only for visualization."
        self.max_x.label = 'Temporal Start Time(ms)'
        self.max_x.doc = "The maximum value of the x-axis for temporal equation plot. " \
                         "Not persisted, used only for visualization."

    def fill_from_post(self, form_data):
        if self.min_x.name in form_data:
            self.min_x.fill_from_post(form_data)
        if self.max_x.name in form_data:
            self.max_x.fill_from_post(form_data)


@traced
class RegionStimulusController(SpatioTemporalController):
    """
    Control layer for defining Stimulus entities on Regions.
    """
    CONNECTIVITY_FIELD = 'set_connectivity'
    TEMPORAL_FIELD = 'set_temporal'
    DISPLAY_NAME_FIELD = 'set_display_name'
    TEMPORAL_PARAMS_FIELD = 'set_temporal_param'
    base_url = '/spatial/stimulus/region'
    MSG_MISSING_CONNECTIVITY = "There is no structural Connectivity in the current project. " \
                               "Please upload one to continue!"

    @cherrypy.expose
    def set_connectivity(self, **param):
        current_region_stim = common.get_from_session(KEY_REGION_STIMULUS)
        connectivity_form_field = RegionStimulusCreatorForm(common.get_current_project().id).connectivity
        connectivity_form_field.fill_from_post(param)
        current_region_stim.connectivity = connectivity_form_field.value
        conn_index = ABCAdapter.load_entity_by_gid(connectivity_form_field.value.hex)
        current_region_stim.weight = StimuliRegion.get_default_weights(conn_index.number_of_regions)

    @cherrypy.expose
    def set_display_name(self, **param):
        display_name_form_field = StimulusRegionSelectorForm(common.get_current_project().id).display_name
        display_name_form_field.fill_from_post(param)
        if display_name_form_field.value is not None:
            common.add2session(KEY_REGION_STIMULUS_NAME, display_name_form_field.value)

    @cherrypy.expose
    @using_template('form_fields/form_field')
    @handle_error(redirect=False)
    @check_user
    def refresh_subform(self, temporal_equation, mapping_key):
        eq_class = get_ui_name_to_equation_dict().get(temporal_equation)
        current_region_stim = common.get_from_session(KEY_REGION_STIMULUS)
        current_region_stim.temporal = eq_class()

        eq_params_form = SubformHelper.get_subform_for_field_value(temporal_equation, mapping_key)
        # TODO: check eqPrefixes
        return {'adapter_form': eq_params_form, 'equationsPrefixes': self.plotted_equation_prefixes}

    @cherrypy.expose
    def set_temporal_param(self, **param):
        current_region_stim = common.get_from_session(KEY_REGION_STIMULUS)
        eq_param_form_class = get_form_for_equation(type(current_region_stim.temporal))
        eq_param_form = eq_param_form_class()
        eq_param_form.fill_from_trait(current_region_stim.temporal)
        eq_param_form.fill_from_post(param)
        eq_param_form.fill_trait(current_region_stim.temporal)

    def step_1(self):
        """
        Generate the required template dictionary for the first step.
        """
        current_stimuli_region = common.get_from_session(KEY_REGION_STIMULUS)
        selected_stimulus_gid = current_stimuli_region.gid.hex
        project_id = common.get_current_project().id
        region_stim_selector_form = StimulusRegionSelectorForm(project_id)
        region_stim_selector_form.region_stimulus.data = selected_stimulus_gid
        region_stim_selector_form.display_name.data = common.get_from_session(KEY_REGION_STIMULUS_NAME)

        region_stim_creator_form = RegionStimulusCreatorForm(project_id)
        if not hasattr(current_stimuli_region, 'connectivity') or not current_stimuli_region.connectivity:
            conn = try_get_last_datatype(project_id, ConnectivityIndex)
            if conn is None:
                current_stimuli_region.connectivity = uuid.uuid4()
                common.set_error_message(self.MSG_MISSING_CONNECTIVITY)
            else:
                current_stimuli_region.connectivity = uuid.UUID(conn.gid)
        region_stim_creator_form.fill_from_trait(current_stimuli_region)

        template_specification = dict(title="Spatio temporal - Region stimulus")
        template_specification['mainContent'] = 'spatial/stimulus_region_step1_main'
        template_specification['isSingleMode'] = True
        template_specification['regionStimSelectorForm'] = self.render_spatial_form(region_stim_selector_form)
        template_specification['regionStimCreatorForm'] = self.render_spatial_form(region_stim_creator_form)
        template_specification['baseUrl'] = self.base_url
        self.plotted_equation_prefixes = {
            self.CONNECTIVITY_FIELD: region_stim_creator_form.connectivity.name,
            self.DISPLAY_NAME_FIELD: region_stim_selector_form.display_name.name
        }
        template_specification['fieldsWithEvents'] = json.dumps(self.plotted_equation_prefixes)
        template_specification['next_step_url'] = '/spatial/stimulus/region/step_1_submit'
        template_specification['anyScaling'] = 0
        template_specification = self._add_extra_fields_to_interface(template_specification)
        return self.fill_default_attributes(template_specification)

    def step_2(self):
        """
        Generate the required template dictionary for the second step.
        """
        current_region_stimulus = common.get_from_session(KEY_REGION_STIMULUS)
        region_stim_selector_form = StimulusRegionSelectorForm(common.get_current_project().id)
        region_stim_selector_form.region_stimulus.data = current_region_stimulus.gid.hex
        region_stim_selector_form.display_name.data = common.get_from_session(KEY_REGION_STIMULUS_NAME)

        template_specification = dict(title="Spatio temporal - Region stimulus")
        template_specification['mainContent'] = 'spatial/stimulus_region_step2_main'
        template_specification['next_step_url'] = '/spatial/stimulus/region/step_2_submit'
        template_specification['regionStimSelectorForm'] = self.render_adapter_form(region_stim_selector_form)

        default_weights = current_region_stimulus.weight
        if len(default_weights) == 0:
            selected_connectivity = ABCAdapter.load_entity_by_gid(current_region_stimulus.connectivity.hex)
            if selected_connectivity is None:
                common.set_error_message(self.MSG_MISSING_CONNECTIVITY)
                default_weights = numpy.array([])
            else:
                default_weights = StimuliRegion.get_default_weights(selected_connectivity.number_of_regions)

        template_specification['baseUrl'] = self.base_url
        self.plotted_equation_prefixes = {
            self.DISPLAY_NAME_FIELD: region_stim_selector_form.display_name.name
        }
        template_specification['fieldsWithEvents'] = json.dumps(self.plotted_equation_prefixes)
        template_specification['node_weights'] = json.dumps(default_weights.tolist())
        template_specification[common.KEY_PARAMETERS_CONFIG] = False
        template_specification.update(self.display_connectivity(current_region_stimulus.connectivity.hex))
        return self.fill_default_attributes(template_specification)

    def do_step(self, step_idx, from_step=None):
        """
        Go to the step given by :param step_idx. In case the next step is the
        create one (3), we want to remain on the same step as before so that is
        handled differently depending on the :param from_step.
        """
        if int(step_idx) == 1:
            return self.step_1()
        if int(step_idx) == 2:
            return self.step_2()
        if int(step_idx) == 3:
            self.create_stimulus()
            if from_step == 2:
                return self.step_2()
            return self.step_1()

    def _reset_region_stimulus(self):
        new_region_stimulus = RegionStimulusCreatorModel()
        new_region_stimulus.temporal = RegionStimulusCreatorForm.default_temporal()
        # TODO: proper init
        new_region_stimulus.weight = numpy.array([])
        common.add2session(KEY_REGION_STIMULUS, new_region_stimulus)
        common.add2session(KEY_REGION_STIMULUS_NAME, None)

    @expose_page
    def step_1_submit(self, next_step, do_reset=0, **kwargs):
        """
        Any submit from the first step should be handled here. Update the context then
        go to the next step as required. In case a reset is needed create a clear context.
        """
        if int(do_reset) == 1:
            self._reset_region_stimulus()
        return self.do_step(next_step)

    @expose_page
    def step_2_submit(self, next_step, **kwargs):
        """
        Any submit from the second step should be handled here. Update the context and then do 
        the next step as required.
        """
        return self.do_step(next_step, 2)

    @staticmethod
    def display_connectivity(connectivity_gid):
        """
        Generates the html for displaying the connectivity matrix.
        """
        connectivity = ABCAdapter.load_entity_by_gid(connectivity_gid)
        if connectivity is None:
            raise MissingDataException(RegionStimulusController.MSG_MISSING_CONNECTIVITY + "!!")
        current_project = common.get_current_project()
        conn_path = FilesHelper().get_project_folder(current_project, str(connectivity.fk_from_operation))
        connectivity_viewer_params = ConnectivityViewer.get_connectivity_parameters(connectivity, conn_path)

        template_specification = dict()
        template_specification['isSingleMode'] = True
        template_specification.update(connectivity_viewer_params)
        return template_specification

    def create_stimulus(self):
        """
        Creates a stimulus from the given data.
        """
        current_stimulus_region = common.get_from_session(KEY_REGION_STIMULUS)
        display_name = common.get_from_session(KEY_REGION_STIMULUS_NAME)
        current_stimulus_region.display_name = display_name
        region_stimulus_creator = ABCAdapter.build_adapter_from_class(RegionStimulusCreator)
        self.operation_service.fire_operation(region_stimulus_creator, common.get_logged_user(),
                                              common.get_current_project().id, view_model=current_stimulus_region)
        common.set_important_message("The operation for creating the stimulus was successfully launched.")

    @cherrypy.expose
    @handle_error(redirect=False)
    def update_scaling(self, **kwargs):
        """
        Update the scaling according to the UI.
        """
        current_stimuli_region = common.get_from_session(KEY_REGION_STIMULUS)
        try:
            scaling = json.loads(kwargs['scaling'])
            current_stimuli_region.weight = numpy.array(scaling)
            return 'true'
        except Exception as ex:
            self.logger.exception(ex)
            return 'false'

    @expose_fragment('spatial/equation_displayer')
    def get_equation_chart(self, **form_data):
        """
        Returns the html which contains the plot with the temporal equation.
        """
        try:
            plot_form = TemporalPlotForm()
            if form_data:
                plot_form.fill_from_post(form_data)

            min_x, max_x, ui_message = self.get_x_axis_range(plot_form.min_x.value, plot_form.max_x.value)
            current_stimuli_region = common.get_from_session(KEY_REGION_STIMULUS)
            series_data, display_ui_message = current_stimuli_region.temporal.get_series_data(min_range=min_x,
                                                                                              max_range=max_x)
            all_series = self.get_series_json(series_data, 'Temporal')

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

    def _load_existent_region_stimuli(self, region_stimulus_gid):
        existent_region_stimulus_index = dao.get_datatype_by_gid(region_stimulus_gid)
        stimuli_region = RegionStimulusCreatorModel()

        stimuli_region_path = h5.path_for_stored_index(existent_region_stimulus_index)
        with StimuliRegionH5(stimuli_region_path) as stimuli_region_h5:
            stimuli_region_h5.load_into(stimuli_region)

        dummy_gid = uuid.UUID(existent_region_stimulus_index.fk_connectivity_gid)
        stimuli_region.connectivity = dummy_gid

        common.add2session(KEY_REGION_STIMULUS, stimuli_region)
        common.add2session(KEY_REGION_STIMULUS_NAME, existent_region_stimulus_index.user_tag_1)
        return stimuli_region

    @expose_page
    def load_region_stimulus(self, region_stimulus_gid, from_step=None):
        """
        Loads the interface for the selected region stimulus.
        """
        self._load_existent_region_stimuli(region_stimulus_gid)
        return self.do_step(from_step)

    @expose_page
    def reset_region_stimulus(self, from_step):
        """
        Just reload default data as if stimulus is None. 
        
        from_step
            not actually used here since when the user selects None
            from the stimulus entities select we want to take him back to step 1
            always. Kept just for compatibility with the normal load entity of a 
            stimulus where we want to stay in the same page.

        """
        self._reset_region_stimulus()
        return self.do_step(1)

    def _add_extra_fields_to_interface(self, input_list):
        """
        The fields that have to be added to the existent
        adapter interface should be added in this method.
        """
        temporal_plot_list_form = TemporalPlotForm()
        input_list['temporalPlotInputList'] = self.render_adapter_form(temporal_plot_list_form)
        return input_list

    def fill_default_attributes(self, template_dictionary):
        """
        Overwrite base controller to add required parameters for adapter templates.
        """
        template_dictionary['loadExistentEntityUrl'] = LOAD_EXISTING_URL
        template_dictionary['resetToDefaultUrl'] = RELOAD_DEFAULT_PAGE_URL
        msg, msg_type = common.get_message_from_session()
        template_dictionary['displayedMessage'] = msg
        template_dictionary['messageType'] = msg_type
        return SpatioTemporalController.fill_default_attributes(self, template_dictionary, subsection='regionstim')
