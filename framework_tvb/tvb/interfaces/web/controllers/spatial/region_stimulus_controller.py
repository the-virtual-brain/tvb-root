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

import json
import uuid
import cherrypy
import numpy
from tvb.adapters.creators.stimulus_creator import RegionStimulusCreatorForm, RegionStimulusCreator
from tvb.adapters.datatypes.db.patterns import StimuliRegionIndex
from tvb.adapters.datatypes.h5.patterns_h5 import StimuliRegionH5
from tvb.adapters.simulator.equation_forms import get_ui_name_to_equation_dict, get_form_for_equation, \
    get_ui_name_for_equation
from tvb.adapters.visualizers.connectivity import ConnectivityViewer
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.file.simulator.configurations_h5 import SimulatorConfigurationH5
from tvb.core.entities.storage import dao
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import Form, DataTypeSelectField, SimpleStrField, prepare_prefixed_name_for_field
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.equations import Linear
from tvb.datatypes.patterns import StimuliRegion
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.decorators import handle_error, expose_page, expose_fragment, using_template, \
    check_user
from tvb.interfaces.web.controllers.spatial.base_spatio_temporal_controller import SpatioTemporalController
from tvb.interfaces.web.controllers.spatial.surface_model_parameters_controller import EquationPlotForm

LOAD_EXISTING_URL = '/spatial/stimulus/region/load_region_stimulus'
RELOAD_DEFAULT_PAGE_URL = '/spatial/stimulus/region/reset_region_stimulus'

KEY_REGION_STIMULUS = "stim-region"
KEY_REGION_STIMULUS_NAME = "stim-region-name"


class StimulusRegionFirstFragment(Form):
    include_next_button = True

    def __init__(self, project_id):
        super(StimulusRegionFirstFragment, self).__init__()
        self.project_id = project_id
        self.region_stimulus = DataTypeSelectField(StimuliRegionIndex, self, name='region_stimulus',
                                                   label='Load Region Stimulus')
        self.display_name = SimpleStrField(self, name='display_name', label='Display name')

    @using_template('spatial/spatial_fragment')
    def __str__(self):
        return {'form': self, 'next_action': '/spatial/stimulus/region/set_region_stimulus',
                'include_previous_button': False, 'include_next_button': self.include_next_button,
                'legend': 'Loaded stimulus'}


class RegionStimulusController(SpatioTemporalController):
    """
    Control layer for defining Stimulus entities on Regions.
    """
    PREFIX_TEMPORAL_EQUATION_PARAMS = '_' + RegionStimulusCreator.KEY_TEMPORAL

    def __init__(self):
        SpatioTemporalController.__init__(self)
        # if any field that starts with one of the following prefixes is changed than the equation chart will be redrawn
        self.fields_prefixes = [self.PREFIX_TEMPORAL_EQUATION_PARAMS, '_min_x', '_max_x']
        self.equation_choices = get_ui_name_to_equation_dict()

    @cherrypy.expose
    @using_template("spatial/spatial_fragment")
    @handle_error(redirect=False)
    @check_user
    def set_region_stimulus(self, **data):
        first_fragment = StimulusRegionFirstFragment(common.get_current_project().id)
        first_fragment.fill_from_post(data)

        if not first_fragment.region_stimulus.value:
            stimuli_region = StimuliRegion()
            dummy_connectivity = Connectivity()
            stimuli_region.connectivity = dummy_connectivity
            stimuli_region.temporal = Linear()
        else:
            stimuli_region = self._load_existent_region_stimuli(first_fragment.region_stimulus.value)

        common.add2session(KEY_REGION_STIMULUS, stimuli_region)
        common.add2session(KEY_REGION_STIMULUS_NAME, first_fragment.display_name.value)

        next_fragment = RegionStimulusCreatorForm(self.equation_choices, common.get_current_project().id)
        next_fragment.fill_from_trait(stimuli_region)
        return {'form': next_fragment, 'next_action': '/spatial/stimulus/region/set_equation',
                'previous_form_action_url': '/spatial/stimulus/region/set_region_stimulus',
                'include_previous_button': True, 'include_next_button': True, 'legend': 'Stimulus parameters'}

    @cherrypy.expose
    @using_template("spatial/spatial_fragment")
    @handle_error(redirect=False)
    @check_user
    def set_equation(self, **data):
        current_stimuli_region = common.get_from_session(KEY_REGION_STIMULUS)
        form = RegionStimulusCreatorForm(self.equation_choices, common.get_current_project().id)
        form.fill_from_post(data)

        form_connectivity_gid = uuid.UUID(form.connectivity.value)
        if current_stimuli_region.connectivity.gid != form_connectivity_gid:
            current_stimuli_region.connectivity.gid = form_connectivity_gid
            conn_idx = dao.get_datatype_by_gid(form.connectivity.value)
            current_stimuli_region.weight = current_stimuli_region.get_default_weights(conn_idx.number_of_regions)

        form_temporal_value = form.temporal.value
        if type(current_stimuli_region.temporal) != form_temporal_value:
            current_stimuli_region.temporal = form_temporal_value()

        next_form = get_form_for_equation(form.temporal.value)(prefix=self.PREFIX_TEMPORAL_EQUATION_PARAMS)
        next_form.fill_from_trait(current_stimuli_region.temporal)
        return {'form': next_form, 'previous_form_action_url': '/spatial/stimulus/region/set_equation',
                'include_next_button': False, 'include_previous_button': True}

    def step_1(self):
        """
        Generate the required template dictionary for the first step.
        """
        current_stimuli_region = common.get_from_session(KEY_REGION_STIMULUS)
        selected_stimulus_gid = current_stimuli_region.gid.hex
        left_side_form = StimulusRegionFirstFragment(common.get_current_project().id)
        left_side_form.region_stimulus.data = selected_stimulus_gid
        template_specification = dict(title="Spatio temporal - Region stimulus")
        template_specification['mainContent'] = 'spatial/stimulus_region_step1_main'
        template_specification['isSingleMode'] = True
        template_specification['existentEntitiesInputList'] = left_side_form
        template_specification['equationViewerUrl'] = '/spatial/stimulus/region/get_equation_chart'
        template_specification['fieldsPrefixes'] = json.dumps(self.fields_prefixes)
        template_specification['next_step_url'] = '/spatial/stimulus/region/step_1_submit'
        template_specification['anyScaling'] = 0
        template_specification = self._add_extra_fields_to_interface(template_specification)
        return self.fill_default_attributes(template_specification)

    def step_2(self):
        """
        Generate the required template dictionary for the second step.
        """
        current_region_stimulus = common.get_from_session(KEY_REGION_STIMULUS)
        left_side_form = StimulusRegionFirstFragment(common.get_current_project().id)
        left_side_form.region_stimulus.data = current_region_stimulus.gid.hex
        left_side_form.include_next_button = False

        template_specification = dict(title="Spatio temporal - Region stimulus")
        template_specification['mainContent'] = 'spatial/stimulus_region_step2_main'
        template_specification['next_step_url'] = '/spatial/stimulus/region/step_2_submit'
        template_specification['existentEntitiesInputList'] = left_side_form

        default_weights = current_region_stimulus.weight
        if len(default_weights) == 0:
            selected_connectivity = ABCAdapter.load_entity_by_gid(current_region_stimulus.connectivity.gid.hex)
            default_weights = StimuliRegion.get_default_weights(selected_connectivity.number_of_regions)

        template_specification['node_weights'] = json.dumps(default_weights.tolist())
        template_specification[common.KEY_PARAMETERS_CONFIG] = False
        template_specification.update(self.display_connectivity(current_region_stimulus.connectivity.gid.hex))
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

    @expose_page
    def step_1_submit(self, next_step, do_reset=0, **kwargs):
        """
        Any submit from the first step should be handled here. Update the context then
        go to the next step as required. In case a reset is needed create a clear context.
        """
        if int(do_reset) == 1:
            new_region_stimulus = StimuliRegion()
            common.add2session(KEY_REGION_STIMULUS, new_region_stimulus)
        else:
            stimuli_region = common.get_from_session(KEY_REGION_STIMULUS)
            last_fragment = get_form_for_equation(type(stimuli_region.temporal))(self.PREFIX_TEMPORAL_EQUATION_PARAMS)
            last_fragment.fill_from_post(kwargs)
            last_fragment.fill_trait(stimuli_region.temporal)
        return self.do_step(next_step)

    @expose_page
    def step_2_submit(self, next_step, **kwargs):
        """
        Any submit from the second step should be handled here. Update the context and then do 
        the next step as required.
        """
        context = common.get_from_session(KEY_REGION_STIMULUS)
        # context.equation_kwargs[DataTypeMetaData.KEY_TAG_1] = kwargs[DataTypeMetaData.KEY_TAG_1]
        return self.do_step(next_step, 2)

    @staticmethod
    def display_connectivity(connectivity_gid):
        """
        Generates the html for displaying the connectivity matrix.
        """
        connectivity = ABCAdapter.load_entity_by_gid(connectivity_gid)

        current_project = common.get_current_project()
        file_handler = FilesHelper()
        conn_path = file_handler.get_project_folder(current_project, str(connectivity.fk_from_operation))

        connectivity_viewer_params = ConnectivityViewer.get_connectivity_parameters(connectivity, conn_path)

        template_specification = dict()
        template_specification['isSingleMode'] = True
        template_specification.update(connectivity_viewer_params)
        return template_specification

    def _prepare_operation_params(self, stimului_region):
        params_dict = {RegionStimulusCreator.KEY_CONNECTIVITY: stimului_region.connectivity.gid.hex,
                       RegionStimulusCreator.KEY_WEIGHT: stimului_region.weight.tolist(),
                       RegionStimulusCreator.KEY_TEMPORAL: get_ui_name_for_equation(type(stimului_region.temporal))
                       }
        for param_key, param_val in stimului_region.temporal.parameters.items():
            param_full_key = prepare_prefixed_name_for_field(RegionStimulusCreator.KEY_TEMPORAL, param_key)
            params_dict.update({param_full_key: str(param_val)})
        return params_dict

    def create_stimulus(self):
        """
        Creates a stimulus from the given data.
        """
        current_stimulus_region = common.get_from_session(KEY_REGION_STIMULUS)
        region_stimulus_creator = ABCAdapter.build_adapter_from_class(RegionStimulusCreator)
        params_dict = self._prepare_operation_params(current_stimulus_region)
        self.flow_service.fire_operation(region_stimulus_creator, common.get_logged_user(),
                                         common.get_current_project().id, **params_dict)
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
            min_x, max_x, ui_message = self.get_x_axis_range(form_data['_min_x'], form_data['_max_x'])
            if '_temporal' not in form_data:
                current_stimuli_region = common.get_from_session(KEY_REGION_STIMULUS)
            series_data, display_ui_message = current_stimuli_region.spatial.get_series_data(min_range=min_x,
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
        stimuli_region = StimuliRegion()

        stimuli_region_path = h5.path_for_stored_index(existent_region_stimulus_index)
        with StimuliRegionH5(stimuli_region_path) as stimuli_region_h5:
            stimuli_region_h5.load_into(stimuli_region)
            temporal_gid = stimuli_region_h5.temporal.load()

        equation_h5 = SimulatorConfigurationH5(stimuli_region_path)
        stimuli_region.temporal = equation_h5.load_from_reference(temporal_gid)

        dummy_connectivity = Connectivity()
        dummy_connectivity.gid = uuid.UUID(existent_region_stimulus_index.connectivity_gid)
        stimuli_region.connectivity = dummy_connectivity
        return stimuli_region

    @expose_page
    def load_region_stimulus(self, region_stimulus_gid, from_step=None):
        """
        Loads the interface for the selected region stimulus.
        """
        stimuli_region = self._load_existent_region_stimuli(region_stimulus_gid)
        common.add2session(KEY_REGION_STIMULUS, stimuli_region)
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
        default_stimuli_region = StimuliRegion()
        common.add2session(KEY_REGION_STIMULUS, default_stimuli_region)
        return self.do_step(1)

    @staticmethod
    def _add_extra_fields_to_interface(input_list):
        """
        The fields that have to be added to the existent
        adapter interface should be added in this method.
        """

        class TemporalPlotForm(EquationPlotForm):
            def __init__(self):
                super(TemporalPlotForm, self).__init__()
                self.min_x.label = 'Temporal Start Time(ms)'
                self.min_x.doc = "The minimum value of the x-axis for temporal equation plot. " \
                                 "Not persisted, used only for visualization."
                self.max_x.label = 'Temporal Start Time(ms)'
                self.max_x.doc = "The maximum value of the x-axis for temporal equation plot. " \
                                 "Not persisted, used only for visualization."

        temporal_plot_list_form = TemporalPlotForm()
        input_list['temporalPlotInputList'] = temporal_plot_list_form
        return input_list

    def fill_default_attributes(self, template_dictionary):
        """
        Overwrite base controller to add required parameters for adapter templates.
        """
        # context = common.get_from_session(KEY_REGION_CONTEXT)
        # default = context.equation_kwargs.get(DataTypeMetaData.KEY_TAG_1, '')
        template_dictionary["entitiySavedName"] = [{'name': DataTypeMetaData.KEY_TAG_1, "disabled": "False",
                                                    'label': 'Display name', 'type': 'str', "default": ''}]
        template_dictionary['loadExistentEntityUrl'] = LOAD_EXISTING_URL
        template_dictionary['resetToDefaultUrl'] = RELOAD_DEFAULT_PAGE_URL
        msg, msg_type = common.get_message_from_session()
        template_dictionary['displayedMessage'] = msg
        template_dictionary['messageType'] = msg_type
        return SpatioTemporalController.fill_default_attributes(self, template_dictionary, subsection='regionstim')
