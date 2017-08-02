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
import cherrypy
from tvb.adapters.visualizers.connectivity import ConnectivityViewer
from tvb.basic.traits.parameters_factory import collapse_params
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.adapters.input_tree import InputTreeManager
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.entities.transient.context_stimulus import RegionStimulusContext
from tvb.core.entities.transient.context_stimulus import SCALING_PARAMETER, CONNECTIVITY_PARAMETER
from tvb.datatypes.equations import Equation
from tvb.datatypes.patterns import StimuliRegion
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.decorators import handle_error, expose_page, expose_fragment
from tvb.interfaces.web.controllers.spatial.base_spatio_temporal_controller import SpatioTemporalController


REGION_STIMULUS_CREATOR_MODULE = "tvb.adapters.creators.stimulus_creator"
REGION_STIMULUS_CREATOR_CLASS = "RegionStimulusCreator"

LOAD_EXISTING_URL = '/spatial/stimulus/region/load_region_stimulus'
RELOAD_DEFAULT_PAGE_URL = '/spatial/stimulus/region/reset_region_stimulus'

KEY_REGION_CONTEXT = "stim-region-ctx"



class RegionStimulusController(SpatioTemporalController):
    """
    Control layer for defining Stimulus entities on Regions.
    """

    def __init__(self):
        SpatioTemporalController.__init__(self)
        #if any field that starts with one of the following prefixes is changed than the equation chart will be redrawn
        self.fields_prefixes = ['temporal', 'min_x', 'max_x']


    def step_1(self):
        """
        Generate the required template dictionary for the first step.
        """
        context = common.get_from_session(KEY_REGION_CONTEXT)
        right_side_interface, any_scaling = self._get_stimulus_interface()
        selected_stimulus_gid = context.selected_stimulus
        left_side_interface = self.get_select_existent_entities('Load Region Stimulus:',
                                                                StimuliRegion, selected_stimulus_gid)
        #add interface to session, needed for filters
        self.add_interface_to_session(left_side_interface, right_side_interface['inputList'])
        template_specification = dict(title="Spatio temporal - Region stimulus")
        template_specification['mainContent'] = 'spatial/stimulus_region_step1_main'
        template_specification['isSingleMode'] = True
        template_specification.update(right_side_interface)
        template_specification['existentEntitiesInputList'] = left_side_interface
        template_specification['equationViewerUrl'] = '/spatial/stimulus/region/get_equation_chart'
        template_specification['fieldsPrefixes'] = json.dumps(self.fields_prefixes)
        template_specification['next_step_url'] = '/spatial/stimulus/region/step_1_submit'
        template_specification['anyScaling'] = any_scaling
        return self.fill_default_attributes(template_specification)


    def step_2(self):
        """
        Generate the required template dictionary for the second step.
        """
        context = common.get_from_session(KEY_REGION_CONTEXT)
        selected_stimulus_gid = context.selected_stimulus
        left_side_interface = self.get_select_existent_entities('Load Region Stimulus:',
                                                                StimuliRegion, selected_stimulus_gid)
        template_specification = dict(title="Spatio temporal - Region stimulus")
        template_specification['mainContent'] = 'spatial/stimulus_region_step2_main'
        template_specification['next_step_url'] = '/spatial/stimulus/region/step_2_submit'
        template_specification['existentEntitiesInputList'] = left_side_interface
        default_weights = context.get_weights()
        if len(default_weights) == 0:
            selected_connectivity = ABCAdapter.load_entity_by_gid(context.get_session_connectivity())
            default_weights = StimuliRegion.get_default_weights(selected_connectivity.number_of_regions)
        template_specification['node_weights'] = json.dumps(default_weights)
        template_specification[common.KEY_PARAMETERS_CONFIG] = False
        template_specification.update(self.display_connectivity(context.get_session_connectivity()))
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
            new_context = RegionStimulusContext()
            common.add2session(KEY_REGION_CONTEXT, new_context)
        context = common.get_from_session(KEY_REGION_CONTEXT)
        if kwargs.get(CONNECTIVITY_PARAMETER) != context.get_session_connectivity():
            context.set_weights([])
        context.equation_kwargs = kwargs
        return self.do_step(next_step)


    @expose_page
    def step_2_submit(self, next_step, **kwargs):
        """
        Any submit from the second step should be handled here. Update the context and then do 
        the next step as required.
        """
        context = common.get_from_session(KEY_REGION_CONTEXT)
        context.equation_kwargs[DataTypeMetaData.KEY_TAG_1] = kwargs[DataTypeMetaData.KEY_TAG_1]
        return self.do_step(next_step, 2)


    @staticmethod
    def display_connectivity(connectivity_gid):
        """
        Generates the html for displaying the connectivity matrix.
        """
        connectivity = ABCAdapter.load_entity_by_gid(connectivity_gid)
        connectivity_viewer_params = ConnectivityViewer.get_connectivity_parameters(connectivity)

        template_specification = dict()
        template_specification['isSingleMode'] = True
        template_specification.update(connectivity_viewer_params)
        return template_specification


    def create_stimulus(self):
        """
        Creates a stimulus from the given data.
        """
        context = common.get_from_session(KEY_REGION_CONTEXT)
        local_connectivity_creator = self.get_creator_and_interface(REGION_STIMULUS_CREATOR_MODULE,
                                                                    REGION_STIMULUS_CREATOR_CLASS, StimuliRegion())[0]
        context.equation_kwargs.update({'weight': json.dumps(context.get_weights())})
        self.flow_service.fire_operation(local_connectivity_creator, common.get_logged_user(),
                                         common.get_current_project().id, **context.equation_kwargs)
        common.set_important_message("The operation for creating the stimulus was successfully launched.")


    def _get_stimulus_interface(self):
        """
        Returns a dictionary which contains the data needed
        for creating the interface for a stimulus.
        """
        context = common.get_from_session(KEY_REGION_CONTEXT)
        input_list = self.get_creator_and_interface(REGION_STIMULUS_CREATOR_MODULE,
                                                    REGION_STIMULUS_CREATOR_CLASS, StimuliRegion())[1]
        context.equation_kwargs.update({SCALING_PARAMETER: context.get_weights()})
        input_list = InputTreeManager.fill_defaults(input_list, context.equation_kwargs)
        input_list, any_scaling = self._remove_scaling(input_list)
        template_specification = {'inputList': input_list, common.KEY_PARAMETERS_CONFIG: False}
        return self._add_extra_fields_to_interface(template_specification), any_scaling


    @staticmethod
    def _remove_scaling(input_list):
        """
        Remove the scaling entry from the UI since we no longer use them in the first step.
        """
        result = []
        any_scaling = False
        for entry in input_list:
            if entry[ABCAdapter.KEY_NAME] != SCALING_PARAMETER:
                result.append(entry)
            if entry[ABCAdapter.KEY_NAME] == SCALING_PARAMETER:
                scaling_values = entry[ABCAdapter.KEY_DEFAULT]
                for scaling in scaling_values:
                    if float(scaling) > 0:
                        any_scaling = True
                        break
        return result, any_scaling


    @cherrypy.expose
    @handle_error(redirect=False)
    def update_scaling(self, **kwargs):
        """
        Update the scaling according to the UI.
        """
        context = common.get_from_session(KEY_REGION_CONTEXT)
        try:
            scaling = json.loads(kwargs['scaling'])
            context.set_weights(scaling)
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
            form_data = collapse_params(form_data, ['temporal'])
            min_x, max_x, ui_message = self.get_x_axis_range(form_data['min_x'], form_data['max_x'])
            equation = Equation.build_equation_from_dict('temporal', form_data)
            series_data, display_ui_message = equation.get_series_data(min_range=min_x, max_range=max_x)
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
            return {'allSeries': None, 'errorMsg': ex.message}


    @expose_page
    def load_region_stimulus(self, region_stimulus_gid, from_step=None):
        """
        Loads the interface for the selected region stimulus.
        """
        selected_region_stimulus = ABCAdapter.load_entity_by_gid(region_stimulus_gid)
        temporal_eq = selected_region_stimulus.temporal
        spatial_eq = selected_region_stimulus.spatial
        connectivity = selected_region_stimulus.connectivity
        weights = selected_region_stimulus.weight

        temporal_eq_type = temporal_eq.__class__.__name__
        spatial_eq_type = spatial_eq.__class__.__name__
        default_dict = {'temporal': temporal_eq_type, 'spatial': spatial_eq_type,
                        'connectivity': connectivity.gid, 'weight': json.dumps(weights)}
        for param in temporal_eq.parameters:
            prepared_name = 'temporal_parameters_option_' + str(temporal_eq_type)
            prepared_name = prepared_name + '_parameters_parameters_' + str(param)
            default_dict[prepared_name] = str(temporal_eq.parameters[param])
        for param in spatial_eq.parameters:
            prepared_name = 'spatial_parameters_option_' + str(spatial_eq_type) + '_parameters_parameters_' + str(param)
            default_dict[prepared_name] = str(spatial_eq.parameters[param])

        input_list = self.get_creator_and_interface(REGION_STIMULUS_CREATOR_MODULE,
                                                    REGION_STIMULUS_CREATOR_CLASS, StimuliRegion())[1]
        input_list = InputTreeManager.fill_defaults(input_list, default_dict)
        context = common.get_from_session(KEY_REGION_CONTEXT)
        context.reset()
        context.update_from_interface(input_list)
        context.equation_kwargs[DataTypeMetaData.KEY_TAG_1] = selected_region_stimulus.user_tag_1
        context.set_active_stimulus(region_stimulus_gid)

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
        context = common.get_from_session(KEY_REGION_CONTEXT)
        context.reset()
        return self.do_step(1)


    @staticmethod
    def _add_extra_fields_to_interface(input_list):
        """
        The fields that have to be added to the existent
        adapter interface should be added in this method.
        """
        temporal_iface = []
        min_x = {'name': 'min_x', 'label': 'Temporal Start Time(ms)', 'type': 'str', "disabled": "False", "default": 0,
                 "description": "The minimum value of the x-axis for temporal equation plot. "
                                "Not persisted, used only for visualization."}
        max_x = {'name': 'max_x', 'label': 'Temporal End Time(ms)', 'type': 'str', "disabled": "False", "default": 100,
                 "description": "The maximum value of the x-axis for temporal equation plot. "
                                "Not persisted, used only for visualization."}
        temporal_iface.append(min_x)
        temporal_iface.append(max_x)

        input_list['temporalPlotInputList'] = temporal_iface
        return input_list


    def fill_default_attributes(self, template_dictionary):
        """
        Overwrite base controller to add required parameters for adapter templates.
        """
        context = common.get_from_session(KEY_REGION_CONTEXT)
        default = context.equation_kwargs.get(DataTypeMetaData.KEY_TAG_1, '')
        template_dictionary["entitiySavedName"] = [{'name': DataTypeMetaData.KEY_TAG_1, "disabled": "False",
                                                    'label': 'Display name', 'type': 'str', "default": default}]
        template_dictionary['loadExistentEntityUrl'] = LOAD_EXISTING_URL
        template_dictionary['resetToDefaultUrl'] = RELOAD_DEFAULT_PAGE_URL
        msg, msg_type = common.get_message_from_session()
        template_dictionary['displayedMessage'] = msg
        template_dictionary['messageType'] = msg_type
        return SpatioTemporalController.fill_default_attributes(self, template_dictionary, subsection='regionstim')
