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
import numpy
import copy

from tvb.core.adapters.input_tree import InputTreeManager
from tvb.datatypes.patterns import StimuliSurface
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.transient.context_stimulus import SurfaceStimulusContext, SURFACE_PARAMETER
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.decorators import expose_page, expose_json, expose_fragment
from tvb.interfaces.web.controllers.spatial.base_spatio_temporal_controller import SpatioTemporalController
from tvb.interfaces.web.controllers.spatial.base_spatio_temporal_controller import PARAM_SURFACE


SURFACE_STIMULUS_CREATOR_MODULE = "tvb.adapters.creators.stimulus_creator"
SURFACE_STIMULUS_CREATOR_CLASS = "SurfaceStimulusCreator"
KEY_STIMULUS = "session_stimulus"
LOAD_EXISTING_URL = '/spatial/stimulus/surface/load_surface_stimulus'
RELOAD_DEFAULT_PAGE_URL = '/spatial/stimulus/surface/reload_default'
CHUNK_SIZE = 20

KEY_SURFACE_CONTEXT = "stim-surface-ctx"


class SurfaceStimulusController(SpatioTemporalController):
    """
    Control layer for defining Stimulus entities on a cortical surface.
    """

    def __init__(self):
        SpatioTemporalController.__init__(self)
        #if any field that starts with one of the following prefixes is changed
        # than the temporal equation chart will be redrawn
        self.temporal_fields_prefixes = ['temporal', 'min_tmp_x', 'max_tmp_x']
        self.spatial_fields_prefixes = ['spatial', 'min_space_x', 'max_space_x']


    def step_1(self):
        """
        Used for generating the interface which allows the user to define a stimulus.
        """
        context = common.get_from_session(KEY_SURFACE_CONTEXT)
        right_side_interface = self._get_stimulus_interface()
        selected_stimulus_gid = context.get_selected_stimulus()
        left_side_interface = self.get_select_existent_entities('Load Surface Stimulus:', StimuliSurface,
                                                                selected_stimulus_gid)
        #add interface to session, needed for filters
        self.add_interface_to_session(left_side_interface, right_side_interface['inputList'])
        template_specification = dict(title="Spatio temporal - Surface stimulus")
        template_specification['existentEntitiesInputList'] = left_side_interface
        template_specification['mainContent'] = 'spatial/stimulus_surface_step1_main'
        template_specification.update(right_side_interface)
        template_specification['temporalEquationViewerUrl'] = '/spatial/stimulus/surface/get_temporal_equation_chart'
        template_specification['temporalFieldsPrefixes'] = json.dumps(self.temporal_fields_prefixes)
        template_specification['spatialEquationViewerUrl'] = '/spatial/stimulus/surface/get_spatial_equation_chart'
        template_specification['spatialFieldsPrefixes'] = json.dumps(self.spatial_fields_prefixes)
        template_specification['next_step_url'] = '/spatial/stimulus/surface/step_1_submit'
        return self.fill_default_attributes(template_specification)


    def step_2(self):
        """
        Used for generating the interface which allows the user to define a stimulus.
        """
        context = common.get_from_session(KEY_SURFACE_CONTEXT)
        selected_stimulus_gid = context.get_selected_stimulus()
        left_side_interface = self.get_select_existent_entities('Load Surface Stimulus:', StimuliSurface,
                                                                selected_stimulus_gid)
        template_specification = dict(title="Spatio temporal - Surface stimulus")
        template_specification['existentEntitiesInputList'] = left_side_interface
        template_specification['mainContent'] = 'spatial/stimulus_surface_step2_main'
        template_specification['next_step_url'] = '/spatial/stimulus/surface/step_2_submit'
        template_specification['loadExistentEntityUrl'] = LOAD_EXISTING_URL
        template_specification['resetToDefaultUrl'] = RELOAD_DEFAULT_PAGE_URL
        template_specification['surfaceGID'] = context.get_session_surface()
        template_specification[common.KEY_PARAMETERS_CONFIG] = False
        template_specification['definedFocalPoints'] = json.dumps(context.focal_points_list)
        template_specification.update(self.display_surface(context.get_session_surface()))
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
            if self.create_stimulus():
                common.set_info_message("Successfully created a new stimulus.")
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
            new_context = SurfaceStimulusContext()
            common.add2session(KEY_SURFACE_CONTEXT, new_context)
        context = common.get_from_session(KEY_SURFACE_CONTEXT)
        if kwargs.get(SURFACE_PARAMETER) != context.get_session_surface():
            context.set_focal_points('[]')
        context.update_eq_kwargs(kwargs)
        return self.do_step(next_step)


    @expose_page
    def step_2_submit(self, next_step, **kwargs):
        """
        Any submit from the second step should be handled here. Update the context and then do 
        the next step as required.
        """
        context = common.get_from_session(KEY_SURFACE_CONTEXT)
        submited_focal_points = kwargs['defined_focal_points']
        context.set_focal_points(submited_focal_points)
        context.equation_kwargs[DataTypeMetaData.KEY_TAG_1] = kwargs[DataTypeMetaData.KEY_TAG_1]
        return self.do_step(next_step, 2)


    def create_stimulus(self):
        """
        Creates a stimulus from the given data.
        """
        try:
            context = common.get_from_session(KEY_SURFACE_CONTEXT)
            surface_stimulus_creator = self.get_creator_and_interface(SURFACE_STIMULUS_CREATOR_MODULE,
                                                                      SURFACE_STIMULUS_CREATOR_CLASS,
                                                                      StimuliSurface())[0]
            self.flow_service.fire_operation(surface_stimulus_creator, common.get_logged_user(),
                                             common.get_current_project().id, **context.equation_kwargs)
            common.set_important_message("The operation for creating the stimulus was successfully launched.")
            context.selected_stimulus = None

        except (NameError, ValueError, SyntaxError):
            common.set_error_message("The operation failed due to invalid parameter input.")
            return False
        except Exception as ex:
            common.set_error_message(ex.message)
            return False
        return True


    @expose_page
    def load_surface_stimulus(self, surface_stimulus_gid, from_step):
        """
        Loads the interface for the selected surface stimulus.
        """
        context = common.get_from_session(KEY_SURFACE_CONTEXT)
        selected_surface_stimulus = ABCAdapter.load_entity_by_gid(surface_stimulus_gid)
        temporal_eq = selected_surface_stimulus.temporal
        spatial_eq = selected_surface_stimulus.spatial
        surface = selected_surface_stimulus.surface
        focal_points_surface = selected_surface_stimulus.focal_points_surface
        focal_points_triangles = selected_surface_stimulus.focal_points_triangles

        temporal_eq_type = temporal_eq.__class__.__name__
        spatial_eq_type = spatial_eq.__class__.__name__
        default_dict = {'temporal': temporal_eq_type, 'spatial': spatial_eq_type,
                        'surface': surface.gid, 'focal_points_surface': json.dumps(focal_points_surface),
                        'focal_points_triangles': json.dumps(focal_points_triangles)}
        for param in temporal_eq.parameters:
            prepared_name = 'temporal_parameters_option_' + str(temporal_eq_type)
            prepared_name = prepared_name + '_parameters_parameters_' + str(param)
            default_dict[prepared_name] = str(temporal_eq.parameters[param])
        for param in spatial_eq.parameters:
            prepared_name = 'spatial_parameters_option_' + str(spatial_eq_type) + '_parameters_parameters_' + str(param)
            default_dict[prepared_name] = str(spatial_eq.parameters[param])

        default_dict[DataTypeMetaData.KEY_TAG_1] = selected_surface_stimulus.user_tag_1

        input_list = self.get_creator_and_interface(SURFACE_STIMULUS_CREATOR_MODULE,
                                                    SURFACE_STIMULUS_CREATOR_CLASS, StimuliSurface(),
                                                    lock_midpoint_for_eq=[1])[1]
        input_list = InputTreeManager.fill_defaults(input_list, default_dict)
        context.reset()
        context.update_from_interface(input_list)
        context.equation_kwargs[DataTypeMetaData.KEY_TAG_1] = selected_surface_stimulus.user_tag_1
        context.set_active_stimulus(surface_stimulus_gid)
        return self.do_step(from_step)


    @expose_page
    def reload_default(self, from_step):
        """
        Just reload default data as if stimulus is None. 
        
        from_step:
            not actually used here since when the user selects None
            from the stimulus entities select we want to take him back to step 1
            always. Kept just for compatibility with the normal load entity of a 
            stimulus where we want to stay in the same page.

        """
        context = common.get_from_session(KEY_SURFACE_CONTEXT)
        context.reset()
        return self.do_step(1)


    @expose_json
    def view_stimulus(self, focal_points):
        """
        Just create the stimulus to view the actual data, don't store to db.
        Hold the entity in session without the surface, so the next time you need
        data just get from that one.
        """
        try:
            context = common.get_from_session(KEY_SURFACE_CONTEXT)
            context.set_focal_points(focal_points)
            kwargs = copy.deepcopy(context.equation_kwargs)
            surface_stimulus_creator = self.get_creator_and_interface(SURFACE_STIMULUS_CREATOR_MODULE,
                                                                      SURFACE_STIMULUS_CREATOR_CLASS,
                                                                      StimuliSurface())[0]
            min_time = float(kwargs.get('min_tmp_x', 0))
            max_time = float(kwargs.get('max_tmp_x', 100))
            kwargs = surface_stimulus_creator.prepare_ui_inputs(kwargs)
            stimulus = surface_stimulus_creator.launch(**kwargs)
            surface_gid = common.get_from_session(PARAM_SURFACE)
            surface = ABCAdapter.load_entity_by_gid(surface_gid)
            stimulus.surface = surface
            stimulus.configure_space()
            time = numpy.arange(min_time, max_time, 1)
            time = time[numpy.newaxis, :]
            stimulus.configure_time(time)
            data = []
            max_value = numpy.max(stimulus())
            min_value = numpy.min(stimulus())
            for i in range(min(CHUNK_SIZE, stimulus.temporal_pattern.shape[1])):
                step_data = stimulus(i).tolist()
                data.append(step_data)
            stimulus.surface = surface.gid
            common.add2session(KEY_STIMULUS, stimulus)
            result = {'status': 'ok', 'max': max_value, 'min': min_value,
                      'data': data, "time_min": min_time, "time_max": max_time, "chunk_size": CHUNK_SIZE}
            return result
        except (NameError, ValueError, SyntaxError):
            return {'status': 'error',
                    'errorMsg': "Could not generate stimulus data. Some of the parameters hold invalid characters."}
        except Exception as ex:
            return {'allSeries': 'error', 'errorMsg': ex.message}


    def fill_default_attributes(self, template_specification):
        """
        Add some entries that are used in both steps then fill the default required attributes.
        """
        context = common.get_from_session(KEY_SURFACE_CONTEXT)
        template_specification["entitiySavedName"] = [
            {'name': DataTypeMetaData.KEY_TAG_1, 'label': 'Display name', 'type': 'str',
             "disabled": "False", "default": context.equation_kwargs.get(DataTypeMetaData.KEY_TAG_1, '')}]
        template_specification['loadExistentEntityUrl'] = LOAD_EXISTING_URL
        template_specification['resetToDefaultUrl'] = RELOAD_DEFAULT_PAGE_URL
        msg, msg_type = common.get_message_from_session()
        template_specification['displayedMessage'] = msg
        template_specification['messageType'] = msg_type
        return super(SurfaceStimulusController, self).fill_default_attributes(template_specification,
                                                                              subsection='surfacestim')


    @expose_json
    def get_stimulus_chunk(self, chunk_idx):
        """
        Get the next chunk of the stimulus data.
        """
        stimulus = common.get_from_session(KEY_STIMULUS)
        surface_gid = common.get_from_session(PARAM_SURFACE)
        chunk_idx = int(chunk_idx)
        if stimulus.surface.gid != surface_gid:
            raise Exception("TODO: Surface changed while visualizing stimulus. See how to handle this.")
        data = []
        for idx in range(chunk_idx * CHUNK_SIZE,
                         min((chunk_idx + 1) * CHUNK_SIZE, stimulus.temporal_pattern.shape[1]), 1):
            data.append(stimulus(idx).tolist())
        return data


    @expose_fragment('spatial/equation_displayer')
    def get_temporal_equation_chart(self, **form_data):
        """
        Returns the HTML which contains the chart in which is plotted the temporal equation.
        """
        try:
            min_x, max_x, ui_message = self.get_x_axis_range(form_data['min_tmp_x'], form_data['max_tmp_x'])
            surface_stimulus_creator = self.get_creator_and_interface(SURFACE_STIMULUS_CREATOR_MODULE,
                                                                      SURFACE_STIMULUS_CREATOR_CLASS,
                                                                      StimuliSurface())[0]
            form_data = surface_stimulus_creator.prepare_ui_inputs(form_data, validation_required=False)
            equation = surface_stimulus_creator.get_temporal_equation(form_data)
            series_data, display_ui_message = equation.get_series_data(min_range=min_x, max_range=max_x)
            all_series = self.get_series_json(series_data, "Temporal")

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


    @expose_fragment('spatial/equation_displayer')
    def get_spatial_equation_chart(self, **form_data):
        """
        Returns the HTML which contains the chart in which is plotted the temporal equation.
        """
        try:
            min_x, max_x, ui_message = self.get_x_axis_range(form_data['min_space_x'], form_data['max_space_x'])
            surface_stimulus_creator = self.get_creator_and_interface(SURFACE_STIMULUS_CREATOR_MODULE,
                                                                      SURFACE_STIMULUS_CREATOR_CLASS,
                                                                      StimuliSurface())[0]
            form_data = surface_stimulus_creator.prepare_ui_inputs(form_data, validation_required=False)
            equation = surface_stimulus_creator.get_spatial_equation(form_data)
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
            return {'allSeries': None, 'errorMsg': ex.message}


    def _get_stimulus_interface(self):
        """
        Returns a dictionary which contains the interface for a surface stimulus.
        """
        context = common.get_from_session(KEY_SURFACE_CONTEXT)
        input_list = self.get_creator_and_interface(SURFACE_STIMULUS_CREATOR_MODULE,
                                                    SURFACE_STIMULUS_CREATOR_CLASS, StimuliSurface(),
                                                    lock_midpoint_for_eq=[1])[1]
        input_list = InputTreeManager.fill_defaults(input_list, context.equation_kwargs)
        input_list, focal_points_list = self._remove_focal_points(input_list)
        input_list = self.prepare_entity_interface(input_list)
        input_list['selectedFocalPoints'] = focal_points_list
        return self._add_extra_fields_to_interface(input_list)


    @staticmethod
    def _remove_focal_points(input_list):
        """
        Remove the focal points entries from the UI since we no longer use them in the first step.
        """
        result = []
        focal_points_triangles = None
        for entry in input_list:
            if entry[ABCAdapter.KEY_NAME] not in ('focal_points_triangles', 'focal_points_surface'):
                result.append(entry)
            if entry[ABCAdapter.KEY_NAME] == 'focal_points_triangles' and len(entry[ABCAdapter.KEY_DEFAULT]):
                focal_points_triangles = json.loads(entry[ABCAdapter.KEY_DEFAULT])
        return result, focal_points_triangles


    @staticmethod
    def _add_extra_fields_to_interface(input_list):
        """
        The fields that have to be added to the existent
        adapter interface should be added in this method.
        """
        context = common.get_from_session(KEY_SURFACE_CONTEXT)

        temporal_iface = []
        min_tmp_x = {'name': 'min_tmp_x', 'label': 'Temporal Start Time(ms)', 'type': 'str', "disabled": "False",
                     "default": context.equation_kwargs.get('min_tmp_x', 0),
                     "description": "The minimum value of the x-axis for temporal equation plot. "
                                    "Not persisted, used only for visualization."}
        max_tmp_x = {'name': 'max_tmp_x', 'label': 'Temporal End Time(ms)', 'type': 'str', "disabled": "False",
                     "default": context.equation_kwargs.get('max_tmp_x', 100),
                     "description": "The maximum value of the x-axis for temporal equation plot. "
                                    "Not persisted, used only for visualization."}
        temporal_iface.append(min_tmp_x)
        temporal_iface.append(max_tmp_x)

        spatial_iface = []
        min_space_x = {'name': 'min_space_x', 'label': 'Spatial Start Distance(mm)', 'type': 'str', "disabled": "False",
                       "default": context.equation_kwargs.get('min_space_x', 0),
                       "description": "The minimum value of the x-axis for spatial equation plot."}
        max_space_x = {'name': 'max_space_x', 'label': 'Spatial End Distance(mm)', 'type': 'str', "disabled": "False",
                       "default": context.equation_kwargs.get('max_space_x', 100),
                       "description": "The maximum value of the x-axis for spatial equation plot."}
        spatial_iface.append(min_space_x)
        spatial_iface.append(max_space_x)

        input_list['spatialPlotInputList'] = spatial_iface
        input_list['temporalPlotInputList'] = temporal_iface
        return input_list
    
    