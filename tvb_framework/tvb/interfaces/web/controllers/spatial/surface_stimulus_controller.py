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
.. moduleauthor:: Paula Popa <paula.popa@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""

import json
import cherrypy
import numpy

from tvb.adapters.creators.stimulus_creator import *
from tvb.adapters.datatypes.h5.patterns_h5 import StimuliSurfaceH5
from tvb.adapters.forms.equation_forms import get_form_for_equation
from tvb.adapters.forms.equation_plot_forms import EquationTemporalPlotForm, EquationSpatialPlotForm
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.load import try_get_last_datatype, load_entity_by_gid
from tvb.core.neocom import h5
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.autologging import traced
from tvb.interfaces.web.controllers.decorators import expose_page, expose_json, expose_fragment
from tvb.interfaces.web.controllers.spatial.base_spatio_temporal_controller import SpatioTemporalController

LOAD_EXISTING_URL = SpatioTemporalController.build_path('/spatial/stimulus/surface/load_surface_stimulus')
RELOAD_DEFAULT_PAGE_URL = SpatioTemporalController.build_path('/spatial/stimulus/surface/reload_default')
CHUNK_SIZE = 20

KEY_TMP_FORM = "temporal-form"


@traced
class SurfaceStimulusController(SpatioTemporalController):
    """
    Control layer for defining Stimulus entities on a cortical surface.
    """
    # These 4 strings are used on client-side to set onchange events on form fields
    SURFACE_FIELD = 'set_surface'
    SPATIAL_FIELD = 'set_spatial'
    TEMPORAL_FIELD = 'set_temporal'
    DISPLAY_NAME_FIELD = 'set_display_name'
    SPATIAL_PARAMS_FIELD = 'set_spatial_param'
    TEMPORAL_PARAMS_FIELD = 'set_temporal_param'
    base_url = '/spatial/stimulus/surface'

    @cherrypy.expose
    def set_surface(self, **param):
        current_surface_stim = common.get_from_session(KEY_SURFACE_STIMULUS)
        surface_form_field = SurfaceStimulusCreatorForm().surface
        surface_form_field.fill_from_post(param)
        current_surface_stim.surface = surface_form_field.value
        self._reset_focal_points(current_surface_stim)

    @cherrypy.expose
    def set_spatial_param(self, **param):
        current_surface_stim = common.get_from_session(KEY_SURFACE_STIMULUS)
        eq_param_form_class = get_form_for_equation(type(current_surface_stim.spatial))
        eq_param_form = eq_param_form_class()
        eq_param_form.fill_from_trait(current_surface_stim.spatial)
        eq_param_form.fill_from_post(param)
        eq_param_form.fill_trait(current_surface_stim.spatial)

    @cherrypy.expose
    def set_temporal_param(self, **param):
        current_surface_stim = common.get_from_session(KEY_SURFACE_STIMULUS)
        eq_param_form_class = get_form_for_equation(type(current_surface_stim.temporal))
        eq_param_form = eq_param_form_class()
        eq_param_form.fill_from_trait(current_surface_stim.temporal)
        eq_param_form.fill_from_post(param)
        eq_param_form.fill_trait(current_surface_stim.temporal)

    @cherrypy.expose
    def set_display_name(self, **param):
        display_name_form_field = StimulusSurfaceSelectorForm().display_name
        display_name_form_field.fill_from_post(param)
        if display_name_form_field.value is not None:
            current_surface_stim = common.get_from_session(KEY_SURFACE_STIMULUS)
            current_surface_stim.display_name = display_name_form_field.value

    def step_1(self):
        """
        Used for generating the interface which allows the user to define a stimulus.
        """
        current_surface_stim = common.get_from_session(KEY_SURFACE_STIMULUS)
        project_id = common.get_current_project().id
        surface_stim_selector_form = self.algorithm_service.prepare_adapter_form(
            form_instance=StimulusSurfaceSelectorForm(), project_id=common.get_current_project().id)
        surface_stim_selector_form.surface_stimulus.data = current_surface_stim.gid.hex
        surface_stim_creator_form = self.algorithm_service.prepare_adapter_form(
            form_instance=SurfaceStimulusCreatorForm(), project_id=common.get_current_project().id)
        if not hasattr(current_surface_stim, 'surface') or not current_surface_stim.surface:
            default_surface_index = try_get_last_datatype(project_id, SurfaceIndex,
                                                          SurfaceStimulusCreatorForm.get_filters())
            if default_surface_index is None:
                common.set_error_message(self.MSG_MISSING_SURFACE)
                current_surface_stim.surface = uuid.uuid4()
            else:
                current_surface_stim.surface = uuid.UUID(default_surface_index.gid)
        surface_stim_creator_form.fill_from_trait(current_surface_stim)
        surface_stim_selector_form.display_name.data = current_surface_stim.display_name

        template_specification = dict(title="Spatio temporal - Surface stimulus")
        template_specification['surfaceStimulusSelectForm'] = self.render_spatial_form(surface_stim_selector_form)
        template_specification['surfaceStimulusCreateForm'] = self.render_spatial_form(surface_stim_creator_form)
        self.plotted_equation_prefixes = {
            self.SURFACE_FIELD: surface_stim_creator_form.surface.name,
            self.DISPLAY_NAME_FIELD: surface_stim_selector_form.display_name.name
        }
        template_specification['mainContent'] = 'spatial/stimulus_surface_step1_main'
        template_specification['baseUrl'] = self.base_url
        template_specification['spatialFieldsPrefixes'] = json.dumps(self.plotted_equation_prefixes)
        template_specification['next_step_url'] = self.build_path('/spatial/stimulus/surface/step_1_submit')
        template_specification['definedFocalPoints'] = current_surface_stim.focal_points_triangles.tolist()
        template_specification = self._add_extra_fields_to_interface(template_specification)
        return self.fill_default_attributes(template_specification)

    def step_2(self):
        """
        Used for generating the interface which allows the user to define a stimulus.
        """
        current_surface_stim = common.get_from_session(KEY_SURFACE_STIMULUS)
        template_specification = dict(title="Spatio temporal - Surface stimulus")
        surface_stim_selector_form = self.algorithm_service.prepare_adapter_form(
            form_instance=StimulusSurfaceSelectorForm(), project_id=common.get_current_project().id)
        surface_stim_selector_form.display_name.data = current_surface_stim.display_name
        surface_stim_selector_form.surface_stimulus.data = current_surface_stim.gid.hex
        template_specification['surfaceStimulusSelectForm'] = self.render_adapter_form(surface_stim_selector_form)
        template_specification['mainContent'] = 'spatial/stimulus_surface_step2_main'
        template_specification['next_step_url'] = self.build_path('/spatial/stimulus/surface/step_2_submit')
        template_specification['loadExistentEntityUrl'] = LOAD_EXISTING_URL
        template_specification['resetToDefaultUrl'] = RELOAD_DEFAULT_PAGE_URL
        template_specification['surfaceGID'] = current_surface_stim.surface.hex
        template_specification[common.KEY_PARAMETERS_CONFIG] = False
        template_specification['definedFocalPoints'] = current_surface_stim.focal_points_triangles.tolist()
        plotted_equation_prefixes = {
            self.DISPLAY_NAME_FIELD: surface_stim_selector_form.display_name.name
        }
        template_specification['baseUrl'] = self.base_url
        template_specification['spatialFieldsPrefixes'] = json.dumps(plotted_equation_prefixes)
        template_specification.update(self.display_surface(current_surface_stim.surface.hex))
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

    def _reset_focal_points(self, surface_stimuli):
        surface_stimuli.focal_points_triangles = numpy.array([], dtype=int)

    def _reset_session_stimuli(self):
        new_surface_stim = SurfaceStimulusCreatorModel()
        new_surface_stim.temporal = SurfaceStimulusCreatorForm.default_temporal.instance
        new_surface_stim.spatial = SurfaceStimulusCreatorForm.default_spatial.instance
        self._reset_focal_points(new_surface_stim)
        common.add2session(KEY_SURFACE_STIMULUS, new_surface_stim)
        common.add2session(KEY_TMP_FORM, EquationTemporalPlotForm())

    @expose_page
    def step_1_submit(self, next_step, do_reset=0, **kwargs):
        """
        Any submit from the first step should be handled here. Update the context then
        go to the next step as required. In case a reset is needed create a clear context.
        """
        if int(do_reset) == 1:
            self._reset_session_stimuli()
        return self.do_step(next_step)

    @expose_page
    def step_2_submit(self, next_step, **kwargs):
        """
        Any submit from the second step should be handled here. Update the context and then do 
        the next step as required.
        """
        current_surface_stim = common.get_from_session(KEY_SURFACE_STIMULUS)
        submited_focal_points = kwargs['defined_focal_points']
        current_surface_stim.focal_points_triangles = numpy.array(json.loads(submited_focal_points))
        return self.do_step(next_step, 2)

    def create_stimulus(self):
        """
        Creates a stimulus from the given data.
        """
        try:
            current_surface_stim = common.get_from_session(KEY_SURFACE_STIMULUS)
            surface_stimulus_creator = ABCAdapter.build_adapter_from_class(SurfaceStimulusCreator)
            self.operation_service.fire_operation(surface_stimulus_creator, common.get_logged_user(),
                                                  common.get_current_project().id, view_model=current_surface_stim)
            common.set_important_message("The operation for creating the stimulus was successfully launched.")

        except (NameError, ValueError, SyntaxError):
            common.set_error_message("The operation failed due to invalid parameter input.")
            return False
        except Exception as ex:
            common.set_error_message(ex)
            return False
        return True

    @expose_page
    def load_surface_stimulus(self, surface_stimulus_gid, from_step):
        """
        Loads the interface for the selected surface stimulus.
        """
        surface_stim_index = load_entity_by_gid(surface_stimulus_gid)
        surface_stim_h5_path = h5.path_for_stored_index(surface_stim_index)
        existent_surface_stim = SurfaceStimulusCreatorModel()
        with StimuliSurfaceH5(surface_stim_h5_path) as surface_stim_h5:
            surface_stim_h5.load_into(existent_surface_stim)

        existent_surface_stim.surface = uuid.UUID(surface_stim_index.fk_surface_gid)
        existent_surface_stim.display_name = surface_stim_index.user_tag_1

        common.add2session(KEY_SURFACE_STIMULUS, existent_surface_stim)
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
        self._reset_session_stimuli()
        return self.do_step(1)

    @expose_json
    def view_stimulus(self, focal_points):
        """
        Just create the stimulus to view the actual data, don't store to db.
        Hold the entity in session without the surface, so the next time you need
        data just get from that one.
        """
        try:
            current_surface_stim = common.get_from_session(KEY_SURFACE_STIMULUS)
            current_surface_stim.focal_points_triangles = numpy.array(json.loads(focal_points))
            min_time = common.get_from_session(KEY_TMP_FORM).min_tmp_x.value or 0
            max_time = common.get_from_session(KEY_TMP_FORM).max_tmp_x.value or 100

            stimuli_surface = SurfaceStimulusCreator().prepare_stimuli_surface_from_view_model(current_surface_stim, True)
            stimuli_surface.configure_space()
            time = numpy.arange(min_time, max_time, 1)
            time = time[numpy.newaxis, :]
            stimuli_surface.configure_time(time)
            current_surface_stim._temporal_pattern = stimuli_surface.temporal_pattern
            current_surface_stim._spatial_pattern = stimuli_surface.spatial_pattern

            data = []
            max_value = numpy.max(stimuli_surface())
            min_value = numpy.min(stimuli_surface())
            for i in range(min(CHUNK_SIZE, stimuli_surface.temporal_pattern.shape[1])):
                step_data = stimuli_surface(i).tolist()
                data.append(step_data)
            result = {'status': 'ok', 'max': max_value, 'min': min_value,
                      'data': data, "time_min": min_time, "time_max": max_time, "chunk_size": CHUNK_SIZE}
            return result
        except (NameError, ValueError, SyntaxError):
            return {'status': 'error',
                    'errorMsg': "Could not generate stimulus data. Some of the parameters hold invalid characters."}
        except Exception as ex:
            return {'allSeries': 'error', 'errorMsg': ex}

    def fill_default_attributes(self, template_specification):
        """
        Add some entries that are used in both steps then fill the default required attributes.
        """
        template_specification['loadExistentEntityUrl'] = LOAD_EXISTING_URL
        template_specification['resetToDefaultUrl'] = RELOAD_DEFAULT_PAGE_URL
        return super(SurfaceStimulusController, self).fill_default_attributes(template_specification,
                                                                              subsection='surfacestim')

    @expose_json
    def get_stimulus_chunk(self, chunk_idx):
        """
        Get the next chunk of the stimulus data.
        """
        stimulus = common.get_from_session(KEY_SURFACE_STIMULUS)
        chunk_idx = int(chunk_idx)
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
            temporal_form = EquationTemporalPlotForm()
            if form_data:
                temporal_form.fill_from_post(form_data)
                common.add2session(KEY_TMP_FORM, temporal_form)

            min_x, max_x, ui_message = self.get_x_axis_range(temporal_form.min_tmp_x.value,
                                                             temporal_form.max_tmp_x.value)
            equation = common.get_from_session(KEY_SURFACE_STIMULUS).temporal
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
            return {'allSeries': None, 'errorMsg': ex}

    @expose_fragment('spatial/equation_displayer')
    def get_spatial_equation_chart(self, **form_data):
        """
        Returns the HTML which contains the chart in which is plotted the spatial equation.
        """
        try:
            spatial_form = EquationSpatialPlotForm()
            if form_data:
                spatial_form.fill_from_post(form_data)

            min_x, max_x, ui_message = self.get_x_axis_range(spatial_form.min_space_x.value,
                                                             spatial_form.max_space_x.value)
            equation = common.get_from_session(KEY_SURFACE_STIMULUS).spatial
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

    def _add_extra_fields_to_interface(self, input_list):
        """
        The fields that have to be added to the existent
        adapter interface should be added in this method.
        """
        input_list['spatialPlotInputList'] = self.render_adapter_form(EquationSpatialPlotForm())
        input_list['temporalPlotInputList'] = self.render_adapter_form(EquationTemporalPlotForm())
        return input_list
