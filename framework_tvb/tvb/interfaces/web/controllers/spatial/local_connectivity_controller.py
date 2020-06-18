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
.. moduleauthor:: Paula Popa <paula.popa@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""

import json
import uuid
import cherrypy
from tvb.adapters.creators.local_connectivity_creator import *
from tvb.adapters.datatypes.h5.local_connectivity_h5 import LocalConnectivityH5
from tvb.adapters.datatypes.h5.surface_h5 import SurfaceH5
from tvb.adapters.simulator.equation_forms import get_form_for_equation
from tvb.adapters.simulator.subform_helper import SubformHelper
from tvb.adapters.simulator.subforms_mapping import get_ui_name_to_equation_dict, GAUSSIAN_EQUATION, \
    DOUBLE_GAUSSIAN_EQUATION, SIGMOID_EQUATION
from tvb.core.entities.load import try_get_last_datatype
from tvb.core.neocom import h5
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.autologging import traced
from tvb.interfaces.web.controllers.base_controller import BaseController
from tvb.interfaces.web.controllers.common import MissingDataException
from tvb.interfaces.web.controllers.decorators import check_user, handle_error, using_template
from tvb.interfaces.web.controllers.decorators import expose_fragment, expose_page, expose_json
from tvb.interfaces.web.controllers.spatial.base_spatio_temporal_controller import SpatioTemporalController

NO_OF_CUTOFF_POINTS = 20

LOAD_EXISTING_URL = '/spatial/localconnectivity/load_local_connectivity'
RELOAD_DEFAULT_PAGE_URL = '/spatial/localconnectivity/reset_local_connectivity'

# Between steps/pages we keep a LocalConnectivityCreatorModel in session at this key
KEY_LCONN = "local-conn"


@traced
class LocalConnectivityController(SpatioTemporalController):
    """
    Controller layer for displaying/creating a LocalConnectivity entity.
    """
    # These 4 strings are used on client-side to set onchange events on form fields
    SURFACE_FIELD = 'set_surface'
    EQUATION_FIELD = 'set_equation'
    CUTOFF_FIELD = 'set_cutoff_value'
    DISPLAY_NAME_FIELD = 'set_display_name'
    EQUATION_PARAMS_FIELD = 'set_equation_param'
    base_url = '/spatial/localconnectivity'

    def __init__(self):
        SpatioTemporalController.__init__(self)
        self.plotted_equation_prefixes = {}
        ui_name_to_equation_dict = get_ui_name_to_equation_dict()
        self.possible_equations = {GAUSSIAN_EQUATION: ui_name_to_equation_dict.get(GAUSSIAN_EQUATION),
                                   DOUBLE_GAUSSIAN_EQUATION: ui_name_to_equation_dict.get(DOUBLE_GAUSSIAN_EQUATION),
                                   SIGMOID_EQUATION: ui_name_to_equation_dict.get(SIGMOID_EQUATION)}

    @expose_page
    def step_1(self, do_reset=0, **kwargs):
        """
        Generate the html for the first step of the local connectivity page.
        :param do_reset: Boolean telling to start from empty page or not
        :param kwargs: not actually used, but parameters are still submitted from UI since we just\
               use the same js function for this.
        """
        project_id = common.get_current_project().id

        if int(do_reset) == 1:
            new_lconn = LocalConnectivityCreatorModel()
            default_surface_index = try_get_last_datatype(project_id, SurfaceIndex,
                                                          LocalConnectivityCreatorForm.get_filters())
            if default_surface_index:
                new_lconn.surface = uuid.UUID(default_surface_index.gid)
            else:
                # Surface is required in model and we should keep it like this, but we also want to
                new_lconn.surface = uuid.uuid4()
                common.set_error_message(self.MSG_MISSING_SURFACE)
            common.add2session(KEY_LCONN, new_lconn)

        current_lconn = common.get_from_session(KEY_LCONN)
        existent_lcon_form = LocalConnectivitySelectorForm(project_id=project_id)
        existent_lcon_form.existentEntitiesSelect.data = current_lconn.gid.hex
        configure_lcon_form = LocalConnectivityCreatorForm(self.possible_equations, project_id=project_id)
        configure_lcon_form.fill_from_trait(current_lconn)
        current_lconn.equation = configure_lcon_form.spatial.value()

        template_specification = dict(title="Surface - Local Connectivity")
        template_specification['mainContent'] = 'spatial/local_connectivity_step1_main'
        template_specification['inputList'] = self.render_spatial_form(configure_lcon_form)
        template_specification['displayCreateLocalConnectivityBtn'] = True
        template_specification['loadExistentEntityUrl'] = LOAD_EXISTING_URL
        template_specification['resetToDefaultUrl'] = RELOAD_DEFAULT_PAGE_URL
        template_specification['existentEntitiesInputList'] = self.render_spatial_form(existent_lcon_form)
        template_specification['submit_parameters_url'] = '/spatial/localconnectivity/create_local_connectivity'
        template_specification['equationViewerUrl'] = '/spatial/localconnectivity/get_equation_chart'
        template_specification['baseUrl'] = self.base_url

        self.plotted_equation_prefixes = {self.SURFACE_FIELD: configure_lcon_form.surface.name,
                                          self.EQUATION_FIELD: configure_lcon_form.spatial.name,
                                          self.CUTOFF_FIELD: configure_lcon_form.cutoff.name,
                                          self.DISPLAY_NAME_FIELD: configure_lcon_form.display_name.name,
                                          self.EQUATION_PARAMS_FIELD: configure_lcon_form.spatial.subform_field.name[1:]}

        template_specification['equationsPrefixes'] = json.dumps(self.plotted_equation_prefixes)
        template_specification['next_step_url'] = '/spatial/localconnectivity/step_2'
        return self.fill_default_attributes(template_specification)

    @cherrypy.expose
    @using_template('form_fields/form_field')
    @handle_error(redirect=False)
    @check_user
    def refresh_subform(self, equation, mapping_key):
        eq_class = get_ui_name_to_equation_dict().get(equation)
        current_lconn = common.get_from_session(KEY_LCONN)
        current_lconn.equation = eq_class()

        eq_params_form = SubformHelper.get_subform_for_field_value(equation, mapping_key)
        return {'adapter_form': eq_params_form, 'equationsPrefixes': self.plotted_equation_prefixes}

    @cherrypy.expose
    def set_equation_param(self, **param):
        current_lconn = common.get_from_session(KEY_LCONN)
        eq_param_form_class = get_form_for_equation(type(current_lconn.equation))
        eq_param_form = eq_param_form_class()
        eq_param_form.fill_from_trait(current_lconn.equation)
        eq_param_form.fill_from_post(param)
        eq_param_form.fill_trait(current_lconn.equation)

    @cherrypy.expose
    def set_cutoff_value(self, **param):
        current_lconn = common.get_from_session(KEY_LCONN)
        cutoff_form_field = LocalConnectivityCreatorForm(self.possible_equations).cutoff
        cutoff_form_field.fill_from_post(param)
        current_lconn.cutoff = cutoff_form_field.value

    @cherrypy.expose
    def set_surface(self, **param):
        current_lconn = common.get_from_session(KEY_LCONN)
        surface_form_field = LocalConnectivityCreatorForm(self.possible_equations).surface
        surface_form_field.fill_from_post(param)
        current_lconn.surface = surface_form_field.value

    @cherrypy.expose
    def set_display_name(self, **param):
        display_name_form_field = LocalConnectivityCreatorForm(self.possible_equations).display_name
        display_name_form_field.fill_from_post(param)
        if display_name_form_field.value is not None:
            lconn = common.get_from_session(KEY_LCONN)
            lconn.display_name = display_name_form_field.value

    @expose_page
    def step_2(self, **kwargs):
        """
        Generate the html for the second step of the local connectivity page.
        :param kwargs: not actually used, but parameters are still submitted from UI since we just\
               use the same js function for this.
        """
        current_lconn = common.get_from_session(KEY_LCONN)
        left_side_form = LocalConnectivitySelectorForm(project_id=common.get_current_project().id)
        left_side_form.existentEntitiesSelect.data = current_lconn.gid.hex
        template_specification = dict(title="Surface - Local Connectivity")
        template_specification['mainContent'] = 'spatial/local_connectivity_step2_main'
        template_specification['existentEntitiesInputList'] = self.render_adapter_form(left_side_form)
        template_specification['loadExistentEntityUrl'] = LOAD_EXISTING_URL
        template_specification['resetToDefaultUrl'] = RELOAD_DEFAULT_PAGE_URL
        template_specification['next_step_url'] = '/spatial/localconnectivity/step_1'
        msg, _ = common.get_message_from_session()
        template_specification['displayedMessage'] = msg
        if current_lconn is not None:
            selected_local_conn = ABCAdapter.load_entity_by_gid(current_lconn.gid.hex)
            template_specification.update(self.display_surface(selected_local_conn.fk_surface_gid))
            template_specification['no_local_connectivity'] = False
            template_specification['minValue'] = selected_local_conn.matrix_non_zero_min
            template_specification['maxValue'] = selected_local_conn.matrix_non_zero_max
        else:
            template_specification['no_local_connectivity'] = True
        template_specification[common.KEY_PARAMETERS_CONFIG] = False
        return self.fill_default_attributes(template_specification)

    @cherrypy.expose
    @handle_error(redirect=False)
    @check_user
    def create_local_connectivity(self, **kwargs):
        """
        Used for creating and storing a local connectivity.
        """
        current_lconn = common.get_from_session(KEY_LCONN)
        local_connectivity_creator = ABCAdapter.build_adapter_from_class(LocalConnectivityCreator)
        self.operation_service.fire_operation(local_connectivity_creator, common.get_logged_user(),
                                              common.get_current_project().id, view_model=current_lconn)
        common.set_important_message("The operation for creating the local connectivity was successfully launched.")
        return self.step_1()

    @cherrypy.expose
    @handle_error(redirect=False)
    @check_user
    def load_local_connectivity(self, local_connectivity_gid, from_step=None):
        """
        Loads an existing local connectivity.
        """
        lconn_index = ABCAdapter.load_entity_by_gid(local_connectivity_gid)
        existent_lconn = LocalConnectivityCreatorModel()
        lconn_h5_path = h5.path_for_stored_index(lconn_index)
        with LocalConnectivityH5(lconn_h5_path) as lconn_h5:
            lconn_h5.load_into(existent_lconn)

        existent_lconn.surface = uuid.UUID(lconn_index.fk_surface_gid)

        common.add2session(KEY_LCONN, existent_lconn)
        existent_lconn.display_name = lconn_index.user_tag_1

        if existent_lconn.equation:
            msg = "Successfully loaded existent entity gid=%s" % (local_connectivity_gid,)
        else:
            msg = "There is no equation specified for this local connectivity. "
            msg += "The default equation is displayed into the spatial field."
        common.set_message(msg)

        if int(from_step) == 1:
            return self.step_1()
        if int(from_step) == 2:
            return self.step_2()

    @cherrypy.expose
    @handle_error(redirect=False)
    @check_user
    def reset_local_connectivity(self, from_step):
        """
        Reset the context and reset to the first step. This method is called when the None entry is
        selected from the select.
        :param from_step: is not used in local connectivity case since we don't want to remain in
        step 2 in case none was selected. We are keeping it so far to remain compatible with the
        stimulus pages.
        """
        return self.step_1(do_reset=1)

    def fill_default_attributes(self, template_dictionary):
        """
        Overwrite base controller to add required parameters for adapter templates.
        """
        template_dictionary[common.KEY_SECTION] = 'connectivity'
        template_dictionary[common.KEY_SUB_SECTION] = 'local'
        template_dictionary[common.KEY_SUBMENU_LIST] = self.connectivity_submenu
        template_dictionary[common.KEY_INCLUDE_RESOURCES] = 'spatial/included_resources'
        BaseController.fill_default_attributes(self, template_dictionary)
        return template_dictionary

    @expose_json
    def compute_data_for_gradient_view(self, local_connectivity_gid, selected_triangle):
        """
        When the user loads an existent local connectivity and he picks a vertex from the used surface, this
        method computes the data needed for drawing a gradient view corresponding to that vertex.

        Returns a json which contains the data needed for drawing a gradient view for the selected vertex.
        """
        lconn_index = ABCAdapter.load_entity_by_gid(local_connectivity_gid)
        triangle_index = int(selected_triangle)

        surface_indx = ABCAdapter.load_entity_by_gid(lconn_index.fk_surface_gid)
        surface_h5 = h5.h5_file_for_index(surface_indx)
        assert isinstance(surface_h5, SurfaceH5)
        vertex_index = int(surface_h5.triangles[triangle_index][0])

        lconn_h5 = h5.h5_file_for_index(lconn_index)
        assert isinstance(lconn_h5, LocalConnectivityH5)
        lconn_matrix = lconn_h5.matrix.load()
        picked_data = list(lconn_matrix[vertex_index].toarray().squeeze())
        lconn_h5.close()

        result = []
        number_of_split_slices = surface_h5.number_of_split_slices.load()
        if number_of_split_slices <= 1:
            result.append(picked_data)
        else:
            for slice_number in range(number_of_split_slices):
                start_idx, end_idx = surface_h5.get_slice_vertex_boundaries(slice_number)
                result.append(picked_data[start_idx:end_idx])

        surface_h5.close()
        result = {'data': json.dumps(result)}
        return result

    @staticmethod
    def get_series_json(ideal_case, average_case, worst_case, best_case, vertical_line):
        """ Gather all the separate data arrays into a single flot series. """
        return json.dumps([
            {"data": ideal_case, "lines": {"lineWidth": 1}, "label": "Theoretical case", "color": "rgb(52, 255, 25)"},
            {"data": average_case, "lines": {"lineWidth": 1}, "label": "Most probable", "color": "rgb(148, 0, 179)"},
            {"data": worst_case, "lines": {"lineWidth": 1}, "label": "Worst case", "color": "rgb(0, 0, 255)"},
            {"data": best_case, "lines": {"lineWidth": 1}, "label": "Best case", "color": "rgb(122, 122, 0)"},
            {"data": vertical_line, "points": {"show": True, "radius": 1}, "label": "Cut-off distance",
             "color": "rgb(255, 0, 0)"}
        ])

    @expose_fragment('spatial/equation_displayer')
    def get_equation_chart(self):
        """
        Returns the html which contains the plot with the equations
        specified into 'plotted_equations_prefixes' field.
        """
        try:
            # This should be called once at first rendering and once for any change event on form fields used
            # in computation: equation, equation params, surface, cutoff
            current_lconn = common.get_from_session(KEY_LCONN)
            surface_gid = current_lconn.surface.hex
            surface = ABCAdapter.load_entity_by_gid(surface_gid)
            if surface is None:
                raise MissingDataException(self.MSG_MISSING_SURFACE + "!!!")
            max_x = current_lconn.cutoff
            if max_x <= 0:
                max_x = 50
            equation = current_lconn.equation
            # What we want
            ideal_case_series, _ = equation.get_series_data(0, 2 * max_x)

            # What we'll mostly get
            avg_res = 2 * int(max_x / surface.edge_mean_length)
            step = max_x * 2 / (avg_res - 1)
            average_case_series, _ = equation.get_series_data(0, 2 * max_x, step)

            # It can be this bad
            worst_res = 2 * int(max_x / surface.edge_max_length)
            step = 2 * max_x / (worst_res - 1)
            worst_case_series, _ = equation.get_series_data(0, 2 * max_x, step)

            # This is as good as it gets...
            best_res = 2 * int(max_x / surface.edge_min_length)
            step = 2 * max_x / (best_res - 1)
            best_case_series, _ = equation.get_series_data(0, 2 * max_x, step)

            max_y = -1000000000
            min_y = 10000000000
            for case in ideal_case_series:
                if min_y > case[1]:
                    min_y = case[1]
                if min_y > case[1]:
                    min_y = case[1]
                if max_y < case[1]:
                    max_y = case[1]
                if max_y < case[1]:
                    max_y = case[1]
            vertical_line = []
            vertical_step = (max_y - min_y) / NO_OF_CUTOFF_POINTS
            for i in range(NO_OF_CUTOFF_POINTS):
                vertical_line.append([max_x, min_y + i * vertical_step])
            all_series = self.get_series_json(ideal_case_series, average_case_series, worst_case_series,
                                              best_case_series, vertical_line)

            return {'allSeries': all_series, 'prefix': 'spatial', "message": None}
        except NameError as ex:
            self.logger.exception(ex)
            return {'allSeries': None, 'errorMsg': "Incorrect parameters for equation passed."}
        except SyntaxError as ex:
            self.logger.exception(ex)
            return {'allSeries': None, 'errorMsg': "Some of the parameters hold invalid characters."}
        except Exception as ex:
            self.logger.exception(ex)
            return {'allSeries': None, 'errorMsg': ex}
