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

from tvb.core.adapters.input_tree import InputTreeManager
from tvb.datatypes.local_connectivity import LocalConnectivity
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.base_controller import BaseController
from tvb.interfaces.web.controllers.decorators import check_user, handle_error
from tvb.interfaces.web.controllers.decorators import expose_fragment, expose_page, expose_json
from tvb.interfaces.web.controllers.spatial.base_spatio_temporal_controller import SpatioTemporalController
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.entities.transient.context_local_connectivity import ContextLocalConnectivity


NO_OF_CUTOFF_POINTS = 20

LOCAL_CONN_CREATOR_MODULE = "tvb.adapters.creators.local_connectivity_creator"
LOCAL_CONN_CREATOR_CLASS = "LocalConnectivityCreator"

LOAD_EXISTING_URL = '/spatial/localconnectivity/load_local_connectivity'
RELOAD_DEFAULT_PAGE_URL = '/spatial/localconnectivity/reset_local_connectivity'

KEY_LCONN_CONTEXT = "local-conn-ctx"


class LocalConnectivityController(SpatioTemporalController):
    """
    Controller layer for displaying/creating a LocalConnectivity entity.
    """

    def __init__(self):
        SpatioTemporalController.__init__(self)
        self.plotted_equations_prefixes = ['equation', 'surface']


    @expose_page
    def step_1(self, do_reset=0, **kwargs):
        """
        Generate the html for the first step of the local connectivity page. 
        :param do_reset: Boolean telling to start from empty page or not
        :param kwargs: not actually used, but parameters are still submitted from UI since we just\
               use the same js function for this. TODO: do this in a smarter way
        """
        if int(do_reset) == 1:
            new_context = ContextLocalConnectivity()
            common.add2session(KEY_LCONN_CONTEXT, new_context)
        context = common.get_from_session(KEY_LCONN_CONTEXT)
        right_side_interface = self._get_lconn_interface()
        left_side_interface = self.get_select_existent_entities('Load Local Connectivity', LocalConnectivity,
                                                                context.selected_entity)
        # add interface to session, needed for filters
        self.add_interface_to_session(left_side_interface, right_side_interface['inputList'])
        template_specification = dict(title="Surface - Local Connectivity")
        template_specification['mainContent'] = 'spatial/local_connectivity_step1_main'
        template_specification.update(right_side_interface)
        template_specification['displayCreateLocalConnectivityBtn'] = True
        template_specification['loadExistentEntityUrl'] = LOAD_EXISTING_URL
        template_specification['resetToDefaultUrl'] = RELOAD_DEFAULT_PAGE_URL
        template_specification['existentEntitiesInputList'] = left_side_interface
        template_specification['submit_parameters_url'] = '/spatial/localconnectivity/create_local_connectivity'
        template_specification['equationViewerUrl'] = '/spatial/localconnectivity/get_equation_chart'
        template_specification['equationsPrefixes'] = json.dumps(self.plotted_equations_prefixes)
        template_specification['next_step_url'] = '/spatial/localconnectivity/step_2'
        msg, msg_type = common.get_message_from_session()
        template_specification['displayedMessage'] = msg
        return self.fill_default_attributes(template_specification)


    @expose_page
    def step_2(self, **kwargs):
        """
        Generate the html for the second step of the local connectivity page.
        :param kwargs: not actually used, but parameters are still submitted from UI since we just\
               use the same js function for this. TODO: do this in a smarter way
        """
        context = common.get_from_session(KEY_LCONN_CONTEXT)
        left_side_interface = self.get_select_existent_entities('Load Local Connectivity:', LocalConnectivity,
                                                                context.selected_entity)
        template_specification = dict(title="Surface - Local Connectivity")
        template_specification['mainContent'] = 'spatial/local_connectivity_step2_main'
        template_specification['existentEntitiesInputList'] = left_side_interface
        template_specification['loadExistentEntityUrl'] = LOAD_EXISTING_URL
        template_specification['resetToDefaultUrl'] = RELOAD_DEFAULT_PAGE_URL
        template_specification['next_step_url'] = '/spatial/localconnectivity/step_1'
        msg, _ = common.get_message_from_session()
        template_specification['displayedMessage'] = msg
        context = common.get_from_session(KEY_LCONN_CONTEXT)
        if context.selected_entity is not None:
            selected_local_conn = ABCAdapter.load_entity_by_gid(context.selected_entity)
            template_specification.update(self.display_surface(selected_local_conn.surface.gid))
            template_specification['no_local_connectivity'] = False
            min_value, max_value = selected_local_conn.get_min_max_values()
            template_specification['minValue'] = min_value
            template_specification['maxValue'] = max_value
        else:
            template_specification['no_local_connectivity'] = True
        template_specification[common.KEY_PARAMETERS_CONFIG] = False
        return self.fill_default_attributes(template_specification)


    def _get_lconn_interface(self, default_surface_gid=None):
        """
        Returns a dictionary which contains the interface for a local connectivity. In case
        the selected entity from the context is not populated just get the defaults, else load
        the template from the context.
        """
        context = common.get_from_session(KEY_LCONN_CONTEXT)
        if context.selected_entity is None:
            input_list = self.get_creator_and_interface(LOCAL_CONN_CREATOR_MODULE,
                                                        LOCAL_CONN_CREATOR_CLASS, LocalConnectivity(),
                                                        lock_midpoint_for_eq=[1])[1]
            input_list = self._add_extra_fields_to_interface(input_list)
            return self.prepare_entity_interface(input_list)
        else:
            return self.get_template_from_context()


    @cherrypy.expose
    @handle_error(redirect=False)
    @check_user
    def create_local_connectivity(self, **kwargs):
        """
        Used for creating and storing a local connectivity.
        """
        context = common.get_from_session(KEY_LCONN_CONTEXT)
        local_connectivity_creator = self.get_creator_and_interface(LOCAL_CONN_CREATOR_MODULE,
                                                                    LOCAL_CONN_CREATOR_CLASS, LocalConnectivity())[0]
        self.flow_service.fire_operation(local_connectivity_creator, common.get_logged_user(),
                                         common.get_current_project().id, **kwargs)
        common.set_important_message("The operation for creating the local connectivity was successfully launched.")
        context.reset()
        return self.step_1()


    @cherrypy.expose
    @handle_error(redirect=False)
    @check_user
    def load_local_connectivity(self, local_connectivity_gid, from_step=None):
        """
        Loads the interface for an existing local connectivity.
        """
        context = common.get_from_session(KEY_LCONN_CONTEXT)
        context.selected_entity = local_connectivity_gid
        msg = "Successfully loaded existent entity gid=%s" % (local_connectivity_gid,)
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
        context = common.get_from_session(KEY_LCONN_CONTEXT)
        context.reset()
        return self.step_1()


    def get_template_from_context(self):
        """
        Return the parameters for the local connectivity in case one is stored in context. Load the entity
        and use it to populate the defaults from the interface accordingly.
        """
        context = common.get_from_session(KEY_LCONN_CONTEXT)
        selected_local_conn = ABCAdapter.load_entity_by_gid(context.selected_entity)
        cutoff = selected_local_conn.cutoff
        equation = selected_local_conn.equation
        surface = selected_local_conn.surface

        default_dict = {'surface': surface.gid, 'cutoff': cutoff}
        if equation is not None:
            equation_type = equation.__class__.__name__
            default_dict['equation'] = equation_type
            for param in equation.parameters:
                prepared_name = 'equation_parameters_option_' + str(equation_type)
                prepared_name = prepared_name + '_parameters_parameters_' + str(param)
                default_dict[prepared_name] = equation.parameters[param]
        else:
            msg = "There is no equation specified for this local connectivity. "
            msg += "The default equation is displayed into the spatial field."
            self.logger.warning(msg)
            common.set_info_message(msg)

        default_dict[DataTypeMetaData.KEY_TAG_1] = selected_local_conn.user_tag_1

        input_list = self.get_creator_and_interface(LOCAL_CONN_CREATOR_MODULE,
                                                    LOCAL_CONN_CREATOR_CLASS, LocalConnectivity(),
                                                    lock_midpoint_for_eq=[1])[1]
        input_list = self._add_extra_fields_to_interface(input_list)
        input_list = InputTreeManager.fill_defaults(input_list, default_dict)

        template_specification = {'inputList': input_list, common.KEY_PARAMETERS_CONFIG: False,
                                  'equationViewerUrl': '/spatial/localconnectivity/get_equation_chart',
                                  'equationsPrefixes': json.dumps(self.plotted_equations_prefixes)}
        return template_specification


    @staticmethod
    def _add_extra_fields_to_interface(input_list):
        """
        The fields that have to be added to the existent
        adapter interface should be added in this method.
        """
        display_name = {'name': DataTypeMetaData.KEY_TAG_1, 'label': 'Display name',
                        'type': 'str', "disabled": "False"}
        input_list.append(display_name)
        return input_list


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
        selected_local_conn = ABCAdapter.load_entity_by_gid(local_connectivity_gid)
        surface = selected_local_conn.surface
        triangle_index = int(selected_triangle)
        vertex_index = int(surface.triangles[triangle_index][0])
        picked_data = list(selected_local_conn.matrix[vertex_index].toarray().squeeze())

        result = []
        if surface.number_of_split_slices <= 1:
            result.append(picked_data)
        else:
            for slice_number in range(surface.number_of_split_slices):
                start_idx, end_idx = surface._get_slice_vertex_boundaries(slice_number)
                result.append(picked_data[start_idx:end_idx])

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
    def get_equation_chart(self, **form_data):
        """
        Returns the html which contains the plot with the equations
        specified into 'plotted_equations_prefixes' field.
        """
        if len(form_data['equation']) > 0 and form_data['equation'] is not None:
            try:
                context = common.get_from_session(KEY_LCONN_CONTEXT)
                surface_gid = form_data['surface']
                if context.selected_surface is None or context.selected_surface.gid != surface_gid:
                    surface = ABCAdapter.load_entity_by_gid(surface_gid)
                else:
                    surface = context.selected_surface
                local_connectivity_creator = self.get_creator_and_interface(LOCAL_CONN_CREATOR_MODULE,
                                                                            LOCAL_CONN_CREATOR_CLASS,
                                                                            LocalConnectivity())[0]
                max_x = float(form_data["cutoff"])
                if max_x <= 0:
                    max_x = 50
                form_data = local_connectivity_creator.prepare_ui_inputs(form_data, validation_required=False)
                equation = local_connectivity_creator.get_lconn_equation(form_data)
                # What we want
                ideal_case_series, _ = equation.get_series_data(0, 2 * max_x)

                # What we'll mostly get
                avg_res = 2 * int(max_x / surface.edge_length_mean)
                step = max_x * 2 / (avg_res - 1)
                average_case_series, _ = equation.get_series_data(0, 2 * max_x, step)

                # It can be this bad
                worst_res = 2 * int(max_x / surface.edge_length_max)
                step = 2 * max_x / (worst_res - 1)
                worst_case_series, _ = equation.get_series_data(0, 2 * max_x, step)

                # This is as good as it gets...
                best_res = 2 * int(max_x / surface.edge_length_min)
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

                return {'allSeries': all_series, 'prefix': self.plotted_equations_prefixes[0], "message": None}
            except NameError as ex:
                self.logger.exception(ex)
                return {'allSeries': None, 'errorMsg': "Incorrect parameters for equation passed."}
            except SyntaxError as ex:
                self.logger.exception(ex)
                return {'allSeries': None, 'errorMsg': "Some of the parameters hold invalid characters."}
            except Exception as ex:
                self.logger.exception(ex)
                return {'allSeries': None, 'errorMsg': ex.message}

        return {'allSeries': None, 'errorMsg': "Equation should not be None for a valid local connectivity."}
