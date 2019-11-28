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
Control code for Main-Burst Page.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import json
import copy
import cherrypy
import formencode
from tvb.simulator.common import total_ms
import tvb.core.entities.model
from formencode import validators
from cgi import FieldStorage
from cherrypy._cpreqbody import Part
from cherrypy.lib.static import serve_fileobj
from tvb.config.init.introspector_registry import IntrospectionRegistry
from tvb.basic.profile import TvbProfile
from tvb.core.utils import generate_guid, string2bool
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.services.import_service import ImportService
from tvb.adapters.exporters.export_manager import ExportManager
from tvb.core.services.exceptions import BurstServiceException
from tvb.core.services.burst_service import BurstService, LAUNCH_NEW
from tvb.core.services.workflow_service import WorkflowService
from tvb.core.services.operation_service import RANGE_PARAMETER_1, RANGE_PARAMETER_2
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.burst.base_controller import BurstBaseController
from tvb.interfaces.web.controllers.decorators import context_selected, check_user, settings, handle_error
from tvb.interfaces.web.controllers.decorators import expose_page, expose_fragment, expose_json
from tvb.interfaces.web.controllers.flow_controller import KEY_CONTROLLS, SelectedAdapterContext

PORTLET_STEP_SEPARATOR = "____"
BURST_NAME = 'burstName'



class BurstController(BurstBaseController):
    """
    Controller class for Burst-Pages.
    """


    def __init__(self):
        BurstBaseController.__init__(self)
        self.burst_service = BurstService()
        self.workflow_service = WorkflowService()
        self.context = SelectedAdapterContext()

        ## Cache simulator Tree and Algorithm for performance issues.
        self.cached_simulator_algorithm = self.flow_service.get_algorithm_by_module_and_class(IntrospectionRegistry.SIMULATOR_MODULE,
                                                                                              IntrospectionRegistry.SIMULATOR_CLASS)


    @property
    @context_selected
    def cached_simulator_input_tree(self):
        """
        Cache Simulator's input tree, for performance issues.
        Anyway, without restart, the introspected tree will not be different on multiple executions.
        :returns: Simulator's Input Tree (copy from cache or just loaded)
        """
        cached_simulator_tree = common.get_from_session(common.KEY_CACHED_SIMULATOR_TREE)
        if cached_simulator_tree is None:
            cached_simulator_tree = self.flow_service.prepare_adapter(self.cached_simulator_algorithm)
            common.add2session(common.KEY_CACHED_SIMULATOR_TREE, cached_simulator_tree)
        return copy.deepcopy(cached_simulator_tree)


    @expose_page
    @settings
    @context_selected
    def index(self):
        """Get on burst main page"""
        # todo : reuse load_burst here for consistency.
        template_specification = dict(mainContent="burst/main_burst", title="Simulation Cockpit",
                                      baseUrl=TvbProfile.current.web.BASE_URL,
                                      includedResources='project/included_resources')
        portlets_list = self.burst_service.get_available_portlets()
        session_stored_burst = common.get_from_session(common.KEY_BURST_CONFIG)
        if session_stored_burst is None or session_stored_burst.id is None:
            if session_stored_burst is None:
                session_stored_burst = self.burst_service.new_burst_configuration(common.get_current_project().id)
                common.add2session(common.KEY_BURST_CONFIG, session_stored_burst)

            adapter_interface = self.cached_simulator_input_tree
            if session_stored_burst is not None:
                current_data = session_stored_burst.get_all_simulator_values()[0]
                adapter_interface = InputTreeManager.fill_defaults(adapter_interface, current_data, True)
                ### Add simulator tree to session to be available in filters
                self.context.add_adapter_to_session(self.cached_simulator_algorithm, adapter_interface, current_data)
            template_specification['inputList'] = adapter_interface

        selected_portlets = session_stored_burst.update_selected_portlets()
        template_specification['burst_list'] = self.burst_service.get_available_bursts(common.get_current_project().id)
        template_specification['portletList'] = portlets_list
        template_specification['selectedPortlets'] = json.dumps(selected_portlets)
        template_specification['draw_hidden_ranges'] = True
        template_specification['burstConfig'] = session_stored_burst

        ### Prepare PSE available metrics
        ### We put here all available algorithms, because the metrics select area is a generic one, 
        ### and not loaded with every Burst Group change in history.
        algorithm = self.flow_service.get_algorithm_by_module_and_class(IntrospectionRegistry.MEASURE_METRICS_MODULE,
                                                                        IntrospectionRegistry.MEASURE_METRICS_CLASS)
        adapter_instance = ABCAdapter.build_adapter(algorithm)
        if adapter_instance is not None and hasattr(adapter_instance, 'available_algorithms'):
            template_specification['available_metrics'] = [metric_name for metric_name
                                                           in adapter_instance.available_algorithms]
        else:
            template_specification['available_metrics'] = []

        template_specification[common.KEY_PARAMETERS_CONFIG] = False
        template_specification[common.KEY_SECTION] = 'burst'
        return self.fill_default_attributes(template_specification)

    @cherrypy.expose
    @handle_error(redirect=False)
    def get_selected_burst(self):
        """
        Return the burst that is currently stored in session.
        This is one alternative to 'chrome-back problem'.
        """
        session_burst = common.get_from_session(common.KEY_BURST_CONFIG)
        if session_burst.id:
            return str(session_burst.id)
        else:
            return 'None'


    @expose_fragment('burst/portlet_configure_parameters')
    def get_portlet_configurable_interface(self, index_in_tab):
        """
        From the position given by the tab index and the index from that tab, 
        get the portlet configuration and build the configurable interface
        for that portlet.
        """
        burst_config = common.get_from_session(common.KEY_BURST_CONFIG)
        tab_index = burst_config.selected_tab
        portlet_config = burst_config.tabs[tab_index].portlets[int(index_in_tab)]
        portlet_interface = self.burst_service.build_portlet_interface(portlet_config, common.get_current_project().id)

        full_portlet_input_tree = []
        for entry in portlet_interface:
            full_portlet_input_tree.extend(entry.interface)
        self.context.add_portlet_to_session(full_portlet_input_tree)

        portlet_interface = {"adapters_list": portlet_interface,
                             common.KEY_PARAMETERS_CONFIG: False,
                             common.KEY_SESSION_TREE: self.context.KEY_PORTLET_CONFIGURATION}
        return self.fill_default_attributes(portlet_interface)


    @expose_fragment('burst/portlets_preview')
    def portlet_tab_display(self, **data):
        """
        When saving a new configuration of tabs, check if any of the old 
        portlets are still present, and if that is the case use their 
        parameters configuration. 
        
        For all the new portlets add entries in the burst configuration. 
        Also remove old portlets that are no longer saved.
        """
        tab_portlets_list = json.loads(data['tab_portlets_list'])
        burst_config = common.get_from_session(common.KEY_BURST_CONFIG)
        selected_tab_idx = burst_config.selected_tab
        for tab_idx in range(len(tab_portlets_list)):
            current_tab = burst_config.tabs[tab_idx]
            ### When configuration already exists, and new portlets          #####
            ### are selected, first check if any configuration was saved for #####
            ### each portlet and if that is the case, use it. If none is present #
            ### create a new one.                                              ###
            for idx_in_tab in range(len(tab_portlets_list[tab_idx])):
                portlet_id = tab_portlets_list[tab_idx][idx_in_tab][0]
                portlet_name = tab_portlets_list[tab_idx][idx_in_tab][1]
                if portlet_id >= 0:
                    saved_config = current_tab.portlets[idx_in_tab]
                    if saved_config is None or saved_config.portlet_id != portlet_id:
                        current_tab.portlets[idx_in_tab] = self.burst_service.new_portlet_configuration(portlet_id,
                                                                                    tab_idx, idx_in_tab, portlet_name)
                    else:
                        saved_config.visualizer.ui_name = portlet_name
                else:
                    current_tab.portlets[idx_in_tab] = None
            #For generating the HTML get for each id the corresponding portlet
        selected_tab_portlets = []
        saved_selected_tab = burst_config.tabs[selected_tab_idx]
        for portlet in saved_selected_tab.portlets:
            if portlet:
                portlet_id = int(portlet.portlet_id)
                portlet_entity = self.burst_service.get_portlet_by_id(portlet_id)
                portlet_entity.name = portlet.name
                selected_tab_portlets.append(portlet_entity)

        return {'portlet_tab_list': selected_tab_portlets}


    @expose_fragment('burst/portlets_preview')
    def get_configured_portlets(self):
        """
        Return the portlets for one given tab. This is used when changing
        from tab to tab and selecting which portlets will be displayed.
        """
        burst_config = common.get_from_session(common.KEY_BURST_CONFIG)
        if burst_config is None:
            return {'portlet_tab_list': []}

        tab_idx = burst_config.selected_tab
        tab_portlet_list = []
        for portlet_cfg in burst_config.tabs[int(tab_idx)].portlets:
            if portlet_cfg is not None:
                portlet_entity = self.burst_service.get_portlet_by_id(portlet_cfg.portlet_id)
                portlet_entity.name = portlet_cfg.name
                tab_portlet_list.append(portlet_entity)
        return {'portlet_tab_list': tab_portlet_list}


    @expose_json
    def change_selected_tab(self, tab_nr):
        """
        Set :param tab_nr: as the currently selected tab in the stored burst
        configuration. 
        """
        common.get_from_session(common.KEY_BURST_CONFIG).selected_tab = int(tab_nr)


    @expose_json
    def get_portlet_session_configuration(self):
        """
        Get the current configuration of portlets stored in session for this burst,
        as a json.
        """
        burst_entity = common.get_from_session(common.KEY_BURST_CONFIG)
        returned_configuration = burst_entity.update_selected_portlets()
        return returned_configuration


    @cherrypy.expose
    @handle_error(redirect=False)
    def save_parameters(self, index_in_tab, **data):
        """
        Save parameters
        
        :param tab_nr: the index of the selected tab
        :param index_in_tab: the index of the configured portlet in the selected tab
        :param data: the {"portlet_parameters": json_string} Where json_string is a Jsonified dictionary
            {"name": value}, representing the configuration of the current portlet
        
        Having these inputs, current method updated the configuration of the portlet in the
        corresponding tab position form the burst configuration in session.
        """
        burst_config = common.get_from_session(common.KEY_BURST_CONFIG)
        tab_nr = burst_config.selected_tab
        old_portlet_config = burst_config.tabs[int(tab_nr)].portlets[int(index_in_tab)]
        data = json.loads(data['portlet_parameters'])

        # Replace all void entries with 'None'
        for entry in data:
            if data[entry] == '':
                data[entry] = None

        need_relaunch = self.burst_service.update_portlet_configuration(old_portlet_config, data)
        if need_relaunch:
            #### Reset Burst Configuration into an entity not persisted (id = None for all)
            common.add2session(common.KEY_BURST_CONFIG, burst_config.clone())
            return "relaunchView"
        else:
            self.workflow_service.store_workflow_step(old_portlet_config.visualizer)
            return "noRelaunch"

    @expose_json
    def get_history_status(self, **data):
        """
        For each burst id received, get the status and return it.
        """
        return self.burst_service.update_history_status(json.loads(data['burst_ids']))


    @cherrypy.expose
    @handle_error(redirect=False)
    @check_user
    def cancel_or_remove_burst(self, burst_id):
        """
        Cancel or Remove the burst entity given by burst_id.
        :returns 'reset-new': When currently selected burst was removed. JS will need to reset selection to a new entry
        :returns 'canceled': When current burst was still running and was just stopped.
        :returns 'done': When no action is required on the client.
        """
        burst_id = int(burst_id)
        session_burst = common.get_from_session(common.KEY_BURST_CONFIG)
        removed = self.burst_service.cancel_or_remove_burst(burst_id)
        if removed:
            if session_burst.id == burst_id:
                return "reset-new"
            return 'done'
        else:
            # Burst was stopped since it was running
            return 'canceled'


    @expose_json
    def get_selected_portlets(self):
        """
        Get the selected portlets for the loaded burst.
        """
        burst = common.get_from_session(common.KEY_BURST_CONFIG)
        return burst.update_selected_portlets()


    @cherrypy.expose
    @handle_error(redirect=False)
    def get_visualizers_for_operation_id(self, op_id, width, height):
        """
        Method called from parameters exploration page in case a burst with a range of parameters
        for the simulator was launched. 
        :param op_id: the selected operation id from the parameter space exploration.
        :param width: the width of the right side display
        :param height: the height of the right side display
        
        Given these parameters first get the workflow to which op_id belongs, then load the portlets
        from that workflow as the current burst configuration. Width and height are used to get the
        proper sizes for the visualization iFrames. 
        """
        burst_config = common.get_from_session(common.KEY_BURST_CONFIG)
        burst_config = self.burst_service.load_tab_configuration(burst_config, op_id)
        common.add2session(common.KEY_BURST_CONFIG, burst_config)
        return self.load_configured_visualizers(width, height)


    @expose_fragment("burst/portlets_view")
    def load_configured_visualizers(self, width='800', height='600'):
        """
        Load all the visualization steps for this tab. Width and height represent
        the dimensions of the right side Div, so that we can compute for each iFrame
        the maximum size it's visualizer can take.
        """
        burst = common.get_from_session(common.KEY_BURST_CONFIG)
        selected_tab = burst.selected_tab
        tab_portlet_list = []
        for portlet_cfg in burst.tabs[int(selected_tab)].portlets:
            if portlet_cfg is not None:
                tab_portlet_list.append(self.__portlet_config2portlet_entity(portlet_cfg))
        return {'status': burst.status, 'portlet_tab_list': tab_portlet_list,
                'max_width': int(width), 'max_height': int(height), 'model': tvb.core.entities.model}


    @expose_fragment("burst/portlet_visualization_template")
    def check_status_for_visualizer(self, selected_tab, index_in_tab, width='800', height='600'):
        """
        This call is used to check on a regular basis if the data for a certain portlet is 
        available for visualization. Should return the status and the HTML to be displayed.
        """
        burst = common.get_from_session(common.KEY_BURST_CONFIG)
        target_portlet = burst.tabs[int(selected_tab)].portlets[int(index_in_tab)]
        target_portlet = self.__portlet_config2portlet_entity(target_portlet)
        template_dict = {'portlet_entity': target_portlet, 'model': tvb.core.entities.model,
                         'width': int(width), 'height': int(height)}
        return template_dict

    @expose_fragment("burst/base_portlets_iframe")
    def launch_visualization(self, index_in_tab, frame_width, frame_height):
        """
        Launch the visualization for this tab and index in tab. The width and height represent the maximum of the inner 
        visualization canvas so that it can fit in the iFrame.
        """
        result = {}
        try:
            burst = common.get_from_session(common.KEY_BURST_CONFIG)
            visualizer = burst.tabs[burst.selected_tab].portlets[int(index_in_tab)].visualizer
            result = self.burst_service.launch_visualization(visualizer, float(frame_width),
                                                             float(frame_height), True)[0]
            result['launch_success'] = True
        except Exception as ex:
            result['launch_success'] = False
            result['error_msg'] = str(ex)
            self.logger.exception("Could not launch Portlet Visualizer...")

        return self.fill_default_attributes(result)

    @expose_page
    @context_selected
    @settings
    def launch_full_visualizer(self, index_in_tab):
        """
        Launch the full scale visualizer from a small preview from the burst cockpit.
        """
        burst = common.get_from_session(common.KEY_BURST_CONFIG)
        selected_tab = burst.selected_tab
        visualizer = burst.tabs[selected_tab].portlets[int(index_in_tab)].visualizer
        result, input_data = self.burst_service.launch_visualization(visualizer, is_preview=False)
        algorithm = self.flow_service.get_algorithm_by_identifier(visualizer.fk_algorithm)

        if common.KEY_TITLE not in result:
            result[common.KEY_TITLE] = algorithm.displayname

        result[common.KEY_ADAPTER] = algorithm.id
        result[common.KEY_OPERATION_ID] = None
        result[common.KEY_INCLUDE_RESOURCES] = 'flow/included_resources'
        ## Add required field to input dictionary and return it so that it can be used ##
        ## for top right control.                                                    ####
        input_data[common.KEY_ADAPTER] = algorithm.id

        if common.KEY_PARENT_DIV not in result:
            result[common.KEY_PARENT_DIV] = ''
        self.context.add_adapter_to_session(algorithm, None, copy.deepcopy(input_data))

        self._populate_section(algorithm, result, True)
        result[common.KEY_DISPLAY_MENU] = True
        result[common.KEY_BACK_PAGE] = "/burst"
        result[common.KEY_SUBMIT_LINK] = self.get_url_adapter(algorithm.fk_category,
                                                              algorithm.id, 'burst')
        if KEY_CONTROLLS not in result:
            result[KEY_CONTROLLS] = ''
        return self.fill_default_attributes(result)


    def __portlet_config2portlet_entity(self, portlet_cfg):
        """
        From a portlet configuration as it is stored in session, update status and add the index in 
        tab so we can properly display it in the burst page.
        """
        portlet_entity = self.burst_service.get_portlet_by_id(portlet_cfg.portlet_id)
        portlet_status, error_msg = self.burst_service.get_portlet_status(portlet_cfg)
        portlet_entity.error_msg = error_msg
        portlet_entity.status = portlet_status
        portlet_entity.name = portlet_cfg.name
        portlet_entity.index_in_tab = portlet_cfg.index_in_tab
        portlet_entity.td_gid = generate_guid()
        return portlet_entity

    @cherrypy.expose
    @handle_error(redirect=False)
    @check_user
    def export(self, burst_id):
        export_manager = ExportManager()
        export_json = export_manager.export_burst(burst_id)

        result_name = "tvb_simulation_" + str(burst_id) + ".json"
        return serve_fileobj(export_json, "application/x-download", "attachment", result_name)


    @expose_fragment("overlay")
    def get_upload_overlay(self):
        template_specification = self.fill_overlay_attributes(None, "Upload", "Simulation JSON",
                                                              "burst/upload_burst_overlay", "dialog-upload")
        return self.fill_default_attributes(template_specification)


    @cherrypy.expose
    @handle_error(redirect=True)
    @check_user
    @settings
    def load_burst_from_json(self, **data):
        """Upload Burst from previously exported JSON file"""
        self.logger.debug("Uploading ..." + str(data))

        try:
            upload_param = "uploadedfile"
            if upload_param in data and data[upload_param]:

                upload_param = data[upload_param]
                if isinstance(upload_param, FieldStorage) or isinstance(upload_param, Part):
                    if not upload_param.file:
                        raise BurstServiceException("Please select a valid JSON file.")
                    upload_param = upload_param.file.read()

                upload_param = json.loads(upload_param)
                prj_id = common.get_current_project().id
                importer = ImportService()
                burst_entity = importer.load_burst_entity(upload_param, prj_id)
                common.add2session(common.KEY_BURST_CONFIG, burst_entity)

        except Exception as excep:
            self.logger.warning(excep.message)
            common.set_error_message(excep.message)

        raise cherrypy.HTTPRedirect('/burst/')
