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
This file will handle Projects related part.
This represents the Controller part (from MVC).

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import cherrypy
import formencode
from formencode import validators
from simplejson import JSONEncoder
from cherrypy.lib.static import serve_file
from tvb.adapters.exporters.export_manager import ExportManager
from tvb.config.init.introspector_registry import IntrospectionRegistry
import tvb.core.entities.model.model_operation as model
from tvb.core.entities.transient import graph_structures
from tvb.core.entities.filters.factory import StaticFiltersFactory
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.services.operation_service import OperationService
from tvb.core.services.project_service import ProjectService
from tvb.core.services.import_service import ImportService
from tvb.core.services.exceptions import ServicesBaseException, ProjectServiceException
from tvb.core.services.exceptions import RemoveDataTypeException
from tvb.core.utils import string2bool
from tvb.interfaces.web.controllers.autologging import traced
from tvb.interfaces.web.entities.context_overlay import OverlayTabDefinition
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.decorators import settings, check_user, handle_error
from tvb.interfaces.web.controllers.decorators import expose_page, expose_json, expose_fragment
from tvb.interfaces.web.controllers.base_controller import BaseController
from tvb.interfaces.web.controllers.flow_controller import FlowController


@traced('generate_call_out_control', exclude=True)
class ProjectController(BaseController):
    """
    Displays pages which deals with Project data management.
    """

    PRROJECTS_FOR_LINK_KEY = "projectsforlink"
    PRROJECTS_LINKED_KEY = "projectslinked"
    KEY_OPERATION_FILTERS = "operationfilters"

    def __init__(self):
        super(ProjectController, self).__init__()
        self.project_service = ProjectService()


    @expose_page
    @settings
    def index(self):
        """
        Display project main-menu. Choose one project to work with.
        """
        current_project = common.get_current_project()
        if current_project is None:
            raise cherrypy.HTTPRedirect("/project/viewall")
        template_specification = dict(mainContent="project/project_submenu", title="TVB Project Menu")
        return self.fill_default_attributes(template_specification)


    @expose_page
    @settings
    def viewall(self, create=False, page=1, selected_project_id=None, **_):
        """
        Display all existent projects. Choose one project to work with.
        """
        page = int(page)
        if cherrypy.request.method == 'POST' and create:
            raise cherrypy.HTTPRedirect('/project/editone')
        current_user_id = common.get_logged_user().id

        ## Select project if user choose one.
        if selected_project_id is not None:
            try:
                selected_project = self.project_service.find_project(selected_project_id)
                self._mark_selected(selected_project)
            except ProjectServiceException as excep:
                self.logger.error(excep)
                self.logger.warning("Could not select project: " + str(selected_project_id))
                common.set_error_message("Could not select project: " + str(selected_project_id))

        #Prepare template response
        prjs, pages_no = self.project_service.retrieve_projects_for_user(current_user_id, page)
        template_specification = dict(mainContent="project/viewall", title="Available TVB Projects",
                                      projectsList=prjs, page_number=page, total_pages=pages_no)
        return self.fill_default_attributes(template_specification, 'list')


    @cherrypy.expose
    @handle_error(redirect=True)
    @check_user
    @settings
    def projectupload(self, **data):
        """Upload Project from TVB ZIP."""
        self.logger.debug("Uploading ..." + str(data))
        try:
            upload_param = "uploadedfile"
            if upload_param in data and data[upload_param]:
                import_service = ImportService()
                import_service.import_project_structure(data[upload_param], common.get_logged_user().id)
        except ServicesBaseException as excep:
            self.logger.warning(excep.message)
            common.set_error_message(excep.message)
        raise cherrypy.HTTPRedirect('/project/viewall')


    def _remove_project(self, project_id):
        """Private method for removing project."""
        try:
            self.project_service.remove_project(project_id)
        except ServicesBaseException as exc:
            self.logger.error("Could not delete project!")
            self.logger.exception(exc)
            common.set_error_message(exc.message)
        prj = common.get_current_project()
        if prj is not None and prj.id == int(project_id):
            common.remove_from_session(common.KEY_PROJECT)


    def _persist_project(self, data, project_id, is_create, current_user):
        """Private method to persist"""
        data = EditForm().to_python(data)
        saved_project = self.project_service.store_project(current_user, is_create, project_id, **data)
        selected_project = common.get_current_project()
        if len(self.project_service.retrieve_projects_for_user(current_user.id, 1)) == 1:
            selected_project = saved_project
        if selected_project is None or (saved_project.id == selected_project.id):
            self._mark_selected(saved_project)


    @expose_page
    @settings
    def editone(self, project_id=None, cancel=False, save=False, delete=False, **data):
        """
        Create or change Project. When project_id is empty we create a 
        new entity, otherwise we are to edit and existent one.
        """
        if cherrypy.request.method == 'POST' and cancel:
            raise cherrypy.HTTPRedirect('/project')
        if cherrypy.request.method == 'POST' and delete:
            self._remove_project(project_id)
            raise cherrypy.HTTPRedirect('/project/viewall')

        current_user = common.get_logged_user()
        is_create = False
        if project_id is None or not int(project_id):
            is_create = True
            data["administrator"] = current_user.display_name
            admin_username = current_user.username
        else:
            current_project = self.project_service.find_project(project_id)
            if not save:
                # Only when we do not have submitted data,
                # populate fields with initial values for edit.
                data = dict(name=current_project.name, description=current_project.description)
            data["administrator"] = current_project.administrator.display_name
            admin_username = current_project.administrator.username
            self._mark_selected(current_project)
        data["project_id"] = project_id

        template_specification = dict(mainContent="project/editone", data=data, isCreate=is_create,
                                      title="Create new project" if is_create else "Edit " + data["name"],
                                      editUsersEnabled=(current_user.username == admin_username))
        try:
            if cherrypy.request.method == 'POST' and save:
                common.remove_from_session(common.KEY_PROJECT)
                common.remove_from_session(common.KEY_CACHED_SIMULATOR_TREE)
                self._persist_project(data, project_id, is_create, current_user)
                raise cherrypy.HTTPRedirect('/project/viewall')
        except formencode.Invalid as excep:
            self.logger.debug(str(excep))
            template_specification[common.KEY_ERRORS] = excep.unpack_errors()
        except ProjectServiceException as excep:
            self.logger.debug(str(excep))
            common.set_error_message(excep.message)
            raise cherrypy.HTTPRedirect('/project/viewall')

        all_users, members, pages = self.user_service.get_users_for_project(current_user.username, project_id)
        template_specification['usersList'] = all_users
        template_specification['usersMembers'] = [m.id for m in members]
        template_specification['usersPages'] = pages
        template_specification['usersCurrentPage'] = 1
        return self.fill_default_attributes(template_specification, 'properties')


    @expose_fragment('project/project_members')
    def getmemberspage(self, page, project_id=None):
        """Retrieve a new page of Project members."""
        current_name = common.get_logged_user().username
        all_users, members, _ = self.user_service.get_users_for_project(current_name, project_id, int(page))
        edit_enabled = True
        if project_id is not None:
            current_project = self.project_service.find_project(project_id)
            edit_enabled = (current_name == current_project.administrator.username)
        return dict(usersList=all_users, usersMembers=[m.id for m in members],
                    usersCurrentPage=page, editUsersEnabled=edit_enabled)


    @expose_json
    def set_visibility(self, entity_type, entity_gid, to_de_relevant):
        """
        Method used for setting the relevancy/visibility on a DataType(Group)/Operation(Group.
        """
        to_de_relevant = string2bool(to_de_relevant)
        is_operation, is_group = False, False
        if entity_type == graph_structures.NODE_OPERATION_TYPE:
            is_group = False
            is_operation = True
        elif entity_type == graph_structures.NODE_OPERATION_GROUP_TYPE:
            is_group = True
            is_operation = True

        if is_operation:
            self.project_service.set_operation_and_group_visibility(entity_gid, to_de_relevant, is_group)
        else:
            self.project_service.set_datatype_visibility(entity_gid, to_de_relevant)


    @expose_page
    @settings
    def viewoperations(self, project_id=None, page=1, filtername=None, reset_filters=None):
        """
        Display table of operations for a given project selected
        """
        if (project_id is None) or (not int(project_id)):
            raise cherrypy.HTTPRedirect('/project')

        ## Toggle filters
        filters = self.__get_operations_filters()
        selected_filters = None
        for my_filter in filters:
            if cherrypy.request.method == 'POST' and (filtername is not None):
                if reset_filters:
                    my_filter.selected = False
                elif my_filter.display_name == filtername:
                    my_filter.selected = not my_filter.selected
            if my_filter.selected:
                selected_filters = my_filter + selected_filters
        ## Iterate one more time, to update counters
        for my_filter in filters:
            if not my_filter.selected:
                new_count = self.project_service.count_filtered_operations(project_id, my_filter + selected_filters)
                my_filter.passes_count = new_count
            else:
                my_filter.passes_count = ''

        page = int(page)
        project, total_op_count, filtered_ops, pages_no = self.project_service.retrieve_project_full(
            project_id, selected_filters, page)
        ## Select current project
        self._mark_selected(project)

        template_specification = dict(mainContent="project/viewoperations", project=project,
                                      title='Past operations for " ' + project.name + '"', operationsList=filtered_ops,
                                      total_op_count=total_op_count, total_pages=pages_no, page_number=page,
                                      filters=filters, no_filter_selected=(selected_filters is None), model=model)
        return self.fill_default_attributes(template_specification, 'operations')
    
    
    @expose_fragment("call_out_project")
    def generate_call_out_control(self):
        """
        Returns the content of a confirmation dialog, with a given question. 
        """
        self.update_operations_count()
        return {'selectedProject': common.get_current_project()}


    def __get_operations_filters(self):
        """
        Filters for VIEW_ALL_OPERATIONS page.
        Get from session currently selected filters, or build a new set of filters.
        """
        session_filtes = common.get_from_session(self.KEY_OPERATION_FILTERS)
        if session_filtes:
            return session_filtes

        else:
            sim_group = self.algorithm_service.get_algorithm_by_module_and_class(IntrospectionRegistry.SIMULATOR_MODULE,
                                                                                 IntrospectionRegistry.SIMULATOR_CLASS)
            new_filters = StaticFiltersFactory.build_operations_filters(sim_group, common.get_logged_user().id)
            common.add2session(self.KEY_OPERATION_FILTERS, new_filters)
            return new_filters


    @expose_fragment("overlay_confirmation")
    def show_confirmation_overlay(self, **data):
        """
        Returns the content of a confirmation dialog, with a given question.
        """
        if not data:
            data = {}
        question = data.get('question', "Are you sure ?")
        data['question'] = question
        return self.fill_default_attributes(data)

    @expose_fragment("overlay")
    def get_datatype_details(self, entity_gid, back_page='null', exclude_tabs=None):
        """
        Returns the HTML which contains the details for the given dataType.
        :param back_page: if different from 'null' (the default) it will redirect to it after saving metedata changes
        """
        if exclude_tabs is None:
            exclude_tabs = []
        selected_project = common.get_current_project()
        datatype_details, states, entity = self.project_service.get_datatype_details(entity_gid)

        # Load DataType categories
        current_type = datatype_details.data_type
        datatype_gid = datatype_details.gid
        categories, has_operations_warning = {}, False
        if not entity.invalid:
            categories, has_operations_warning = self.algorithm_service.get_launchable_algorithms(datatype_gid)

        is_group = False
        if datatype_details.operation_group_id is not None:
            is_group = True

        # Retrieve links
        linkable_projects_dict = self._get_linkable_projects_dict(entity.id)
        # Load all exporters
        exporters = {}
        if not entity.invalid:
            exporters = ExportManager().get_exporters_for_data(entity)
        is_relevant = entity.visible

        template_specification = {"entity_gid": entity_gid,
                                  "nodeFields": datatype_details.get_ui_fields(),
                                  "allStates": states,
                                  "project": selected_project,
                                  "categories": categories,
                                  "exporters": exporters,
                                  "datatype_id": entity.id,
                                  "isGroup": is_group,
                                  "isRelevant": is_relevant,
                                  "nodeType": 'datatype',
                                  "backPageIdentifier": back_page}
        template_specification.update(linkable_projects_dict)

        overlay_class = "can-browse editor-node node-type-" + str(current_type).lower()
        if is_relevant:
            overlay_class += " node-relevant"
        else:
            overlay_class += " node_irrelevant"
        overlay_title = current_type
        if datatype_details.datatype_tag_1:
            overlay_title += " " + datatype_details.datatype_tag_1

        tabs = []
        overlay_indexes = []
        if "Metadata" not in exclude_tabs:
            tabs.append(OverlayTabDefinition("Metadata", "metadata"))
            overlay_indexes.append(0)
        if "Analyzers" not in exclude_tabs:
            tabs.append(OverlayTabDefinition("Analyzers", "analyzers", enabled=categories and 'Analyze' in categories))
            overlay_indexes.append(1)
        if "Visualizers" not in exclude_tabs:
            tabs.append(OverlayTabDefinition("Visualizers", "visualizers", enabled=categories and 'View' in categories))
            overlay_indexes.append(2)

        enable_link_tab = False
        if (not entity.invalid) and (linkable_projects_dict is not None):
            projects_for_link = linkable_projects_dict.get(self.PRROJECTS_FOR_LINK_KEY)
            if projects_for_link is not None and len(projects_for_link) > 0:
                enable_link_tab = True
            projects_linked = linkable_projects_dict.get(self.PRROJECTS_LINKED_KEY)
            if projects_linked is not None and len(projects_linked) > 0:
                enable_link_tab = True
        if "Links" not in exclude_tabs:
            tabs.append(OverlayTabDefinition("Links", "link_to", enabled=enable_link_tab))
            overlay_indexes.append(3)
        if "Export" not in exclude_tabs:
            tabs.append(OverlayTabDefinition("Export", "export", enabled=(exporters and len(exporters) > 0)))
            overlay_indexes.append(4)
        if "Derived DataTypes" not in exclude_tabs:
            tabs.append(OverlayTabDefinition("Derived DataTypes", "result_dts",
                                             enabled=self.project_service.count_datatypes_generated_from(entity_gid)))
            overlay_indexes.append(5)
        template_specification = self.fill_overlay_attributes(template_specification, "DataType Details",
                                                              overlay_title, "project/details_datatype_overlay",
                                                              overlay_class, tabs, overlay_indexes)
        template_specification = FlowController().fill_default_attributes(template_specification)
        if has_operations_warning:
            template_specification[common.KEY_MESSAGE] = 'Not all operations could be loaded for this input DataType.' \
                                                         ' Contact the admin to check the logs!'
            template_specification[common.KEY_MESSAGE_TYPE] = "warningMessage"
        return template_specification


    @expose_fragment('project/linkable_projects')
    def get_linkable_projects(self, datatype_id, is_group, entity_gid):
        """
        Returns the HTML which displays the link-able projects for the given dataType
        """
        template_specification = self._get_linkable_projects_dict(datatype_id)
        template_specification["entity_gid"] = entity_gid
        template_specification["isGroup"] = is_group
        return template_specification


    def _get_linkable_projects_dict(self, datatype_id):
        """" UI ready dictionary with projects in which current DataType can be linked."""
        self.logger.debug("Searching projects to link for DT " + str(datatype_id))
        for_link, linked = self.project_service.get_linkable_projects_for_user(common.get_logged_user().id, datatype_id)

        projects_for_link, linked_projects = None, None
        if for_link:
            projects_for_link = {}
            for project in for_link:
                projects_for_link[project.id] = project.name

        if linked:
            linked_projects = {}
            for project in linked:
                linked_projects[project.id] = project.name

        template_specification = {self.PRROJECTS_FOR_LINK_KEY: projects_for_link,
                                  self.PRROJECTS_LINKED_KEY: linked_projects,
                                  "datatype_id": datatype_id}
        return template_specification

    @expose_fragment("overlay")
    def get_operation_details(self, entity_gid, is_group=False, back_page='burst'):
        """
        Returns the HTML which contains the details for the given operation.
        """
        if string2bool(str(is_group)):
            # we have an OperationGroup entity.
            template_specification = self._compute_operation_details(entity_gid, True)
            # I expect that all the operations from a group are visible or not
            template_specification["nodeType"] = graph_structures.NODE_OPERATION_GROUP_TYPE

        else:
            # we have a simple Operation
            template_specification = self._compute_operation_details(entity_gid)
            template_specification["displayRelevantButton"] = True
            template_specification["nodeType"] = graph_structures.NODE_OPERATION_TYPE

        template_specification["backPageIdentifier"] = back_page
        overlay_class = "can-browse editor-node node-type-" + template_specification["nodeType"]
        if template_specification["isRelevant"]:
            overlay_class += " node-relevant"
        else:
            overlay_class += " node_irrelevant"

        template_specification = self.fill_overlay_attributes(template_specification, "Details", "Operation",
                                                              "project/details_operation_overlay", overlay_class)
        return FlowController().fill_default_attributes(template_specification)


    def _compute_operation_details(self, entity_gid, is_group=False):
        """
        Returns a dictionary which contains the details for the given operation.
        """
        selected_project = common.get_current_project()
        op_details = self.project_service.get_operation_details(entity_gid, is_group)
        operation_id = op_details.operation_id

        display_reload_btn = True
        operation = OperationService.load_operation(operation_id)

        if (operation.fk_operation_group is not None) or (operation.burst is not None):
            display_reload_btn = False
        else:
            op_categ_id = operation.algorithm.fk_category
            raw_categories = self.algorithm_service.get_raw_categories()
            for category in raw_categories:
                if category.id == op_categ_id:
                    display_reload_btn = False
                    break

        template_specification = {"entity_gid": entity_gid,
                                  "nodeFields": op_details.get_ui_fields(),
                                  "operationId": operation_id,
                                  "displayReloadBtn": display_reload_btn,
                                  "project": selected_project,
                                  "isRelevant": operation.visible}
        return template_specification


    def get_project_structure_grouping(self):
        user = common.get_logged_user()
        return user.get_project_structure_grouping()


    def set_project_structure_grouping(self, first, second):
        user = common.get_logged_user()
        user.set_project_structure_grouping(first, second)
        self.user_service.edit_user(user)

    @expose_page
    @settings
    def editstructure(self, project_id=None, last_selected_tab="treeTab", first_level=None,
                      second_level=None, filter_input="", visibility_filter=None, **_ignored):
        """
        Return the page skeleton for displaying the project structure.
        """
        try:
            int(project_id)
        except (ValueError, TypeError):
            raise cherrypy.HTTPRedirect('/project')

        if first_level is None or second_level is None:
            first_level, second_level = self.get_project_structure_grouping()

        selected_project = self.project_service.find_project(project_id)
        self._mark_selected(selected_project)
        data = self.project_service.get_filterable_meta()
        filters = StaticFiltersFactory.build_datatype_filters(selected=visibility_filter)
        template_specification = dict(mainContent="project/structure",
                                      title=selected_project.name,
                                      project=selected_project, data=data,
                                      lastSelectedTab=last_selected_tab, firstLevelSelection=first_level,
                                      secondLevelSelection=second_level, filterInputValue=filter_input, filters=filters)
        return self.fill_default_attributes(template_specification, 'data')


    @expose_fragment("overlay")
    def get_data_uploader_overlay(self, project_id):
        """
        Returns the html which displays a dialog which allows the user
        to upload certain data into the application.
        """
        upload_algorithms = self.algorithm_service.get_upload_algorithms()

        flow_controller = FlowController()
        algorithms_interface = {}
        tabs = []

        for algorithm in upload_algorithms:
            adapter_template = flow_controller.get_adapter_template(project_id, algorithm.id, True, None)
            algorithms_interface['template_for_algo_' + str(algorithm.id)] = adapter_template
            tabs.append(OverlayTabDefinition(algorithm.displayname, algorithm.subsection_name,
                                             description=algorithm.description))

        template_specification = self.fill_overlay_attributes(None, "Upload", "Upload data for this project",
                                                              "project/upload_data_overlay", "dialog-upload",
                                                              tabs_vertical=tabs)
        template_specification['uploadAlgorithms'] = upload_algorithms
        template_specification['projectId'] = project_id
        template_specification['algorithmsInterface'] = algorithms_interface

        return flow_controller.fill_default_attributes(template_specification)


    @expose_fragment("overlay")
    def get_project_uploader_overlay(self):
        """
        Returns the html which displays a dialog which allows the user
        to upload an entire project.
        """
        template_specification = self.fill_overlay_attributes(None, "Upload", "Project structure",
                                                              "project/upload_project_overlay", "dialog-upload")

        return FlowController().fill_default_attributes(template_specification)


    @expose_page
    def launchloader(self, project_id, algorithm_id, cancel=False, **data):
        """ 
        Start Upload mechanism
        """
        success_link = "/project/editstructure/" + str(project_id)
        # do not allow GET
        if cherrypy.request.method != 'POST' or cancel:
            raise cherrypy.HTTPRedirect(success_link)
        try:
            int(project_id)
            int(algorithm_id)
        except (ValueError, TypeError):
            raise cherrypy.HTTPRedirect(success_link)

        project = self.project_service.find_project(project_id)
        algorithm = self.algorithm_service.get_algorithm_by_identifier(algorithm_id)
        FlowController().execute_post(project.id, success_link, algorithm.fk_category, algorithm, **data)

        raise cherrypy.HTTPRedirect(success_link)


    @cherrypy.expose
    @handle_error(redirect=False)
    @check_user
    def readjsonstructure(self, project_id, visibility_filter=StaticFiltersFactory.FULL_VIEW,
                          first_level=None, second_level=None, filter_value=None):
        """
        AJAX exposed method. 
        Will return the complete JSON for Project's structure, or filtered tree
        (filter only Relevant entities or Burst only Data).
        """
        if first_level is None or second_level is None:
            first_level, second_level = self.get_project_structure_grouping()
        else:
            self.set_project_structure_grouping(first_level, second_level)

        selected_filter = StaticFiltersFactory.build_datatype_filters(single_filter=visibility_filter)
        if project_id == 'undefined':
            project_id = common.get_current_project().id
        project = self.project_service.find_project(project_id)
        json_structure = self.project_service.get_project_structure(project, selected_filter,
                                                                    first_level, second_level, filter_value)
        # This JSON encoding is necessary, otherwise we will get an error
        # from JSTree library while trying to load with AJAX 
        # the content of the tree.     
        encoder = JSONEncoder()
        return encoder.iterencode(json_structure)


    @cherrypy.expose
    @handle_error(redirect=False)
    @check_user
    def createlink(self, link_data, project_id, is_group):
        """
        Delegate the creation of the actual link to the flow service.
        """
        if not string2bool(str(is_group)):
            self.algorithm_service.create_link([link_data], project_id)
        else:
            all_data = self.project_service.get_datatype_in_group(link_data)
            # Link all Dts in group and the DT_Group entity
            data_ids = [data.id for data in all_data]
            data_ids.append(int(link_data))
            self.algorithm_service.create_link(data_ids, project_id)


    @cherrypy.expose
    @handle_error(redirect=False)
    @check_user
    def removelink(self, link_data, project_id, is_group):
        """
        Delegate the creation of the actual link to the flow service.
        """
        if not string2bool(str(is_group)):
            self.algorithm_service.remove_link(link_data, project_id)
        else:
            all_data = self.project_service.get_datatype_in_group(link_data)
            for data in all_data:
                self.algorithm_service.remove_link(data.id, project_id)
            self.algorithm_service.remove_link(int(link_data), project_id)


    @cherrypy.expose
    @handle_error(redirect=False)
    @check_user
    def noderemove(self, project_id, node_gid):
        """
        AJAX exposed method, to execute operation of data removal.
        """
        try:
            if node_gid is None:
                return "Remove can only be applied on a Node with GID!"
            self.logger.debug("Removing data with GID=" + str(node_gid))
            self.project_service.remove_datatype(project_id, node_gid)
        except RemoveDataTypeException as excep:
            self.logger.exception("Could not execute operation Node Remove!")
            return excep.message
        except ServicesBaseException as excep:
            self.logger.exception("Could not execute operation Node Remove!")
            return excep.message
        return None


    @cherrypy.expose
    @handle_error(redirect=False)
    @check_user
    def updatemetadata(self, **data):
        """ Submit MetaData edited for DataType(Group) or Operation(Group). """
        try:

            self.project_service.update_metadata(data)

        except ServicesBaseException as excep:
            self.logger.error("Could not execute MetaData update!")
            self.logger.exception(excep)
            common.set_error_message(excep.message)
            return excep.message


    @cherrypy.expose
    @handle_error(redirect=False)
    @check_user
    def downloaddata(self, data_gid, export_module):
        """ Export the data to a default path of TVB_STORAGE/PROJECTS/project_name """
        current_prj = common.get_current_project()
        # Load data by GID
        entity = ABCAdapter.load_entity_by_gid(data_gid)
        # Do real export
        export_mng = ExportManager()
        file_name, file_path, delete_file = export_mng.export_data(entity, export_module, current_prj)
        if delete_file:
            # We force parent folder deletion because export process generated it.
            self.mark_file_for_delete(file_path, True)

        self.logger.debug("Data exported in file: " + str(file_path))
        return serve_file(file_path, "application/x-download", "attachment", file_name)


    @cherrypy.expose
    @handle_error(redirect=False)
    @check_user
    def downloadproject(self, project_id):
        """
        Export the data from a whole project.
        """
        current_project = self.project_service.find_project(project_id)
        export_mng = ExportManager()
        export_file = export_mng.export_project(current_project)

        # Register export file for delete when download complete
        # We force parent folder deletion because export process generated it.
        self.mark_file_for_delete(export_file, True)

        return serve_file(export_file, "application/x-download", "attachment")


    #methods related to data structure - graph

    @expose_json
    def create_json(self, item_gid, item_type, visibility_filter):
        """
        Method used for creating a JSON representation of a graph.
        """
        selected_filter = StaticFiltersFactory.build_datatype_filters(single_filter=visibility_filter)
        project = common.get_current_project()

        is_upload_operation = (item_type == graph_structures.NODE_OPERATION_TYPE) and \
                              (self.project_service.is_upload_operation(item_gid) or item_gid == "firstOperation")
        if is_upload_operation:
            graph_branches = []
            uploader_operations = self.project_service.get_all_operations_for_uploaders(project.id)
            for operation in uploader_operations:
                dt_outputs = self.project_service.get_results_for_operation(operation.id, selected_filter)
                dt_outputs = self._create_datatype_nodes(dt_outputs)
                parent_op = self._create_operation_nodes([operation], item_gid)
                branch = graph_structures.GraphComponent([], parent_op, dt_outputs, [])
                graph_branches.append(branch)
            graph = graph_structures.FullGraphStructure(graph_branches)
            return graph.prepare_for_json()

        dt_inputs, parent_op, dt_outputs, op_inputs = [], [], [], []
        if item_type == graph_structures.NODE_OPERATION_TYPE:
            dt_inputs = self.project_service.get_datatype_and_datatypegroup_inputs_for_operation(item_gid, selected_filter)
            parent_op = self.project_service.load_operation_by_gid(item_gid)
            dt_outputs = self.project_service.get_results_for_operation(parent_op.id, selected_filter)
            #create graph nodes
            dt_inputs, parent_op, dt_outputs, op_inputs = self._create_nodes(dt_inputs, [parent_op],
                                                                             dt_outputs, [], item_gid)

        elif item_type == graph_structures.NODE_OPERATION_GROUP_TYPE:
            parent_op_group = self.project_service.get_operation_group_by_gid(item_gid)
            dt_inputs = self.project_service.get_datatypes_inputs_for_operation_group(parent_op_group.id,
                                                                                      selected_filter)
            datatype_group = self.project_service.get_datatypegroup_by_op_group_id(parent_op_group.id)
            datatype = self.project_service.get_datatype_by_id(datatype_group.id)

            dt_inputs = self._create_datatype_nodes(dt_inputs)
            parent_op = graph_structures.NodeStructure.build_structure_for_operation_group(parent_op_group.gid)
            parent_op.selected = True
            parent_op = [parent_op]
            if selected_filter.display_name == StaticFiltersFactory.RELEVANT_VIEW and datatype.visible is False:
                dt_outputs = []
            else:
                dt_outputs = self._create_datatype_nodes([datatype])

        elif item_type == graph_structures.NODE_DATATYPE_TYPE:
            selected_dt = ABCAdapter.load_entity_by_gid(item_gid)
            if self.project_service.is_datatype_group(item_gid):
                datatype_group = self.project_service.get_datatypegroup_by_gid(selected_dt.gid)
                parent_op_group = self.project_service.get_operation_group_by_id(datatype_group.fk_operation_group)
                dt_inputs = self.project_service.get_datatypes_inputs_for_operation_group(parent_op_group.id,
                                                                                          selected_filter)
                op_inputs = self.project_service.get_operations_for_datatype_group(selected_dt.id, selected_filter)
                op_inputs_in_groups = self.project_service.get_operations_for_datatype_group(selected_dt.id,
                                                                                             selected_filter,
                                                                                             only_in_groups=True)
                # create graph nodes
                dt_inputs, parent_op, dt_outputs, op_inputs = self._create_nodes(dt_inputs, [], [selected_dt],
                                                                                 op_inputs, item_gid)
                parent_op = [graph_structures.NodeStructure.build_structure_for_operation_group(parent_op_group.gid)]
                op_inputs_in_groups = self._create_operation_group_nodes(op_inputs_in_groups)
                op_inputs.extend(op_inputs_in_groups)
            else:
                parent_op = OperationService.load_operation(selected_dt.fk_from_operation)
                dt_inputs = self.project_service.get_datatype_and_datatypegroup_inputs_for_operation(parent_op.gid,
                                                                                               selected_filter)
                op_inputs = self.project_service.get_operations_for_datatype(selected_dt.gid, selected_filter)
                op_inputs_in_groups = self.project_service.get_operations_for_datatype(selected_dt.gid, selected_filter,
                                                                                       only_in_groups=True)
                dt_outputs = self.project_service.get_results_for_operation(parent_op.id, selected_filter)
                # create graph nodes
                dt_inputs, parent_op, dt_outputs, op_inputs = self._create_nodes(dt_inputs, [parent_op], dt_outputs,
                                                                                 op_inputs, item_gid)
                op_inputs_in_groups = self._create_operation_group_nodes(op_inputs_in_groups)
                op_inputs.extend(op_inputs_in_groups)

        else:
            self.logger.error("Invalid item type: " + str(item_type))
            raise Exception("Invalid item type.")

        branch = graph_structures.GraphComponent(dt_inputs, parent_op, dt_outputs, op_inputs)
        graph = graph_structures.FullGraphStructure([branch])
        return graph.prepare_for_json()

    def _create_nodes(self, dt_inputs, parent_op, dt_outputs, op_inputs, item_gid=None):
        """Expected a list of DataTypes, Parent Operation, Outputs, and returns NodeStructure entities."""
        dt_inputs = self._create_datatype_nodes(dt_inputs, item_gid)
        parent_op = self._create_operation_nodes(parent_op, item_gid)
        dt_outputs = self._create_datatype_nodes(dt_outputs, item_gid)
        op_inputs = self._create_operation_nodes(op_inputs, item_gid)
        return dt_inputs, parent_op, dt_outputs, op_inputs


    @staticmethod
    def _create_datatype_nodes(datatypes_list, selected_item_gid=None):
        """ Expects a list of DataTypes and returns a list of NodeStructures """
        nodes = []
        if datatypes_list is None:
            return nodes
        for data_type in datatypes_list:
            node = graph_structures.NodeStructure.build_structure_for_datatype(data_type.gid)
            if data_type.gid == selected_item_gid:
                node.selected = True
            nodes.append(node)
        return nodes


    @staticmethod
    def _create_operation_nodes(operations_list, selected_item_gid=None):
        """
        Expects a list of operations and returns a list of NodeStructures
        """
        nodes = []
        for operation in operations_list:
            node = graph_structures.NodeStructure.build_structure_for_operation(operation)
            if operation.gid == selected_item_gid:
                node.selected = True
            nodes.append(node)
        return nodes


    def _create_operation_group_nodes(self, operations_list, selected_item_gid=None):
        """
        Expects a list of operations that are part of some operation groups.
        """
        groups = dict()
        for operation in operations_list:
            if operation.fk_operation_group not in groups:
                group = self.project_service.get_operation_group_by_id(operation.fk_operation_group)
                groups[group.id] = group.gid
        nodes = []
        for _, group in groups.items():
            node = graph_structures.NodeStructure.build_structure_for_operation_group(group)
            if group == selected_item_gid:
                node.selected = True
            nodes.append(node)
        return nodes


    def fill_default_attributes(self, template_dictionary, subsection='project'):
        """
        Overwrite base controller to add required parameters for adapter templates.
        """
        template_dictionary[common.KEY_SECTION] = 'project'
        template_dictionary[common.KEY_SUB_SECTION] = subsection
        template_dictionary[common.KEY_INCLUDE_RESOURCES] = 'project/included_resources'
        BaseController.fill_default_attributes(self, template_dictionary)
        return template_dictionary


class EditForm(formencode.Schema):
    """
    Validate creation of a Project entity. 
    """
    invalis_name_msg = "Please enter a name composed only of letters, numbers and underscores."
    name = formencode.All(validators.UnicodeString(not_empty=True),
                          validators.PlainText(messages={'invalid': invalis_name_msg}))
    description = validators.UnicodeString()
    users = formencode.foreach.ForEach(formencode.validators.Int())
    administrator = validators.UnicodeString(not_empty=False)
    project_id = validators.UnicodeString(not_empty=False)
    visited_pages = validators.UnicodeString(not_empty=False)



