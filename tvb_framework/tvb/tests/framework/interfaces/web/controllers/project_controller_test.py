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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import pytest
import cherrypy
import tvb.interfaces.web.controllers.common as common
from sqlalchemy.orm.exc import NoResultFound
from tvb.core.entities.storage import dao
from tvb.interfaces.web.controllers.project.project_controller import ProjectController
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.interfaces.web.controllers.base_controller_test import BaseTransactionalControllerTest


class TestProjectController(BaseTransactionalControllerTest):
    """ Unit tests for ProjectController """

    def transactional_setup_method(self):
        """
        Sets up the environment for testing;
        creates a `ProjectController`
        """
        self.init()
        self.project_c = ProjectController()

    def transactional_teardown_method(self):
        """ Cleans the testing environment """
        self.cleanup()

    def test_index_no_project(self):
        """
        Index with no project selected should redirect to viewall page.
        """
        del cherrypy.session[common.KEY_PROJECT]
        self._expect_redirect('/project/viewall', self.project_c.index)

    def test_index(self):
        """
        Verifies that result dictionary has the expected keys / values
        """
        result = self.project_c.index()
        assert result['mainContent'] == "project/project_submenu"
        assert result[common.KEY_PROJECT].id == self.test_project.id
        assert result['subsection_name'] == 'project'
        assert result[common.KEY_USER].id == self.test_user.id

    def test_viewall_valid_data(self):
        """
        Create a bunch of projects and check that they are returned correctly.
        """
        project1 = TestFactory.create_project(self.test_user, 'prj1')
        TestFactory.create_project(self.test_user, 'prj2')
        TestFactory.create_project(self.test_user, 'prj3')
        result = self.project_c.viewall(selected_project_id=project1.id)
        projects_list = result['projectsList']
        # Use this old version of SET builder, otherwise it will fain on Python 2.6
        assert set([prj.name for prj in projects_list]) == {'prj1', 'prj2', 'prj3', 'Test'}
        assert result['page_number'] == 1
        assert result[common.KEY_PROJECT].name == 'prj1'

    def test_viewall_invalid_projectid(self):
        """
        Try to pass on an invalid id for the selected project.
        """
        result = self.project_c.viewall(selected_project_id='invalid')
        assert result[common.KEY_MESSAGE_TYPE] == common.TYPE_ERROR
        assert result[common.KEY_PROJECT].id == self.test_project.id

    def test_viewall_post_create(self):
        """
        Test that you are redirected to edit project page in case of correct post.
        """
        cherrypy.request.method = "POST"
        self._expect_redirect('/project/editone', self.project_c.viewall, create=True)

    def test_editone_cancel(self):
        """
        Test that cancel redirects to appropriate page.
        """
        cherrypy.request.method = "POST"
        self._expect_redirect('/project', self.project_c.editone, cancel=True)

    def test_editone_remove(self):
        """
        Test that a project is indeed deleted.
        """
        cherrypy.request.method = "POST"
        self._expect_redirect('/project/viewall', self.project_c.editone,
                              self.test_project.id, delete=True)
        with pytest.raises(NoResultFound):
            dao.get_project_by_id(self.test_project.id)

    def test_editone_create(self):
        """
        Create a new project using the editone page.
        """
        data = dict(name="newly_created",
                    description="Some test descript.",
                    users=[],
                    administrator=self.test_user.username,
                    visited_pages=None,
                    max_operation_size=None,
                    disable_imports=False)
        cherrypy.request.method = "POST"
        self._expect_redirect('/project/viewall', self.project_c.editone, save=True, **data)
        projects = dao.get_projects_for_user(self.test_user.id)
        assert len(projects) == 2

    def test_getmemberspage(self):
        """
        Get the first page of the members page.
        """
        users_count = dao.get_all_users(is_count=True)
        user = TestFactory.create_user('usr', 'display', 'pass')
        test_project = TestFactory.create_project(user, 'new_name')
        result = self.project_c.getmemberspage(1, test_project.id)
        assert result['usersMembers'] == [user.id]
        # Same users as before should be available since we created new one
        # as owned for the project.
        assert len(result['usersList']) == users_count

    def test_set_visibility_datatype(self, dummy_datatype_index_factory):
        """
        Set datatype visibility to true and false and check results are updated.
        """
        datatype = dummy_datatype_index_factory()
        assert datatype.visible
        self.project_c.set_visibility('datatype', datatype.gid, 'False')
        datatype = dao.get_datatype_by_gid(datatype.gid)
        assert not datatype.visible
        self.project_c.set_visibility('datatype', datatype.gid, 'True')
        datatype = dao.get_datatype_by_gid(datatype.gid)
        assert datatype.visible

    def test_set_visibility_operation(self, operation_factory):
        """
        Same flow of operations as per test_set_visibilty_datatype just for
        operation entity.
        """
        operation = operation_factory()
        assert operation.visible
        self.project_c.set_visibility('operation', operation.gid, 'False')
        operation = dao.get_operation_by_gid(operation.gid)
        assert not operation.visible
        self.project_c.set_visibility('operation', operation.gid, 'True')
        operation = dao.get_operation_by_gid(operation.gid)
        assert operation.visible

    def test_viewoperations(self, operation_factory):
        """ 
        Test the viewoperations from projectcontroller.
        """
        operation = operation_factory(test_user=self.test_user, test_project=self.test_project)
        result_dict = self.project_c.viewoperations(self.test_project.id)
        operation_list = result_dict['operationsList']
        assert len(operation_list) == 1
        assert operation_list[0]['id'] == str(operation.id)
        assert 'no_filter_selected' in result_dict
        assert 'total_op_count' in result_dict

    def test_get_datatype_details(self, dummy_datatype_index_factory):
        """
        Check for various field in the datatype details dictionary.
        """
        datatype = dummy_datatype_index_factory()
        dt_details = self.project_c.get_datatype_details(datatype.gid)
        assert dt_details['datatype_id'] == datatype.id
        assert dt_details['entity_gid'] == datatype.gid
        assert not dt_details['isGroup']
        assert dt_details['isRelevant']
        assert len(dt_details['overlay_indexes']) == len(dt_details['overlay_tabs_horizontal'])

    def test_get_linkable_projects(self, dummy_datatype_index_factory):
        """
        Test get linkable project, no projects linked so should just return none.
        """
        datatype = dummy_datatype_index_factory()
        result_dict = self.project_c.get_linkable_projects(datatype.id, False, False)
        assert result_dict['projectslinked'] is None
        assert result_dict['datatype_id'] == datatype.id

    def test_get_operation_details(self, operation_factory):
        """
        Verifies result dictionary has the expected keys / values after call to
        `get_operation_details(...`
        """
        operation = operation_factory(test_user=self.test_user, test_project=self.test_project)
        result_dict = self.project_c.get_operation_details(operation.gid)
        assert result_dict['entity_gid'] == operation.gid
        assert result_dict['nodeType'] == 'operation'
        operation_dict = result_dict['nodeFields'][1]
        assert operation_dict['burst_name']['value'] == ''
        assert operation_dict['count']['value'] == 1
        assert operation_dict['gid']['value'] == operation.gid
        assert operation_dict['operation_id']['value'] == operation.id

    def test_editstructure_invalid_proj(self):
        self._expect_redirect('/project', self.project_c.editstructure, None)

    def test_editproject_valid(self):
        """
        Pass valid project to edit structure and check some entries from result dict.
        """
        result_dict = self.project_c.editstructure(self.test_project.id)
        assert result_dict['mainContent'] == 'project/structure'
        assert result_dict['firstLevelSelection'] == 'Data_State'
        assert result_dict['secondLevelSelection'] == 'Data_Subject'
