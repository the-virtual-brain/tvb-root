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
"""
import os
import shutil
import pytest
import tvb_data
from tvb.basic.profile import TvbProfile
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.file.xml_metadata_handlers import XMLReader
from tvb.core.entities.model import model_datatype, model_project, model_operation
from tvb.core.entities.storage import dao
from tvb.core.entities.transient.context_overlay import DataTypeOverlayDetails
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.neocom import h5
from tvb.core.services.exceptions import ProjectServiceException
from tvb.core.services.algorithm_service import AlgorithmService
from tvb.core.services.project_service import ProjectService, PROJECTS_PAGE_SIZE
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.tests.framework.core.factory import TestFactory, ExtremeTestFactory
from tvb.tests.framework.datatypes.datatype1 import Datatype1

NR_USERS = 20
MAX_PROJ_PER_USER = 8


class TestProjectService(TransactionalTestCase):
    """
    This class contains tests for the tvb.core.services.project_service module.
    """

    def transactional_setup_method(self):
        """
        Reset the database before each test.
        """
        self.project_service = ProjectService()
        self.structure_helper = FilesHelper()
        self.test_user = TestFactory.create_user()

    def transactional_teardown_method(self):
        """
        Remove project folders and clean up database.
        """
        created_projects = dao.get_projects_for_user(self.test_user.id)
        for project in created_projects:
            self.structure_helper.remove_project_structure(project.name)
        self.delete_project_folders()

    def test_create_project_happy_flow(self):

        user1 = TestFactory.create_user('test_user1')
        user2 = TestFactory.create_user('test_user2')
        initial_projects = dao.get_projects_for_user(self.test_user.id)
        assert len(initial_projects) == 0, "Database reset probably failed!"

        TestFactory.create_project(self.test_user, 'test_project', "description", users=[user1.id, user2.id])

        resulting_projects = dao.get_projects_for_user(self.test_user.id)
        assert len(resulting_projects) == 1, "Project with valid data not inserted!"
        project = resulting_projects[0]
        assert project.name == "test_project", "Invalid retrieved project name"
        assert project.description == "description", "Description do no match"

        users_for_project = dao.get_members_of_project(project.id)
        for user in users_for_project:
            assert user.id in [user1.id, user2.id, self.test_user.id], "Users not stored properly."
        assert os.path.exists(os.path.join(TvbProfile.current.TVB_STORAGE, FilesHelper.PROJECTS_FOLDER,
                                           "test_project")), "Folder for project was not created"

    def test_create_project_empty_name(self):
        """
        Creating a project with an empty name.
        """
        data = dict(name="", description="test_description", users=[])
        initial_projects = dao.get_projects_for_user(self.test_user.id)
        assert len(initial_projects) == 0, "Database reset probably failed!"
        with pytest.raises(ProjectServiceException):
            self.project_service.store_project(self.test_user, True, None, **data)

    def test_edit_project_happy_flow(self):
        """
        Standard flow for editing an existing project.
        """
        selected_project = TestFactory.create_project(self.test_user, 'test_proj')
        proj_root = self.structure_helper.get_project_folder(selected_project)
        initial_projects = dao.get_projects_for_user(self.test_user.id)
        assert len(initial_projects) == 1, "Database initialization probably failed!"

        edited_data = dict(name="test_project", description="test_description", users=[])
        edited_project = self.project_service.store_project(self.test_user, False, selected_project.id, **edited_data)
        assert not os.path.exists(proj_root), "Previous folder not deleted"
        proj_root = self.structure_helper.get_project_folder(edited_project)
        assert os.path.exists(proj_root), "New folder not created!"
        assert selected_project.name != edited_project.name, "Project was no changed!"

    def test_edit_project_unexisting(self):
        """
        Trying to edit an un-existing project.
        """
        selected_project = TestFactory.create_project(self.test_user, 'test_proj')
        self.structure_helper.get_project_folder(selected_project)
        initial_projects = dao.get_projects_for_user(self.test_user.id)
        assert len(initial_projects) == 1, "Database initialization probably failed!"
        data = dict(name="test_project", description="test_description", users=[])
        with pytest.raises(ProjectServiceException):
            self.project_service.store_project(self.test_user, False, 99, **data)

    def test_find_project_happy_flow(self):
        """
        Standard flow for finding a project by it's id.
        """
        initial_projects = dao.get_projects_for_user(self.test_user.id)
        assert len(initial_projects) == 0, "Database reset probably failed!"
        inserted_project = TestFactory.create_project(self.test_user, 'test_project')
        assert self.project_service.find_project(inserted_project.id) is not None, "Project not found !"
        dao_returned_project = dao.get_project_by_id(inserted_project.id)
        service_returned_project = self.project_service.find_project(inserted_project.id)
        assert dao_returned_project.id == service_returned_project.id, \
            "Data returned from service is different from data returned by DAO."
        assert dao_returned_project.name == service_returned_project.name, \
            "Data returned from service is different than  data returned by DAO."
        assert dao_returned_project.description == service_returned_project.description, \
            "Data returned from service is different from data returned by DAO."
        assert dao_returned_project.members == service_returned_project.members, \
            "Data returned from service is different from data returned by DAO."

    def test_find_project_unexisting(self):
        """
        Searching for an un-existing project.
        """
        data = dict(name="test_project", description="test_description", users=[])
        initial_projects = dao.get_projects_for_user(self.test_user.id)
        assert len(initial_projects) == 0, "Database reset probably failed!"
        project = self.project_service.store_project(self.test_user, True, None, **data)
        # fetch a likely non-existing project. Previous project id plus a 'big' offset
        with pytest.raises(ProjectServiceException):
            self.project_service.find_project(project.id + 1033)

    def test_retrieve_projects_for_user(self):
        """
        Test for retrieving the projects for a given user. One page only.
        """
        initial_projects = self.project_service.retrieve_projects_for_user(self.test_user.id)[0]
        assert len(initial_projects) == 0, "Database was not reset properly!"
        TestFactory.create_project(self.test_user, 'test_proj')
        TestFactory.create_project(self.test_user, 'test_proj1')
        TestFactory.create_project(self.test_user, 'test_proj2')
        user1 = TestFactory.create_user('another_user')
        TestFactory.create_project(user1, 'test_proj3')
        projects = self.project_service.retrieve_projects_for_user(self.test_user.id)[0]
        assert len(projects) == 3, "Projects not retrieved properly!"
        for project in projects:
            assert project.name != "test_project3", "This project should not have been retrieved"

    def test_retrieve_1project_3usr(self):
        """
        One user as admin, two users as members, getting projects for admin and for any of
        the members should return one.
        """
        member1 = TestFactory.create_user("member1")
        member2 = TestFactory.create_user("member2")
        TestFactory.create_project(self.test_user, 'Testproject', users=[member1.id, member2.id])
        projects = self.project_service.retrieve_projects_for_user(self.test_user.id, 1)[0]
        assert len(projects) == 1, "Projects not retrieved properly!"
        projects = self.project_service.retrieve_projects_for_user(member1.id, 1)[0]
        assert len(projects) == 1, "Projects not retrieved properly!"
        projects = self.project_service.retrieve_projects_for_user(member2.id, 1)[0]
        assert len(projects) == 1, "Projects not retrieved properly!"

    def test_retrieve_3projects_3usr(self):
        """
        Three users, 3 projects. Structure of db:
        proj1: {admin: user1, members: [user2, user3]}
        proj2: {admin: user2, members: [user1]}
        proj3: {admin: user3, members: [user1, user2]}
        Check valid project returns for all the users.
        """
        member1 = TestFactory.create_user("member1")
        member2 = TestFactory.create_user("member2")
        member3 = TestFactory.create_user("member3")
        TestFactory.create_project(member1, 'TestProject1', users=[member2.id, member3.id])
        TestFactory.create_project(member2, 'TestProject2', users=[member1.id])
        TestFactory.create_project(member3, 'TestProject3', users=[member1.id, member2.id])
        projects = self.project_service.retrieve_projects_for_user(member1.id, 1)[0]
        assert len(projects) == 3, "Projects not retrieved properly!"
        projects = self.project_service.retrieve_projects_for_user(member2.id, 1)[0]
        assert len(projects) == 3, "Projects not retrieved properly!"
        projects = self.project_service.retrieve_projects_for_user(member3.id, 1)[0]
        assert len(projects) == 2, "Projects not retrieved properly!"

    def test_retrieve_projects_random(self):
        """
        Generate a large number of users/projects, and validate the results.
        """
        ExtremeTestFactory.generate_users(NR_USERS, MAX_PROJ_PER_USER)
        for i in range(NR_USERS):
            current_user = dao.get_user_by_name("gen" + str(i))
            expected_projects = ExtremeTestFactory.VALIDATION_DICT[current_user.id]
            if expected_projects % PROJECTS_PAGE_SIZE == 0:
                expected_pages = expected_projects / PROJECTS_PAGE_SIZE
                exp_proj_per_page = PROJECTS_PAGE_SIZE
            else:
                expected_pages = expected_projects // PROJECTS_PAGE_SIZE + 1
                exp_proj_per_page = expected_projects % PROJECTS_PAGE_SIZE
            if expected_projects == 0:
                expected_pages = 0
                exp_proj_per_page = 0
            projects, pages = self.project_service.retrieve_projects_for_user(current_user.id, expected_pages)
            assert len(projects) == exp_proj_per_page, "Projects not retrieved properly! Expected:" + \
                                                       str(exp_proj_per_page) + "but got:" + str(len(projects))
            assert pages == expected_pages, "Pages not retrieved properly!"

        for folder in os.listdir(TvbProfile.current.TVB_STORAGE):
            full_path = os.path.join(TvbProfile.current.TVB_STORAGE, folder)
            if os.path.isdir(full_path) and folder.startswith('Generated'):
                shutil.rmtree(full_path)

    def test_retrieve_projects_page2(self):
        """
        Test for retrieving the second page projects for a given user.
        """
        for i in range(PROJECTS_PAGE_SIZE + 3):
            TestFactory.create_project(self.test_user, 'test_proj' + str(i))
        projects, pages = self.project_service.retrieve_projects_for_user(self.test_user.id, 2)
        assert len(projects) == (PROJECTS_PAGE_SIZE + 3) % PROJECTS_PAGE_SIZE, "Pagination inproper."
        assert pages == 2, 'Wrong number of pages retrieved.'

    def test_retrieve_projects_and_del(self):
        """
        Test for retrieving the second page projects for a given user.
        """
        created_projects = []
        for i in range(PROJECTS_PAGE_SIZE + 1):
            created_projects.append(TestFactory.create_project(self.test_user, 'test_proj' + str(i)))
        projects, pages = self.project_service.retrieve_projects_for_user(self.test_user.id, 2)
        assert len(projects) == (PROJECTS_PAGE_SIZE + 1) % PROJECTS_PAGE_SIZE, "Pagination improper."
        assert pages == (PROJECTS_PAGE_SIZE + 1) // PROJECTS_PAGE_SIZE + 1, 'Wrong number of pages'
        self.project_service.remove_project(created_projects[1].id)
        projects, pages = self.project_service.retrieve_projects_for_user(self.test_user.id, 2)
        assert len(projects) == 0, "Pagination improper."
        assert pages == 1, 'Wrong number of pages retrieved.'
        projects, pages = self.project_service.retrieve_projects_for_user(self.test_user.id, 1)
        assert len(projects) == PROJECTS_PAGE_SIZE, "Pagination improper."
        assert pages == 1, 'Wrong number of pages retrieved.'

    def test_empty_project_has_zero_disk_size(self):
        TestFactory.create_project(self.test_user, 'test_proj')
        projects, pages = self.project_service.retrieve_projects_for_user(self.test_user.id)
        assert 0 == projects[0].disk_size
        assert '0.0 KiB' == projects[0].disk_size_human

    def test_project_disk_size(self):
        project1 = TestFactory.create_project(self.test_user, 'test_proj1')
        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_66.zip')
        TestFactory.import_zip_connectivity(self.test_user, project1, zip_path, 'testSubject')

        project2 = TestFactory.create_project(self.test_user, 'test_proj2')
        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_76.zip')
        TestFactory.import_zip_connectivity(self.test_user, project2, zip_path, 'testSubject')

        projects = self.project_service.retrieve_projects_for_user(self.test_user.id)[0]
        assert projects[0].disk_size != projects[1].disk_size, "projects should have different size"

        for project in projects:
            assert 0 != project.disk_size
            assert '0.0 KiB' != project.disk_size_human

            prj_folder = self.structure_helper.get_project_folder(project)
            actual_disk_size = self.compute_recursive_h5_disk_usage(prj_folder)[0]

            ratio = float(actual_disk_size) / project.disk_size
            msg = "Real disk usage: %s The one recorded in the db : %s" % (actual_disk_size, project.disk_size)
            assert ratio < 1.4, msg

    def test_get_linkable_projects(self):
        """
        Test for retrieving the projects for a given user.
        """
        initial_projects = self.project_service.retrieve_projects_for_user(self.test_user.id)[0]
        assert len(initial_projects) == 0, "Database was not reset!"
        test_proj = []
        user1 = TestFactory.create_user("another_user")
        for i in range(4):
            test_proj.append(TestFactory.create_project(self.test_user if i < 3 else user1, 'test_proj' + str(i)))

        project_storage = self.structure_helper.get_project_folder(test_proj[0])

        operation = TestFactory.create_operation(test_user=self.test_user, test_project=test_proj[0])

        project_storage = os.path.join(project_storage, str(operation.id))
        os.makedirs(project_storage)
        datatype = dao.store_entity(model_datatype.DataType(module="test_data", subject="subj1",
                                                            state="test_state", operation_id=operation.id))
        linkable = self.project_service.get_linkable_projects_for_user(self.test_user.id, str(datatype.id))[0]
        assert len(linkable) == 2, "Wrong count of link-able projects!"
        proj_names = [project.name for project in linkable]
        assert test_proj[1].name in proj_names
        assert test_proj[2].name in proj_names
        assert not test_proj[3].name in proj_names

    def test_remove_project_happy_flow(self):
        """
        Standard flow for deleting a project.
        """
        inserted_project = TestFactory.create_project(self.test_user, 'test_proj')
        project_root = self.structure_helper.get_project_folder(inserted_project)
        projects = dao.get_projects_for_user(self.test_user.id)
        assert len(projects) == 1, "Initializations failed!"
        assert os.path.exists(project_root), "Something failed at insert time!"
        self.project_service.remove_project(inserted_project.id)
        projects = dao.get_projects_for_user(self.test_user.id)
        assert len(projects) == 0, "Project was not deleted!"
        assert not os.path.exists(project_root), "Root folder not deleted!"

    def test_remove_project_wrong_id(self):
        """
        Flow for deleting a project giving an un-existing id.
        """
        TestFactory.create_project(self.test_user, 'test_proj')
        projects = dao.get_projects_for_user(self.test_user.id)
        assert len(projects) == 1, "Initializations failed!"
        with pytest.raises(ProjectServiceException):
            self.project_service.remove_project(99)

    def __check_meta_data(self, expected_meta_data, new_datatype):
        """Validate Meta-Data"""
        mapp_keys = {DataTypeMetaData.KEY_SUBJECT: "subject", DataTypeMetaData.KEY_STATE: "state"}
        for key, value in expected_meta_data.items():
            if key in mapp_keys:
                assert value == getattr(new_datatype, mapp_keys[key])
            elif key == DataTypeMetaData.KEY_OPERATION_TAG:
                if DataTypeMetaData.KEY_OP_GROUP_ID in expected_meta_data:
                    # We have a Group to check
                    op_group = new_datatype.parent_operation.fk_operation_group
                    op_group = dao.get_generic_entity(model_operation.OperationGroup, op_group)[0]
                    assert value == op_group.name
                else:
                    assert value == new_datatype.parent_operation.user_group

    def test_remove_project_node(self):
        """
        Test removing of a node from a project.
        """
        inserted_project, gid, op = TestFactory.create_value_wrapper(self.test_user)
        project_to_link = model_project.Project("Link", self.test_user.id, "descript")
        project_to_link = dao.store_entity(project_to_link)
        exact_data = dao.get_datatype_by_gid(gid)
        assert exact_data is not None, "Initialization problem!"
        dao.store_entity(model_datatype.Links(exact_data.id, project_to_link.id))

        vw_h5_path = h5.path_for_stored_index(exact_data)
        assert os.path.exists(vw_h5_path)

        if dao.get_system_user() is None:
            dao.store_entity(model_operation.User(TvbProfile.current.web.admin.SYSTEM_USER_NAME,
                                                  TvbProfile.current.web.admin.SYSTEM_USER_NAME, None, None, True,
                                                  None))

        self.project_service._remove_project_node_files(inserted_project.id, gid)

        assert not os.path.exists(vw_h5_path)
        exact_data = dao.get_datatype_by_gid(gid)
        assert exact_data  is not None, "Data should still be in DB, because of links"
        vw_h5_path_new = h5.path_for_stored_index(exact_data)
        assert os.path.exists(vw_h5_path_new)
        assert vw_h5_path_new != vw_h5_path

        self.project_service._remove_project_node_files(project_to_link.id, gid)
        assert dao.get_datatype_by_gid(gid) is None

    def test_update_meta_data_simple(self):
        """
        Test the new update metaData for a simple data that is not part of a group.
        """
        inserted_project, gid, _ = TestFactory.create_value_wrapper(self.test_user)
        new_meta_data = {DataTypeOverlayDetails.DATA_SUBJECT: "new subject",
                         DataTypeOverlayDetails.DATA_STATE: "second_state",
                         DataTypeOverlayDetails.CODE_GID: gid,
                         DataTypeOverlayDetails.CODE_OPERATION_TAG: 'new user group'}
        self.project_service.update_metadata(new_meta_data)

        new_datatype = dao.get_datatype_by_gid(gid)
        self.__check_meta_data(new_meta_data, new_datatype)

        # TODO Change bellow to check the Operation ViewModel H5 instead of the old Operation.xml
        op_path = FilesHelper().get_operation_meta_file_path(inserted_project.name, new_datatype.parent_operation.id)
        op_meta = XMLReader(op_path).read_metadata()
        assert op_meta['user_group'] == 'new user group', 'UserGroup not updated!'

    def test_update_meta_data_group(self, datatype_group_factory):
        """
        Test the new update metaData for a group of dataTypes.
        """
        group = datatype_group_factory()

        new_meta_data = {DataTypeOverlayDetails.DATA_SUBJECT: "new subject",
                         DataTypeOverlayDetails.DATA_STATE: "updated_state",
                         DataTypeOverlayDetails.CODE_OPERATION_GROUP_ID: group.id,
                         DataTypeOverlayDetails.CODE_OPERATION_TAG: 'newGroupName'}
        self.project_service.update_metadata(new_meta_data)
        datatypes = dao.get_datatype_in_group(group.id)
        for datatype in datatypes:
            new_datatype = dao.get_datatype_by_id(datatype.id)
            assert group.id == new_datatype.parent_operation.fk_operation_group
            new_group = dao.get_generic_entity(model_operation.OperationGroup, group.id)[0]
            assert new_group.name == "newGroupName"
            self.__check_meta_data(new_meta_data, new_datatype)

    def _create_datatypes(self, dt_factory, nr_of_dts):
        for idx in range(nr_of_dts):
            dt = Datatype1()
            dt.row1 = "value%i" % (idx,)
            dt.row2 = "value%i" % (idx + 1,)
            dt_factory._store_datatype(dt)

    def test_retrieve_project_full(self, dummy_datatype_index_factory):
        """
        Tests full project information is retrieved by method `ProjectService.retrieve_project_full(...)`
        """

        project = TestFactory.create_project(self.test_user)
        operation = TestFactory.create_operation(test_user=self.test_user, test_project=project)

        dummy_datatype_index_factory(project=project, operation=operation)
        dummy_datatype_index_factory(project=project, operation=operation)
        dummy_datatype_index_factory(project=project, operation=operation)

        _, ops_nr, operations, pages_no = self.project_service.retrieve_project_full(project.id)
        assert ops_nr == 1, "DataType Factory should only use one operation to store all it's datatypes."
        assert pages_no == 1, "DataType Factory should only use one operation to store all it's datatypes."
        resulted_dts = operations[0]['results']
        assert len(resulted_dts) == 3, "3 datatypes should be created."

    def test_get_project_structure(self, datatype_group_factory, dummy_datatype_index_factory, project_factory,
                                   user_factory):
        """
        Tests project structure is as expected and contains all datatypes
        """
        SELF_DTS_NUMBER = 3

        user = user_factory()
        project = project_factory(user)
        dt_group = datatype_group_factory(project=project)

        link_ids, expected_links = [], []
        # Prepare link towards a simple DT
        dt_to_link = dummy_datatype_index_factory(state="RAW_DATA")
        link_ids.append(dt_to_link.id)
        expected_links.append(dt_to_link.gid)

        # Prepare links towards a full DT Group, but expecting only the DT_Group in the final tree
        link_gr = dt_group
        dts = dao.get_datatype_in_group(datatype_group_id=link_gr.id)
        link_ids.extend([dt_to_link.id for dt_to_link in dts])
        link_ids.append(link_gr.id)
        expected_links.append(link_gr.gid)

        # Prepare link towards a single DT inside a group, and expecting to find the DT in the final tree
        link_gr = dt_group
        dt_to_link = dao.get_datatype_in_group(datatype_group_id=link_gr.id)[0]
        link_ids.append(dt_to_link.id)
        expected_links.append(dt_to_link.gid)

        # Actually create the links from Prj2 into Prj1
        AlgorithmService().create_link(link_ids, project.id)

        # Retrieve the raw data used to compose the tree (for easy parsing)
        dts_in_tree = dao.get_data_in_project(project.id)
        dts_in_tree = [dt.gid for dt in dts_in_tree]
        # Retrieve the tree json (for trivial validations only, as we can not decode)
        node_json = self.project_service.get_project_structure(project, None, DataTypeMetaData.KEY_STATE,
                                                               DataTypeMetaData.KEY_SUBJECT, None)

        assert len(expected_links) + SELF_DTS_NUMBER + 2 == len(dts_in_tree), "invalid number of nodes in tree"
        assert not link_gr.gid in dts_in_tree, "DT_group where a single DT is linked is not expected."
        assert dt_group.gid in dts_in_tree, "DT_Group should be in the Project Tree!"
        assert dt_group.gid in node_json, "DT_Group should be in the Project Tree JSON!"

        project_dts = dao.get_datatypes_in_project(project.id)
        for dt in project_dts:
            if dt.fk_datatype_group is not None:
                assert not dt.gid in node_json, "DTs part of a group should not be"
                assert not dt.gid in dts_in_tree, "DTs part of a group should not be"
            else:
                assert dt.gid in node_json, "Simple DTs and DT_Groups should be"
                assert dt.gid in dts_in_tree, "Simple DTs and DT_Groups should be"

        for link_gid in expected_links:
            assert link_gid in node_json, "Expected Link not present"
            assert link_gid in dts_in_tree, "Expected Link not present"
