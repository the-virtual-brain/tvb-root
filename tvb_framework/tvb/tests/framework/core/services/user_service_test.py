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
from tvb.basic.profile import TvbProfile
from tvb.core.entities.model import model_project
from tvb.core.entities.storage import dao
from tvb.core.services.exceptions import UsernameException
from tvb.core.services.project_service import ProjectService
from tvb.core.services.user_service import UserService, MEMBERS_PAGE_SIZE
from tvb.core.utils import hash_password
from tvb.tests.framework.core.base_testcase import TransactionalTestCase


class MainSenderDummy(object):
    @staticmethod
    def send(address_from, address_to, email_subject, email_content):
        """
        Overwrite sending of emails for test
        """
        if address_from is None or "@" not in address_from:
            raise Exception("Invalid FROM email address!")
        if address_to is None or '@' not in address_to:
            raise Exception("Invalid TO email address!")


import tvb.core.services.user_service as userservice

userservice.email_sender = MainSenderDummy


class TestUserService(TransactionalTestCase):
    """
    This class contains tests for the tvb.core.services.user_service module.
    """
    NOT_EXISTENT_PROJECT_ID = 43

    def transactional_setup_method(self):
        """
        Reset the database before each test .
        """
        self.clean_database()
        self.user_service = UserService()
        self.user_service.create_user(username=TvbProfile.current.web.admin.ADMINISTRATOR_NAME,
                                      display_name=TvbProfile.current.web.admin.ADMINISTRATOR_DISPLAY_NAME,
                                      password=TvbProfile.current.web.admin.ADMINISTRATOR_PASSWORD,
                                      email=TvbProfile.current.web.admin.ADMINISTRATOR_EMAIL,
                                      role=model_project.ROLE_ADMINISTRATOR, skip_import=True)
        available_users = dao.get_all_users()
        if len(available_users) != 1:
            raise AssertionError("Something went wrong with database initialization!")

    def transactional_teardown_method(self):
        """
        Reset database at test finish.
        """
        self.delete_project_folders()

    def test_create_user_happy_flow(self):
        """
        Standard flow for creating a user.
        """
        initial_user_count = dao.get_all_users()
        data = dict(username="test_user", display_name="test_name", password=hash_password("test_password"),
                    email="test_user@tvb.org", role="user", comment="")
        self.user_service.create_user(**data)
        final_user_count = dao.get_all_users()
        assert len(initial_user_count) == len(final_user_count) - 1, "User count was not increased after create."
        inserted_user = dao.get_user_by_name("test_user")
        assert inserted_user.password == hash_password("test_password"), "Incorrect password"
        assert inserted_user.email == "test_user@tvb.org", "The email inserted is not correct."
        assert inserted_user.role == "user", "The role inserted is not correct."
        assert not inserted_user.validated, "User validation  is not correct."

    def test_create_user_empty_password(self):
        """
        Try to create a user with an empty password field.
        """
        data = dict(username="test_user", display_name="test_name", password="", email="test_user@tvb.org", role="user",
                    comment="")
        with pytest.raises(UsernameException):
            self.user_service.create_user(**data)

    def test_create_user_no_password(self):
        """
        Try to create a user with no password data.
        """
        data = dict(username="test_user", display_name="test_name", email="test_user@tvb.org", role="user", comment="")
        with pytest.raises(UsernameException):
            self.user_service.create_user(**data)

    def test_create_user_empty_username(self):
        """
        Try to create a user with an empty username field.
        """
        data = dict(username="", display_name="test_name", password="test_pass", email="test_user@tvb.org", role="user",
                    comment="")
        with pytest.raises(UsernameException):
            self.user_service.create_user(**data)

    def test_create_user_no_username(self):
        """
        Try to create a user with no username data.
        """
        data = dict(password="test_pass", display_name="test_name", email="test_user@tvb.org", role="user", comment="")
        with pytest.raises(UsernameException):
            self.user_service.create_user(**data)

    def test_create_user_no_email(self):
        """
        Try to create a user with an empty email field.
        """
        data = dict(username="test_username", display_name="test_name", password="test_password", email="", role="user",
                    comment="")
        with pytest.raises(UsernameException):
            self.user_service.create_user(**data)

    def test_reset_password_happy_flow(self):
        """
        Test method for the reset password method. Happy flow.
        """
        data = dict(username="test_user", display_name="test_name", password=hash_password("test_password"),
                    email="test_user@tvb.org", role="user", comment="")
        self.user_service.create_user(**data)
        inserted_user = dao.get_user_by_name("test_user")
        assert inserted_user.password == hash_password("test_password"), "Incorrect password"
        reset_pass_data = dict(username="test_user", email="test_user@tvb.org")
        self.user_service.reset_password(**reset_pass_data)
        inserted_user = dao.get_user_by_name("test_user")
        assert inserted_user.password != hash_password("test_password"), "Password not reset for some reason!"

    def test_reset_pass_wrong_email(self):
        """
        Test method for the reset password method. Email is not valid,
        should raise exception
        """
        data = dict(username="test_user", display_name="test_name", password=hash_password("test_password"),
                    email="test_user@tvb.org", role="user", comment="")
        self.user_service.create_user(**data)
        inserted_user = dao.get_user_by_name("test_user")
        assert inserted_user.password == hash_password("test_password"), "Incorrect password"
        reset_pass_data = dict(username="test_user", email="wrong_mail@tvb.org")
        with pytest.raises(UsernameException):
            self.user_service.reset_password(**reset_pass_data)

    def test_change_password_happy_flow(self):
        """
        Test method for the change password method. Happy flow.
        """
        inserted_user = self._prepare_user_for_change_pwd()
        self.user_service.edit_user(inserted_user, hash_password("test_password"))
        changed_user = dao.get_user_by_name("test_user")
        assert changed_user.password == hash_password("new_test_password"), "The password did not change."

    def test_change_password_wrong_old(self):
        """
        Test method for the change password method. Old password is wrong, should return false.
        """
        inserted_user = self._prepare_user_for_change_pwd()
        params = dict(edited_user=inserted_user, old_password=hash_password("wrong_old_pwd"))
        with pytest.raises(UsernameException):
            self.user_service.edit_user(**params)
        user = dao.get_user_by_name("test_user")
        assert user.password == hash_password("test_password"), "The password should have not been changed!"

    def _prepare_user_for_change_pwd(self):
        """Private method to prepare password change operation"""
        data = dict(username="test_user", display_name="test_name", password=hash_password("test_password"),
                    email="test_user@tvb.org", role="user", comment="")
        self.user_service.create_user(**data)
        self.user_service.validate_user("test_user")
        inserted_user = dao.get_user_by_name("test_user")
        assert inserted_user.password == hash_password("test_password"), "The password inserted is not correct."
        inserted_user.password = hash_password("new_test_password")
        return inserted_user

    def test_is_username_valid(self):
        """
        Test the method that checks if a userName is valid or not (if it already exists
        in the database the userName is not valid).
        """
        user = model_project.User("test_user", "test_name", "test_pass", "test_mail@tvb.org", False, "user")
        dao.store_entity(user)
        assert not self.user_service.is_username_valid("test_user"), "Should be False but got True"
        assert self.user_service.is_username_valid("test_user2"), "Should be True but got False"

    def test_validate_user_happy_flow(self):
        """
        Standard flow for a validate user action.
        """
        user = model_project.User("test_user", "test_name", "test_pass", "test_mail@tvb.org", False, "user")
        dao.store_entity(user)
        assert self.user_service.validate_user("test_user"), "Validation failed when it shouldn't have."

    def test_validate_user_validated(self):
        """
        Flow for trying to validate a user that was already validated.
        """
        user = model_project.User("test_user", "test_name", "test_pass", "test_mail@tvb.org", True, "user")
        dao.store_entity(user)
        assert not self.user_service.validate_user("test_user"), "Validation invalid."

    def test_validate_user_non_existent(self):
        """
        Flow for trying to validate a user that doesn't exist in the database.
        """
        user = model_project.User("test_user", "test_name", "test_pass", "test_mail@tvb.org", True, "user")
        dao.store_entity(user)
        assert not self.user_service.validate_user("test_user2"), "Validation done even tho user is non-existent"

    def test_check_login_happy_flow(self):
        """
        Standard login flow with a valid username and password.
        """
        user = model_project.User("test_user", 'test_name', hash_password("test_pass"), "test_mail@tvb.org", True,
                                  "user")
        dao.store_entity(user)
        available_users = dao.get_all_users()
        assert 2 == len(available_users)
        assert self.user_service.check_login("test_user", "test_pass") is not None, "Login failed when it shouldn't."

    def test_check_login_bad_pass(self):
        """
        Flow for entering a bad/invalid password.
        """
        user = model_project.User("test_user", 'test_user_name', hash_password("test_pass"), "test_mail@tvb.org", True,
                                  "user")
        dao.store_entity(user)
        available_users = dao.get_all_users()
        assert 2 == len(available_users)
        assert self.user_service.check_login("test_user", "bad_pass") is None, "Login succeeded with bad password."

    def test_check_login_bad_user(self):
        """
        Flow for entering a bad/invalid username.
        """
        user = model_project.User("test_user", 'test_name', hash_password("test_pass"), "test_mail@tvb.org", True,
                                  "user")
        dao.store_entity(user)
        available_users = dao.get_all_users()
        assert 2 == len(available_users)
        assert self.user_service.check_login("bad_user", "test_pass") is None, "Login succeeded with bad userName."

    def test_get_users_for_project(self):
        """
        Get all members of a project except the current user.
        """
        user_ids = []
        for i in range(5):
            user = model_project.User("test_user" + str(i), "test_user_no" + str(i), "pass", "test_mail@tvb.org")
            user = dao.store_entity(user)
            user_ids.append(user.id)
        admin = dao.get_user_by_name("test_user1")
        member1 = dao.get_user_by_name("test_user2")
        member2 = dao.get_user_by_name("test_user4")
        data = dict(name="test_proj", description="test_desc", users=[member1.id, member2.id], max_operation_size=None,
                    disable_imports=False)
        project = ProjectService().store_project(admin, True, None, **data)
        all_users, members, pag = self.user_service.get_users_for_project(admin.username, project.id)
        assert len(members) == 3, "More members than there should be."
        assert len(all_users) == 5
        assert pag == 1, "Invalid total pages number."

        admin_found_member = False
        for user in members:
            if user.username == admin.username:
                admin_found_member = True
        assert admin_found_member, "Admin is expected to be a project member"

        admin_found_editable = False
        for user in all_users:
            if user.username == admin.username:
                admin_found_editable = True
        assert not admin_found_editable, "Admin membership should not be editable"

    def test_get_members_pages(self):
        """
        Create many users (more than one page of members.
        Create a project and asign all Users as members.
        Test that 2 pages or Project Members are retrieved.
        Now remove extra users, to have only one page of members for the project.
        """
        user_ids = []
        for i in range(MEMBERS_PAGE_SIZE + 3):
            user = model_project.User("test_user_no" + str(i), "test_user_no" + str(i), "pass", "test_mail@tvb.org")
            user = dao.store_entity(user)
            user_ids.append(user.id)

        admin = dao.get_user_by_name("test_user_no1")
        data = dict(name='test_proj', description='test_desc', users=user_ids, max_operation_size=None,
                    disable_imports=False)
        project = ProjectService().store_project(admin, True, None, **data)

        page_users, all_users, pag = self.user_service.get_users_for_project(admin.username, project.id, 2)
        assert len(page_users) == (MEMBERS_PAGE_SIZE + 3) % MEMBERS_PAGE_SIZE
        assert len(all_users) == MEMBERS_PAGE_SIZE + 3, 'Not all members returned'
        assert pag == 2, 'Invalid page number returned'

        for i in range(3):
            user = dao.get_user_by_name("test_user_no" + str(i + 2))
            self.user_service.delete_user(user.id)

        page_users, all_users, pag = self.user_service.get_users_for_project("test_user_no1", project.id, 2)
        assert len(page_users) == 0, 'Paging not working properly'
        assert len(all_users) == MEMBERS_PAGE_SIZE, 'Not all members returned'
        assert pag == 1, 'Invalid page number returned'

        page_users, all_users, pag = self.user_service.get_users_for_project("test_user_no1", project.id, 1)
        assert len(page_users) == MEMBERS_PAGE_SIZE, 'Paging not working properly'
        assert len(all_users) == MEMBERS_PAGE_SIZE, 'Not all members returned'
        assert pag == 1, 'Invalid page number returned'

    def test_edit_user_happy_flow(self):
        """
        Test the method of editing a user.
        """
        data = dict(username="test_user", display_name="test_name", password=hash_password("test_password"),
                    email="test_user@tvb.org", role="user", comment="")
        self.user_service.create_user(**data)
        inserted_user = dao.get_user_by_name("test_user")
        assert inserted_user.password == hash_password("test_password"), "Incorrect password"
        inserted_user.role = "new_role"
        inserted_user.validated = 1
        self.user_service.edit_user(inserted_user)
        changed_user = dao.get_user_by_name("test_user")
        assert changed_user.role == "new_role", "role unchanged"
        assert changed_user.validated == 1, "user not validated"

    def test_get_users_when_no_projects(self):
        """
        Assert exception is thrown when no project is found gor the given ID.
        """
        with pytest.raises(UsernameException):
            self.user_service.get_users_for_project(TvbProfile.current.web.admin.ADMINISTRATOR_NAME,
                                                    self.NOT_EXISTENT_PROJECT_ID)
