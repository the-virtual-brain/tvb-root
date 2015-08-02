# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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

import unittest
from hashlib import md5
from tvb.core.entities import model
from tvb.basic.profile import TvbProfile
from tvb.core.entities.storage import dao
from tvb.core.services.exceptions import UsernameException
from tvb.core.services.user_service import UserService, USERS_PAGE_SIZE
from tvb.core.services.project_service import ProjectService
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



class UserServiceTest(TransactionalTestCase):
    """
    This class contains tests for the tvb.core.services.user_service module.
    """
    NOT_EXISTENT_PROJECT_ID = 43

    def setUp(self):
        """
        Reset the database before each test .
        """
        self.clean_database()
        self.user_service = UserService()
        self.user_service.create_user(username=TvbProfile.current.web.admin.ADMINISTRATOR_NAME,
                                      password=TvbProfile.current.web.admin.ADMINISTRATOR_PASSWORD,
                                      email=TvbProfile.current.web.admin.ADMINISTRATOR_EMAIL,
                                      role=model.ROLE_ADMINISTRATOR)
        available_users = dao.get_all_users()
        if len(available_users) != 1:
            self.fail("Something went wrong with database initialization!")


    def tearDown(self):
        """
        Reset database at test finish.
        """
        self.delete_project_folders()


    def test_create_user_happy_flow(self):
        """
        Standard flow for creating a user.
        """
        initial_user_count = dao.get_all_users()
        data = dict(username="test_user", password=md5("test_password").hexdigest(),
                    email="test_user@tvb.org", role="user", comment="")
        self.user_service.create_user(**data)
        final_user_count = dao.get_all_users()
        self.assertEqual(len(initial_user_count), len(final_user_count) - 1,
                         "User count was not increased after create.")
        inserted_user = dao.get_user_by_name("test_user")
        self.assertEqual(inserted_user.password, md5("test_password").hexdigest(), "Incorrect password")
        self.assertEqual(inserted_user.email, "test_user@tvb.org", "The email inserted is not correct.")
        self.assertEqual(inserted_user.role, "user", "The role inserted is not correct.")
        self.assertFalse(inserted_user.validated, "User validation  is not correct.")


    def test_create_user_empty_password(self):
        """
        Try to create a user with an empty password field.
        """
        data = dict(username="test_user", password="", email="test_user@tvb.org", role="user", comment="")
        self.assertRaises(UsernameException, self.user_service.create_user, **data)


    def test_create_user_no_password(self):
        """
        Try to create a user with no password data.
        """
        data = dict(username="test_user", email="test_user@tvb.org", role="user", comment="")
        self.assertRaises(UsernameException, self.user_service.create_user, **data)


    def test_create_user_empty_username(self):
        """
        Try to create a user with an empty username field.
        """
        data = dict(username="", password="test_pass", email="test_user@tvb.org", role="user", comment="")
        self.assertRaises(UsernameException, self.user_service.create_user, **data)


    def test_create_user_no_username(self):
        """
        Try to create a user with no username data.
        """
        data = dict(password="test_pass", email="test_user@tvb.org", role="user", comment="")
        self.assertRaises(UsernameException, self.user_service.create_user, **data)


    def test_create_user_no_email(self):
        """
        Try to create a user with an empty email field.
        """
        data = dict(username="test_username", password="test_password",
                    email="", role="user", comment="")
        self.assertRaises(UsernameException, self.user_service.create_user, **data)


    def test_reset_password_happy_flow(self):
        """
        Test method for the reset password method. Happy flow.
        """
        data = dict(username="test_user", password=md5("test_password").hexdigest(),
                    email="test_user@tvb.org", role="user", comment="")
        self.user_service.create_user(**data)
        inserted_user = dao.get_user_by_name("test_user")
        self.assertEqual(inserted_user.password, md5("test_password").hexdigest(), "Incorrect password")
        reset_pass_data = dict(username="test_user", email="test_user@tvb.org")
        self.user_service.reset_password(**reset_pass_data)
        inserted_user = dao.get_user_by_name("test_user")
        self.assertNotEqual(inserted_user.password, md5("test_password"), "Password not reset for some reason!")


    def test_reset_pass_wrong_email(self):
        """
        Test method for the reset password method. Email is not valid, 
        should raise exception
        """
        data = dict(username="test_user", password=md5("test_password").hexdigest(),
                    email="test_user@tvb.org", role="user", comment="")
        self.user_service.create_user(**data)
        inserted_user = dao.get_user_by_name("test_user")
        self.assertEqual(inserted_user.password, md5("test_password").hexdigest(), "Incorrect password")
        reset_pass_data = dict(username="test_user", email="wrong_mail@tvb.org")
        self.assertRaises(UsernameException, self.user_service.reset_password, **reset_pass_data)


    def test_change_password_happy_flow(self):
        """
        Test method for the change password method. Happy flow.
        """
        inserted_user = self._prepare_user_for_change_pwd()
        self.user_service.edit_user(inserted_user, md5("test_password").hexdigest())
        changed_user = dao.get_user_by_name("test_user")
        self.assertEqual(changed_user.password, md5("new_test_password").hexdigest(),
                         "The password did not change.")


    def test_change_password_wrong_old(self):
        """
        Test method for the change password method. Old password is wrong, should return false.
        """
        inserted_user = self._prepare_user_for_change_pwd()
        params = dict(edited_user=inserted_user, old_password=md5("wrong_old_pwd").hexdigest())
        self.assertRaises(UsernameException, self.user_service.edit_user, **params)
        user = dao.get_user_by_name("test_user")
        self.assertEqual(user.password, md5("test_password").hexdigest(),
                         "The password should have not been changed!")


    def _prepare_user_for_change_pwd(self):
        """Private method to prepare password change operation"""
        data = dict(username="test_user", password=md5("test_password").hexdigest(),
                    email="test_user@tvb.org", role="user", comment="")
        self.user_service.create_user(**data)
        self.user_service.validate_user("test_user")
        inserted_user = dao.get_user_by_name("test_user")
        self.assertEqual(inserted_user.password, md5("test_password").hexdigest(),
                         "The password inserted is not correct.")
        inserted_user.password = md5('new_test_password').hexdigest()
        return inserted_user


    def test_is_username_valid(self):
        """
        Test the method that checks if a userName is valid or not (if it already exists
        in the database the userName is not valid).
        """
        user = model.User("test_user", "test_pass", "test_mail@tvb.org", False, "user")
        dao.store_entity(user)
        self.assertFalse(self.user_service.is_username_valid("test_user"), "Should be False but got True")
        self.assertTrue(self.user_service.is_username_valid("test_user2"), "Should be True but got False")


    def test_validate_user_happy_flow(self):
        """
        Standard flow for a validate user action.
        """
        user = model.User("test_user", "test_pass", "test_mail@tvb.org", False, "user")
        dao.store_entity(user)
        self.assertTrue(self.user_service.validate_user("test_user"), "Validation failed when it shouldn't have.")


    def test_validate_user_validated(self):
        """
        Flow for trying to validate a user that was already validated.
        """
        user = model.User("test_user", "test_pass", "test_mail@tvb.org", True, "user")
        dao.store_entity(user)
        self.assertFalse(self.user_service.validate_user("test_user"), "Validation invalid.")


    def test_validate_user_non_existent(self):
        """
        Flow for trying to validate a user that doesn't exist in the database.
        """
        user = model.User("test_user", "test_pass", "test_mail@tvb.org", True, "user")
        dao.store_entity(user)
        self.assertFalse(self.user_service.validate_user("test_user2"), "Validation done even tho user is non-existent")


    def test_check_login_happy_flow(self):
        """
        Standard login flow with a valid username and password.
        """
        user = model.User("test_user", md5("test_pass").hexdigest(), "test_mail@tvb.org", True, "user")
        dao.store_entity(user)
        available_users = dao.get_all_users()
        if len(available_users) != 2:
            self.fail("Something went wrong with database reset!")
        self.assertTrue(self.user_service.check_login("test_user", "test_pass")
                        is not None, "Login failed when it shouldn't.")


    def test_check_login_bad_pass(self):
        """
        Flow for entering a bad/invalid password.
        """
        user = model.User("test_user", md5("test_pass").hexdigest(), "test_mail@tvb.org", True, "user")
        dao.store_entity(user)
        available_users = dao.get_all_users()
        if len(available_users) != 2:
            self.fail("Something went wrong with database reset!")
        self.assertTrue(self.user_service.check_login("test_user", "bad_pass") is None,
                        "Login succeeded with bad password.")


    def test_check_login_bad_user(self):
        """
        Flow for entering a bad/invalid username.
        """
        user = model.User("test_user", md5("test_pass").hexdigest(), "test_mail@tvb.org", True, "user")
        dao.store_entity(user)
        available_users = dao.get_all_users()
        if len(available_users) != 2:
            self.fail("Something went wrong with database reset!")
        self.assertTrue(self.user_service.check_login("bad_user", "test_pass") is None,
                        "Login succeeded with bad userName.")


    def test_get_users_for_project(self):
        """
        Get all members of a project except the current user.
        """
        user_1 = model.User("test_user1", "test_pass", "test_mail1@tvb.org", False, "user")
        dao.store_entity(user_1)
        user_2 = model.User("test_user2", "test_pass", "test_mail2@tvb.org", False, "user")
        dao.store_entity(user_2)
        user_3 = model.User("test_user3", "test_pass", "test_mail2@tvb.org", False, "user")
        dao.store_entity(user_3)
        user_4 = model.User("test_user4", "test_pass", "test_mail2@tvb.org", False, "user")
        dao.store_entity(user_4)
        user_5 = model.User("test_user5", "test_pass", "test_mail2@tvb.org", False, "user")
        dao.store_entity(user_5)
        admin = dao.get_user_by_name("test_user1")
        member1 = dao.get_user_by_name("test_user2")
        member2 = dao.get_user_by_name("test_user5")
        data = dict(name="test_proj", description="test_desc", users=[member1.id, member2.id])
        project = ProjectService().store_project(admin, True, None, **data)
        all_users, members, pag = self.user_service.get_users_for_project(admin.username, project.id)
        self.assertEquals(len(members), 2, "More members than there should be.")
        self.assertEquals(len(all_users), 5, "Admin should not be viewed as member. "
                                             "Neither should users that were not part of the project's users list.")
        self.assertEqual(pag, 1, "Invalid total pages number.")
        for user in all_users:
            self.assertNotEqual(user.username, admin.username, "Admin is in members!")


    def test_get_users_second_page(self):
        """
        Try to get the second page of users for a given project
        """
        for i in range(USERS_PAGE_SIZE + 3):
            exec 'user_' + str(i) + '= model.User("test_user' + str(
                i) + '", "test_pass", "test_mail@tvb.org", False, "user")'
            exec "dao.store_entity(user_" + str(i) + ")"
        for i in range(USERS_PAGE_SIZE + 3):
            exec 'member' + str(i) + '=dao.get_user_by_name("test_user' + str(i) + '")'
        admin = dao.get_user_by_name("test_user1")
        data = dict(name='test_proj', description='test_desc',
                    users=[eval('member' + str(i) + '.id') for i in range(USERS_PAGE_SIZE + 3)])
        project = ProjectService().store_project(admin, True, None, **data)
        page_users, all_users, pag = self.user_service.get_users_for_project(admin.username, project.id, 2)
        self.assertEqual(len(page_users), (USERS_PAGE_SIZE + 3) % USERS_PAGE_SIZE)
        self.assertEqual(len(all_users), USERS_PAGE_SIZE + 3, 'Not all members returned')
        self.assertEqual(pag, 2, 'Invalid page number returned')


    def test_get_users_second_page_del(self):
        """
        Try to get the second page of users for a given project where only one user on last page.
        Then delete that user.
        """
        for i in range(USERS_PAGE_SIZE + 1):
            exec 'user_' + str(i) + '= model.User("test_user' + str(i) + \
                 '", "test_pass", "test_mail@tvb.org", False, "user")'
            exec "dao.store_entity(user_" + str(i) + ")"
        for i in range(USERS_PAGE_SIZE + 1):
            exec 'member' + str(i) + '=dao.get_user_by_name("test_user' + str(i) + '")'

        admin = dao.get_user_by_name("test_user1")
        data = dict(name='test_proj', description='test_desc',
                    users=[eval('member' + str(i) + '.id') for i in range(USERS_PAGE_SIZE + 1)])
        project = ProjectService().store_project(admin, True, None, **data)
        page_users, all_users, pag = self.user_service.get_users_for_project(admin.username, project.id, 2)
        self.assertEqual(len(page_users), 1, 'Paging not working properly')
        self.assertEqual(len(all_users), USERS_PAGE_SIZE + 1, 'Not all members returned')
        self.assertEqual(pag, 2, 'Invalid page number returned')
        self.user_service.delete_user(member2.id)
        page_users, all_users, pag = self.user_service.get_users_for_project(admin.username, project.id, 2)
        self.assertEqual(len(page_users), 0, 'Paging not working properly')
        self.assertEqual(len(all_users), USERS_PAGE_SIZE, 'Not all members returned')
        self.assertEqual(pag, 1, 'Invalid page number returned')
        page_users, all_users, pag = self.user_service.get_users_for_project(admin.username, project.id, 1)
        self.assertEqual(len(page_users), USERS_PAGE_SIZE, 'Paging not working properly')
        self.assertEqual(len(all_users), USERS_PAGE_SIZE, 'Not all members returned')
        self.assertEqual(pag, 1, 'Invalid page number returned')


    def test_edit_user_happy_flow(self):
        """
        Test the method of editing a user.
        """
        data = dict(username="test_user", password=md5("test_password").hexdigest(),
                    email="test_user@tvb.org", role="user", comment="")
        self.user_service.create_user(**data)
        inserted_user = dao.get_user_by_name("test_user")
        self.assertEqual(inserted_user.password, md5("test_password").hexdigest(), "Incorrect password")
        inserted_user.role = "new_role"
        inserted_user.validated = 1
        self.user_service.edit_user(inserted_user)
        changed_user = dao.get_user_by_name("test_user")
        self.assertEqual(changed_user.role, "new_role", "role unchanged")
        self.assertEqual(changed_user.validated, 1, "user not validated")


    def test_get_users_when_no_projects(self):
        """
        Assert exception is thrown when no project is found gor the given ID.
        """
        self.assertRaises(UsernameException, self.user_service.get_users_for_project,
                          TvbProfile.current.web.admin.ADMINISTRATOR_NAME, self.NOT_EXISTENT_PROJECT_ID)



def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(UserServiceTest))
    return test_suite



if __name__ == "__main__":
    #So you can run tests individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)
        
        
        
        