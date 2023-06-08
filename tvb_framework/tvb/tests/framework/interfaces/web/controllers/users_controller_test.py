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

import os
import cherrypy
from tvb.core.utils import hash_password
from tvb.storage.h5.utils import string2bool
from tvb.tests.framework.interfaces.web.controllers.base_controller_test import BaseTransactionalControllerTest
from tvb.tests.framework.core.factory import TestFactory
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.users_controller import UserController
from tvb.basic.profile import TvbProfile
from tvb.core.entities.model.model_project import User, ROLE_ADMINISTRATOR
from tvb.core.entities.storage import dao
from tvb.core.entities.model.model_project import UserPreferences


class TestUsersController(BaseTransactionalControllerTest):
    """Unit test for UserController"""

    def transactional_setup_method(self):
        """
        Sets up the testing environment;
        creates a `UserController`
        """
        self.clean_database()
        self.init(user_role=ROLE_ADMINISTRATOR)
        self.user_c = UserController()

    def transactional_teardown_method(self):
        """ Cleans the testing environment """
        self.cleanup()
        self.clean_database()

    def test_index_valid_post(self):
        """
        Tests for a valid redirect on user login
        """
        user = User('valid_user', 'name', hash_password('valid_pass'), 'mail@mail.com', True, 'CLINICIAN')
        dao.store_entity(user)
        login_data = {'username': 'valid_user', 'password': 'valid_pass'}
        cherrypy.request.method = "POST"
        self._expect_redirect('/user/profile', self.user_c.index, **login_data)

    def test_profile_logout(self):
        """
        Simulate a logout and make sure we are redirected to the logout page.
        """
        cherrypy.request.method = "POST"
        self._expect_redirect('/user/logout', self.user_c.profile, logout=True)

    def test_profile_no_settings(self):
        """
        Delete the dummy tvb settings file and make sure we are redirected to the settings page.
        """
        os.remove(TvbProfile.current.TVB_CONFIG_FILE)
        TvbProfile.current.manager.stored_settings = None
        self._expect_redirect('/settings/settings', self.user_c.profile)

    def test_profile_get(self):
        """
        Simulate a GET request and make sure all required information for the user profile
        page are returned.
        """
        cherrypy.request.method = "GET"
        template_dict = self.user_c.profile()
        self._check_default_attributes(template_dict, {}, {})
        assert template_dict[common.KEY_USER].id == self.test_user.id

    def test_profile_edit(self):
        """
        Simulate a edit of the email and check that data is actually changed.
        """
        edited_data = {'email': 'jira1.tvb@gmail.com'}
        cherrypy.request.method = "POST"
        self.user_c.profile(save=True, **edited_data)
        user = dao.get_user_by_id(self.test_user.id)
        assert 'jira1.tvb@gmail.com' == user.email

    def test_logout(self):
        """
        Test that a logout removes the user from session.
        """
        cherrypy.request.config = {'tools.sessions.name': 'session_id'}
        cherrypy.serving.response.cookie['session_id'] = 1
        self._expect_redirect('/user', self.user_c.logout)
        assert common.KEY_USER not in cherrypy.session, "User should be removed after logout."

    def test_switch_online_help(self):
        """
        Test the switch_online_help method and make sure it adds corresponding entry to UserPreferences.
        """
        self._expect_redirect('/user/profile', self.user_c.switch_online_help)
        assert not string2bool(self.test_user.preferences[UserPreferences.ONLINE_HELP_ACTIVE]), \
            "Online help should be switched to False."

    def test_register_cancel(self):
        """
        Test cancel on registration page.
        """
        self._expect_redirect('/user', self.user_c.register, cancel=True)

    def test_register_post_valid(self):
        """
        Test with valid data and check user is created.
        """
        cherrypy.request.method = "POST"
        data = dict(username="registered_user",
                    display_name="display_name",
                    password="pass",
                    password2="pass",
                    email="email@email.com",
                    comment="This is some dummy comment",
                    role="CLINICIAN")
        self._expect_redirect('/user', self.user_c.register, **data)
        stored_user = dao.get_user_by_name('registered_user')
        assert stored_user is not None, "New user should be saved."

    def test_register_post_invalid_data(self):
        """
        Check that errors field from template is filled in case invalid data is submitted.
        """
        cherrypy.request.method = "POST"
        # Data invalid missing username
        data = dict(password="pass",
                    password2="pass",
                    email="email@email.com",
                    comment="This is some dummy comment",
                    role="CLINICIAN")
        template_dict = self.user_c.register(**data)
        assert template_dict[common.KEY_ERRORS] != {}, "Errors should contain some data."

    def test_create_new_cancel(self):
        """
        Test that a cancel brings you back to usermanagement.
        """
        self._expect_redirect('/user/usermanagement', self.user_c.create_new, cancel=True)

    def test_create_new_valid_post(self):
        """
        Test that a valid create new post will actually create a new user in database.
        """
        data = dict(username="registered_user",
                    display_name="display_name",
                    password="pass",
                    password2="pass",
                    email="email@email.com",
                    comment="This is some dummy comment",
                    role="CLINICIAN")
        cherrypy.request.method = "POST"
        self._expect_redirect('/user/usermanagement', self.user_c.create_new, **data)
        created_user = dao.get_user_by_name("registered_user")
        assert created_user is not None, "Should have a new user created."

    def test_usermanagement_no_access(self):
        """
        Since we need to be admin to access this, we should get redirect to /tvb.
        """
        self.test_user.role = "TEST"
        self._expect_redirect('/tvb', self.user_c.usermanagement)
        assert cherrypy.session[common.KEY_MESSAGE_TYPE] == common.TYPE_ERROR

    def test_usermanagement_cancel(self):
        """
        Test that a cancel redirects us to a corresponding page.
        """
        self.test_user.role = "ADMINISTRATOR"
        self.test_user = dao.store_entity(self.test_user)
        cherrypy.session[common.KEY_USER] = self.test_user
        self._expect_redirect('/user/profile', self.user_c.usermanagement, cancel=True)

    def test_usermanagement_post_valid(self):
        """
        Create a valid post and check that user is created.
        """
        self.test_user.role = "ADMINISTRATOR"
        self.test_user = dao.store_entity(self.test_user)
        cherrypy.session[common.KEY_USER] = self.test_user
        TestFactory.create_user(username="to_be_deleted")
        TestFactory.create_user(username="to_validate", validated=False)
        user_before_delete = dao.get_user_by_name("to_be_deleted")
        assert user_before_delete is not None
        user_before_validation = dao.get_user_by_name("to_validate")
        assert not user_before_validation.validated
        data = {"delete_%i" % user_before_delete.id: True,
                "role_%i" % user_before_validation.id: "ADMINISTRATOR",
                "validate_%i" % user_before_validation.id: True}
        self.user_c.usermanagement(do_persist=True, **data)
        user_after_delete = dao.get_user_by_id(user_before_delete.id)
        assert user_after_delete is None, "User should be deleted."
        user_after_validation = dao.get_user_by_id(user_before_validation.id)
        assert user_after_validation.validated, "User should be validated now."
        assert user_after_validation.role == "ADMINISTRATOR", "Role has not changed."

    def test_recoverpassword_cancel(self):
        """
        Test that cancel redirects to user page.
        """
        cherrypy.request.method = "POST"
        self._expect_redirect('/user', self.user_c.recoverpassword, cancel=True)

    def test_recoverpassword_valid_post(self):
        """
        Test a valid password recovery.
        """
        cherrypy.request.method = "POST"
        data = {"email": self.test_user.email,
                "username": self.test_user.username,
                }
        self._expect_redirect("/user", self.user_c.recoverpassword, **data)
        assert cherrypy.session[common.KEY_MESSAGE_TYPE] == common.TYPE_INFO, \
            "Info message informing successfull reset should be present"

    def test_validate_valid(self):
        """
        Pass a valid user and test that it is actually validate.
        """
        self.test_user.role = "ADMINISTRATOR"
        self.test_user = dao.store_entity(self.test_user)
        cherrypy.session[common.KEY_USER] = self.test_user
        TestFactory.create_user(username="to_validate", validated=False)
        user_before_validation = dao.get_user_by_name("to_validate")
        assert not user_before_validation.validated
        self._expect_redirect('/tvb', self.user_c.validate, user_before_validation.username)
        user_after_validation = dao.get_user_by_id(user_before_validation.id)
        assert user_after_validation.validated, "User should be validated."
        assert cherrypy.session[common.KEY_MESSAGE_TYPE] == common.TYPE_INFO

    def test_validate_invalid(self):
        """
        Pass a valid user and test that it is actually validate.
        """
        unexisting = dao.get_user_by_name("should-not-exist")
        assert unexisting is None, "This user should not exist"
        self._expect_redirect('/tvb', self.user_c.validate, "should-not-exist")
        assert cherrypy.session[common.KEY_MESSAGE_TYPE] == common.TYPE_ERROR

    def _check_default_attributes(self, template_dict, data, errors):
        """
        Check that all the defaults are present in the template dictionary.
        """
        assert template_dict[common.KEY_LINK_ANALYZE] == '/flow/step_analyzers'
        assert template_dict[common.KEY_BACK_PAGE] == False
        assert template_dict[common.KEY_LINK_CONNECTIVITY_TAB] == '/flow/step_connectivity'
        assert template_dict[common.KEY_CURRENT_TAB] == 'nav-user'
        assert template_dict[common.KEY_FORM_DATA] == data
        assert template_dict[common.KEY_ERRORS] == errors
        assert template_dict[common.KEY_INCLUDE_TOOLTIP] is True
