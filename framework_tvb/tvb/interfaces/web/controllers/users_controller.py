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
Here, user related tasks are described.
Basic authentication processes are described here, 
but also user related annotation (checked-logged).

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import json
import os
import ssl
import time
from urllib.request import urlopen

import cherrypy
import formencode
import tvb.interfaces.web
from formencode import validators
from tvb.basic.profile import TvbProfile
from tvb.core.entities.file.files_update_manager import FilesUpdateManager
from tvb.core.services.authorization import AuthorizationManager
from tvb.core.services.exceptions import UsernameException
from tvb.core.services.project_service import ProjectService
from tvb.core.services.texture_to_json import color_texture_to_list
from tvb.core.services.user_service import KEY_AUTH_TOKEN, USERS_PAGE_SIZE
from tvb.core.services.user_service import UserService, KEY_PASSWORD, KEY_EMAIL, KEY_USERNAME, KEY_COMMENT
from tvb.core.utils import format_bytes_human, hash_password
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.base_controller import BaseController
from tvb.interfaces.web.controllers.decorators import check_user, expose_json, check_admin
from tvb.interfaces.web.controllers.decorators import handle_error, using_template, settings, jsonify
from tvb.interfaces.web.entities.context_simulator import SimulatorContext

KEY_SERVER_VERSION = "versionInfo"
KEY_CURRENT_VERSION_FULL = "currentVersionLongText"
KEY_STORAGE_IN_UPDATE = "isStorageInUpdate"


class UserController(BaseController):
    """
    This class takes care of the user authentication and/or register.
    """

    def __init__(self):
        BaseController.__init__(self)
        self.version_info = None

    @cherrypy.expose
    @handle_error(redirect=True)
    @using_template('user/base_user')
    @settings
    def index(self, **data):
        """
        Login page (with or without messages).
        """
        template_specification = dict(mainContent="user/login", title="Login", data=data)
        if cherrypy.request.method == 'POST':
            keycloak_login = TvbProfile.current.KEYCLOAK_LOGIN_ENABLED
            form = LoginForm() if not keycloak_login else KeycloakLoginForm()

            try:
                data = form.to_python(data)
                if keycloak_login:
                    auth_token = data[KEY_AUTH_TOKEN]
                    kc_user_info = AuthorizationManager(
                        TvbProfile.current.KEYCLOAK_WEB_CONFIG).get_keycloak_instance().userinfo(auth_token)
                    user = self.user_service.get_external_db_user(kc_user_info)
                else:
                    username = data[KEY_USERNAME]
                    password = data[KEY_PASSWORD]
                    user = self.user_service.check_login(username, password)
                if user is not None:
                    common.add2session(common.KEY_USER, user)
                    common.set_info_message('Welcome ' + user.display_name)
                    self.logger.debug("User " + user.username + " has just logged in!")
                    if user.selected_project is not None:
                        prj = user.selected_project
                        prj = ProjectService().find_project(prj)
                        self._mark_selected(prj)
                    raise cherrypy.HTTPRedirect('/user/profile')
                elif not keycloak_login:
                    common.set_error_message('Wrong username/password, or user not yet validated...')
                    self.logger.debug("Wrong username " + username + " !!!")
                else:
                    common.set_error_message(
                        'Your account is not validated. Please contact us at support@thevirtualbrain.org for more details')
                    self.logger.debug("Invalidated account")
                    template_specification[common.KEY_ERRORS] = {'invalid_user': True}
            except formencode.Invalid as excep:
                template_specification[common.KEY_ERRORS] = excep.unpack_errors()

        return self.fill_default_attributes(template_specification)

    @cherrypy.expose
    @handle_error(redirect=True)
    @using_template('user/base_user')
    @check_user
    @settings
    def profile(self, logout=False, save=False, **data):
        """
        Display current user's profile page.
        On POST: logout, or save password/email.
        """
        if cherrypy.request.method == 'POST' and logout:
            raise cherrypy.HTTPRedirect('/user/logout')
        template_specification = dict(mainContent="user/profile", title="User Profile")
        user = common.get_logged_user()

        if cherrypy.request.method == 'POST' and save:
            try:
                form = EditUserForm()
                data = form.to_python(data)
                if data.get(KEY_PASSWORD):
                    user.password = hash_password(data[KEY_PASSWORD])
                if data.get(KEY_EMAIL):
                    user.email = data[KEY_EMAIL]
                old_password = None
                if data.get('old_password'):
                    old_password = hash_password(data['old_password'])
                self.user_service.edit_user(user, old_password)
                if old_password:
                    common.set_info_message("Changes Submitted!")
                else:
                    common.set_info_message("Submitted!  No password changed.")
            except formencode.Invalid as excep:
                template_specification[common.KEY_ERRORS] = excep.unpack_errors()
            except UsernameException as excep:
                self.logger.exception(excep)
                user = common.get_logged_user()
                common.add2session(common.KEY_USER, self.user_service.get_user_by_id(user.id))
                common.set_error_message("Could not save changes. Probably wrong old password!!")
        else:
            # Update session user since disk size might have changed from last time to profile.
            user = self.user_service.get_user_by_id(user.id)
            common.add2session(common.KEY_USER, user)

        template_specification['user_used_disk_human'] = format_bytes_human(
            self.user_service.compute_user_generated_disk_size(user.id))
        return self.fill_default_attributes(template_specification)

    @cherrypy.expose
    @using_template('user/silent_check_sso')
    def check_sso(self):
        return {}

    @cherrypy.expose
    @handle_error(redirect=True)
    @check_user
    def logout(self):
        """
        Logging out user and clean session
        """
        user = common.remove_from_session(common.KEY_USER)
        if user is not None:
            self.logger.debug("User " + user.username + " is just logging out!")
        SimulatorContext().clean_project_data_from_session()
        common.set_info_message("Thank you for using The Virtual Brain!")

        common.expire_session()
        raise cherrypy.HTTPRedirect("/user")

    @cherrypy.expose
    @handle_error(redirect=False)
    @jsonify
    def keycloak_web_config(self):
        file_path = TvbProfile.current.KEYCLOAK_WEB_CONFIG
        with open(file_path) as f:
            return json.load(f)

    @cherrypy.expose
    @handle_error(redirect=True)
    @check_user
    def switch_online_help(self):
        """
        Switch flag that displays online helps
        """
        user = common.get_logged_user()

        # Change OnlineHelp Active flag and save user
        user.switch_online_help_state()
        self.user_service.edit_user(user)
        raise cherrypy.HTTPRedirect("/user/profile")

    @expose_json
    def get_viewer_color_scheme(self):
        user = common.get_logged_user()
        return user.get_viewers_color_scheme()

    @expose_json
    def set_viewer_color_scheme(self, color_scheme_name):
        user = common.get_logged_user()
        user.set_viewers_color_scheme(color_scheme_name)
        self.user_service.edit_user(user)

    @expose_json
    def get_color_schemes_json(self):
        cherrypy.response.headers['Cache-Control'] = 'max-age=86400'  # cache for a day
        pth = os.path.join(os.path.dirname(tvb.interfaces.web.__file__), 'static', 'coloring', 'color_schemes.png')
        return color_texture_to_list(pth, 256, 8)

    @cherrypy.expose
    @handle_error(redirect=True)
    @using_template('user/base_user')
    def register(self, cancel=False, **data):
        """
        This register form send an e-mail to the user and to the site admin.
        """
        template_specification = dict(mainContent="user/register", title="Register", data=data)
        redirect = False
        if cherrypy.request.method == 'POST':
            if cancel:
                raise cherrypy.HTTPRedirect('/user')
            try:
                okmessage = self._create_user(**data)
                common.set_info_message(okmessage)
                redirect = True
            except formencode.Invalid as excep:
                template_specification[common.KEY_ERRORS] = excep.unpack_errors()
                redirect = False
            except Exception as excep1:
                self.logger.error("Could not create user:" + data["username"])
                self.logger.exception(excep1)
                common.set_error_message("We are very sorry, but we could not create your user. Most probably is "
                                         "because it was impossible to sent emails. Please try again later...")
                redirect = False

        if redirect:
            # Redirect to login page, with some success message to display
            raise cherrypy.HTTPRedirect('/user')
        else:
            # Stay on the same page
            return self.fill_default_attributes(template_specification)

    @cherrypy.expose
    @handle_error(redirect=True)
    @using_template('user/base_user')
    def create_new(self, cancel=False, **data):
        """
        Create new user with data submitted from UI.
        """
        if cancel:
            raise cherrypy.HTTPRedirect('/user/usermanagement')
        template_specification = dict(mainContent="user/create_new", title="Create New", data=data)
        redirect = False
        if cherrypy.request.method == 'POST':
            try:
                data[KEY_COMMENT] = "Created by administrator."
                # User is created by administrator, should be validated automatically, and credentials
                # should be sent to user by email.
                email_msg = """A TVB account was just created for you by an administrator. 
                \n Your credentials are username=%s, password=%s. 
                \n You can log in here: %s.
                """ % (data[KEY_USERNAME], data[KEY_PASSWORD], TvbProfile.current.web.BASE_URL)
                self._create_user(email_msg=email_msg, validated=True, **data)
                common.set_info_message("New user created successfully.")
                redirect = True
            except formencode.Invalid as excep:
                template_specification[common.KEY_ERRORS] = excep.unpack_errors()
            except Exception as excep:
                self.logger.exception(excep)
                common.set_error_message("We are very sorry, but we could not create your user. Most probably is "
                                         "because it was impossible to send emails. Please try again later...")
        if redirect:
            raise cherrypy.HTTPRedirect('/user/usermanagement')
        else:
            return self.fill_default_attributes(template_specification)

    @cherrypy.expose
    @handle_error(redirect=True)
    @using_template('user/base_user')
    @check_admin
    def usermanagement(self, cancel=False, page=1, do_persist=False, **data):
        """
        Display a table used for user management.
        """
        if cancel:
            raise cherrypy.HTTPRedirect('/user/profile')

        page = int(page)
        if cherrypy.request.method == 'POST' and do_persist:
            not_deleted = 0
            for key in data:
                user_id = int(key.split('_')[1])
                if 'delete_' in key:
                    self.user_service.delete_user(user_id)
                if ("role_" in key) and not (("delete_" + str(user_id)) in data):
                    valid = ("validate_" + str(user_id)) in data
                    user = self.user_service.get_user_by_id(user_id)
                    user.role = data[key] if data[key] != "None" else None
                    user.validated = valid
                    self.user_service.edit_user(user)
                    not_deleted += 1
            # The entire current page was deleted, go to previous page
            if not_deleted == 0 and page > 1:
                page -= 1

        admin_ = common.get_logged_user().username
        except_usernames = [admin_]
        if TvbProfile.current.KEYCLOAK_LOGIN_ENABLED:
            except_usernames.append(TvbProfile.current.web.admin.ADMINISTRATOR_NAME)
        user_list, pages_no = self.user_service.retrieve_users_except(except_usernames, page, USERS_PAGE_SIZE)
        allRoles = [None]
        allRoles.extend(UserService.USER_ROLES)

        template_specification = dict(mainContent="user/user_management", title="Users management", page_number=page,
                                      total_pages=pages_no, userList=user_list, allRoles=allRoles,
                                      data={})
        return self.fill_default_attributes(template_specification)

    @cherrypy.expose
    @handle_error(redirect=True)
    @using_template('user/base_user')
    def recoverpassword(self, cancel=False, **data):
        """
        This form should reset a password for a given userName/email and send a
        notification message to that email.
        """
        template_specification = dict(mainContent="user/recover_password", title="Recover password", data=data)
        redirect = False
        if cherrypy.request.method == 'POST':
            if cancel:
                raise cherrypy.HTTPRedirect('/user')
            form = RecoveryForm()
            try:
                data = form.to_python(data)
                okmessage = self.user_service.reset_password(**data)
                common.set_info_message(okmessage)
                redirect = True
            except formencode.Invalid as excep:
                template_specification[common.KEY_ERRORS] = excep.unpack_errors()
                redirect = False
            except UsernameException as excep1:
                self.logger.exception("Could not reset password!")
                common.set_error_message(excep1.message)
                redirect = False
        if redirect:
            # Redirect to login page, with some success message to display
            raise cherrypy.HTTPRedirect('/user')
        else:
            # Stay on the same page
            return self.fill_default_attributes(template_specification)

    @cherrypy.expose
    @handle_error(redirect=False)
    @jsonify
    def is_storage_ready(self):
        """
        Check if all storage updates are done
        """
        while TvbProfile.current.version.DATA_CHECKED_TO_VERSION < TvbProfile.current.version.DATA_VERSION:
            time.sleep(2)

        return dict(message=FilesUpdateManager.MESSAGE, status=FilesUpdateManager.STATUS)

    @cherrypy.expose
    @handle_error(redirect=True)
    @check_admin
    def validate(self, name):
        """
        A link to this page will be sent to the administrator to validate
        the registration of each user.
        """
        success = self.user_service.validate_user(name)
        if not success:
            common.set_error_message("Problem validating user:" + name + "!! Please check logs.")
            self.logger.error("Problem validating user " + name)
        else:
            common.set_info_message("User Validated successfully and notification email sent!")
        raise cherrypy.HTTPRedirect('/tvb')

    @cherrypy.expose
    def base_url(self, **data):
        if not TvbProfile.current.web.BASE_URL:
            url = data['url']
            self.logger.info("Set base url to {}".format(url))
            TvbProfile.current.web.BASE_URL = url

    def _create_user(self, email_msg=None, validated=False, **data):
        """
        Just create a user given the data input. Do form validation beforehand.
        """
        form = RegisterForm()
        data = form.to_python(data)
        data[KEY_PASSWORD] = hash_password(data[KEY_PASSWORD])
        data['password2'] = hash_password(data['password2'])
        return self.user_service.create_user(email_msg=email_msg, validated=validated, **data)

    def fill_default_attributes(self, template_dictionary):
        """
        Fill into 'template_dictionary' data that we want to have ready in UI.
        """
        template_dictionary = self._populate_version(template_dictionary)
        BaseController.fill_default_attributes(self, template_dictionary)
        template_dictionary[common.KEY_INCLUDE_TOOLTIP] = True
        template_dictionary[common.KEY_WRAP_CONTENT_IN_MAIN_DIV] = False
        template_dictionary[common.KEY_CURRENT_TAB] = 'nav-user'
        template_dictionary[KEY_STORAGE_IN_UPDATE] = (TvbProfile.current.version.DATA_CHECKED_TO_VERSION <
                                                      TvbProfile.current.version.DATA_VERSION)
        return template_dictionary

    def _populate_version(self, template_dictionary):
        """
        Fill in template information about current version available online.
        """
        content = ""
        if self.version_info is None:
            try:
                context = ssl.SSLContext(ssl.PROTOCOL_TLS)
                content = urlopen(TvbProfile.current.web.URL_TVB_VERSION, timeout=7, context=context).read()
                self.version_info = json.loads(content.decode('utf-8'))
                pos = TvbProfile.current.web.URL_TVB_VERSION.find('/tvb')
                self.version_info['url'] = TvbProfile.current.web.URL_TVB_VERSION[:pos]
                self.logger.debug("Read version: " + json.dumps(self.version_info))
            except Exception as excep:
                self.logger.warning("Could not read current version from remote server!")
                self.logger.debug(str(content))
                self.logger.exception(excep)
                self.version_info = {}
        template_dictionary[KEY_SERVER_VERSION] = self.version_info
        template_dictionary[KEY_CURRENT_VERSION_FULL] = TvbProfile.current.version.CURRENT_VERSION
        return template_dictionary


class LoginForm(formencode.Schema):
    """
    Validate for Login UI Form
    """
    empty_msg = 'Please enter a value'
    username = validators.UnicodeString(not_empty=True, use_builtins_gettext=False, messages={'empty': empty_msg})
    password = validators.UnicodeString(not_empty=True, use_builtins_gettext=False, messages={'empty': empty_msg})


class KeycloakLoginForm(formencode.Schema):
    """
        Validate for Login UI Form
    """
    empty_msg = 'Please enter a value'
    auth_token = validators.UnicodeString(not_empty=True, use_builtins_gettext=False, messages={'empty': empty_msg})


class UniqueUsername(formencode.FancyValidator):
    """
    Custom validator to check that a given user-name is unique.
    """

    def _convert_to_python(self, value, state):
        """ Fancy validate for Unique user-name """
        if not UserService().is_username_valid(value):
            raise formencode.Invalid('Please choose another user-name, this one is already in use!', value, state)
        return value


class RegisterForm(formencode.Schema):
    """
    Validate Register Form
    """
    username = formencode.All(validators.UnicodeString(not_empty=True), validators.PlainText(), UniqueUsername())
    display_name = validators.UnicodeString(not_empty=True)
    password = validators.UnicodeString(not_empty=True)
    password2 = validators.UnicodeString(not_empty=True)
    email = validators.Email(not_empty=True)
    comment = validators.UnicodeString()
    role = validators.UnicodeString()
    chained_validators = [validators.FieldsMatch('password', 'password2')]


class RecoveryForm(formencode.Schema):
    """
    Validate Recover Password Form
    """
    email = validators.Email(not_empty=True)
    username = validators.String(not_empty=False)


class EditUserForm(formencode.Schema):
    """   
    Validate fields on user-edit
    """
    old_password = validators.UnicodeString(if_missing=None)
    password = validators.UnicodeString(if_missing=None)
    password2 = validators.UnicodeString(if_missing=None)
    email = validators.Email(if_missing=None)
    chained_validators = [validators.FieldsMatch('password', 'password2'),
                          validators.RequireIfPresent('password', present='old_password')]
