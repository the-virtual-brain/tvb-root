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
Service layer for USER entities. 
   
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import os
import random
from random import randint

import six
import tvb_data
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.config import DEFAULT_PROJECT_GID
from tvb.core.entities.model.model_project import User, ROLE_ADMINISTRATOR, USER_ROLES
from tvb.core.entities.storage import dao
from tvb.core.services import email_sender
from tvb.core.services.exceptions import UsernameException
from tvb.core.services.import_service import ImportService
from tvb.core.services.settings_service import SettingsService
from tvb.core.utils import hash_password

FROM_ADDRESS = 'donotreply@thevirtualbrain.org'
SUBJECT_REGISTER = '[TVB] Registration Confirmation'
SUBJECT_VALIDATE = '[TVB] Account validated'
SUBJECT_RECOVERY = '[TVB] Recover password'
TEXT_RECOVERY = 'Hi %s,\n\n' \
                'According to your recent request, a new password was generated for your user, by the system.\n' \
                'Please login with the below password and change it into one you can easily remember.\n\n ' \
                'The new password is: %s\n\n' \
                'TVB Team.'
TEXT_DISPLAY = "Thank you! Please check your email for further details!"
TEXT_CREATE = (',\n\nYour registration has been notified to the administrators '
               + 'of The Virtual Brain Project; you will receive an email as '
               + 'soon as the administrator has validated your registration.'
               + ' \n\nThank you for registering!\nTVB Team')
TEXT_CREATE_TO_ADMIN = 'New member requires validation. Go to this url to validate '
TEXT_VALIDATED = ',\n\nYour registration has been validated by TVB Administrator, Please proceed with the login at '
KEY_USERNAME = "username"
KEY_PASSWORD = "password"
KEY_EMAIL = "email"
KEY_ROLE = "role"
KEY_COMMENT = "comment"
DEFAULT_PASS_LENGTH = 10
USERS_PAGE_SIZE = 7


class UserService:
    """
    CRUD methods for USER entities are here.
    """
    USER_ROLES = USER_ROLES

    def __init__(self):
        self.logger = get_logger(self.__class__.__module__)

    def create_user(self, username=None, password=None, password2=None,
                    role=None, email=None, comment=None, email_msg=None, validated=False, skip_import=False,
                    external_id=None, skip_sending_email=False):
        """
        Service Layer for creating a new user.
        """
        if (username is None) or len(username) < 1:
            raise UsernameException("Empty UserName!")
        if (password is None) or len(password) < 1:
            raise UsernameException("Empty password!")
        if password2 is None:
            password2 = password
        if password != password2:
            raise UsernameException("Passwords do not match!")

        try:
            user_validated = (role == ROLE_ADMINISTRATOR) or validated
            user = User(username, password, email, user_validated, role, external_id)
            if email_msg is None:
                email_msg = 'Hello ' + username + TEXT_CREATE
            admin_msg = (TEXT_CREATE_TO_ADMIN + username + ' :\n ' + TvbProfile.current.web.BASE_URL +
                         'user/validate/' + username + '\n\n"' + str(comment) + '"')
            self.logger.info("Registering user " + username + " !")

            if role != ROLE_ADMINISTRATOR and email is not None and not skip_sending_email:
                admins = UserService.get_administrators()
                admin = admins[randint(0, len(admins) - 1)]
                if admin.email is not None and (admin.email != TvbProfile.current.web.admin.DEFAULT_ADMIN_EMAIL):
                    # Do not send validation email in case default admin email remained unchanged
                    email_sender.send(FROM_ADDRESS, admin.email, SUBJECT_REGISTER, admin_msg)
                    self.logger.debug("Email sent to:" + admin.email + " for validating user:" + username + " !")
                email_sender.send(FROM_ADDRESS, email, SUBJECT_REGISTER, email_msg)
                self.logger.debug("Email sent to:" + email + " for notifying new user:" + username + " !")

            user = dao.store_entity(user)

            if role == ROLE_ADMINISTRATOR and not skip_import:
                to_upload = os.path.join(os.path.dirname(tvb_data.__file__), "Default_Project.zip")
                if not os.path.exists(to_upload):
                    self.logger.warning("Could not find DEFAULT PROJECT at path %s. You might want to import it "
                                        "yourself. See TVB documentation about where to find it!" % to_upload)
                    return TEXT_DISPLAY
                ImportService().import_project_structure(to_upload, user.id)
            else:
                try:
                    default_prj_id = dao.get_project_by_gid(DEFAULT_PROJECT_GID).id
                    dao.add_members_to_project(default_prj_id, [user.id])
                except Exception:
                    self.logger.warning(
                        "Could not link user_id: %d with project_gid: %s " % (user.id, DEFAULT_PROJECT_GID))

            return TEXT_DISPLAY
        except Exception as excep:
            self.logger.exception("Could not create user!")
            raise UsernameException(str(excep))

    def reset_password(self, **data):
        """
        Service Layer for resetting a password.
        """
        if (KEY_EMAIL not in data) or len(data[KEY_EMAIL]) < 1:
            raise UsernameException("Empty Email!")

        old_pass, user = None, None
        try:
            email = data[KEY_EMAIL]
            name_hint = data[KEY_USERNAME]
            user = dao.get_user_by_email(email, name_hint)
            if user is None:
                raise UsernameException("No singular user could be found for the given data!")

            old_pass = user.password
            new_pass = ''.join(chr(randint(48, 122)) for _ in range(DEFAULT_PASS_LENGTH))
            user.password = hash_password(new_pass)
            self.edit_user(user, old_pass)
            self.logger.info("Resetting password for email : " + email)
            email_sender.send(FROM_ADDRESS, email, SUBJECT_RECOVERY, TEXT_RECOVERY % (user.username, new_pass))
            return TEXT_DISPLAY
        except Exception as excep:
            if old_pass and len(old_pass) > 1 and user:
                user.password = old_pass
                dao.store_entity(user)
            self.logger.exception("Could not change user password!")
            raise UsernameException(excep)

    @staticmethod
    def is_username_valid(name):
        """
        Service layer for checking if a given UserName is unique or not.
        """
        users_no = dao.count_users_for_name(name)
        if users_no > 0:
            return False
        return True

    def validate_user(self, name='', user_id=None):
        """
        Service layer for editing a user and validating the account.
        """
        try:
            if user_id:
                user = dao.get_user_by_id(user_id)
            else:
                user = dao.get_user_by_name(name)
            if user is None or user.validated:
                self.logger.debug("UserName not found or already validated:" + name)
                return False
            user.validated = True
            user = dao.store_entity(user)
            self.logger.debug("Sending validation email for userName=" + name + " to address=" + user.email)
            email_sender.send(FROM_ADDRESS, user.email, SUBJECT_VALIDATE,
                              "Hello " + name + TEXT_VALIDATED + TvbProfile.current.web.BASE_URL + "user/")
            self.logger.info("User:" + name + " was validated successfully" + " and notification email sent!")
            return True
        except Exception as excep:
            self.logger.warning('Could not validate user:')
            self.logger.warning('WARNING : ' + str(excep))
            return False

    @staticmethod
    def check_login(username, password):
        """
        Service layer to check if given UserName and Password are according to DB.
        """
        user = dao.get_user_by_name(username)
        if user is not None and user.password == hash_password(password) and user.validated:
            return user
        else:
            return None

    def get_users_for_project(self, user_name, project_id, page=1):
        """
        Return tuple: (All Users except the project administrator, Project Members).
        Parameter "user_name" is the current user.
        Parameter "user_name" is used for new projects (project_id is None).
        When "project_id" not None, parameter "user_name" is ignored.
        """
        try:
            admin_name = user_name
            if project_id is not None:
                project = dao.get_project_by_id(project_id)
                if project is not None:
                    admin_name = project.administrator.username
            all_users, total_pages = self.retrieve_all_users(admin_name, page)
            members = dao.get_members_of_project(project_id)
            return all_users, members, total_pages
        except Exception as excep:
            self.logger.exception("Invalid userName or project identifier")
            raise UsernameException(str(excep))

    @staticmethod
    def retrieve_all_users(username, current_page=1):
        """
        Return all users from the database except the given user
        """
        start_idx = USERS_PAGE_SIZE * (current_page - 1)
        total = dao.get_all_users(username, is_count=True)
        user_list = dao.get_all_users(username, start_idx, USERS_PAGE_SIZE)
        pages_no = total // USERS_PAGE_SIZE + (1 if total % USERS_PAGE_SIZE else 0)
        return user_list, pages_no

    @staticmethod
    def fetch_all_users(page_start=0, page_size=USERS_PAGE_SIZE):
        """
        Return all users from the database without pagination
        """
        return dao.get_all_users(page_size=page_size, page_start=page_start)

    def edit_user(self, edited_user, old_password=None):
        """
        Retrieve a user by and id, then modify it's role and validate status.
        """
        if edited_user.validated:
            self.validate_user(user_id=edited_user.id)
        user = dao.get_user_by_id(edited_user.id)
        user.role = edited_user.role
        user.validated = edited_user.validated
        if old_password is not None:
            if user.password == old_password:
                user.password = edited_user.password
            else:
                raise UsernameException("Invalid old password!")
        user.email = edited_user.email
        for key, value in six.iteritems(edited_user.preferences):
            user.preferences[key] = value
        dao.store_entity(user)
        if user.is_administrator():
            TvbProfile.current.manager.add_entries_to_config_file({SettingsService.KEY_ADMIN_EMAIL: user.email,
                                                                   SettingsService.KEY_ADMIN_PWD: user.password})

    def delete_user(self, user_id):
        """
        Delete a user with a given ID.
        Return True when successfully, or False."""
        try:
            dao.remove_entity(User, user_id)
            return True
        except Exception as excep:
            self.logger.exception(excep)
            return False

    @staticmethod
    def get_administrators():
        """Retrieve system administrators.
        Will be used for sending emails, for example."""
        return dao.get_administrators()

    @staticmethod
    def save_project_to_user(user_id, project_id):
        """
        Mark for current user that the given project is the last one selected.
        """
        user = dao.get_user_by_id(user_id)
        user.selected_project = project_id
        dao.store_entity(user)

    @staticmethod
    def get_user_by_id(user_id):
        """
        Retrieves a user by its id.
        """
        return dao.get_user_by_id(user_id)

    @staticmethod
    def get_user_by_name(username):
        """
        Retrieves a user by its username.
        """
        return dao.get_user_by_name(username)

    @staticmethod
    def get_user_by_external_id(external_id):
        """
        Retrieves a user by its external id.
        """
        return dao.get_user_by_external_id(external_id)

    @staticmethod
    def compute_user_generated_disk_size(user_id):
        return dao.compute_user_generated_disk_size(user_id)

    def create_external_service_user(self, user_data):
        external_id = user_data['sub']
        email = user_data['email'] if 'email' in user_data else None
        self.create_user(external_id, hash_password(''.join(random.sample(external_id, len(external_id)))),
                         external_id=external_id, email=email,
                         validated=True, skip_sending_email=True)
        return self.get_user_by_external_id(external_id)
