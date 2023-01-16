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
import flask
import formencode
from flask_restx import Resource
from tvb.core.services.authorization import AuthorizationManager
from tvb.interfaces.rest.commons.exceptions import InvalidInputException
from tvb.interfaces.rest.commons.status_codes import HTTP_STATUS_CREATED
from tvb.interfaces.rest.commons.strings import FormKeyInput, Strings
from tvb.interfaces.rest.server.facades.project_facade import ProjectFacade
from tvb.interfaces.rest.server.facades.user_facade import UserFacade
from tvb.interfaces.rest.server.request_helper import get_current_user
from tvb.interfaces.rest.server.resources.rest_resource import RestResource

USERS_PAGE_SIZE = 30


class LoginUserResource(Resource):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def post(self):
        """
        Authorize user in the configured Keycloak server
        :return a dict which contains user's tokens
        """
        try:
            data = flask.request.json
            code = data[FormKeyInput.CODE.value]
            redirect_uri = data[FormKeyInput.REDIRECT_URI.value]
            return AuthorizationManager.get_keycloak_instance().token(code=code,
                                                                      grant_type=["authorization_code"],
                                                                      redirect_uri=redirect_uri)
        except KeyError:
            raise InvalidInputException("Invalid input.")

    def put(self):
        """
        Refresh user's token
        :return: new token
        """
        data = flask.request.json
        try:
            refresh_token = data[FormKeyInput.KEYCLOAK_REFRESH_TOKEN.value]
            return AuthorizationManager.get_keycloak_instance().refresh_token(refresh_token)
        except KeyError:
            raise InvalidInputException("Invalid refresh token input.")

    def delete(self):
        """
        Logout user. Invalidate token
        :return:
        """
        data = flask.request.json
        try:
            refresh_token = data[FormKeyInput.KEYCLOAK_REFRESH_TOKEN.value]
            return AuthorizationManager.get_keycloak_instance().logout(refresh_token)
        except KeyError:
            raise InvalidInputException("Invalid refresh token input.")


class GetUsersResource(RestResource):
    def get(self):
        """
        :return: a list of TVB users
        """
        page_number = self.extract_page_number()

        user_dto_list, pages_no = UserFacade.get_users(get_current_user().username, page_number, USERS_PAGE_SIZE)
        return {"users": user_dto_list, "pages_no": pages_no}


class GetProjectsListResource(RestResource):

    def get(self):
        """
        :return a list of logged user's projects
        """
        return ProjectFacade.retrieve_logged_user_projects(get_current_user().id)

    def post(self):
        """
        Create a new project linked to the current user
        """
        input_data = flask.request.json
        try:
            project_name = input_data[FormKeyInput.CREATE_PROJECT_NAME.value]
            project_description = input_data[FormKeyInput.CREATE_PROJECT_DESCRIPTION.value] \
                if FormKeyInput.CREATE_PROJECT_DESCRIPTION.value in input_data else ""
            try:
                project_gid = ProjectFacade().create_project(get_current_user(), project_name, project_description)
                return project_gid, HTTP_STATUS_CREATED
            except formencode.Invalid as exception:
                raise InvalidInputException(exception.msg)
        except KeyError:

            raise InvalidInputException("Invalid create project input.")


class LinksResource(Resource):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def get(self):
        redirect_uri = flask.request.args.get(FormKeyInput.REDIRECT_URI.value)
        if redirect_uri is None:
            raise InvalidInputException(message="Invalid redirect uri")
        keycloak_instance = AuthorizationManager.get_keycloak_instance()
        auth_url = keycloak_instance.auth_url(redirect_uri) + "&scope=openid profile email"
        account_url = keycloak_instance.connection.base_url + "realms/{}/account".format(keycloak_instance.realm_name)
        return {
            Strings.AUTH_URL.value: auth_url,
            Strings.ACCOUNT_URL.value: account_url
        }
