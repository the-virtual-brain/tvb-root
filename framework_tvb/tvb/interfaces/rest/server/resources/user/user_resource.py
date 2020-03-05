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
import formencode
import flask
from flask_restplus import Resource
from tvb.core.services.project_service import ProjectService
from tvb.interfaces.rest.commons.dtos import ProjectDto
from tvb.interfaces.rest.commons.exceptions import InvalidInputException
from tvb.interfaces.rest.commons.status_codes import HTTP_STATUS_CREATED
from tvb.interfaces.rest.commons.strings import FormKeyInput
from tvb.interfaces.rest.server.resources.rest_resource import RestResource
from tvb.interfaces.rest.server.security.authorization import get_current_user, AuthorizationManager

USERS_PAGE_SIZE = 1000


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
            return AuthorizationManager.get_keycloak_instance().token(data[FormKeyInput.USERS_USERNAME.value],
                                                                      data[FormKeyInput.USERS_PASSWORD.value])
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


class GetProjectsListResource(RestResource):

    def get(self):
        """
        :return a list of logged user's projects
        """
        user = get_current_user()
        projects = ProjectService.retrieve_all_user_projects(user_id=user.id)
        return [ProjectDto(project) for project in projects]

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
                db_project = ProjectService().store_project(get_current_user(), True, None, name=project_name,
                                                            description=project_description, users=[])
                return db_project.gid, HTTP_STATUS_CREATED
            except formencode.Invalid as excep:
                raise InvalidInputException(excep.msg)
        except KeyError:
            raise InvalidInputException("Invalid create project input.")
