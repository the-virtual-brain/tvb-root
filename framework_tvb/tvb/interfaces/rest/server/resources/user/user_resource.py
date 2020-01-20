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
from tvb.core.services.project_service import ProjectService
from tvb.core.services.user_service import UserService
from tvb.interfaces.rest.commons.dtos import UserDto, ProjectDto
from tvb.interfaces.rest.commons.exceptions import InvalidIdentifierException
from tvb.interfaces.rest.server.resources.rest_resource import RestResource


class GetUsersResource(RestResource):

    def get(self):
        """
        :return a list of system's users
        """
        users = UserService.fetch_all_users()
        return [UserDto(user) for user in users]


class GetProjectsListResource(RestResource):

    def get(self, username):
        """
        :return a list of user's projects
        """
        user = UserService.get_user_by_name(username)
        if user is None:
            raise InvalidIdentifierException('No user registered with username: %s' % username)
        projects = ProjectService.retrieve_all_user_projects(user_id=user.id)
        return [ProjectDto(project) for project in projects]
