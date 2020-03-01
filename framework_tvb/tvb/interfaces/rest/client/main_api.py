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
import requests
from tvb.interfaces.rest.commons.strings import Strings


class MainApi:
    """
    Base API class which will be inherited by all the API specific subclasses
    """

    def __init__(self, server_url, auth_token=''):
        # type: (str, str) -> None
        """
        Rest server url, where all rest calls will be made
        :param server_url: REST server base URL
        :param auth_token: Keycloak authorization token
        """
        self.server_url = server_url + "/" + Strings.BASE_PATH.value
        self.authorization_token = auth_token

    def build_request_url(self, url):
        return self.server_url + url

    def secured_request(self):
        """
        Build a secured request protected by the authorization token set before, used in the entire session
        :return: secured requests session
        """

        authorization_header = {Strings.AUTH_HEADER.value: Strings.BEARER.value + self.authorization_token}
        with requests.Session() as request_session:
            request_session.headers.update(authorization_header)
            return request_session
