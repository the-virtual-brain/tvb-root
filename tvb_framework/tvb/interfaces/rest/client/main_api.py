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
from datetime import datetime, timedelta

import requests
from tvb.interfaces.rest.client.client_decorators import handle_response
from tvb.interfaces.rest.commons.strings import Strings, RestLink, FormKeyInput


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
        self.token_expiry_date = None
        self.refresh_token = None

    def build_request_url(self, url):
        return self.server_url + url

    def secured_request(self):
        """
        If we have an expiration date and the token is expired we make a refresh call then we build a secured request
        based on the refreshed token.
        :return: secured request session
        """
        if self.token_expiry_date is not None and datetime.now() >= self.token_expiry_date \
                and self.refresh_token is not None:
            refresh_token_response = self._refresh_token()
            self.update_tokens(refresh_token_response)
        return self._build_request()

    def _build_request(self):
        """
        Build a secured request protected by the authorization token set before, used in the entire session
        :return: secured requests session
        """

        authorization_header = {Strings.AUTH_HEADER.value: Strings.BEARER.value + self.authorization_token}
        with requests.Session() as request_session:
            request_session.headers.update(authorization_header)
            return request_session

    @handle_response
    def _refresh_token(self):
        return self._build_request().put(self.build_request_url(RestLink.LOGIN.compute_url(True)), json={
            FormKeyInput.KEYCLOAK_REFRESH_TOKEN.value: self.refresh_token,
        })

    def update_tokens(self, response):
        self.refresh_token = response['refresh_token']
        self.authorization_token = response['access_token']
        expires_in = response['expires_in']
        self.token_expiry_date = datetime.now() + timedelta(seconds=expires_in)
