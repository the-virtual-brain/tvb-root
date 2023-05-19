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
import threading
import urllib.parse as parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from time import sleep

from tvb.interfaces.rest.client.client_decorators import handle_response
from tvb.interfaces.rest.client.main_api import MainApi
from tvb.interfaces.rest.commons.dtos import ProjectDto
from tvb.interfaces.rest.commons.status_codes import HTTP_STATUS_REDIRECT, HTTP_STATUS_BAD_REQUEST
from tvb.interfaces.rest.commons.strings import RestLink, FormKeyInput, RestNamespace, Strings


class UserApi(MainApi):
    @handle_response
    def login(self, code, redirect_uri):
        response = self.secured_request().post(self.build_request_url(RestLink.LOGIN.compute_url(True)), json={
            FormKeyInput.CODE.value: code,
            FormKeyInput.REDIRECT_URI.value: redirect_uri
        })
        return response

    @handle_response
    def logout(self):
        response = self.secured_request().delete(self.build_request_url(RestLink.LOGIN.compute_url(True)), json={
            FormKeyInput.KEYCLOAK_REFRESH_TOKEN.value: self.refresh_token,
        })
        return response

    @handle_response
    def get_projects_list(self):
        response = self.secured_request().get(self.build_request_url(RestLink.PROJECTS.compute_url(True)))
        return response, ProjectDto

    @handle_response
    def get_users(self, page):
        return self.secured_request().get(self.build_request_url(RestNamespace.USERS.value), params={
            Strings.PAGE_NUMBER: page
        })

    @handle_response
    def create_project(self, project_name, project_description):
        return self.secured_request().post(self.build_request_url(RestLink.PROJECTS.compute_url(True)), json={
            FormKeyInput.CREATE_PROJECT_NAME.value: project_name,
            FormKeyInput.CREATE_PROJECT_DESCRIPTION.value: project_description
        })

    @handle_response
    def _get_urls(self, redirect_uri):
        return self.secured_request().get(self.build_request_url(RestLink.USEFUL_URLS.compute_url(True)), params={
            FormKeyInput.REDIRECT_URI.value: redirect_uri
        })

    def browser_login(self, login_callback_port, open_browser_function):
        host = "127.0.0.1"
        redirect_uri = "http://{}:{}".format(host, login_callback_port)

        # Fetch account and authorization urls
        urls = self._get_urls(redirect_uri)
        account_url = urls[Strings.ACCOUNT_URL.value]
        auth_url = urls[Strings.AUTH_URL.value]

        class LoginCallbackApp(BaseHTTPRequestHandler):
            """
            An handler of request for the keycloak redirect
            """

            authorization_code = None
            has_error = False

            def do_GET(self):
                try:
                    LoginCallbackApp.authorization_code = parse.parse_qs(self.path[2:])['code'][0]
                    self.send_response(HTTP_STATUS_REDIRECT)
                    self.send_header('Location', account_url)
                    self.end_headers()
                except KeyError:
                    LoginCallbackApp.has_error = True
                    self.send_response(HTTP_STATUS_BAD_REQUEST)

        # Configure and start a basic HTTP server
        httpd = HTTPServer((host, login_callback_port), LoginCallbackApp)
        th = threading.Thread(target=httpd.serve_forever)
        th.start()
        open_browser_function(auth_url)
        while LoginCallbackApp.authorization_code is None and not LoginCallbackApp.has_error:
            # Wait until login is performed
            sleep(1)
        httpd.shutdown()

        # Ask keycloak server for tokens using the authorization code
        return self.login(code=LoginCallbackApp.authorization_code, redirect_uri=redirect_uri)
