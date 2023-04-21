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

import json
from keycloak import KeycloakOpenID


class AuthorizationManagerMeta(type):
    """
    Metaclass used to generate the singleton instance
    """

    _instance = None

    def __call__(cls, config_file=''):
        if cls._instance is None:
            cls._instance = super(AuthorizationManagerMeta, cls).__call__(config_file)
        return cls._instance


class AuthorizationManager(metaclass=AuthorizationManagerMeta):
    """
    Keycloak configuration class
    """

    def __init__(self, config_file=''):
        """
        :param config_file: Path to the generated keycloak configuration file
        """
        self.config_file = config_file
        self.realm = None
        self.auth_server_url = None
        self.client_id = None
        self.secret_key = None
        self._load_configs()
        self.keycloak = KeycloakOpenID(self.auth_server_url, self.realm, self.client_id, self.secret_key)

    def _load_configs(self):
        """
        Read keycloak parameters from the configuration file
        """

        if self.config_file == '':
            raise RuntimeError("Empty keycloak config path.")
        try:
            with open(self.config_file) as f:
                config = json.load(f)
                self.realm = config['realm']
                self.auth_server_url = config['auth-server-url']
                self.client_id = config['resource']
                try:
                    self.secret_key = config['credentials']['secret']
                except KeyError:
                    self.secret_key = None
        except OSError:
            raise RuntimeError("Failed to read Keycloak configuration file from {}".format(self.config_file))

    @staticmethod
    def get_keycloak_instance():
        return AuthorizationManager().keycloak
