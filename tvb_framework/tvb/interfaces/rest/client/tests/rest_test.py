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

from keycloak import KeycloakOpenID
from tvb.interfaces.rest.client.examples.fire_simulation import fire_simulation_example
from tvb.interfaces.rest.client.examples.launch_operation import launch_operation_examples
from tvb.interfaces.rest.client.tvb_client import TVBClient
from tvb.tests.framework.core.base_testcase import TransactionalTestCase


class TestRestService(TransactionalTestCase):

    def transactional_setup_method(self):
        self.tvb_client = TVBClient("http://rest-server:9090")
        keycloak_instance = KeycloakOpenID("https://keycloak.codemart.ro/auth/", "TVB", "tvb-tests")
        self.tvb_client.update_auth_token(keycloak_instance.token("tvb_user", "pass")['access_token'])

    def test_launch_operation(self):
        launch_operation_examples(self.tvb_client)

    def test_fire_simulation(self):
        fire_simulation_example(self.tvb_client)
