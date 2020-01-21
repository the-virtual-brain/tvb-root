# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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

from io import BytesIO
import flask
import pytest
from tvb.interfaces.rest.commons.exceptions import InvalidIdentifierException, BadRequestException
from tvb.interfaces.rest.server.resources.simulator.simulation_resource import FireSimulationResource
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from werkzeug.datastructures import FileStorage


class TestSimulationResource(TransactionalTestCase):

    def transactional_setup_method(self):
        self.simulation_resource = FireSimulationResource()

    def test_server_fire_simulation_inexistent_gid(self, mocker):
        project_gid = "inexistent-gid"
        dummy_file = FileStorage(BytesIO(b"test"), 'test.zip')
        # Mock flask.request.files to return a dictionary
        request_mock = mocker.patch.object(flask, 'request')
        request_mock.files = {'file': dummy_file}

        with pytest.raises(InvalidIdentifierException): self.simulation_resource.post(project_gid)

    def test_server_fire_simulation_no_file(self, mocker):
        # Mock flask.request.files to return a dictionary
        request_mock = mocker.patch.object(flask, 'request')
        request_mock.files = {}

        with pytest.raises(BadRequestException): self.simulation_resource.post('')

    def test_server_fire_simulation_bad_extension(self, mocker):
        dummy_file = FileStorage(BytesIO(b"test"), 'test.txt')
        # Mock flask.request.files to return a dictionary
        request_mock = mocker.patch.object(flask, 'request')
        request_mock.files = {'file': dummy_file}

        with pytest.raises(BadRequestException): self.simulation_resource.post('')
