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

import os
from io import BytesIO

import flask
import pytest
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.model.model_operation import Operation
from tvb.core.services.simulator_serializer import SimulatorSerializer
from tvb.core.services.simulator_service import SimulatorService
from tvb.interfaces.rest.commons.exceptions import InvalidIdentifierException
from tvb.interfaces.rest.commons.strings import RequestFileKey
from tvb.interfaces.rest.server.resources.simulator.simulation_resource import FireSimulationResource
from tvb.simulator.simulator import Simulator
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.interfaces.rest.base_resource_test import RestResourceTest
from werkzeug.datastructures import FileStorage


class TestSimulationResource(RestResourceTest):

    def transactional_setup_method(self):
        self.test_user = TestFactory.create_user('Rest_User')
        self.test_project = TestFactory.create_project(self.test_user, 'Rest_Project', users=[self.test_user.id])
        self.simulation_resource = FireSimulationResource()
        self.files_helper = FilesHelper()

    def test_server_fire_simulation_inexistent_gid(self, mocker):
        self._mock_user(mocker)
        project_gid = "inexistent-gid"
        dummy_file = FileStorage(BytesIO(b"test"), 'test.zip')
        # Mock flask.request.files to return a dictionary
        request_mock = mocker.patch.object(flask, 'request')
        request_mock.files = {RequestFileKey.SIMULATION_FILE_KEY.value: dummy_file}

        with pytest.raises(InvalidIdentifierException): self.simulation_resource.post(project_gid=project_gid)

    def test_server_fire_simulation_no_file(self, mocker):
        self._mock_user(mocker)
        # Mock flask.request.files to return a dictionary
        request_mock = mocker.patch.object(flask, 'request')
        request_mock.files = {}

        with pytest.raises(InvalidIdentifierException): self.simulation_resource.post(project_gid='')

    def test_server_fire_simulation_bad_extension(self, mocker):
        self._mock_user(mocker)
        dummy_file = FileStorage(BytesIO(b"test"), 'test.txt')
        # Mock flask.request.files to return a dictionary
        request_mock = mocker.patch.object(flask, 'request')
        request_mock.files = {RequestFileKey.SIMULATION_FILE_KEY.value: dummy_file}

        with pytest.raises(InvalidIdentifierException): self.simulation_resource.post(project_gid='')

    def test_server_fire_simulation(self, mocker, connectivity_factory):
        self._mock_user(mocker)
        input_folder = self.files_helper.get_project_folder(self.test_project)
        sim_dir = os.path.join(input_folder, 'test_sim')
        if not os.path.isdir(sim_dir):
            os.makedirs(sim_dir)

        simulator = Simulator()
        simulator.connectivity = connectivity_factory()
        sim_serializer = SimulatorSerializer()
        sim_serializer.serialize_simulator(simulator, None, sim_dir)

        zip_filename = os.path.join(input_folder, RequestFileKey.SIMULATION_FILE_NAME.value)
        FilesHelper().zip_folder(zip_filename, sim_dir)

        # Mock flask.request.files to return a dictionary
        request_mock = mocker.patch.object(flask, 'request')
        fp = open(zip_filename, 'rb')
        request_mock.files = {RequestFileKey.SIMULATION_FILE_KEY.value: FileStorage(fp, os.path.basename(zip_filename))}

        def launch_sim(self, user_id, project, algorithm, zip_folder_path, simulator_file):
            return Operation('', '', '', {})

        # Mock simulation launch and current user
        mocker.patch.object(SimulatorService, 'prepare_simulation_on_server', launch_sim)

        operation_gid, status = self.simulation_resource.post(project_gid=self.test_project.gid)
        fp.close()

        assert type(operation_gid) is str
        assert status == 201

    def transactional_teardown_method(self):
        self.files_helper.remove_project_structure(self.test_project.name)
