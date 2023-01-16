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

import os
from io import BytesIO
import flask
import pytest

from tvb.core.entities.file.simulator.view_model import SimulatorAdapterModel
from tvb.core.entities.model.model_operation import Operation
from tvb.core.neocom import h5
from tvb.core.services.simulator_service import SimulatorService
from tvb.interfaces.rest.commons.exceptions import InvalidIdentifierException
from tvb.interfaces.rest.commons.strings import RequestFileKey
from tvb.interfaces.rest.server.resources.simulator.simulation_resource import FireSimulationResource
from tvb.storage.storage_interface import StorageInterface
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.interfaces.rest.base_resource_test import RestResourceTest
from werkzeug.datastructures import FileStorage


class TestSimulationResource(RestResourceTest):

    def transactional_setup_method(self):
        self.test_user = TestFactory.create_user('Rest_User')
        self.test_project = TestFactory.create_project(self.test_user, 'Rest_Project', users=[self.test_user.id])
        self.simulation_resource = FireSimulationResource()
        self.storage_interface = StorageInterface()

    def test_server_fire_simulation_inexistent_gid(self, mocker):
        self._mock_user(mocker)
        project_gid = "inexistent-gid"
        dummy_file = FileStorage(BytesIO(b"test"), 'test.zip')
        # Mock flask.request.files to return a dictionary
        request_mock = mocker.patch.object(flask, 'request', spec={})
        request_mock.files = {RequestFileKey.SIMULATION_FILE_KEY.value: dummy_file}

        with pytest.raises(InvalidIdentifierException): self.simulation_resource.post(project_gid=project_gid)

    def test_server_fire_simulation_no_file(self, mocker):
        self._mock_user(mocker)
        # Mock flask.request.files to return a dictionary
        request_mock = mocker.patch.object(flask, 'request', spec={})
        request_mock.files = {}

        with pytest.raises(InvalidIdentifierException): self.simulation_resource.post(project_gid='')

    def test_server_fire_simulation_bad_extension(self, mocker):
        self._mock_user(mocker)
        dummy_file = FileStorage(BytesIO(b"test"), 'test.txt')
        # Mock flask.request.files to return a dictionary
        request_mock = mocker.patch.object(flask, 'request', spec={})
        request_mock.files = {RequestFileKey.SIMULATION_FILE_KEY.value: dummy_file}

        with pytest.raises(InvalidIdentifierException): self.simulation_resource.post(project_gid='')

    def test_server_fire_simulation(self, mocker, connectivity_factory):
        self._mock_user(mocker)
        input_folder = self.storage_interface.get_project_folder(self.test_project.name)
        sim_dir = os.path.join(input_folder, 'test_sim')
        if not os.path.isdir(sim_dir):
            os.makedirs(sim_dir)

        simulator = SimulatorAdapterModel()
        simulator.connectivity = connectivity_factory().gid
        h5.store_view_model(simulator, sim_dir)

        zip_filename = os.path.join(input_folder, RequestFileKey.SIMULATION_FILE_NAME.value)
        self.storage_interface.write_zip_folder(zip_filename, sim_dir)

        # Mock flask.request.files to return a dictionary
        request_mock = mocker.patch.object(flask, 'request', spec={})
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
