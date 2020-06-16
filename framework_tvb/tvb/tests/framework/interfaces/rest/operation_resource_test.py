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
from uuid import UUID

import flask
import pytest
import tvb_data
from tvb.adapters.analyzers.fourier_adapter import FFTAdapterModel
from tvb.analyzers.fft import SUPPORTED_WINDOWING_FUNCTIONS
from tvb.basic.exceptions import TVBException
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.neocom import h5
from tvb.core.neotraits.h5 import ViewModelH5
from tvb.core.services.operation_service import OperationService
from tvb.interfaces.rest.commons.exceptions import InvalidIdentifierException, BadRequestException
from tvb.interfaces.rest.commons.strings import Strings, RequestFileKey
from tvb.interfaces.rest.server.resources.operation.operation_resource import GetOperationStatusResource, \
    GetOperationResultsResource, LaunchOperationResource
from tvb.interfaces.rest.server.resources.project.project_resource import GetOperationsInProjectResource
from tvb.tests.framework.core.factory import TestFactory, OperationPossibleStatus
from tvb.tests.framework.interfaces.rest.base_resource_test import RestResourceTest
from werkzeug.datastructures import FileStorage


class TestOperationResource(RestResourceTest):

    def transactional_setup_method(self):
        self.test_user = TestFactory.create_user('Rest_User')
        self.test_project = TestFactory.create_project(self.test_user, 'Rest_Project', users=[self.test_user.id])
        self.operations_resource = GetOperationsInProjectResource()
        self.status_resource = GetOperationStatusResource()
        self.results_resource = GetOperationResultsResource()
        self.launch_resource = LaunchOperationResource()
        self.files_helper = FilesHelper()

    def test_server_get_operation_status_inexistent_gid(self, mocker):
        self._mock_user(mocker)
        operation_gid = "inexistent-gid"
        with pytest.raises(InvalidIdentifierException): self.status_resource.get(operation_gid=operation_gid)

    def test_server_get_operation_status(self, mocker):
        self._mock_user(mocker)
        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_96.zip')
        TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path)

        request_mock = mocker.patch.object(flask, 'request')
        request_mock.args = {Strings.PAGE_NUMBER: '1'}

        operations_and_pages = self.operations_resource.get(project_gid=self.test_project.gid)

        result = self.status_resource.get(operation_gid=operations_and_pages['operations'][0].gid)
        assert type(result) is str
        assert result in OperationPossibleStatus

    def test_server_get_operation_results_inexistent_gid(self, mocker):
        self._mock_user(mocker)
        operation_gid = "inexistent-gid"
        with pytest.raises(InvalidIdentifierException): self.results_resource.get(operation_gid=operation_gid)

    def test_server_get_operation_results(self, mocker):
        self._mock_user(mocker)
        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_96.zip')
        TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path)

        request_mock = mocker.patch.object(flask, 'request')
        request_mock.args = {Strings.PAGE_NUMBER: '1'}

        operations_and_pages = self.operations_resource.get(project_gid=self.test_project.gid)

        result = self.results_resource.get(operation_gid=operations_and_pages['operations'][0].gid)
        assert type(result) is list
        assert len(result) == 1

    def test_server_get_operation_results_failed_operation(self, mocker):
        self._mock_user(mocker)
        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_90.zip')
        with pytest.raises(TVBException):
            TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path)

        request_mock = mocker.patch.object(flask, 'request')
        request_mock.args = {Strings.PAGE_NUMBER: '1'}

        operations_and_pages = self.operations_resource.get(project_gid=self.test_project.gid)

        result = self.results_resource.get(operation_gid=operations_and_pages['operations'][0].gid)
        assert type(result) is list
        assert len(result) == 0

    def test_server_launch_operation_no_file(self, mocker):
        self._mock_user(mocker)
        # Mock flask.request.files to return a dictionary
        request_mock = mocker.patch.object(flask, 'request')
        request_mock.files = {}

        with pytest.raises(InvalidIdentifierException): self.launch_resource.post(project_gid='', algorithm_module='',
                                                                           algorithm_classname='')

    def test_server_launch_operation_wrong_file_extension(self, mocker):
        self._mock_user(mocker)
        dummy_file = FileStorage(BytesIO(b"test"), 'test.txt')
        # Mock flask.request.files to return a dictionary
        request_mock = mocker.patch.object(flask, 'request')
        request_mock.files = {RequestFileKey.LAUNCH_ANALYZERS_MODEL_FILE.value: dummy_file}

        with pytest.raises(InvalidIdentifierException): self.launch_resource.post(project_gid='', algorithm_module='',
                                                                           algorithm_classname='')

    def test_server_launch_operation_inexistent_gid(self, mocker):
        self._mock_user(mocker)
        project_gid = "inexistent-gid"
        dummy_file = FileStorage(BytesIO(b"test"), 'test.h5')
        # Mock flask.request.files to return a dictionary
        request_mock = mocker.patch.object(flask, 'request')
        request_mock.files = {RequestFileKey.LAUNCH_ANALYZERS_MODEL_FILE.value: dummy_file}

        with pytest.raises(InvalidIdentifierException): self.launch_resource.post(project_gid=project_gid,
                                                                                  algorithm_module='',
                                                                                  algorithm_classname='')

    def test_server_launch_operation_inexistent_algorithm(self, mocker):
        self._mock_user(mocker)
        inexistent_algorithm = "inexistent-algorithm"

        dummy_file = FileStorage(BytesIO(b"test"), 'test.h5')
        # Mock flask.request.files to return a dictionary
        request_mock = mocker.patch.object(flask, 'request')
        request_mock.files = {RequestFileKey.LAUNCH_ANALYZERS_MODEL_FILE.value: dummy_file}

        with pytest.raises(InvalidIdentifierException): self.launch_resource.post(project_gid=self.test_project.gid,
                                                                                  algorithm_module=inexistent_algorithm,
                                                                                  algorithm_classname='')

    def test_server_launch_operation(self, mocker, time_series_index_factory):
        self._mock_user(mocker)
        algorithm_module = "tvb.adapters.analyzers.fourier_adapter"
        algorithm_class = "FourierAdapter"

        input_ts_index = time_series_index_factory()

        fft_model = FFTAdapterModel()
        fft_model.time_series = UUID(input_ts_index.gid)
        fft_model.window_function = list(SUPPORTED_WINDOWING_FUNCTIONS)[0]

        input_folder = self.files_helper.get_project_folder(self.test_project)
        view_model_h5_path = h5.path_for(input_folder, ViewModelH5, fft_model.gid)

        view_model_h5 = ViewModelH5(view_model_h5_path, fft_model)
        view_model_h5.store(fft_model)
        view_model_h5.close()

        # Mock flask.request.files to return a dictionary
        request_mock = mocker.patch.object(flask, 'request')
        fp = open(view_model_h5_path, 'rb')
        request_mock.files = {
            RequestFileKey.LAUNCH_ANALYZERS_MODEL_FILE.value: FileStorage(fp, os.path.basename(view_model_h5_path))}

        # Mock launch_operation() call and current_user
        mocker.patch.object(OperationService, 'launch_operation')

        operation_gid, status = self.launch_resource.post(project_gid=self.test_project.gid,
                                                          algorithm_module=algorithm_module,
                                                          algorithm_classname=algorithm_class)

        fp.close()

        assert type(operation_gid) is str
        assert len(operation_gid) > 0

    def transactional_teardown_method(self):
        self.files_helper.remove_project_structure(self.test_project.name)
