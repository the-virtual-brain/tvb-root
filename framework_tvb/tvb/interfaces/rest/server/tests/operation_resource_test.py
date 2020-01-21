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
import tvb_data
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.uploaders.zip_connectivity_importer import ZIPConnectivityImporterModel
from tvb.basic.exceptions import TVBException
from tvb.core.neocom import h5
from tvb.core.neotraits._h5core import ViewModelH5
from tvb.interfaces.rest.commons.exceptions import InvalidIdentifierException, BadRequestException
from tvb.interfaces.rest.server.resources.operation.operation_resource import GetOperationStatusResource, \
    GetOperationResultsResource, LaunchOperationResource
from tvb.interfaces.rest.server.resources.project.project_resource import GetOperationsInProjectResource
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.tests.framework.core.factory import TestFactory, OperationPossibleStatus
from werkzeug.datastructures import FileStorage


class TestOperationResource(TransactionalTestCase):

    def transactional_setup_method(self):
        self.operations_resource = GetOperationsInProjectResource()
        self.status_resource = GetOperationStatusResource()
        self.results_resource = GetOperationResultsResource()
        self.launch_resource = LaunchOperationResource()

    def test_server_get_operation_status_inexistent_gid(self):
        operation_gid = "inexistent-gid"
        with pytest.raises(InvalidIdentifierException): self.status_resource.get(operation_gid)

    def test_server_get_operation_status(self):
        test_user = TestFactory.create_user('Rest_User')
        test_project_with_data = TestFactory.create_project(test_user, 'Rest_Project')
        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_96.zip')
        TestFactory.import_zip_connectivity(test_user, test_project_with_data, zip_path)

        operations = self.operations_resource.get(test_project_with_data.gid)

        result = self.status_resource.get(operations[0].gid)
        assert type(result) is str
        assert result in OperationPossibleStatus

    def test_server_get_operation_results_inexistent_gid(self):
        operation_gid = "inexistent-gid"
        with pytest.raises(InvalidIdentifierException): self.results_resource.get(operation_gid)

    def test_server_get_operation_results(self):
        test_user = TestFactory.create_user('Rest_User')
        test_project_with_data = TestFactory.create_project(test_user, 'Rest_Project')
        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_96.zip')
        TestFactory.import_zip_connectivity(test_user, test_project_with_data, zip_path)

        operations = self.operations_resource.get(test_project_with_data.gid)

        result = self.results_resource.get(operations[0].gid)
        assert type(result) is list
        assert len(result) == 1

    def test_server_get_operation_results_failed_operation(self):
        test_user = TestFactory.create_user('Rest_User')
        test_project_with_data = TestFactory.create_project(test_user, 'Rest_Project')
        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_90.zip')
        with pytest.raises(TVBException):
            TestFactory.import_zip_connectivity(test_user, test_project_with_data, zip_path)

        operations = self.operations_resource.get(test_project_with_data.gid)

        result = self.results_resource.get(operations[0].gid)
        assert type(result) is list
        assert len(result) == 0

    def test_server_launch_operation_no_file(self, mocker):
        # Mock flask.request.files to return a dictionary
        request_mock = mocker.patch.object(flask, 'request')
        request_mock.files = {}

        with pytest.raises(BadRequestException): self.launch_resource.post('', '', '')

    def test_server_launch_operation_wrong_file_extension(self, mocker):
        dummy_file = FileStorage(BytesIO(b"test"), 'test.txt')
        # Mock flask.request.files to return a dictionary
        request_mock = mocker.patch.object(flask, 'request')
        request_mock.files = {'file': dummy_file}

        with pytest.raises(BadRequestException): self.launch_resource.post('', '', '')

    def test_server_launch_operation_inexistent_gid(self, mocker):
        project_gid = "inexistent-gid"
        dummy_file = FileStorage(BytesIO(b"test"), 'test.h5')
        # Mock flask.request.files to return a dictionary
        request_mock = mocker.patch.object(flask, 'request')
        request_mock.files = {'file': dummy_file}

        with pytest.raises(InvalidIdentifierException): self.launch_resource.post(project_gid, '', '')

    def test_server_launch_operation_inexistent_algorithm(self, mocker):
        inexistent_algorithm = "inexistent-algorithm"
        test_user = TestFactory.create_user('Rest_User')
        test_project = TestFactory.create_project(test_user, 'Rest_Project')

        dummy_file = FileStorage(BytesIO(b"test"), 'test.h5')
        # Mock flask.request.files to return a dictionary
        request_mock = mocker.patch.object(flask, 'request')
        request_mock.files = {'file': dummy_file}

        with pytest.raises(InvalidIdentifierException): self.launch_resource.post(test_project.gid,
                                                                                  inexistent_algorithm, '')

    def test_server_launch_operation(self, mocker):
        algorithm_module = "tvb.adapters.uploaders.zip_connectivity_importer"
        algorithm_class = "ZIPConnectivityImporter"
        test_user = TestFactory.create_user('Rest_User')
        test_project = TestFactory.create_project(test_user, 'Rest_Project')

        importer_model = ZIPConnectivityImporterModel()
        importer_model.uploaded = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity',
                                               'connectivity_96.zip')

        view_model_h5_path = h5.path_for('', ViewModelH5, importer_model.gid)

        view_model_h5 = ViewModelH5(view_model_h5_path, importer_model)
        view_model_h5.store(importer_model)
        view_model_h5.close()

        # Mock flask.request.files to return a dictionary
        request_mock = mocker.patch.object(flask, 'request')
        fp = open(view_model_h5_path, 'rb')
        request_mock.files = {'file': FileStorage(fp, view_model_h5_path)}

        operation_gid = self.launch_resource.post(test_project.gid, algorithm_module, algorithm_class)
        fp.close()

        assert type(operation_gid) is str
        assert len(operation_gid) > 0

        result = self.results_resource.get(operation_gid)
        assert type(result) is list
        assert result[0].type == ConnectivityIndex().display_type
