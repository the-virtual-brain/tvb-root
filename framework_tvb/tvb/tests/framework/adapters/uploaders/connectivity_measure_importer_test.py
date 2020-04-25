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

"""
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

import pytest
import os.path
import tvb_data
from tvb.adapters.uploaders.connectivity_measure_importer import ConnectivityMeasureImporterModel
from tvb.adapters.uploaders.connectivity_measure_importer import ConnectivityMeasureImporter
from tvb.adapters.datatypes.db.graph import ConnectivityMeasureIndex
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.services.exceptions import OperationException
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.adapters.uploaders import test_data


class TestConnectivityMeasureImporter(TransactionalTestCase):
    """
    Unit-tests for ConnectivityMeasureImporter
    """

    def transactional_setup_method(self):
        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_66.zip')
        self.test_user = TestFactory.create_user('Test_User_CM')
        self.test_project = TestFactory.create_project(self.test_user, "Test_Project_CM")
        self.connectivity = TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path, "John")

    def transactional_teardown_method(self):
        FilesHelper().remove_project_structure(self.test_project.name)

    def _import(self, import_file_name):
        path = os.path.join(os.path.dirname(test_data.__file__), import_file_name)

        view_model = ConnectivityMeasureImporterModel()
        view_model.data_file = path
        view_model.dataset_name = "M"
        view_model.connectivity = self.connectivity.gid
        TestFactory.launch_importer(ConnectivityMeasureImporter, view_model, self.test_user, self.test_project.id)

    def test_happy_flow(self):
        assert 0 == TestFactory.get_entity_count(self.test_project, ConnectivityMeasureIndex)
        self._import('mantini_networks.mat')
        assert 6 == TestFactory.get_entity_count(self.test_project, ConnectivityMeasureIndex)

    def test_connectivity_mismatch(self):
        with pytest.raises(OperationException):
            self._import('mantini_networks_33.mat')
