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

"""
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

from os import path

import pytest
import tvb_data
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.uploaders.csv_connectivity_importer import CSVConnectivityImporter
from tvb.adapters.uploaders.csv_connectivity_importer import CSVConnectivityParser, CSVConnectivityImporterModel
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.neocom import h5
from tvb.core.services.exceptions import OperationException
from tvb.storage.storage_interface import StorageInterface
from tvb.tests.framework.core.base_testcase import BaseTestCase
from tvb.tests.framework.core.factory import TestFactory

TEST_SUBJECT_A = "TEST_SUBJECT_A"
TEST_SUBJECT_B = "TEST_SUBJECT_B"


class TestCSVConnectivityParser(BaseTestCase):
    BASE_PTH = path.join(path.dirname(tvb_data.__file__), 'dti_pipeline_toronto')

    def test_parse_happy(self):
        cap_pth = path.join(self.BASE_PTH, 'output_ConnectionDistanceMatrix.csv')

        with open(cap_pth) as f:
            result_conn = CSVConnectivityParser(f).result_conn
            assert [0, 61.7082, 50.7576, 76.4214] == result_conn[0][:4]
            for i in range(len(result_conn)):
                assert 0 == result_conn[i][i]


class TestCSVConnectivityImporter(BaseTestCase):
    """
    Unit-tests for csv connectivity importer.
    """

    def setup_method(self):
        self.test_user = TestFactory.create_user()
        self.test_project = TestFactory.create_project(self.test_user)
        self.storage_interface = StorageInterface()

    def teardown_method(self):
        """
        Clean-up tests data
        """
        self.clean_database()

    def _import_csv_test_connectivity(self, reference_connectivity_gid, subject):
        ### First prepare input data:
        data_dir = path.abspath(path.dirname(tvb_data.__file__))

        toronto_dir = path.join(data_dir, 'dti_pipeline_toronto')
        weights = path.join(toronto_dir, 'output_ConnectionCapacityMatrix.csv')
        tracts = path.join(toronto_dir, 'output_ConnectionDistanceMatrix.csv')
        weights_tmp = weights + '.tmp'
        tracts_tmp = tracts + '.tmp'
        self.storage_interface.copy_file(weights, weights_tmp)
        self.storage_interface.copy_file(tracts, tracts_tmp)

        view_model = CSVConnectivityImporterModel()
        view_model.weights = weights_tmp
        view_model.tracts = tracts_tmp
        view_model.data_subject = subject
        view_model.input_data = reference_connectivity_gid
        TestFactory.launch_importer(CSVConnectivityImporter, view_model, self.test_user, self.test_project, False)

    def test_happy_flow_import(self):
        """
        Test that importing a CFF generates at least one DataType in DB.
        """

        zip_path = path.join(path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_96.zip')
        TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path, subject=TEST_SUBJECT_A)

        field = FilterChain.datatype + '.subject'
        filters = FilterChain('', [field], [TEST_SUBJECT_A], ['=='])
        reference_connectivity_index = TestFactory.get_entity(self.test_project, ConnectivityIndex, filters)

        dt_count_before = TestFactory.get_entity_count(self.test_project, ConnectivityIndex)

        self._import_csv_test_connectivity(reference_connectivity_index.gid, TEST_SUBJECT_B)

        dt_count_after = TestFactory.get_entity_count(self.test_project, ConnectivityIndex)
        assert dt_count_before + 1 == dt_count_after

        filters = FilterChain('', [field], [TEST_SUBJECT_B], ['like'])
        imported_connectivity_index = TestFactory.get_entity(self.test_project, ConnectivityIndex, filters)

        # check relationship between the imported connectivity and the reference
        assert reference_connectivity_index.number_of_regions == imported_connectivity_index.number_of_regions
        assert not reference_connectivity_index.number_of_connections == imported_connectivity_index.number_of_connections

        reference_connectivity = h5.load_from_index(reference_connectivity_index)
        imported_connectivity = h5.load_from_index(imported_connectivity_index)

        assert not (reference_connectivity.weights == imported_connectivity.weights).all()
        assert not (reference_connectivity.tract_lengths == imported_connectivity.tract_lengths).all()

        assert (reference_connectivity.centres == imported_connectivity.centres).all()
        assert (reference_connectivity.orientations == imported_connectivity.orientations).all()
        assert (reference_connectivity.region_labels == imported_connectivity.region_labels).all()

    def test_bad_reference(self):
        zip_path = path.join(path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_66.zip')
        TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path)
        field = FilterChain.datatype + '.subject'
        filters = FilterChain('', [field], [TEST_SUBJECT_A], ['!='])
        bad_reference_connectivity = TestFactory.get_entity(self.test_project, ConnectivityIndex, filters)

        with pytest.raises(OperationException):
            self._import_csv_test_connectivity(bad_reference_connectivity.gid, TEST_SUBJECT_A)
