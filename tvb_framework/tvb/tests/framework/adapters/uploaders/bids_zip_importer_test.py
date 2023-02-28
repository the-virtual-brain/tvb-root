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
import pytest
from tvb.adapters.uploaders.bids_importer import BIDSImporterModel, BIDSImporter
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.surface import SurfaceIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.adapters.datatypes.db.graph import CorrelationCoefficientsIndex
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.tests.framework.core.base_testcase import BaseTestCase
from tvb.tests.framework.core.factory import TestFactory

try:
    import tvb_data.bids

    BIDS_DATA = os.path.join(os.path.dirname(tvb_data.bids.__file__), 'bids_derivatives_dataset.zip')
    BIDS_DATA_FOUND = os.path.exists(BIDS_DATA)

except ImportError:
    BIDS_DATA_FOUND = False


@pytest.mark.skipif(not BIDS_DATA_FOUND, reason="Older or incomplete tvb_data")
class TestBIDSImporter(BaseTestCase):
    """
    Unit-tests for BIDS importer.
    """

    def setup_method(self):
        """
        Set-up DB before each test
        """
        self.test_user = TestFactory.create_user('Zip_BIDS_User')
        self.test_project = TestFactory.create_project(self.test_user, 'Zip_BIDS_Project')
        self.zip_file_path = BIDS_DATA

    def teardown_method(self):
        """
        Clean-up tests data
        """
        self.clean_database()

    def _import(self, import_file_path=None):
        view_model = BIDSImporterModel()
        view_model.data_file = import_file_path
        TestFactory.launch_importer(BIDSImporter, view_model, self.test_user, self.test_project, False)

    def _import_zip_bids(self, user=None, project=None, zip_path=None, subject=DataTypeMetaData.DEFAULT_SUBJECT,
                         same_process=True):
        if user is None:
            user = self.test_user
        if project is None:
            project = self.test_project
        if zip_path is None:
            zip_path = self.zip_file_path

        view_model = BIDSImporterModel()
        view_model.uploaded = zip_path
        view_model.data_subject = subject
        TestFactory.launch_importer(BIDSImporter, view_model, user, project, same_process)

    def test_import_bids_dataset(self):
        """
        Dataset used in this test has 1 Connectivity, 1 Surface, 2 TimeSeries, 2 CorrelationCoefficients
        Test that importing a BIDS dataset generates corresponding DataType in DB.
        """
        conn_count_before = TestFactory.get_entity_count(self.test_project, ConnectivityIndex)
        surface_count_before = TestFactory.get_entity_count(self.test_project, SurfaceIndex)
        times_count_before = TestFactory.get_entity_count(self.test_project, TimeSeriesIndex)
        coef_count_before = TestFactory.get_entity_count(self.test_project, CorrelationCoefficientsIndex)

        self._import_zip_bids(subject="John", same_process=False)

        conn_count_after = TestFactory.get_entity_count(self.test_project, ConnectivityIndex)
        surface_count_after = TestFactory.get_entity_count(self.test_project, SurfaceIndex)
        times_count_after = TestFactory.get_entity_count(self.test_project, TimeSeriesIndex)
        coef_count_after = TestFactory.get_entity_count(self.test_project, CorrelationCoefficientsIndex)

        assert conn_count_before + 1 == conn_count_after
        assert surface_count_before + 1 == surface_count_after
        assert times_count_before + 2 == times_count_after
        assert coef_count_before + 2 == coef_count_after

    def test_import_bids_zip_has_expected_surface_values(self):
        self._import_zip_bids(subject="John", same_process=False)
        surface = TestFactory._assert_one_more_datatype(self.test_project, SurfaceIndex)
        expected_vertices = 10000
        expected_triangles = 10000

        assert surface.number_of_vertices == expected_vertices
        assert surface.number_of_triangles == expected_triangles
