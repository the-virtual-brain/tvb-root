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
.. moduleauthor:: Gabriel Florea <gabriel.florea@codemart.ro>
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
"""
import json
import os

import numpy
import tvb_data
import tvb_data.nifti as demo_data
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.region_mapping import RegionVolumeMappingIndex
from tvb.adapters.datatypes.db.structural import StructuralMRIIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesVolumeIndex
from tvb.adapters.uploaders.nifti_importer import NIFTIImporterModel, NIFTIImporter
from tvb.core.entities.load import load_entity_by_gid
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5
from tvb.core.services.exceptions import OperationException
from tvb.tests.framework.core.base_testcase import BaseTestCase
from tvb.tests.framework.core.factory import TestFactory


class TestNIFTIImporter(BaseTestCase):
    """
    Unit-tests for NIFTI importer.
    """

    NII_FILE = os.path.join(os.path.dirname(demo_data.__file__), 'minimal.nii')
    GZ_NII_FILE = os.path.join(os.path.dirname(demo_data.__file__), 'minimal.nii.gz')
    TIMESERIES_NII_FILE = os.path.join(os.path.dirname(demo_data.__file__), 'time_series_152.nii.gz')
    WRONG_NII_FILE = os.path.abspath(__file__)
    TXT_FILE = os.path.join(os.path.dirname(demo_data.__file__), 'volume_mapping/mapping_FS_76.txt')

    DEFAULT_ORIGIN = [[0.0, 0.0, 0.0]]
    UNKNOWN_STR = "unknown"

    def setup_method(self):
        self.test_user = TestFactory.create_user('Nifti_Importer_User')
        self.test_project = TestFactory.create_project(self.test_user, "Nifti_Importer_Project")

    def teardown_method(self):
        """
        Clean-up tests data
        """
        self.clean_database()

    def _import(self, import_file_path=None, expected_result_class=StructuralMRIIndex, connectivity_gid=None):
        """
        This method is used for importing data in NIFIT format
        :param import_file_path: absolute path of the file to be imported
        """
        view_model = NIFTIImporterModel()
        view_model.data_file = import_file_path
        view_model.mappings_file = self.TXT_FILE
        view_model.apply_corrections = True
        view_model.connectivity = connectivity_gid
        view_model.data_subject = "Bla Bla"

        TestFactory.launch_importer(NIFTIImporter, view_model, self.test_user, self.test_project, False)

        dts, count = dao.get_values_of_datatype(self.test_project.id, expected_result_class, None)
        assert 1, count == "Project should contain only one data type."

        result = load_entity_by_gid(dts[0][2])
        assert result is not None, "Result should not be none"
        return result

    def test_import_demo_ts(self):
        """
        This method tests import of a NIFTI file.
        """
        time_series_index = self._import(self.TIMESERIES_NII_FILE, TimeSeriesVolumeIndex)

        # Since self.assertAlmostEquals is not available on all machine
        # We compare floats as following
        assert abs(1.0 - time_series_index.sample_period) <= 0.001
        assert "sec" == str(time_series_index.sample_period_unit)
        assert time_series_index.title.startswith("NIFTI")

        dimension_labels = time_series_index.labels_ordering
        assert dimension_labels is not None
        assert 4 == len(json.loads(dimension_labels))

        volume = h5.load_from_gid(time_series_index.fk_volume_gid)

        assert numpy.equal(self.DEFAULT_ORIGIN, volume.origin).all()
        assert "mm" == volume.voxel_unit

    def test_import_nii_without_time_dimension(self):
        """
        This method tests import of a NIFTI file.
        """
        structural_mri_index = self._import(self.NII_FILE)
        assert "T1" == structural_mri_index.weighting

        structural_mri = h5.load_from_index(structural_mri_index)

        data_shape = structural_mri.array_data.shape
        assert 3 == len(data_shape)
        assert 64 == data_shape[0]
        assert 64 == data_shape[1]
        assert 10 == data_shape[2]

        volume = h5.load_from_gid(structural_mri_index.fk_volume_gid)

        assert numpy.equal(self.DEFAULT_ORIGIN, volume.origin).all()
        assert numpy.equal([3.0, 3.0, 3.0], volume.voxel_size).all()
        assert self.UNKNOWN_STR == volume.voxel_unit

    def test_import_nifti_compressed(self):
        """
        This method tests import of a NIFTI file compressed in GZ format.
        """
        structure = self._import(self.GZ_NII_FILE)
        assert "T1" == structure.weighting

    def test_import_region_mapping(self):
        """
        This method tests import of a NIFTI file compressed in GZ format.
        """
        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_76.zip')
        TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path, "John")
        to_link_conn = TestFactory.get_entity(self.test_project, ConnectivityIndex)

        mapping_index = self._import(self.GZ_NII_FILE, RegionVolumeMappingIndex, to_link_conn.gid)
        mapping = h5.load_from_index(mapping_index)

        assert -1 <= mapping.array_data.min()
        assert mapping.array_data.max() < to_link_conn.number_of_regions
        assert to_link_conn.gid == mapping_index.fk_connectivity_gid

        volume = h5.load_from_gid(mapping_index.fk_volume_gid)
        assert numpy.equal(self.DEFAULT_ORIGIN, volume.origin).all()
        assert numpy.equal([3.0, 3.0, 3.0], volume.voxel_size).all()
        assert self.UNKNOWN_STR == volume.voxel_unit

    def test_import_wrong_nii_file(self):
        """
        This method tests import of a file in a wrong format
        """
        try:
            self._import(self.WRONG_NII_FILE)
            raise AssertionError("Import should fail in case of a wrong NIFTI format.")
        except OperationException:
            # Expected exception
            pass
