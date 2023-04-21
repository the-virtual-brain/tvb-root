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

import shutil
import time
import uuid
import pytest
import os

from tvb.adapters.uploaders.bids_importer import BIDSImporter
from tvb.interfaces.rest.bids_monitor.bids_data_builder import BIDSDataBuilder
from tvb.interfaces.rest.bids_monitor.bids_dir_monitor import BIDSDirWatcher
from tvb.storage.storage_interface import StorageInterface

try:
    import tvb_data.bids

    BIDS_DATA_FOUND = True
    BIDS_DATA_DIR = os.path.join(tvb_data.bids.__path__[0], 'BIDS_DEMO_DATASET')
except ImportError:
    BIDS_DATA_FOUND = False


@pytest.mark.skipif(not BIDS_DATA_FOUND, reason="Older or incomplete tvb_data")
class TestBIDSDataBuilder:
    TS_TO_COPY = 'sub-01_desc-50healthy-delta-speed20-G0.05-bold_ts.tsv'
    TS_JSON_TO_COPY = 'sub-01_desc-50healthy-delta-speed20-G0.05-bold_ts.json'

    def setup(self):
        self.ts_folder = os.path.join(BIDS_DATA_DIR, 'sub-01', 'ts')
        self.ts_copied = os.path.join(self.ts_folder, 'sub-01_desc-50healthy-delta-speed20-G0.05-bold_ts_cpy.tsv')
        self.ts_json_copied = os.path.join(self.ts_folder, 'sub-01_desc-50healthy-delta-speed20-G0.05-bold_ts_cpy.json')
        self.bids_connectivity_zip = None
        self.bids_timeseries_zip = None
        self.bids_spatial_zip = None
        self.bids_surface_zip = None
        self.dataset_zip = None

    def test_create_bids_dataset(self):
        """
        Dataset used in this test has 1 Connectivity, 1 Surface, 1 TimeSeries and 1 Functional Connectivity
        Test that creating a dataset for a single datatype contains all required files are not
        """
        self.bids_connectivity_zip = BIDSDataBuilder(BIDSImporter.NET_TOKEN, BIDS_DATA_DIR).create_dataset_subjects()
        self.bids_timeseries_zip = BIDSDataBuilder(BIDSImporter.TS_TOKEN, BIDS_DATA_DIR).create_dataset_subjects()
        self.bids_spatial_zip = BIDSDataBuilder(BIDSImporter.SPATIAL_TOKEN, BIDS_DATA_DIR).create_dataset_subjects()
        self.bids_surface_zip = BIDSDataBuilder(BIDSImporter.COORDS_TOKEN, BIDS_DATA_DIR).create_dataset_subjects()

        bids_connectivity_test_data = self.extract_data(self.bids_connectivity_zip)
        bids_timeseries_test_data = self.extract_data(self.bids_timeseries_zip)
        bids_spatial_test_data = self.extract_data(self.bids_spatial_zip)
        bids_surface_test_data = self.extract_data(self.bids_surface_zip)

        expected_bids_connectivity_data = [
            'sub-01_desc-50healthy_distances.json',
            'sub-01_desc-50healthy_distances.tsv',
            'sub-01_desc-50healthy_weights.json',
            'sub-01_desc-50healthy_weights.tsv',
            'desc-50healthy_labels.json',
            'desc-50healthy_labels.tsv',
            'desc-50healthy_nodes.json',
            'desc-50healthy_nodes.tsv'
        ]
        expected_bids_connectivity_data.sort()

        expected_bids_timeseries_data = [
            'desc-50healthy-alpha-bold_times.json',
            'desc-50healthy-alpha-bold_times.tsv',
            'desc-50healthy-delta-bold_times.json',
            'desc-50healthy-delta-bold_times.tsv',
            'desc-50healthy_labels.json',
            'desc-50healthy_labels.tsv',
            'desc-50healthy_nodes.json',
            'desc-50healthy_nodes.tsv',
            'desc-50healthy.json',
            'desc-50healthy.xml',
            'desc-50healthy-alpha-speed20-G0.05.json',
            'desc-50healthy-alpha-speed20-G0.05.xml',
            'desc-50healthy-delta-speed20-G0.05.json',
            'desc-50healthy-delta-speed20-G0.05.xml',
            'sub-01_desc-50healthy_distances.json',
            'sub-01_desc-50healthy_distances.tsv',
            'sub-01_desc-50healthy_weights.json',
            'sub-01_desc-50healthy_weights.tsv',
            'sub-01_desc-50healthy-alpha-speed20-G0.05-bold_ts.json',
            'sub-01_desc-50healthy-alpha-speed20-G0.05-bold_ts.tsv',
            self.TS_JSON_TO_COPY,
            self.TS_TO_COPY
        ]
        expected_bids_timeseries_data.sort()

        expected_bids_spatial_data = [
            'desc-50healthy_labels.json',
            'desc-50healthy_labels.tsv',
            'desc-50healthy_nodes.json',
            'desc-50healthy_nodes.tsv',
            'desc-50healthy.json',
            'desc-50healthy.xml',
            'desc-50healthy-alpha-speed20-G0.05.json',
            'desc-50healthy-alpha-speed20-G0.05.xml',
            'desc-50healthy-delta.json',
            'desc-50healthy-delta.xml',
            'sub-01_desc-50healthy_distances.json',
            'sub-01_desc-50healthy_distances.tsv',
            'sub-01_desc-50healthy_weights.json',
            'sub-01_desc-50healthy_weights.tsv',
            'sub-01_desc-50healthy-alpha-speed20-G0.05sim_fc.json',
            'sub-01_desc-50healthy-alpha-speed20-G0.05sim_fc.tsv',
            'sub-01_desc-50healthy-emp_fc.json',
            'sub-01_desc-50healthy-emp_fc.tsv'
        ]
        expected_bids_spatial_data.sort()

        expected_bids_surface_data = [
            'sub-01_desc-cortex_faces.json',
            'sub-01_desc-cortex_faces.tsv',
            'sub-01_desc-cortex_vertices.json',
            'sub-01_desc-cortex_vertices.tsv',
            'sub-01_desc-cortex_vnormals.json',
            'sub-01_desc-cortex_vnormals.tsv'
        ]
        expected_bids_surface_data.sort()

        assert bids_connectivity_test_data == expected_bids_connectivity_data
        assert bids_timeseries_test_data == expected_bids_timeseries_data
        assert bids_spatial_test_data == expected_bids_spatial_data
        assert bids_surface_test_data == expected_bids_surface_data

    def extract_data(self, path):
        self.storage_interface = StorageInterface()
        temp_path = self.get_storage_path(path)
        files_list = self.storage_interface.unpack_zip(path, temp_path)
        self.storage_interface.remove_folder(temp_path)
        return self.extract_file_names(files_list)

    def get_storage_path(self, zip_dir):
        base_dir = os.path.dirname(zip_dir)
        temp_path = os.path.join(base_dir, str(uuid.uuid4()))
        os.mkdir(temp_path)
        return temp_path

    def extract_file_names(self, files_list):
        for i in range(0, len(files_list)):
            files_list[i] = os.path.split(files_list[i])[1]
        files_list.sort()
        return files_list

    def test_monitor_bids_dataset(self):
        """
        Dataset used in this test has 1 Connectivity, 1 Surface, 1 TimeSeries and 1 Functional Connectivity.
        Test that adding a new file to the TS directory will create a dataset with correct files.
        First TS directory will be moved to the parent directory of subject folder and
        then we'll move it back to the subject directory.
        This way we can check if BIDSDirWatcher is building only when files are added inside subject directories.

        """
        bids_dir_watcher = BIDSDirWatcher(
            DIRECTORY_TO_WATCH=BIDS_DATA_DIR,
            UPLOAD_TRIGGER_INTERVAL=1,
            IMPORT_DATA_IN_TVB=False
        )

        bids_dir_watcher.init_watcher()

        # copy new file to subject folder
        shutil.copy(os.path.join(self.ts_folder, self.TS_TO_COPY), self.ts_copied)
        shutil.copy(os.path.join(self.ts_folder, self.TS_JSON_TO_COPY), self.ts_json_copied)

        while True:
            time.sleep(0.5)
            if bids_dir_watcher.current_dataset_loc != "":
                bids_dir_watcher.end_watcher()
                break

        self.dataset_zip = bids_dir_watcher.current_dataset_loc
        filenames_list = TestBIDSDataBuilder().extract_data(bids_dir_watcher.current_dataset_loc)

        expected_bids_timeseries_files = [
            'desc-50healthy-delta-bold_times.json',
            'desc-50healthy-delta-bold_times.tsv',
            'desc-50healthy_labels.json',
            'desc-50healthy_labels.tsv',
            'desc-50healthy_nodes.json',
            'desc-50healthy_nodes.tsv',
            'desc-50healthy.json',
            'desc-50healthy.xml',
            'desc-50healthy-delta-speed20-G0.05.json',
            'desc-50healthy-delta-speed20-G0.05.xml',
            'sub-01_desc-50healthy_distances.json',
            'sub-01_desc-50healthy_distances.tsv',
            'sub-01_desc-50healthy_weights.json',
            'sub-01_desc-50healthy_weights.tsv',
            os.path.basename(self.ts_copied),
            os.path.basename(self.ts_json_copied)
        ]
        expected_bids_timeseries_files.sort()

        assert filenames_list == expected_bids_timeseries_files

    def teardown_method(self):
        for pth in [self.bids_connectivity_zip, self.bids_timeseries_zip, self.bids_spatial_zip, self.bids_surface_zip,
                    self.ts_copied, self.ts_json_copied, self.dataset_zip]:
            if pth and os.path.exists(pth):
                os.remove(pth)
