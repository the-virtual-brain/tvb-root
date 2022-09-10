import uuid
import pytest
import os

from tvb.interfaces.rest.bids_monitoring.bids_data_builder import BIDSDataBuilder
from tvb.adapters.uploaders.bids_importer import BIDSImporter
from tvb.storage.storage_interface import StorageInterface

try:
    import tvb_data.bids
    BIDS_DATA_FOUND = True
    BIDS_DATA_DIR = os.path.join(os.path.dirname(tvb_data.bids.__file__), 'BIDS_DEMO_DATASET')
except ImportError:
    BIDS_DATA_FOUND = False


@pytest.mark.skipif(not BIDS_DATA_FOUND, reason="Older or incomplete tvb_data")
class TestBIDSDataBuilder:

    def setup_test_data(self):
        self.bids_connectivity_data = [
            'sub-01_desc-50healthy_distances.json',
            'sub-01_desc-50healthy_distances.tsv',
            'sub-01_desc-50healthy_weights.json',
            'sub-01_desc-50healthy_weights.tsv',
            'desc-50healthy_labels.json',
            'desc-50healthy_labels.tsv',
            'desc-50healthy_nodes.json',
            'desc-50healthy_nodes.tsv'
        ]
        self.bids_timeseries_data = [
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
            'sub-01_desc-50healthy-delta-speed20-G0.05-bold_ts.json',
            'sub-01_desc-50healthy-delta-speed20-G0.05-bold_ts.tsv'
        ]
        self.bids_spatial_data = [
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
        self.bids_surface_data = [
            'sub-01_desc-cortex_faces.json',
            'sub-01_desc-cortex_faces.tsv',
            'sub-01_desc-cortex_vertices.json',
            'sub-01_desc-cortex_vertices.tsv',
            'sub-01_desc-cortex_vnormals.json',
            'sub-01_desc-cortex_vnormals.tsv'
        ]
        self.bids_connectivity_data.sort()
        self.bids_timeseries_data.sort()
        self.bids_surface_data.sort()
        self.bids_spatial_data.sort()

    def test_create_bids_dataset(self):
        """
        Dataset used in this test has 1 Connectivity, 1 Surface, 1 TimeSeries and 1 Functional Connectivity
        Test that creating a dataset for a single datatype contains all required files are not
        """
        self.setup_test_data()
        bids_connectivity_zip = BIDSDataBuilder(
            BIDSImporter.NET_TOKEN, BIDS_DATA_DIR).create_dataset_subjects()
        bids_timeseries_zip = BIDSDataBuilder(
            BIDSImporter.TS_TOKEN, BIDS_DATA_DIR).create_dataset_subjects()
        bids_spatial_zip = BIDSDataBuilder(
            BIDSImporter.SPATIAL_TOKEN, BIDS_DATA_DIR).create_dataset_subjects()
        bids_surface_zip = BIDSDataBuilder(
            BIDSImporter.COORDS_TOKEN, BIDS_DATA_DIR).create_dataset_subjects()

        bids_connectivity_test_data = self.extract_data(bids_connectivity_zip)
        bids_timeseries_test_data = self.extract_data(bids_timeseries_zip)
        bids_spatial_test_data = self.extract_data(bids_spatial_zip)
        bids_surface_test_data = self.extract_data(bids_surface_zip)

        assert bids_connectivity_test_data == self.bids_connectivity_data
        assert bids_timeseries_test_data == self.bids_timeseries_data
        assert bids_spatial_test_data == self.bids_spatial_data
        assert bids_surface_test_data == self.bids_surface_data


    def extract_data(self, path):
        self.storage_interface = StorageInterface()
        temp_path = self.get_storage_path(path)
        files_list = self.storage_interface.unpack_zip(path, temp_path)
        self.storage_interface.remove_folder(temp_path)
        # self.storage_interface.remove_files(path)
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

