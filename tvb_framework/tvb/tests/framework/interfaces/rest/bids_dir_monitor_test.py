from fileinput import filename
import pytest
from tvb.interfaces.rest.bids_monitoring.bids_dir_monitor import BIDSDirWatcher
import os
import time
import shutil

from tvb.tests.framework.interfaces.rest.bids_data_builder_test import TestBIDSDataBuilder

BIDS_DATA_DIR = "C:/Users/upadh/Documents/GitHub/tvb-root/tvb_framework/tvb/interfaces/rest/client/examples/BIDS_DEMO_DATSET - Copy"
BIDS_DATA_FOUND = True


@pytest.mark.skipif(not BIDS_DATA_FOUND, reason="Older or incomplete tvb_data")
class TestBIDSDirWatcher:

    def setup_test_data(self):
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
        self.bids_timeseries_data.sort()

    def test_monitor_bids_dataset(self):
        """
        Dataset used in this test has 1 Connectivity, 1 Surface, 1 TimeSeries and 1 Functional Connectivity.
        Test that adding a new file to the TS directory will create a dataset with correct files.
        First TS directory will be moved to the parent directory of subject folder and 
        then we'll move it back to the subject directory. 
        This way we can check if BIDSDirWatcher is building only when files are added inside subject directories.

        """
        self.setup_test_data()
        bids_dir_watcher = BIDSDirWatcher(
            DIRECTORY_TO_WATCH=BIDS_DATA_DIR,
            UPLOAD_TRIGGER_INTERVAL=1,
            IMPORT_DATA_IN_TVB=False
        )
        bids_dir_watcher.init_watcher()

        test_folder = os.path.join(BIDS_DATA_DIR, 'sub-01', 'ts')
        # moving TS folder into root
        shutil.move(test_folder, BIDS_DATA_DIR)

        # moving back to subject folder
        shutil.move(os.path.join(BIDS_DATA_DIR, 'ts'), test_folder)

        while True:
            time.sleep(0.5)
            if bids_dir_watcher.current_dataset_loc is not "":
                bids_dir_watcher.end_watcher()
                break
        
        filenames_list = TestBIDSDataBuilder().extract_data(bids_dir_watcher.current_dataset_loc)

        assert filenames_list == self.bids_timeseries_data

        