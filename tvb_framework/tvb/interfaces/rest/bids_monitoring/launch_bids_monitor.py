import sys
from tvb.interfaces.rest.bids_monitoring.bids_data_builder import BIDSDataBuilder
from tvb.interfaces.rest.bids_monitoring.bids_dir_monitor import BIDSDirWatcher
from tvb.adapters.uploaders.bids_importer import BIDSImporter


BIDS_UPLOAD_CONTENT = BIDSImporter.NET_TOKEN
BIDS_DIR = "C:/Users/upadh/Documents/GitHub/tvb-root/tvb_framework/tvb/interfaces/rest/client/examples/BIDS_DEMO_DATSET - Copy"


def get_bids_dir():
    if len(sys.argv) > 0:
        for arg in sys.argv:
            if arg.startswith('--bids-dir'):
                BIDS_DIR = arg.split('=')[1]
    return BIDS_DIR


def build_bids_dataset():
    # A sample code to how to build BIDS dataset for each datatype using BIDSDataBuilder

    bids_data_builder = BIDSDataBuilder(BIDS_UPLOAD_CONTENT, BIDS_DIR)
    print(bids_data_builder.create_dataset_subjects())


def monitor_dir():
    # A sample code to how to monitor a directory using BIDSDirWatcher
    # and build BIDS dataset whenever new files are added

    # Set IMPORT_DATA_IN_TVB to True to enable importing dataset into TVB
    bids_dir_watcher = BIDSDirWatcher(
        DIRECTORY_TO_WATCH=BIDS_DIR,
        UPLOAD_TRIGGER_INTERVAL=20,
        IMPORT_DATA_IN_TVB=True
    )
    bids_dir_watcher.init_watcher()


if __name__ == '__main__':

    # If bids dir is provided as command line args
    BIDS_DIR = get_bids_dir()

    monitor_dir()

    # build_bids_dataset()
