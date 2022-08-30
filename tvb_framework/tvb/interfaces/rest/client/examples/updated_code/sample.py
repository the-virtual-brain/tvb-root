from tvb.basic.neotraits.api import TVBEnum
from tvb.interfaces.rest.client.examples.updated_code.bids_data_builder import BIDSDataBuilder
from tvb.interfaces.rest.client.examples.updated_code.bids_dir_monitor import BIDSDirWatcher

class BIDSUploadDataTypeOptionsEnum(TVBEnum):
    BIDS = 'bids'
    CONNECTIVITY = 'net'
    SURFACE = 'coord'
    FUNCTIONAL_CONNECTIVITY = 'spatial'
    TIME_SERIES = 'ts'

BIDS_UPLOAD_CONTENT = BIDSUploadDataTypeOptionsEnum.TIME_SERIES
BIDS_DIR = "C:/Users/upadh/Documents/GitHub/tvb-root/tvb_framework/tvb/interfaces/rest/client/examples/BIDS_DEMO_DATSET - Copy"

def build_bids_dataset():
    # A sample code to how to build BIDS dataset for each datatype using BIDSDataBuilder

    bids_data_builder = BIDSDataBuilder(BIDS_UPLOAD_CONTENT, BIDS_DIR)   
    print(bids_data_builder.create_dataset_subjects())
 

def monitor_dir():
    # A sample code to how to monitor a directory and build BIDS dataset whenever new files are 
    # added using BIDSDirWatcher

    # Set IMPORT_DATA_IN_TVB to True to enable importing dataset into TVB
    bids_dir_watcher = BIDSDirWatcher(
        DIRECTORY_TO_WATCH = BIDS_DIR,
        UPLOAD_TRIGGER_INTERVAL = 20,
        IMPORT_DATA_IN_TVB = False
    )
    bids_dir_watcher.init_watcher()


if __name__ == '__main__':

    monitor_dir()

    #build_bids_dataset()
