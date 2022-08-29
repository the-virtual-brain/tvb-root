import time
import os
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from tvb.interfaces.rest.client.examples.updated_code.bids_data_builder import BIDSDataBuilder
from tvb.interfaces.rest.client.examples.utils import compute_tvb_data_path, monitor_operation, compute_rest_url
from tvb.interfaces.rest.client.tvb_client import TVBClient
from tvb.adapters.uploaders.bids_importer import BIDSImporter, BIDSImporterModel

from threading import Thread

from tvb.basic.logger.builder import get_logger

logger = get_logger(__name__)

SUBJECT_PREFIX = 'sub'

class BIDSDirWatcher:
    def __init__(self, DIRECTORY_TO_WATCH = None, UPLOAD_TRIGGER_INTERVAL = 20, IMPORT_DATA_IN_TVB = False, TVB_PROJECT_ID = None):
        self.added_files = []
        self.DIRECTORY_TO_WATCH = DIRECTORY_TO_WATCH
        self.UPLOAD_TRIGGER_INTERVAL = UPLOAD_TRIGGER_INTERVAL
        self.IMPORT_DATA_IN_TVB = IMPORT_DATA_IN_TVB
        self.TVB_PROJECT_ID = TVB_PROJECT_ID
    
    def check_data(self):
        if self.DIRECTORY_TO_WATCH is None:
            logger.info("Provided directory to monitor is None")
            return False

    def watchdog_thread(self):
        patterns = ["*.json"]
        ignore_directories = True
        case_sensitive = True
        my_event_handler = PatternMatchingEventHandler(patterns = patterns, ignore_directories = ignore_directories, case_sensitive = case_sensitive)
        my_event_handler.on_created = self.handle_file_added
        go_recursively = True
        my_observer = Observer()
        my_observer.schedule(my_event_handler, self.DIRECTORY_TO_WATCH, recursive = go_recursively)
        my_observer.start()
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt as e:
            my_observer.stop()
            my_observer.join()
    
    def uploader_thread(self):
        try:
            while True:
                time.sleep(self.UPLOAD_TRIGGER_INTERVAL)
                # uploading files currently present in the queue
                self.create_bids_dataset(added_files = self.added_files[:])
                # emptying the queue after uploading them
                self.added_files = []
        except KeyboardInterrupt as e:
            return
    
    def init_watcher(self):
        watchdog_thread = Thread(target = self.watchdog_thread)
        upload_file_thread = Thread(target = self.uploader_thread)
        
        print("Starting watchdog thread...")
        watchdog_thread.start()
        
        print("Starting file uploader thread...")
        upload_file_thread.start()

        if self.IMPORT_DATA_IN_TVB:
            self.tvb_client = TVBClient(compute_rest_url())
            self.tvb_client.browser_login()

    def handle_file_added(self, event):
        print(f"File created! {event.src_path}")
        self.added_files.append(os.path.normpath(event.src_path))
        print("Current queue length {}".format(len(self.added_files)))

    def create_bids_dataset(self, added_files):
        print("Creating BIDS dataset, with {} as intial json files".format(len(added_files)))
        added_files = set(added_files)
        print(added_files)
        
        uploading_files = []
        for path in added_files:
            if self.change_outside_sub_dir(path):
                continue
            uploading_files.append(path)
        
        print("json files to upload")
        print(uploading_files)
        
        if len(uploading_files) == 0: return

        subs_divided_paths = {}

        bids_dir_name = os.path.split(os.path.normpath(self.DIRECTORY_TO_WATCH))[1]
        
        for f in uploading_files:
            path_ar = f.split(bids_dir_name)[1].split(os.sep)
            print(path_ar)
            for i, j in enumerate(path_ar):
                print("{}{}".format(i,j))
                if j.startswith(SUBJECT_PREFIX):
                    if subs_divided_paths.get(j) is None:
                        subs_divided_paths[j] = [f]
                    else:
                        subs_divided_paths[j].append(f)
                    break
        
        print("subject wise pathss")
        print(subs_divided_paths)

        bids_data_builder = BIDSDataBuilder(bids_root_dir = self.DIRECTORY_TO_WATCH, init_json_files = subs_divided_paths)
        bids_zip_file = bids_data_builder.create_dataset_json_files()
        print(bids_zip_file)

        if self.IMPORT_DATA_IN_TVB:
            self.upload_to_tvb(bids_zip_file)

    def upload_to_tvb(self, file_path):
        
        projects_of_user = self.tvb_client.get_project_list()

        if len(projects_of_user) == 0 :
            return
        
        if self.TVB_PROJECT_ID is None:
            self.TVB_PROJECT_ID = projects_of_user[0].gid
        
        print(self.TVB_PROJECT_ID)
        print(projects_of_user[0].name)
        model = BIDSImporterModel()
        model.uploaded = file_path
        operation_gid = self.tvb_client.launch_operation(self.TVB_PROJECT_ID, BIDSImporter, model)
        monitor_operation(self.tvb_client, operation_gid)
        print("Get the result of import...")
        res = self.tvb_client.get_operation_results(operation_gid)
        print(res)
     
    def change_outside_sub_dir(self, file_path):
        print(file_path)
        print(os.path.abspath(file_path))
        print(os.path.normpath(file_path))
        bids_dir_name = os.path.split(os.path.normpath(self.DIRECTORY_TO_WATCH))[1]
        print(bids_dir_name)
        path_ar = file_path.split(bids_dir_name)
        print(path_ar)
        sub_dir = path_ar[1]
        sub_name = os.path.split(sub_dir)[0]
        print(sub_name)
        print(os.path.split(sub_dir))
        print(sub_dir.split(sub_name))
        print(os.path.split(sub_dir.split(sub_name)[0]))
        try:
            if SUBJECT_PREFIX in sub_name:
                sub_dir_ar = sub_dir.split(os.sep)
                for i, j in enumerate(sub_dir_ar):
                    print("{}{}".format(i,j))
                    if j.startswith(SUBJECT_PREFIX):
                        if sub_dir_ar[i+1] in ['ts', 'net', 'coord', 'spatial']:
                            return False
                        break
        except Exception as e:
            print(e)
            logger.err("Exception occurred in checking if added file is in sub directory")
        return True
    
