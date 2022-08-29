
from asyncio.log import logger
import time
from tvb.interfaces.rest.client.examples.updated_code.bids_data_builder import BIDSDataBuilder
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from threading import Thread
import os

from tvb.interfaces.rest.client.examples.updated_code.bids_dir_monitor import BIDSDirWatcher

DIRECTORY_TO_WATCH = "C:/Users/upadh/Documents/GitHub/tvb-root/tvb_framework/tvb/interfaces/rest/client/examples/BIDS_DATASET_MONITOR_DIR/"

SUBJECT_PREFIX = 'sub'

UPLOAD_TRIGGER_INTERVAL = 20

# queue for maintaing all new files added till we trigger the upload in UPLOAD_TRIGGER_INTERVAL seconds
added_files = []

def handle_file_added(event):
    print(f"File created! {event.src_path}")
    added_files.append(os.path.normpath(event.src_path))
    print("Current queue length {}".format(len(added_files)))

def initWatchDog():
    global added_files
    patterns = ["*.json"]
    ignore_patterns = None
    ignore_directories = True
    case_sensitive = True
    my_event_handler = PatternMatchingEventHandler(patterns = patterns, ignore_directories = ignore_directories, case_sensitive = case_sensitive)
    my_event_handler.on_created = handle_file_added
    go_recursively = True
    my_observer = Observer()
    my_observer.schedule(my_event_handler, DIRECTORY_TO_WATCH, recursive=go_recursively)
    my_observer.start()
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt as e:
        my_observer.stop()
        my_observer.join()
  

def initFileUpload():
    global added_files
    try:
        while True:
            time.sleep(UPLOAD_TRIGGER_INTERVAL)
            # uploading files currently present in the queue
            uploadFiles(added_files = added_files[:])
            # emptying the queue after uploading them
            added_files = []
    except KeyboardInterrupt as e:
        return

def change_outside_sub_dir(file_path):
    print(file_path)
    print(os.path.abspath(file_path))
    print(os.path.normpath(file_path))
    bids_dir_name = os.path.split(os.path.normpath(DIRECTORY_TO_WATCH))[1]
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




def uploadFiles(added_files):
    print("Uploading files now.... {}".format(len(added_files)))
    added_files = set(added_files)
    print(added_files)
    
    uploading_files = []
    for path in added_files:
        if change_outside_sub_dir(path):
            continue
        uploading_files.append(path)
    
    print("json files to upload")
    print(uploading_files)
    
    if len(uploading_files) == 0: return

    subs_divided_paths = {}

    bids_dir_name = os.path.split(os.path.normpath(DIRECTORY_TO_WATCH))[1]
    

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

    bids_data_builder = BIDSDataBuilder(bids_root_dir = DIRECTORY_TO_WATCH, init_json_files = subs_divided_paths)

    print(bids_data_builder.create_dataset_json_files())



if __name__ == "__main__":
    # watchdog_thread = Thread(target = initWatchDog)
    # upload_file_thread = Thread(target = initFileUpload)
    
    # print("Starting watchdog thread...")
    # watchdog_thread.start()
    
    # print("Starting file uploader thread...")
    # upload_file_thread.start()

    bids_dir_watcher = BIDSDirWatcher(
        DIRECTORY_TO_WATCH = DIRECTORY_TO_WATCH,
        UPLOAD_TRIGGER_INTERVAL = UPLOAD_TRIGGER_INTERVAL
    )
    bids_dir_watcher.init_watcher()


# few cases:

# 1. when user adds single file
# 2. when user adds more than one new file 
# 3. when user updates subject folders
# 4. when user updates outside subject folders

# 5. to upload only when subject folders are updated
