import os
import json
import uuid
from zipfile import ZipFile

from tvb.adapters.uploaders.csv_connectivity_importer import CSVConnectivityImporterModel
from tvb.adapters.uploaders.csv_connectivity_importer import CSVDelimiterOptionsEnum
from tvb.basic.logger.builder import get_logger
from tvb.datatypes.surfaces import SurfaceTypesEnum
from tvb.interfaces.rest.client.examples.utils import compute_tvb_data_path, monitor_operation, compute_rest_url
from tvb.interfaces.rest.client.tvb_client import TVBClient
from tvb.adapters.uploaders.bids_importer import BIDSImporter, BIDSImporterModel, BIDSUploadDataTypeOptionsEnum

logger = get_logger(__name__)

root_dir = 'BIDS_DEMO_DATSET - Copy'
sub_dir = root_dir + '/sub-01'
algs_dir = os.listdir(sub_dir)
model_mapper = {
    'coord': None,
    'net' : CSVConnectivityImporterModel(),
    'spatial': None,
    'ts': None
}
possible_paths = {
    BIDSUploadDataTypeOptionsEnum.SURFACE: [],
    BIDSUploadDataTypeOptionsEnum.CONNECTIVITY: ["CoordsRows", "CoordsColumns"],
    BIDSUploadDataTypeOptionsEnum.FUNCTIONAL_CONNECTIVITY: ["CoordsRows", "CoordsColumns", "ModelEq", "ModelParam", "Network"],
    BIDSUploadDataTypeOptionsEnum.TIME_SERIES: ["CoordsRows", "CoordsColumns", "ModelEq", "ModelParam", "Network"],
    "common_paths": ["CoordsRows", "CoordsColumns", "ModelEq", "ModelParam", "Network"]
}
#possible_paths["common_paths"] = possible_paths[BIDSUploadDataTypeOptionsEnum.SURFACE] + possible_paths[BIDSUploadDataTypeOptionsEnum.CONNECTIVITY] + possible_paths[BIDSUploadDataTypeOptionsEnum.FUNCTIONAL_CONNECTIVITY] + possible_paths[BIDSUploadDataTypeOptionsEnum.TIME_SERIES]
delimiter_mapper = {
    ' ': CSVDelimiterOptionsEnum.SPACE
}

BIDS_UPLOAD_CONTENT = BIDSUploadDataTypeOptionsEnum.TIME_SERIES
BIDS_DIR_NAME = "BIDS_DEMO_DATSET - Copy"
BIDS_DIR = "C:/Users/upadh/Documents/GitHub/tvb-root/tvb_framework/tvb/interfaces/rest/client/examples/BIDS_DEMO_DATSET - Copy"
SUBJECT_PREFIX = 'sub'


def upload_bids_data(tvb_client_instance, path):
    projects_of_user = tvb_client_instance.get_project_list()
    assert len(projects_of_user) > 0
    project_gid = projects_of_user[0].gid
    print(project_gid)
    print(projects_of_user[0].name)
    model = BIDSImporterModel()
    model.uploaded = path
    model.bids_file_upload_type = BIDSUploadDataTypeOptionsEnum.BIDS
    operation_gid = tvb_client_instance.launch_operation(project_gid, BIDSImporter,
                                                         model)
    monitor_operation(tvb_client_instance, operation_gid)
    print("Get the result of import...")
    connectivity_dto = tvb_client_instance.get_operation_results(operation_gid)[0]
    print(connectivity_dto)

def get_abs_path(bids_root_dir, sub, path):
    if os.path.exists(os.path.abspath(path)):
        return os.path.abspath(path)
    if '../..' in path:
        path1 = path.replace('../..', bids_root_dir)
        if os.path.exists(path1):
            return path1
    elif '..' in path:
        path1 = path.replace('..', bids_root_dir + '/' + sub)
        path2 = path.replace('..', bids_root_dir)
        if os.path.exists(path1):
            return path1
        if os.path.exists(path2):
            return path2
    return path


def create_archive(files_list, zip_name, base_dir):
    base_dir_name = os.path.dirname(base_dir)
    with ZipFile(zip_name, 'w') as myzip:
        for file_name in files_list:
            myzip.write(file_name, arcname=file_name.split(base_dir_name)[1])


def create_bids_dataset(bids_data_to_import, bids_root_dir, bids_file_name):
    
    logger.info("Creating BIDS dataset for {}".format(bids_data_to_import))
    bids_file_base_dir = os.path.abspath(os.path.join(BIDS_DIR, os.pardir))
    temp_bids_dir_name = bids_file_name + '-' + str(uuid.uuid4()).split("-")[4]
    temp_bids_dir = bids_file_base_dir + '/' + temp_bids_dir_name
    temp_bids_zip_dir = temp_bids_dir + '.zip'
    
    files = os.listdir(bids_root_dir)
    subject_folders = []

    # First we find subject parent folders
    for file_name in files:
        if os.path.basename(file_name).startswith(SUBJECT_PREFIX) and os.path.isdir(bids_root_dir + '/' + file_name):
            subject_folders.append(file_name)

    logger.info("Found {} subject folders in the".format(len(subject_folders)))
    # For each subject we read its content and sub-dirs
    for sub in subject_folders:
        sub_contents_path = bids_root_dir + '/' + sub + '/' + bids_data_to_import.value
        sub_contents = os.listdir(sub_contents_path)
        if len(sub_contents) == 0:
            continue

        # Set for keeping only unique file pahts
        # Dict for track of json files which are processed
        # queue is used for reading all other dependencies present in nested json files 
        import_dependencies_paths = set()
        json_files_processed = {}
        paths_queue = []

        # Addding path of all files present in the alg(to import) dir
        for file_name in os.listdir(sub_contents_path):
            import_dependencies_paths.add(os.path.join(sub_contents_path, file_name))
        
        json_files = [os.path.join(sub_contents_path, file_name) for file_name in sub_contents if '.json' in os.path.splitext(file_name)[1]]

        print(json_files)

        # add json files content in abs path in import_dependencies_paths
        
        # Reading all json files in the alg(to import) dir and adding all dependencies path found in the json in the set
        for file in json_files:
            try:
                file_info = json.load(open(get_abs_path(bids_root_dir, sub, file)))
                
                for possible_path_key in possible_paths[bids_data_to_import]:
                    if isinstance(file_info[possible_path_key], list):
                        for path in file_info[possible_path_key]:
                            import_dependencies_paths.add(get_abs_path(bids_root_dir, sub, path))
                    else:
                        import_dependencies_paths.add(get_abs_path(bids_root_dir, sub, file_info[possible_path_key]))
            except Exception as e:       
                print(e.__class__.__name__)
                logger.error("Exception occurred in reading json files: {}".format(e.__class__.__name__))

        for dependency_path in import_dependencies_paths: 
            if os.path.splitext(dependency_path)[1] != '.json':
                continue
            paths_queue.append(dependency_path)
            json_files_processed[dependency_path] = False
            
        # Now, reading all dependencies json files and adding dependency again if found
        while len(paths_queue)!=0:
            path = paths_queue.pop(0)
            if json_files_processed[path] is False:
                json_files_processed[path] = True
                try:
                    json_data = json.load(open(get_abs_path(bids_root_dir, sub, path)))
                    
                    for possible_path_key in possible_paths["common_paths"]:
                        if possible_path_key in json_data.keys() and isinstance(json_data[possible_path_key], list):
                            for path1 in json_data[possible_path_key]:
                                computed_path = get_abs_path(bids_root_dir, sub, path1)
                                import_dependencies_paths.add(computed_path)
                                paths_queue.append(computed_path)
                                if computed_path not in json_files_processed:
                                    json_files_processed[computed_path] = False
                        else:
                            computed_path = get_abs_path(bids_root_dir, sub, json_data[possible_path_key])
                            import_dependencies_paths.add(computed_path)
                            paths_queue.append(computed_path)
                            if computed_path not in json_files_processed:
                                json_files_processed[computed_path] = False
                            
                except Exception as e:
                    print(e.__class__.__name__)
                    logger.error("Exception occurred in reading json files: {}".format(e.__class__.__name__)) 
        

        data_files = set()

        for p in import_dependencies_paths:
            path_ar = p.split('/')
            file_dir = '/'.join(path_ar[0:len(path_ar)-1])
            file_name = path_ar[len(path_ar)-1]
            files_to_copy = [fn for fn in os.listdir(file_dir) if os.path.splitext(file_name)[0] == os.path.splitext(fn)[0]] 
            for f in files_to_copy:
                data_files.add(file_dir + '/' + f)
            
        for p in data_files:
            import_dependencies_paths.add(p)
        
        # Creating zip archive all paths present in the set
        logger.info("Creating ZIP archive of {} files".format(len(import_dependencies_paths)))
        create_archive(import_dependencies_paths, temp_bids_zip_dir, bids_root_dir)

        return temp_bids_zip_dir


if __name__ == '__main__':
    
    zip_file_dir = create_bids_dataset(BIDS_UPLOAD_CONTENT, BIDS_DIR, BIDS_DIR_NAME)
    logger.info("Created ZIP file successfully at {} ".format(zip_file_dir))
 
    # tvb_client = TVBClient(compute_rest_url())
    # tvb_client.browser_login()
    # upload_bids_data(tvb_client, zip_file_dir)


