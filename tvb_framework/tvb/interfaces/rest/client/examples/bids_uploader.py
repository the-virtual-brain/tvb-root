import os
import json

from tvb.adapters.uploaders.csv_connectivity_importer import CSVConnectivityImporterModel
from tvb.adapters.uploaders.csv_connectivity_importer import CSVDelimiterOptionsEnum

from tvb.adapters.analyzers.bct_adapters import BaseBCTModel
from tvb.adapters.analyzers.bct_degree_adapters import Degree
from tvb.adapters.uploaders.region_mapping_importer import RegionMappingImporterModel, RegionMappingImporter
from tvb.adapters.uploaders.zip_connectivity_importer import ZIPConnectivityImporterModel, ZIPConnectivityImporter
from tvb.adapters.uploaders.zip_surface_importer import ZIPSurfaceImporterModel, ZIPSurfaceImporter
from tvb.basic.logger.builder import get_logger
from tvb.datatypes.surfaces import SurfaceTypesEnum
from tvb.interfaces.rest.client.examples.utils import compute_tvb_data_path, monitor_operation, compute_rest_url
from tvb.interfaces.rest.client.tvb_client import TVBClient
from tvb.adapters.uploaders.csv_connectivity_importer import CSVConnectivityImporter, CSVConnectivityImporterForm
from tvb.adapters.uploaders.bids_importer import BIDSImporter, BIDSImporterModel, BIDSUploadDataTypeOptionsEnum



root_dir = 'BIDS_DEMO_DATSET - Copy'

sub_dir = root_dir + '/sub-01'

algs_dir = os.listdir(sub_dir)

model_mapper = {
    'coord': None,
    'net' : CSVConnectivityImporterModel(),
    'spatial': None,
    'ts': None
}

delimiter_mapper = {
    ' ': CSVDelimiterOptionsEnum.SPACE
}

def filter(arr, sub_str):
    return [file_name for file_name in arr if sub_str in file_name]


print(os.listdir(sub_dir))



def upload_bids_data(tvb_client_instance):
    projects_of_user = tvb_client_instance.get_project_list()
    assert len(projects_of_user) > 0

    project_gid = projects_of_user[0].gid

    print(project_gid)
    print(projects_of_user[0].name)

    model = BIDSImporterModel()

    model.uploaded = "C:/Users/upadh/Documents/GitHub/tvb-root/tvb_framework/tvb/interfaces/rest/client/examples/BIDS_DEMO_DATA_SET.zip"
    model.bids_file_upload_type = BIDSUploadDataTypeOptionsEnum.FUNCTIONAL_CONNECTIVITY

    operation_gid = tvb_client_instance.launch_operation(project_gid, BIDSImporter,
                                                         model)
    monitor_operation(tvb_client_instance, operation_gid)

    print("Get the result of import...")
    connectivity_dto = tvb_client_instance.get_operation_results(operation_gid)[0]

    print(connectivity_dto)




if __name__ == '__main__':
    tvb_client = TVBClient(compute_rest_url())

    tvb_client.browser_login()
    upload_bids_data(tvb_client)