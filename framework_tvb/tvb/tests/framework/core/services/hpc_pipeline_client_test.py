import json
import os

from tvb.adapters.creators.pipeline_creator import IPPipelineCreatorModel
from tvb.basic.profile import TvbProfile
from tvb.core.services.backend_clients.hpc_client import HPCClient
from tvb.core.services.backend_clients.hpc_pipeline_client import HPCPipelineClient
from tvb.interfaces.rest.commons.files_helper import create_temp_folder
from tvb.storage.storage_interface import StorageInterface
from tvb.tests.framework.core.factory import TestFactory


# @pytest.mark.skip
def test_hpc_pipeline(operation_factory):
    test_user = TestFactory.create_user()
    test_project = TestFactory.create_project(test_user)

    op = operation_factory(test_user=test_user, test_project=test_project)
    storage_interface = StorageInterface()
    storage_path = storage_interface.get_project_folder(op.project.name,
                                                        str(op.id))
    pipeline_data_zip = os.path.join(storage_path, IPPipelineCreatorModel.PIPELINE_DATASET_FILE)

    temporary_folder = create_temp_folder()
    # storage_interface.write_zip_folder(pipeline_data_zip, temporary_folder)
    storage_interface.write_zip_folder(pipeline_data_zip, '/Users/bvalean/Downloads/Demo_data_pipeline_CON03')

    args_file = os.path.join(storage_path, IPPipelineCreatorModel.PIPELINE_CONFIG_FILE)
    with open(args_file, "w") as outfile:
        json.dump({"dummy_key": "dummy_value"}, outfile)

    # Add correct path to tvb_bin folder and a valid ebrains auth token
    os.environ[HPCClient.TVB_BIN_ENV_KEY] = "/Users/bvalean/WORK/tvb-root/tvb_bin"
    os.environ[HPCClient.CSCS_PROJECT] = "ich012"
    os.environ.setdefault(HPCClient.CSCS_LOGIN_TOKEN_ENV_KEY,
                          'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJfNkZVSHFaSDNIRmVhS0pEZDhXcUx6LWFlZ3kzYXFodVNJZ1RXaTA1U2k0In0.eyJleHAiOjE2Mzc5MTQxODIsImlhdCI6MTYzNzMwOTM4NCwiYXV0aF90aW1lIjoxNjM3MzA5MzgyLCJqdGkiOiI1MzQwMWFiZi1jMWYzLTQ5OGQtODBjNC1mMGEwOTAzYzNhMjkiLCJpc3MiOiJodHRwczovL2lhbS5lYnJhaW5zLmV1L2F1dGgvcmVhbG1zL2hicCIsImF1ZCI6WyJyZWFsbS1tYW5hZ2VtZW50IiwianVweXRlcmh1YiIsImp1cHl0ZXJodWItanNjIiwieHdpa2kiLCJ0ZWFtIiwiZ3JvdXAiXSwic3ViIjoiMjZhNTBmYzgtZGQ4OC00NDFlLWFiNzItZTBmYTE3ZjNjZmI3IiwidHlwIjoiQmVhcmVyIiwiYXpwIjoidHZiLXdlYiIsIm5vbmNlIjoiZDY1ODcyZTEtNTJhYi00M2RiLTgyMGEtYzAxNmM5YjZmOGI3Iiwic2Vzc2lvbl9zdGF0ZSI6IjIxMDYxZDdmLTgwNTgtNGEwYy1hY2Y3LTc5ZmQ4Zjg2Y2RjZSIsImFjciI6IjAiLCJhbGxvd2VkLW9yaWdpbnMiOlsiaHR0cDovL2xvY2FsaG9zdDo4MDgwIiwiaHR0cHM6Ly90dmItdGVzdC5hcHBzLmhicC5ldSIsImh0dHA6Ly90dmItZ3VpLXJvdXRlLXR2Yi5hcHBzLWRldi5oYnAuZXUiLCJodHRwczovL3RoZXZpcnR1YWxicmFpbi5hcHBzLmpzYy5oYnAuZXUiLCJodHRwczovL3RoZXZpcnR1YWxicmFpbi5hcHBzLWRldi5oYnAuZXUiLCJodHRwczovL3RoZXZpcnR1YWxicmFpbi5hcHBzLmhicC5ldSJdLCJzY29wZSI6InByb2ZpbGUgcm9sZXMgZW1haWwgb3BlbmlkIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsIm5hbWUiOiJWYWxlYW4gQm9nZGFuIiwibWl0cmVpZC1zdWIiOiIzMDg1OTYiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJidmFsZWFuIiwiZ2l2ZW5fbmFtZSI6IlZhbGVhbiIsImZhbWlseV9uYW1lIjoiQm9nZGFuIiwiZW1haWwiOiJib2dkYW4udmFsZWFuQGNvZGVtYXJ0LnJvIn0.OAobjT7Sblz3hSvA_hiOxmOpIkSnA2YYGIdxqEe1ppCDE8mHaW8N8Czpl4WQImpsPrf3-Mc6nEarF1AU9FZGYN3ngeC9Jt-oY9CpB0nDXtLzyxG83eC5vxuyXglz6uUeKssm3zuLpxw1BWVPJwrRT3KgsSnMggGHNiYSgckLBAFRg23CJfwwVuX76w_oHSd4VnjLyS9O38cP3oE4FwcCtJVjJN4scOQQpjhWW3zevhDEgiFl7uNs8cOLl_cqfMe9jNj-874ToM3BCc84_zeYhtq3063Zs2NSArsDlBW0GaLf9ttmbQOpYyj-GS-j3iHEkZlgQo759yF9rEf_G-aWcA')
    TvbProfile.current.hpc.HPC_COMPUTE_SITE = 'DAINT-CSCS'

    hpc_pipeline_client = HPCPipelineClient()
    job = hpc_pipeline_client._launch_job_with_pyunicore(op, os.environ[HPCClient.CSCS_LOGIN_TOKEN_ENV_KEY])
    assert job is not None
