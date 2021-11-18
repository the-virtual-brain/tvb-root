import json
import os

from tvb.adapters.creators.pipeline_creator import IPPipelineCreator
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
    pipeline_data_zip = os.path.join(storage_path, IPPipelineCreator.PIPELINE_DATASET_FILE)

    temporary_folder = create_temp_folder()
    # storage_interface.write_zip_folder(pipeline_data_zip, temporary_folder)
    storage_interface.write_zip_folder(pipeline_data_zip, '/Users/bvalean/Downloads/Demo_data_pipeline_CON03')

    args_file = os.path.join(storage_path, "args.json")
    with open(args_file, "w") as outfile:
        json.dump({"dummy_key": "dummy_value"}, outfile)

    # Add correct path to tvb_bin folder and a valid ebrains auth token
    os.environ[HPCClient.TVB_BIN_ENV_KEY] = "/Users/bvalean/WORK/tvb-root/tvb_bin"
    os.environ[HPCClient.CSCS_PROJECT] = "ich012"
    os.environ.setdefault(HPCClient.CSCS_LOGIN_TOKEN_ENV_KEY,
                          'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJfNkZVSHFaSDNIRmVhS0pEZDhXcUx6LWFlZ3kzYXFodVNJZ1RXaTA1U2k0In0.eyJleHAiOjE2MzczMTM0MDIsImlhdCI6MTYzNzA0NzExMywiYXV0aF90aW1lIjoxNjM2NzA4NjAyLCJqdGkiOiJiYzQ4MjNlNi1kM2VkLTRmZWUtYTg1YS0yOTAzNjZkYTQzYTMiLCJpc3MiOiJodHRwczovL2lhbS5lYnJhaW5zLmV1L2F1dGgvcmVhbG1zL2hicCIsImF1ZCI6WyJyZWFsbS1tYW5hZ2VtZW50IiwianVweXRlcmh1YiIsImp1cHl0ZXJodWItanNjIiwieHdpa2kiLCJ0ZWFtIiwiZ3JvdXAiXSwic3ViIjoiMjZhNTBmYzgtZGQ4OC00NDFlLWFiNzItZTBmYTE3ZjNjZmI3IiwidHlwIjoiQmVhcmVyIiwiYXpwIjoidHZiLXdlYiIsIm5vbmNlIjoiYjZlZDk3OTYtMGQ1YS00NWQ1LThjYzEtZGYyYmYyM2RlNGE0Iiwic2Vzc2lvbl9zdGF0ZSI6ImU2OTljZWJmLTUwMGUtNDVhNS05MzMwLTQ0MDFlMWU5OWQwOCIsImFjciI6IjAiLCJhbGxvd2VkLW9yaWdpbnMiOlsiaHR0cDovL2xvY2FsaG9zdDo4MDgwIiwiaHR0cHM6Ly90dmItdGVzdC5hcHBzLmhicC5ldSIsImh0dHA6Ly90dmItZ3VpLXJvdXRlLXR2Yi5hcHBzLWRldi5oYnAuZXUiLCJodHRwczovL3RoZXZpcnR1YWxicmFpbi5hcHBzLmpzYy5oYnAuZXUiLCJodHRwczovL3RoZXZpcnR1YWxicmFpbi5hcHBzLWRldi5oYnAuZXUiLCJodHRwczovL3RoZXZpcnR1YWxicmFpbi5hcHBzLmhicC5ldSJdLCJzY29wZSI6InByb2ZpbGUgcm9sZXMgZW1haWwgb3BlbmlkIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsIm5hbWUiOiJWYWxlYW4gQm9nZGFuIiwibWl0cmVpZC1zdWIiOiIzMDg1OTYiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJidmFsZWFuIiwiZ2l2ZW5fbmFtZSI6IlZhbGVhbiIsImZhbWlseV9uYW1lIjoiQm9nZGFuIiwiZW1haWwiOiJib2dkYW4udmFsZWFuQGNvZGVtYXJ0LnJvIn0.ROAvKK-eFF1CkZj8E8mi6fYRv2ZY1zBOLDbBFr0Qj1MPdmoKn6noDzdV6bE0HTH0EjJchS-8nh1GCkZCHaxTqsIaPoYEsGVg75vs0CLHZwuYZ-59_6xvDzqY8MV1j2vv-Q4RUs9fDeR4NPrNqOuFnSPNDaD-q37YURrqHgBVVSZZH8hpTlUFUFUj2auz1fCzcamwA8hGow8K0I6ZHrMIwCBWO8D0klQho0dqtkws8bHODprT_OphYLyYXwc9ES6YGbCxu8t0kxBj7h5JvuzHr0DtqI0pfO04_El3SJOwbi3eJflLadDMoVzrUcmPGY63Q_-CKKbk7J_JvhmO8yKZGA')
    TvbProfile.current.hpc.HPC_COMPUTE_SITE = 'DAINT-CSCS'

    hpc_pipeline_client = HPCPipelineClient()
    job = hpc_pipeline_client._launch_job_with_pyunicore(op, os.environ[HPCClient.CSCS_LOGIN_TOKEN_ENV_KEY])
    assert job is not None
