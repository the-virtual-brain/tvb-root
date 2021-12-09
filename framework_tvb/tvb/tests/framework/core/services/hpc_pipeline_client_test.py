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
        json.dump({
            "mri_data": "pipeline_dataset.zip",
            "participant_label": "sub-CON03",
            "session_label": "ses-postop",
            "task-label": "rest",
            "parcellation": "destrieux",
            "nr_of_cpus": 1,
            "estimated_time": '10:00:00',
            "mrtrix": True,
            "mrtrix_parameters": {
                "output_verbosity": "2",
                "analysis_level": "preproc",
                "analysis_level_config": {
                    "streamlines": 5,
                    "template_reg": "ants"
                }
            },
            "fmriprep": True,
            "fmriprep_parameters": {
                "analysis_level": "participant",
                "analysis_level_config": {
                    "skip_bids_validation": False,
                    "anat-only": True,
                    "fs-no-reconall": True
                }
            },
            "freesurfer": False,
            "tvbconverter": True
        }

            , outfile)

    # Add correct path to tvb_bin folder and a valid ebrains auth token
    os.environ[HPCClient.TVB_BIN_ENV_KEY] = "/Users/bvalean/WORK/tvb-root/tvb_bin"
    os.environ[HPCClient.CSCS_PROJECT] = "ich012"
    os.environ.setdefault(HPCClient.CSCS_LOGIN_TOKEN_ENV_KEY,
                          'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJfNkZVSHFaSDNIRmVhS0pEZDhXcUx6LWFlZ3kzYXFodVNJZ1RXaTA1U2k0In0.eyJleHAiOjE2Mzk1NTY0OTAsImlhdCI6MTYzODk1MTY5MSwiYXV0aF90aW1lIjoxNjM4OTUxNjkwLCJqdGkiOiJkNzY1MzBjZS04YzlhLTRlMzYtOTMwMS1mNzBiMjZmYTA1MWYiLCJpc3MiOiJodHRwczovL2lhbS5lYnJhaW5zLmV1L2F1dGgvcmVhbG1zL2hicCIsImF1ZCI6WyJyZWFsbS1tYW5hZ2VtZW50IiwianVweXRlcmh1YiIsImp1cHl0ZXJodWItanNjIiwieHdpa2kiLCJ0ZWFtIiwiZ3JvdXAiXSwic3ViIjoiMjZhNTBmYzgtZGQ4OC00NDFlLWFiNzItZTBmYTE3ZjNjZmI3IiwidHlwIjoiQmVhcmVyIiwiYXpwIjoidHZiLXdlYiIsIm5vbmNlIjoiYmExM2Y0Y2QtZmU0MS00YWM1LThhZmEtYmYzNTNlYzdmYmQ1Iiwic2Vzc2lvbl9zdGF0ZSI6ImUxNDJiMGRmLWJkMjYtNDZjMy1iZGQ2LWJmMzdkZDE3NDE4ZSIsImFjciI6IjAiLCJhbGxvd2VkLW9yaWdpbnMiOlsiaHR0cHM6Ly9waXBlbGluZS10dmIuYXBwcy5oYnAuZXUiLCJodHRwOi8vbG9jYWxob3N0OjgwODAiLCJodHRwczovL3R2Yi10ZXN0LmFwcHMuaGJwLmV1IiwiaHR0cHM6Ly90dmItcGlwZWxpbmUuYXBwcy5oYnAuZXUiLCJodHRwczovL3R2Yi1ocGMuYXBwcy5oYnAuZXUiLCJodHRwOi8vdHZiLWd1aS1yb3V0ZS10dmIuYXBwcy1kZXYuaGJwLmV1IiwiaHR0cHM6Ly90aGV2aXJ0dWFsYnJhaW4uYXBwcy5qc2MuaGJwLmV1IiwiaHR0cHM6Ly90aGV2aXJ0dWFsYnJhaW4uYXBwcy1kZXYuaGJwLmV1IiwiaHR0cHM6Ly90aGV2aXJ0dWFsYnJhaW4uYXBwcy5oYnAuZXUiXSwic2NvcGUiOiJwcm9maWxlIHJvbGVzIGVtYWlsIG9wZW5pZCIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJuYW1lIjoiVmFsZWFuIEJvZ2RhbiIsIm1pdHJlaWQtc3ViIjoiMzA4NTk2IiwicHJlZmVycmVkX3VzZXJuYW1lIjoiYnZhbGVhbiIsImdpdmVuX25hbWUiOiJWYWxlYW4iLCJmYW1pbHlfbmFtZSI6IkJvZ2RhbiIsImVtYWlsIjoiYm9nZGFuLnZhbGVhbkBjb2RlbWFydC5ybyJ9.YsZ-GrLsOR6IvWUpa2QTsgwTMbZcwD5EiZtVYZ4ZcAS1AbAbpN3v5-kd-P2tQTnQjM5TCBy_-8OyhbGCiHvjCH0X1JLkaReJ8LcfbOyWkITVUpDUum-gA6Ce1zXlS88YOs5caeIZ7PnrKbVkPvD61pc2o1eBCj1OQXOd6BdeQgbH66sn5RhYLj4iYJjHk_3k9xhAOKSFmKn0kwEEhL7Ml5mOZcRSChrv4-FpRRK6bvYHvtQElNSG3G0XTZvrEEJSarGhZUV3SzbY7tW1UwLzainCpM7VJ2DrUTWAJRULyRnyvZnaaOe1rmxBu575O134xwKt6zjyIEDK9TlFsag4cA')
    TvbProfile.current.hpc.HPC_COMPUTE_SITE = 'DAINT-CSCS'

    hpc_pipeline_client = HPCPipelineClient()
    jobs_list = hpc_pipeline_client._launch_job_with_pyunicore(op, os.environ[HPCClient.CSCS_LOGIN_TOKEN_ENV_KEY])
    assert len(jobs_list) == 2
