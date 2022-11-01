import json
import os

from tvb.adapters.creators.pipeline_creator import IPPipelineCreatorModel
from tvb.basic.profile import TvbProfile
from tvb.core.services.backend_clients.hpc_client_base import HPCClientBase
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
    os.environ[HPCClientBase.TVB_BIN_ENV_KEY] = "/Users/bvalean/WORK/tvb-root/tvb_bin"
    os.environ[HPCClientBase.CSCS_PROJECT] = "ich012"
    os.environ.setdefault(HPCClientBase.CSCS_LOGIN_TOKEN_ENV_KEY,
                          '')
    TvbProfile.current.hpc.HPC_COMPUTE_SITE = 'DAINT-CSCS'

    hpc_pipeline_client = HPCPipelineClient()
    jobs_list = hpc_pipeline_client._launch_job_with_pyunicore(op, os.environ[HPCClientBase.CSCS_LOGIN_TOKEN_ENV_KEY])
    assert len(jobs_list) == 2
